#!/usr/bin/env python3
"""
atlas_encoder.py — Layer 2: the encoder as a versioned, checksum-bound projection module.

The ONLY GPU touch and the ONLY mutable-shared resource in the serving stack.  Its job is narrow:
reads -> normalized vectors, identically to how the INDEX was built.  Everything principled here follows
from one hard invariant and two hazards we measured:

  INVARIANT — vectors must MATCH the index.  The index (embed_shard.py) encodes with a specific load
  sequence (v9 base + v10.9 overlay), `encode_angular_only`, and L2-normalize.  If L2 encodes even
  slightly differently, query vectors miss reference vectors and retrieval silently returns garbage
  (this is how the 1kbp Annoy index shipped broken).  So `encode()` here is byte-for-byte the same math
  as `embed_shard.emb()` — not a reimplementation, the same path.  A parity test asserts it on real data.

  HAZARD 1 — the checksum must not lie.  v10.9 = base + overlay, so `encoder_id` is the hash of the
  EFFECTIVE (composed) weights, NOT the checkpoint filename.  It gets stamped into the index's
  meta.encoder_id at build time; serving asserts its live encoder hashes to the same value before binding
  the port.  This turns the index<->encoder coupling that burned us all session into a one-line invariant.

  HAZARD 2 — the encoder is the sole concurrency point.  A triage tool that never abstains will accept a
  read flood; the 16GB card OOMs if we let it.  So GPU access is serialized behind one lock and chunked to
  a bounded batch — the backpressure lives HERE, at the one place that can hurt.

Split of responsibility (each independently testable):
  Tokenizer.tokenize(dna, ntok) -> token row   — PURE (no torch); the QUERY path (reference was pre-tokenized).
  effective_encoder_id(state_dict) -> sha256    — PURE; the checksum over composed weights.
  EncoderModule.encode(token_rows) -> vectors   — the shared GPU math (lazy torch); byte-identical to the index.

The read POLICY (tile a long sequence into reads, route scale->ntok) is Layer 2.5 (read_policy below) —
kept explicit so the measured deep-ensemble/scale-routing levers don't get smuggled into encode or search.
"""
from __future__ import annotations
import json, hashlib
from dataclasses import dataclass

EMB_DIM = 129
PAD_ID = 3          # [PAD] — matches embed_shard.py PAD and vocab
UNK_ID = 0          # [UNK]
DEFAULT_CHUNK = 512  # GPU batch chunk (numerically identical to any chunk size; bounds peak VRAM)


# ── Tokenizer (PURE) — the query path; exact replica of serve.py `bpe` ──────────────────────────
class Tokenizer:
    """Greedy longest-match BPE over the 4096 vocab.  The reference windows were pre-tokenized with the
    SAME vocab at build time; a query DNA read must be tokenized identically or the encoder sees a
    different distribution than it was trained/indexed on.  Non-special tokens only (specials start '[')."""
    def __init__(self, vocab_path: str):
        self.vocab = json.load(open(vocab_path))
        self.by_len = {}
        for tok, i in self.vocab.items():
            if not tok.startswith("["):                 # exclude [UNK]/[CLS]/[SEP]/[PAD]/[MASK] from matching
                self.by_len.setdefault(len(tok), {})[tok] = i
        self.max_tok_len = max(self.by_len)

    def tokenize(self, dna: str, ntok: int) -> list:
        """DNA string -> list[int] of length exactly ntok (padded with PAD_ID, truncated).  N->A, upper."""
        s = dna.upper().replace("N", "A")
        o = []
        i = 0
        while i < len(s) and len(o) < ntok:
            for L in range(min(self.max_tok_len, len(s) - i), 0, -1):
                seg = s[i:i + L]
                if L in self.by_len and seg in self.by_len[L]:
                    o.append(self.by_len[L][seg]); i += L; break
            else:
                o.append(self.vocab.get(s[i], UNK_ID)); i += 1     # single-char fallback -> its id or UNK
        while len(o) < ntok:
            o.append(PAD_ID)
        return o[:ntok]


# ── effective-weight checksum (PURE) — HAZARD 1 ────────────────────────────────────────────────
def effective_encoder_id(state_dict) -> str:
    """sha256 over the COMPOSED weights (base+overlay), so the id can't lie about what will actually run.
    Deterministic and order-independent: keys sorted, each contributes name|dtype|shape|raw-bytes.
    Accepts torch tensors OR numpy arrays OR raw bytes (so it's testable without torch)."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        v = state_dict[name]
        b, dtype, shape = _to_bytes(v)
        h.update(name.encode()); h.update(b"|"); h.update(str(dtype).encode())
        h.update(b"|"); h.update(str(tuple(shape)).encode()); h.update(b"|"); h.update(b)
    return "sha256:" + h.hexdigest()


def _to_bytes(v):
    if hasattr(v, "detach"):                     # torch tensor
        arr = v.detach().cpu().contiguous().numpy()
        return arr.tobytes(), arr.dtype, arr.shape
    if hasattr(v, "tobytes"):                    # numpy array
        return v.tobytes(), v.dtype, v.shape
    if isinstance(v, (bytes, bytearray)):        # raw bytes (test fixtures)
        return bytes(v), "bytes", (len(v),)
    raise TypeError(f"un-hashable weight type: {type(v)}")


# ── front-door validity gate ────────────────────────────────────────────────────────────────
def sequence_complexity(s: str, k: int = 4) -> float:
    """Normalized k-mer entropy of the ACGT content — the low-complexity signal.  Real genomic DNA sits
    ~0.83-0.99 (measured, n=250: 1%ile 0.831); homopolymers/simple-repeats/degenerate input sit < 0.61.
    Returns 0.0 for empty / no-ACGT (e.g. all-N).  Cheap, encoder-independent — runs before any encode so
    garbage that would collapse near a manifold point (reading as 'confidently typical') never gets there."""
    import math
    from collections import Counter
    t = "".join(c for c in s.upper() if c in "ACGT")
    if len(t) < k:
        return 0.0
    kk = [t[i:i + k] for i in range(len(t) - k + 1)]
    c = Counter(kk); tot = len(kk)
    h = -sum((n / tot) * math.log2(n / tot) for n in c.values())
    hmax = math.log2(min(4 ** k, tot))
    return h / hmax if hmax > 0 else 0.0


# ── read policy (Layer 2.5) — the measured tiling/routing lever, kept explicit ─────────────────
@dataclass
class ReadPolicy:
    """Tile one long sequence into reads, or accept a read list; route bp -> ntok against the index scale.
    This is where the deep-ensemble / scale-routing levers live (NOT inside encode or search) so they stay
    visible and tunable.  Mirrors serve.py to_read_tokens; the ensemble-N budget is applied by the caller."""
    index_bp: int
    index_ntok: int
    max_reads: int = 24                 # ensemble cap: subsample <= this many evenly-spaced tiles (bounds cost)
    min_complexity: float = 0.0         # front-door: drop reads with k-mer entropy < this (0 = off; deploy 0.70)

    def ntok_for(self, bp: int) -> int:
        return max(60, round(bp * self.index_ntok / self.index_bp))

    def reads_from(self, sequence: str | None = None, reads: list | None = None,
                   read_bp: int | None = None):
        bp = read_bp or self.index_bp
        frags = []
        if reads:
            for s in reads:
                s = "".join(c for c in s.upper() if c in "ACGTN")
                if len(s) >= bp:      frags.append(s[:bp])
                elif len(s) >= bp // 2: frags.append(s)          # short read: use as-is
        elif sequence:
            s = "".join(c for c in sequence.upper() if c in "ACGTN")
            for st in range(0, len(s) - bp + 1, bp):             # non-overlapping tiling
                frags.append(s[st:st + bp])
            if not frags and len(s) >= bp // 2:
                frags.append(s)
            # bounded deep-ensemble: a whole genome tiles into hundreds of windows; vote over an EVENLY-SPACED
            # sample (representative, not first-N) so ensembling is the default without unbounded encode cost.
            if len(frags) > self.max_reads:
                idx = sorted({round(i * (len(frags) - 1) / (self.max_reads - 1)) for i in range(self.max_reads)})
                frags = [frags[i] for i in idx]
        # front-door validity gate: drop low-complexity fragments (all-N / homopolymer / simple-repeat) BEFORE
        # they reach the encoder. Applies to both the reads and the sequence path; when it empties the set, the
        # caller's no-frags path returns maximally-novel evidence (garbage -> honest abstention, not false-typical).
        if self.min_complexity > 0.0:
            frags = [f for f in frags if sequence_complexity(f) >= self.min_complexity]
        return frags, bp


# ── the encoder module (GPU; lazy torch) — the shared math + HAZARD 2 guard ─────────────────────
class EncoderModule:
    """Holds the composed model + its effective encoder_id.  `encode` is byte-identical to
    embed_shard.emb().  GPU access serialized behind one lock (the sole concurrency point) and chunked
    to bound peak VRAM (the read-flood backpressure)."""
    def __init__(self, model, encoder_id: str, device: str = "cuda",
                 chunk: int = DEFAULT_CHUNK):
        self.model = model
        self.encoder_id = encoder_id
        self.device = device
        self.chunk = chunk
        import threading
        self._lock = threading.Lock()

    def encode(self, token_rows):
        """list[list[int]] (each length ntok) -> (N, EMB_DIM) float32, L2-normalized.  MATCHES the index."""
        import numpy as np, torch
        rows = list(token_rows)
        if not rows:
            return np.zeros((0, EMB_DIM), "float32")
        out = []
        with self._lock:                                     # single encode queue (HAZARD 2)
            for i in range(0, len(rows), self.chunk):        # chunk bounds peak VRAM; numerics unchanged
                bt = torch.from_numpy(np.asarray(rows[i:i + self.chunk], dtype=np.int64)).to(self.device)
                with torch.no_grad():
                    z = self.model.encode_angular_only(bt)
                z = z.float(); z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-10)
                out.append(z.cpu().numpy().astype("float32"))
        return np.concatenate(out, 0)


def load_effective_encoder(v9_path: str, overlay_path: str | None, device: str = "cuda",
                           model_loader=None):
    """Load v9 base, apply the v10.9 overlay by the EXACT embed_shard sequence, return (EncoderModule).
    The load order is the invariant — any deviation changes the vectors.  encoder_id hashes the result."""
    import torch
    if model_loader is None:
        from model_v15_5 import load_v15_5_model
        model_loader = load_v15_5_model
    m = model_loader(v9_path, device=device)
    if overlay_path:
        s = torch.load(overlay_path, map_location="cpu", weights_only=False); s = s.get("model", s)
        cur = m.state_dict()
        matched = {k: v for k, v in s.items() if k in cur and cur[k].shape == v.shape}
        assert len(matched) >= 150, f"overlay load looks wrong: only {len(matched)} keys matched"
        m.load_state_dict(matched, strict=False)
    m.eval()
    eid = effective_encoder_id(m.state_dict())               # hash of COMPOSED weights (HAZARD 1)
    return EncoderModule(m, eid, device=device)
