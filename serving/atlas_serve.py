#!/usr/bin/env python3
"""
atlas_serve.py — Layer 4: the thin, verify-gated orchestrator.

Holds NO heavy logic — the scoring is L3 (atlas_evidence), the encode math is L2 (atlas_encoder), the map
is the index artifact.  L4's whole job is to WIRE them and to REFUSE to serve a map it can't verify or
match.  The flow is one line per stage:

    reads --(L2 tokenize)--> tokens --(L2 encode)--> qvecs --(L1 search)--> neighbors
          --(rerank seam)--> neighbors --(L3 build_evidence)--> Evidence --(L3 decide @ operating point)--> JSON

Two gates fire at LOAD (never per-request — the invariants are about the artifact, checked once):
  * ENCODER-MATCH.  live encoder.encoder_id must equal meta.encoder_id (the composed-weight checksum from
    L2).  A mismatch means the encoder would project queries into a space the index wasn't built in — the
    exact silent-corruption class.  Present -> assert; absent (legacy artifact) -> loud warn, don't block.
  * VERIFY.  Stage-C-lite self-retrieval (mirrors build_pipeline/verify_index.py): sample indexed genomes,
    re-encode a window, assert domain-match + self-retrieval above threshold.  Fail -> refuse to bind.

Hot-swap: SERVE_INDEX points at a `current` symlink over versioned dirs.  build finishes -> verifies ->
flip the symlink -> reload().  reload() loads+gates a NEW service and swaps it in ONLY if the gates pass,
so a bad artifact can never replace a good running one.  Reversible: flip the symlink back.

The AtlasService.place() orchestration is PURE of torch/faiss (they live behind the injected encoder/index
objects), so it is unit-testable with fakes — see test_atlas_serve.py.  FastAPI is imported lazily in
build_app() so this module imports without a web stack.
"""
import os, json, time  # NOTE: no `from __future__ import annotations` — FastAPI must resolve endpoint
# param annotations (Request) at runtime; stringized annotations + locally-imported fastapi break that.
from atlas_evidence import build_evidence, rerank, fuse, Evidence, RankReadout

# scale-name -> (read_bp, ntok), matching embed_shard.py; lets L4 read BOTH meta schemas
# (interim index carries bp/ntok; full-tree build carries `scale`).
SCALE_TABLE = {"300bp": (300, 60), "1kb": (1000, 200), "2kb": (2000, 400),
               "5kb": (5000, 1000), "20kb": (20000, 4096)}


def meta_bp_ntok(meta: dict):
    """Derive (bp, ntok) from either meta schema."""
    if "bp" in meta and "ntok" in meta:
        return int(meta["bp"]), int(meta["ntok"])
    scale = meta.get("scale")
    if scale in SCALE_TABLE:
        return SCALE_TABLE[scale]
    raise ValueError(f"meta has neither bp/ntok nor a known scale: {meta}")


# ── the pure orchestration (dependency-injected; no torch/faiss here) ───────────────────────────
class AtlasService:
    """Wires L2 + L1 + L3.  Stateless per request; the only state is the loaded artifact + encoder."""
    def __init__(self, tokenizer, encoder, index, taxonomy, read_policy, calib=None, meta=None, k=15):
        self.tokenizer = tokenizer        # L2 Tokenizer
        self.encoder = encoder            # L2 EncoderModule (has .encode and .encoder_id)
        self.index = index                # L1 backend (.search(qvecs, k, exclude) -> [(gid, sim)])
        self.taxonomy = taxonomy          # gid -> {rank: taxon}
        self.read_policy = read_policy    # L2.5 ReadPolicy
        self.calib = calib                # 2D-shaped calibration (or None -> confidence == margin)
        self.meta = meta or {}
        self.k = k

    def place(self, sequence=None, reads=None, read_bp=None,
              min_confidence: float = 0.5, exclude=None) -> dict:
        """One calibrated descent.  Returns the L3 evidence payload at the requested operating point."""
        frags, bp = self.read_policy.reads_from(sequence=sequence, reads=reads, read_bp=read_bp)
        if not frags:
            # decision #8: no usable reads is EVIDENCE (maximally novel), not a 400.
            ev = build_evidence([], self.taxonomy, n_reads=0, calib=self.calib)
            return ev.as_dict(min_confidence)
        ntok = self.read_policy.ntok_for(bp)
        rows = [self.tokenizer.tokenize(f, ntok) for f in frags]
        qvecs = self.encoder.encode(rows)                          # L2 (the one GPU touch)
        neighbors = self.index.search(qvecs, self.k, exclude=exclude)   # L1
        neighbors = rerank(neighbors)                              # identity seam (L3 #6b)
        ev = build_evidence(neighbors, self.taxonomy, n_reads=len(rows),
                            k_neighborhood=self.k, calib=self.calib)
        return ev.as_dict(min_confidence)

    def health(self) -> dict:
        return {"status": "ok", "encoder_id": self.encoder.encoder_id,
                "index": self.meta, "n_taxa": len(self.taxonomy)}


# ── L2.5 router: read length -> reference tier(s) (the measured scale-routing table) ─────────────
class Router:
    """Maps a read length to which index tier(s) answer it, from the smoothness/crossover measurement:
    short reads want the richer-but-closer 5kb tier; the ~5kb crossover fuses; long/native reads want 20kb.
    `bands` is config data (routing.json), so the policy is tunable without code. `mode='fuse'` collapses to
    a single always-fuse band (never-worst, no routing logic)."""
    def __init__(self, bands):
        self.bands = bands                                       # [{"max_bp": 4000, "tiers": ["5kb"]}, ...]
    def route(self, read_bp):
        for b in self.bands:
            if b["max_bp"] is None or read_bp < b["max_bp"]: return list(b["tiers"])
        return list(self.bands[-1]["tiers"])
    @staticmethod
    def default(tier_names):
        """Length-routed SINGLE-tier bands when both 5kb+20kb are present; else single-tier passthrough.
        MEASURED (win2, n=100, 2-4kb sweep): rank-normalized cross-tier FUSION is a NET NEGATIVE — it gives
        the weaker tier equal vote and dilutes the stronger one, landing worst-or-tied at every crossover
        length on BOTH family and genus.  So route to the single best tier, no fused band.  Family crossover
        is ~3kb: below it the 5kb tier edges family (~+.03); at/above it the 20kb tier wins genus and native.
        (The fuse()/force_tiers machinery is retained for eval, just not on the default path.)"""
        if set(tier_names) >= {"5kb", "20kb"}:
            return Router([{"max_bp": 3000, "tiers": ["5kb"]},
                           {"max_bp": None, "tiers": ["20kb"]}])
        return Router([{"max_bp": None, "tiers": list(tier_names)}])


# ── the adaptive-resolution service: one encoder, two index tiers, length-routed + crossover-fused ──
class AdaptiveService:
    """Serves v10.9 as one map at two resolutions.  Per read: route by length -> encode at the tier's ntok
    -> search the tier index(es) -> (fuse rank-normalized in the crossover band) -> pool across reads -> L3.
    Emits the same Evidence payload PLUS `scale.tier_used` and a `cross_scale.agreed` DIAGNOSTIC (surfaced,
    never fed to the confidence — it's redundant with margin, proven)."""
    def __init__(self, tokenizer, encoder, tiers, taxonomy, router, read_policy, calib=None, meta=None, k=15):
        self.tokenizer = tokenizer; self.encoder = encoder
        self.tiers = tiers            # {name: {"index": FaissIndex, "ntok": int, "bp": int, "meta": dict}}
        self.taxonomy = taxonomy; self.router = router; self.read_policy = read_policy
        self.calib = calib; self.meta = meta or {}; self.k = k

    def _family_top(self, neighbors):
        ev = build_evidence(neighbors, self.taxonomy, k_neighborhood=self.k)
        fr = next((r for r in ev.ranks if r.rank == "family"), None)
        return fr.top if fr and not fr.gap else None

    def place(self, sequence=None, reads=None, read_bp=None, min_confidence=0.5, exclude=None,
              force_tiers=None) -> dict:
        bp = read_bp or 20000                                    # default tile = native 20kb
        frags, bp = self.read_policy.reads_from(sequence=sequence, reads=reads, read_bp=bp)
        if not frags:
            return build_evidence([], self.taxonomy, n_reads=0, calib=self.calib).as_dict(min_confidence)
        # force_tiers (eval only) pins the tier(s) instead of the router — for A/B'ing routing vs each single arm.
        tier_names = [t for t in force_tiers if t in self.tiers] if force_tiers else self.router.route(bp)
        per_tier_raw = {}                                        # name -> pooled raw neighbors across reads
        for name in tier_names:
            t = self.tiers[name]
            rows = [self.tokenizer.tokenize(f, t["ntok"]) for f in frags]
            qv = self.encoder.encode(rows)                       # encode at THIS tier's resolution
            per_tier_raw[name] = t["index"].search(qv, self.k, exclude=exclude)
        if len(tier_names) > 1:                                  # crossover -> fuse (rank-normalized)
            pooled = fuse(*[per_tier_raw[n] for n in tier_names])
        else:
            pooled = per_tier_raw[tier_names[0]]                 # single tier -> raw sims (validated)
        pooled = rerank(pooled)
        ev = build_evidence(pooled, self.taxonomy, n_reads=len(frags), k_neighborhood=self.k, calib=self.calib)
        out = ev.as_dict(min_confidence)
        out["scale"] = {"read_bp": bp, "tier_used": "+".join(tier_names)}
        if len(tier_names) > 1:                                  # cross-scale DIAGNOSTIC (not a confidence input)
            fams = {n: self._family_top(per_tier_raw[n]) for n in tier_names}
            out["cross_scale"] = {"agreed": len(set(fams.values())) == 1, "per_tier_family": fams}
        return out

    def health(self) -> dict:
        return {"status": "ok", "encoder_id": self.encoder.encoder_id, "mode": "adaptive",
                "tiers": {n: t["meta"] for n, t in self.tiers.items()},
                "routing": self.router.bands, "n_taxa": len(self.taxonomy)}


# ── gates (torch/faiss/numpy; run once at load) ────────────────────────────────────────────────
def assert_encoder_matches(encoder, meta: dict, strict: bool = False, log=print):
    """ENCODER-MATCH gate.  meta.encoder_id present -> must equal live id; absent -> warn (legacy)."""
    want = meta.get("encoder_id")
    if want is None:
        msg = (f"[gate] meta has NO encoder_id (legacy artifact) — cannot verify encoder<->index binding; "
               f"live id = {encoder.encoder_id}")
        if strict:
            raise RuntimeError(msg + "  (strict mode: refusing to serve unstamped artifact)")
        log("WARN " + msg)
        return False
    if want != encoder.encoder_id:
        raise RuntimeError(f"[gate] ENCODER MISMATCH: index built with {want}, live encoder is "
                           f"{encoder.encoder_id} — queries would land in the wrong space. Refusing to bind.")
    log(f"[gate] encoder-match OK ({want})")
    return True


def verify_gate(encoder, index, taxonomy, tok_dir, ntok, n=32, dom_min=0.85, self_min=0.6,
                seed=0, pad_id=3, log=print):
    """VERIFY gate — Stage-C-lite self-retrieval (mirrors build_pipeline/verify_index.py).  Returns a
    report dict; raises if below threshold (a map that can't find its own members must not serve)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    gids = index.all_gids()
    uniq = np.unique(gids)
    samp = rng.choice(uniq, size=min(n, len(uniq)), replace=False)
    dom_ok = self_ok = tot = 0
    for g in samp:
        g = int(g)
        try:
            acc = taxonomy[g]["accession"]
            tk = np.load(os.path.join(tok_dir, acc + ".npy"), mmap_mode="r")
            t = np.array(tk[0, :ntok]).astype(np.int64)
            if len(t) < ntok:
                t = np.pad(t, (0, ntok - len(t)), constant_values=pad_id)
            e = encoder.encode([t.tolist()])                       # live encoder, same math as the index
            nbrs = [gid for gid, _ in index.search(e, 5)]
            if not nbrs:
                tot += 1; continue
            dom_ok += (taxonomy[nbrs[0]].get("domain") == taxonomy[g].get("domain"))
            self_ok += (g in nbrs)
            tot += 1
        except Exception:
            continue
    dr = dom_ok / max(tot, 1); sr = self_ok / max(tot, 1)
    report = {"n": tot, "domain_match": round(dr, 3), "self_retrieval": round(sr, 3),
              "passed": bool(dr >= dom_min and sr >= self_min)}
    log(f"[gate] verify n={tot}: domain-match {dr:.2f}, self-retrieval {sr:.2f} -> "
        f"{'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise RuntimeError(f"[gate] VERIFY FAILED {report} — quarantine, refusing to bind.")
    return report


# ── L1 backend + loaders (faiss/torch; lazy) ────────────────────────────────────────────────────
class FaissIndex:
    """Layer-1 backend.  Normalized IP == cosine.  Pools neighbors across all query reads."""
    def __init__(self, index_dir):
        import faiss, numpy as np
        self.idx = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        self.gids = np.load(os.path.join(index_dir, "gids.npy"))

    def all_gids(self):
        return self.gids

    def search(self, qvecs, k, exclude=None, over=16):
        import numpy as np
        if qvecs is None or len(qvecs) == 0:
            return []
        ex = set() if exclude is None else ({int(exclude)} if isinstance(exclude, int) else set(int(x) for x in exclude))
        # over-fetch when masking: a large exclude set (e.g. a whole family for held-out eval) can cover the
        # entire top-k, so fetch k*over and keep the first k SURVIVORS per read. No exclude -> exact k (fast path).
        kf = min(k * over, self.idx.ntotal) if ex else k
        D, I = self.idx.search(np.asarray(qvecs, dtype="float32"), kf)
        out = []
        for r in range(len(qvecs)):
            kept = 0
            for j in range(kf):
                nid = I[r, j]
                if nid < 0:
                    continue
                g = int(self.gids[nid])
                if g in ex:
                    continue
                out.append((g, float(D[r, j])))
                kept += 1
                if kept >= k:
                    break
        return out


def load_service(index_dir, v9_path, overlay_path, vocab_path, ftax_path, tok_dir,
                 device="cuda", k=15, strict_encoder=False, run_verify=True, log=print):
    """Load a self-describing index dir + the composed encoder, run BOTH gates, return an AtlasService.
    Raises on gate failure (never binds a bad/mismatched artifact)."""
    import json as _json, numpy as np
    from atlas_encoder import Tokenizer, ReadPolicy, load_effective_encoder

    meta = _json.load(open(os.path.join(index_dir, "meta.json")))
    bp, ntok = meta_bp_ntok(meta)
    tokenizer = Tokenizer(vocab_path)
    encoder = load_effective_encoder(v9_path, overlay_path, device=device)
    log(f"[load] encoder {encoder.encoder_id} on {device}; index meta={meta}")

    assert_encoder_matches(encoder, meta, strict=strict_encoder, log=log)

    index = FaissIndex(index_dir)
    tax_list = _json.load(open(ftax_path))
    taxonomy = {g: tax_list[g] for g in range(len(tax_list))}

    calib_path = os.path.join(index_dir, "calibration.json")
    calib = _json.load(open(calib_path)) if os.path.exists(calib_path) else None
    log(f"[load] calibration: {'loaded' if calib else 'none (confidence == raw margin)'}")

    if run_verify:
        rep = verify_gate(encoder, index, taxonomy, tok_dir, ntok, log=log)
        meta = {**meta, "verify": rep}

    read_policy = ReadPolicy(index_bp=bp, index_ntok=ntok)
    return AtlasService(tokenizer, encoder, index, taxonomy, read_policy, calib=calib, meta=meta, k=k)


def load_adaptive_service(tier_dirs, v9_path, overlay_path, vocab_path, ftax_path, tok_dir,
                          device="cuda", k=15, strict_encoder=False, run_verify=True, routing=None, log=print):
    """Load MULTIPLE self-describing index tiers behind ONE encoder, gate EACH (encoder-match + verify),
    build the length router, return an AdaptiveService.  tier_dirs = {"5kb": dir, "20kb": dir}."""
    import json as _json
    from atlas_encoder import Tokenizer, ReadPolicy, load_effective_encoder
    encoder = load_effective_encoder(v9_path, overlay_path, device=device)   # ONE encoder for all tiers
    log(f"[load] encoder {encoder.encoder_id} on {device}")
    tokenizer = Tokenizer(vocab_path)
    tax_list = _json.load(open(ftax_path)); taxonomy = {g: tax_list[g] for g in range(len(tax_list))}
    tiers, calib = {}, None
    for name, d in tier_dirs.items():
        meta = _json.load(open(os.path.join(d, "meta.json"))); bp, ntok = meta_bp_ntok(meta)
        assert_encoder_matches(encoder, meta, strict=strict_encoder, log=log)   # gate EACH tier
        idx = FaissIndex(d)
        if run_verify:
            meta = {**meta, "verify": verify_gate(encoder, idx, taxonomy, tok_dir, ntok, log=log)}
        tiers[name] = {"index": idx, "ntok": ntok, "bp": bp, "meta": meta, "dir": d}
        cp = os.path.join(d, "calibration.json")
        if calib is None and os.path.exists(cp): calib = _json.load(open(cp))   # margin calib (tier-agnostic)
        log(f"[load] tier '{name}': {meta['scale']} bp={bp} ntok={ntok} ({idx.idx.ntotal:,} vec)")
    router = Router(routing["bands"]) if routing else Router.default(tiers.keys())
    log(f"[load] router bands: {router.bands}; calibration: {'loaded' if calib else 'none'}")
    read_policy = ReadPolicy(index_bp=20000, index_ntok=4096)                # tiling only; ntok comes from tier
    meta = {"mode": "adaptive", "tiers": {n: t["meta"]["scale"] for n, t in tiers.items()}}
    return AdaptiveService(tokenizer, encoder, tiers, taxonomy, router, read_policy, calib=calib, meta=meta, k=k)


# ── FastAPI app (lazy import) ───────────────────────────────────────────────────────────────────
def build_app(service_holder, reload_fn=None):
    """service_holder: a 1-element list holding the live AtlasService (so /admin/reload can swap it).
    reload_fn: () -> AtlasService, loads+gates a fresh service from the (possibly re-pointed) index dir."""
    from fastapi import FastAPI, HTTPException, Header, Request

    app = FastAPI(title="BiosphereAtlas Placement (L4)", version="2.0")
    API_KEY = os.environ.get("SERVE_API_KEY", "")
    # default operating point (the novelty dial). 0.5 keeps legacy behavior; deploys set SERVE_MIN_CONFIDENCE=0.7
    # to halve novel-FAMILY false-commit (.28->.15, measured) at a known-recall cost. Per-request `min_confidence`
    # still overrides. NOTE: measured at FAMILY level; it applies per-rank globally (genus curve not yet validated).
    DEFAULT_MIN_CONF = float(os.environ.get("SERVE_MIN_CONFIDENCE", "0.5"))

    @app.get("/health")
    def health():
        return service_holder[0].health()

    @app.post("/place")
    async def place(request: Request, x_api_key: str = Header(default="")):
        # parse the JSON body directly — robust to pydantic-model annotation resolution under
        # `from __future__ import annotations` (a body model would be mis-read as a query param).
        if API_KEY and x_api_key != API_KEY:
            raise HTTPException(401, "bad api key")
        b = await request.json()
        t0 = time.time()
        # `exclude` (optional): gids to drop from the neighborhood — held-out evaluation (mask self /
        # a whole family) and the "novel relatives excluding my own assembly" query. Never required.
        svc = service_holder[0]
        kw = {"exclude": b.get("exclude")}
        if b.get("force_tiers") is not None and hasattr(svc, "tiers"):
            kw["force_tiers"] = b["force_tiers"]                 # eval-only tier pin (adaptive service only)
        out = svc.place(sequence=b.get("sequence"), reads=b.get("reads"),
                        read_bp=b.get("read_bp"), min_confidence=b.get("min_confidence", DEFAULT_MIN_CONF), **kw)
        out["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return out

    @app.post("/admin/reload")
    def reload(x_api_key: str = Header(default="")):
        """Hot-swap: load+GATE a fresh service from the current index dir; swap in ONLY if gates pass.
        A failing artifact raises inside reload_fn and the OLD service keeps serving (never degraded)."""
        if API_KEY and x_api_key != API_KEY:
            raise HTTPException(401, "bad api key")
        if reload_fn is None:
            raise HTTPException(501, "reload not configured")
        try:
            fresh = reload_fn()                       # load_service raises on gate fail -> old stays live
        except Exception as e:
            raise HTTPException(503, f"reload rejected (gates failed): {e}")
        old = service_holder[0]; service_holder[0] = fresh
        return {"status": "reloaded", "was": old.encoder.encoder_id, "now": fresh.encoder.encoder_id,
                "index": fresh.meta}

    return app


# ── uvicorn entry point (systemd calls `atlas_serve:app`) ──────────────────────────────────────
def create_app():
    """Build the live app from env. Two modes:
      SINGLE   — SERVE_INDEX points at a `current` symlink over versioned dirs; /admin/reload re-reads it.
      ADAPTIVE — SERVE_INDEX_5KB + SERVE_INDEX_20KB both set -> length-routed dual-tier AdaptiveService.
    ADAPTIVE takes precedence when both tier vars are present. Optional SERVE_ROUTING=path.json overrides bands."""
    import json as _json
    common = dict(
        v9_path=os.environ.get("V9_PATH", "/home/rohit/v9_best.pt"),
        overlay_path=os.environ.get("OVERLAY_PATH", "/home/rohit/v10_curvature_field/v10_9_encoder.pt"),
        vocab_path=os.environ.get("VOCAB_PATH", "/home/rohit/biosphere_inference/bpe_vocab.json"),
        ftax_path=os.environ.get("FTAX_PATH", "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"),
        tok_dir=os.environ.get("TOK_DIR", "/zfs_raid/SentryBio/tokenized_4096"),
        device=os.environ.get("SERVE_DEVICE", "cuda"),
        strict_encoder=os.environ.get("STRICT_ENCODER", "0") == "1",
    )
    d5, d20 = os.environ.get("SERVE_INDEX_5KB"), os.environ.get("SERVE_INDEX_20KB")
    if d5 and d20:
        routing = _json.load(open(os.environ["SERVE_ROUTING"])) if os.environ.get("SERVE_ROUTING") else None
        cfg = dict(tier_dirs={"5kb": d5, "20kb": d20}, routing=routing, **common)
        holder = [load_adaptive_service(**cfg)]
        return build_app(holder, reload_fn=lambda: load_adaptive_service(**cfg))
    cfg = dict(index_dir=os.environ["SERVE_INDEX"], **common)
    holder = [load_service(**cfg)]
    return build_app(holder, reload_fn=lambda: load_service(**cfg))


# built at import ONLY when an index is configured (so tests import this module without loading the model/GPU)
_HAS_INDEX = os.environ.get("SERVE_INDEX") or (os.environ.get("SERVE_INDEX_5KB") and os.environ.get("SERVE_INDEX_20KB"))
app = create_app() if _HAS_INDEX else None
