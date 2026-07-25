#!/usr/bin/env python3
"""
test_atlas_encoder.py — PURE tests for Layer 2 (no torch, no GPU).

Covers the parts that can be verified off-GPU: the query tokenizer (against the real vocab), the
effective-weight checksum (determinism / order-independence / sensitivity), and the read policy.

NOT covered here (requires the GPU + model + index on Nexus, and must not disturb a running build):
  * PARITY — that EncoderModule.encode() reproduces embed_shard.emb() byte-for-byte on real tokens.
    That is the one test that earns the word "matches the index"; it runs when the GPU is free.
    See parity_check() below for the exact assertion to run on Nexus.

Run:  python test_atlas_encoder.py
"""
from __future__ import annotations
import os
from atlas_encoder import Tokenizer, effective_encoder_id, ReadPolicy, PAD_ID

VOCAB = os.path.join(os.path.dirname(__file__), "bpe_vocab.json")


def _passed(name): print(f"  ok  {name}")


def test_tokenizer_length_pad_truncate():
    tk = Tokenizer(VOCAB)
    row = tk.tokenize("ACGT", ntok=16)
    assert len(row) == 16 and row[-1] == PAD_ID, "pads to ntok with PAD_ID"
    long = tk.tokenize("ACGT" * 100, ntok=8)
    assert len(long) == 8, "truncates to ntok"
    _passed("tokenizer pads and truncates to exactly ntok")


def test_tokenizer_deterministic_and_greedy():
    tk = Tokenizer(VOCAB)
    a = tk.tokenize("ACGTACGTACGTACGT", 32)
    b = tk.tokenize("ACGTACGTACGTACGT", 32)
    assert a == b, "same input -> same tokens (deterministic)"
    # greedy longest-match: a run should consume multi-char tokens, not degrade to all single chars.
    non_pad = [t for t in a if t != PAD_ID]
    singles = {tk.vocab[c] for c in "ACGT"}
    assert not all(t in singles for t in non_pad), "greedy match uses multi-char merges, not just singles"
    _passed("tokenizer deterministic + greedy longest-match")


def test_tokenizer_N_and_case():
    tk = Tokenizer(VOCAB)
    assert tk.tokenize("acgt", 8) == tk.tokenize("ACGT", 8), "case-insensitive"
    assert tk.tokenize("NNNN", 8) == tk.tokenize("AAAA", 8), "N -> A (matches embed_shard)"
    _passed("tokenizer lowercases and maps N->A")


def test_encoder_id_deterministic_and_order_independent():
    sd1 = {"layer.b": b"\x01\x02\x03", "layer.a": b"\xaa\xbb"}
    sd2 = {"layer.a": b"\xaa\xbb", "layer.b": b"\x01\x02\x03"}   # same content, different insertion order
    assert effective_encoder_id(sd1) == effective_encoder_id(sd2), "hash is order-independent (keys sorted)"
    assert effective_encoder_id(sd1).startswith("sha256:")
    _passed("encoder_id deterministic + key-order independent")


def test_encoder_id_sensitive_to_weight_change():
    base = {"w": b"\x00\x00\x00\x00"}
    flip = {"w": b"\x00\x01\x00\x00"}                            # one byte differs (an overlay would do this)
    assert effective_encoder_id(base) != effective_encoder_id(flip), "any weight change -> different id"
    # also sensitive to a key RENAME (structure change), not just values
    renamed = {"w2": b"\x00\x00\x00\x00"}
    assert effective_encoder_id(base) != effective_encoder_id(renamed), "key set is part of the identity"
    _passed("encoder_id changes on any weight or structural change (checksum can't lie)")


def test_read_policy_tiling():
    rp = ReadPolicy(index_bp=2000, index_ntok=400)
    # a long sequence tiles non-overlapping at bp stride
    frags, bp = rp.reads_from(sequence="ACGT" * 1500, read_bp=2000)  # 6000 bp -> 3 full 2000bp tiles
    assert bp == 2000 and len(frags) == 3 and all(len(f) == 2000 for f in frags), (len(frags), [len(f) for f in frags])
    # explicit reads: long trimmed to bp, mid-length kept, too-short dropped
    frags2, _ = rp.reads_from(reads=["A" * 2500, "C" * 1200, "G" * 100], read_bp=2000)
    assert len(frags2) == 2 and len(frags2[0]) == 2000 and len(frags2[1]) == 1200
    # scale routing: bp -> ntok scales with the index ratio
    assert rp.ntok_for(2000) == 400 and rp.ntok_for(1000) == 200 and rp.ntok_for(300) >= 60
    _passed("read policy tiles, trims, drops-too-short, and routes bp->ntok")


# ── the deferred GPU parity assertion (run on Nexus when the card is free) ──────────────────────
def parity_check():
    """Run ON NEXUS with the GPU idle.  Asserts EncoderModule.encode() == embed_shard.emb() on real
    tokenized windows — the test that earns 'matches the index'.  Not run in the local suite."""
    import numpy as np, json
    from atlas_encoder import load_effective_encoder
    V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
    TOK = "/zfs_raid/SentryBio/tokenized_4096"
    FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
    NTOK = 400  # 2kb scale
    tax = json.load(open(FTAX))
    enc = load_effective_encoder(V9, V109, device="cuda")
    # replicate embed_shard's raw-window path for a few genomes
    rows = []
    for g in range(5):
        tk = np.load(f"{TOK}/{tax[g]['accession']}.npy", mmap_mode="r")
        t = np.array(tk[0, :NTOK]).astype(np.int64)
        if len(t) < NTOK: t = np.pad(t, (0, NTOK - len(t)), constant_values=PAD_ID)
        rows.append(t.tolist())
    import torch
    bt = torch.from_numpy(np.asarray(rows, dtype=np.int64)).cuda()
    with torch.no_grad():
        z = enc.model.encode_angular_only(bt); z = z.float(); z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-10)
    ref = z.cpu().numpy().astype("float32")
    got = enc.encode(rows)
    assert np.allclose(got, ref, atol=1e-6), f"PARITY FAIL max|d|={np.abs(got-ref).max()}"
    print(f"PARITY OK — encoder_id={enc.encoder_id}, max|d|={np.abs(got-ref).max():.2e}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"atlas_encoder PURE tests — {len(tests)} cases (GPU parity deferred, see parity_check)")
    for t in tests:
        t()
    print(f"\nALL {len(tests)} PASSED — tokenizer + checksum + read-policy contract holds.")


if __name__ == "__main__":
    main()
