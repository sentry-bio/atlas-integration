# Serving Deploy Runbook — L4 `atlas_serve`

The principled serving stack for the BiosphereAtlas placement model. Four layers, each a separate file with
its own tests; the seams carry the state, the logic is pure, the artifact is portable, and **nothing serves a
map it can't verify or match.**

| Layer | File | Role | Tests |
|------|------|------|-------|
| L1 — the map | the index dir (`index.faiss` + `gids.npy` + `taxonomy` + `meta.json` [+ `calibration.json`]) | versioned, self-describing artifact — **the product** | verify-gate |
| L2 — encoder | `atlas_encoder.py` | reads→vectors (the one GPU touch), checksum-bound, queue-guarded | `test_atlas_encoder.py` (6) + `parity_check` (GPU) |
| L3 — evidence | `atlas_evidence.py` | pure `(neighbors)→Evidence`; the frozen `scorer_v1`; the dial | `test_atlas_evidence.py` (9) |
| L4 — serve | `atlas_serve.py` | thin verify-gated orchestrator + FastAPI | `test_atlas_serve.py` (5) |

Flow per request: `reads → L2 tokenize → L2 encode → L1 search → rerank(seam) → L3 build_evidence → L3 decide@op → JSON`.

---

## The self-describing index artifact (L1)

A deploy IS an index directory. It carries everything needed to serve it honestly, so it can't be mis-paired:

```
serve_index/<version>/
  index.faiss        # vectors (FlatIP small, IVFFlat >2M)
  gids.npy           # vector i -> gid  (asserted len-equal at build)
  meta.json          # encoder_id, scale/bp/ntok, dim, index_type, verify{...}   ← stamped by verify_index.py
  calibration.json   # (margin,support)->p(correct) per rank   [optional; absent -> confidence == raw margin]
```

Taxonomy is the shared 234K catalog (`genome_taxonomy_full.json`), referenced by gid — the gid↔taxonomy
correspondence is what verify proves. `meta.encoder_id` is the **sha256 of the composed (v9+overlay) weights**,
written by `build_pipeline/verify_index.py` on a passing build — the same hash the serve side computes, so a
mismatch is impossible to miss.

---

## First install (Nexus)

```bash
# 1. serving code lives in /home/rohit/build_pipeline (single source: verify_index + serve import the SAME hash)
#    already deployed: atlas_encoder.py atlas_evidence.py atlas_serve.py bpe_vocab.json

# 2. uvicorn/fastapi in the torch venv — WITHOUT breaking the numpy pin (uvicorn[standard] pulls numpy 2.x,
#    which breaks the faiss ABI). Install bare, then re-assert the pin:
/home/rohit/biosphere-atlas-venv/bin/pip install fastapi uvicorn        # NOT uvicorn[standard]
/home/rohit/biosphere-atlas-venv/bin/pip install 'numpy==1.26.4'        # re-pin, always
/home/rohit/biosphere-atlas-venv/bin/python -c "import numpy,faiss,torch; print(numpy.__version__)"  # must be 1.26.4

# 3. point `current` at the artifact to serve
ln -sfn /zfs_raid/SentryBio/serve_index_v109/2kb /zfs_raid/SentryBio/serve_index/current

# 4. systemd user service (survives logout via linger — matches the cloudflared pattern)
cp atlas-serve.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger rohit
systemctl --user enable --now atlas-serve
journalctl --user -u atlas-serve -f      # watch the two gates fire at startup
```

Health / smoke:
```bash
curl -s localhost:8091/health | jq          # encoder_id, index meta, verify report
curl -s localhost:8091/place -H 'content-type: application/json' \
  -d '{"reads":["ACGT...~2kb..."],"read_bp":2000,"min_confidence":0.5}' | jq
```

---

## The two load-time gates (why a bad deploy can't serve)

Both run once at startup inside `load_service` (never per-request):

1. **ENCODER-MATCH** — live `encoder.encoder_id` vs `meta.encoder_id`.
   - stamped artifact + `STRICT_ENCODER=1` → **hard refuse** on mismatch (queries would land in the wrong space).
   - legacy artifact (no `encoder_id`) → **warn** and serve (set `STRICT_ENCODER=0`). Graduate to `1` after the
     first stamped full-tree build.
2. **VERIFY** — Stage-C-lite self-retrieval (samples indexed genomes, re-encodes a window, asserts
   domain-match ≥ 0.85 **and** self-retrieval ≥ 0.6). Fail → **refuse to bind**. Fast path: trusts a persisted
   `meta.verify.passed` when the encoder matches; else re-runs live.

A failing artifact makes the process exit → systemd retries → it **crash-loops loudly** (visible in journal +
failing `/health`) instead of serving garbage silently. That is the design: loud failure over quiet corruption.

---

## Hot-swap deploy (atomic, reversible)

The index is immutable and versioned, so swapping is a symlink flip + a gated reload — no rebuild-in-place,
no half-states, and the **old service keeps serving until the new one passes its gates**.

```bash
# 1. build the new artifact (does not touch the running server)
cd /home/rohit/build_pipeline
BUILD_DIR=/zfs_raid/SentryBio/serve_index_v109_20kb SCALE=20kb ENCODER=v10.9 \
  SHARD_SIZE=1500 KREF=8 ./run_build.sh        # embed -> merge -> VERIFY stamps encoder_id + report

# 2. confirm it verified before pointing at it
jq '.verify, .encoder_id' /zfs_raid/SentryBio/serve_index_v109_20kb/meta.json   # verify.passed == true

# 3. flip `current` (atomic) and ask the LIVE server to reload+gate the new artifact
ln -sfn /zfs_raid/SentryBio/serve_index_v109_20kb /zfs_raid/SentryBio/serve_index/current
curl -s -X POST localhost:8091/admin/reload -H "x-api-key: $SERVE_API_KEY" | jq
#   -> {"status":"reloaded","was":<old id>,"now":<new id>,...}   on success
#   -> 503 "reload rejected (gates failed)"                      on a bad artifact — OLD service stays live
```

**Rollback** — flip the symlink back and reload; the previous artifact is untouched on disk:
```bash
ln -sfn /zfs_raid/SentryBio/serve_index_v109/2kb /zfs_raid/SentryBio/serve_index/current
curl -s -X POST localhost:8091/admin/reload -H "x-api-key: $SERVE_API_KEY" | jq
```

> Reload rebuilds the encoder too (re-hashes weights, re-runs gates). For a pure index swap on the same
> encoder that's redundant but cheap and keeps the path uniform; a restart (`systemctl --user restart
> atlas-serve`) is the equivalent heavier hammer.

---

## The dial (operating point)

`/place` takes `min_confidence` (0..1). It is **not** retraining or re-searching — it re-`decide()`s the same
evidence at a different threshold, so a caller can trace the whole P-R curve per query. Higher → precision
(the triage-precision end); lower → recall (the novel-tail-sensitivity end). Default 0.5.

---

## Operational cautions (scars, encoded)

- **numpy stays 1.26.4.** 2.x breaks the faiss ABI (intermittent segfaults). Re-assert after ANY pip install.
- **GPU contention.** The 4070 Ti SUPER also drives the workstation display; heavy compute + an active desktop
  session has hung the card. Serving is light (a few reads/request), but schedule big *builds* around desktop use.
- **The encoder is the one shared mutable resource.** GPU access is serialized behind a lock in L2 and chunked
  to bound VRAM — the read-flood backpressure lives there. Don't add a second GPU path.
- **Ports:** L4 v10.9 on `:8091`. (Legacy `serve.py`/`serve_v9_annoy.py` were the pre-refactor path — superseded.)

---

## Still shaped-but-empty (honest TODO, awaiting the full-tree index)

- `calibration.json` — 2D `(margin, support)` schema is wired; the table is a held-out fit that needs the
  20kb index. Until then confidence == raw margin.
- Novelty flag thresholds in `Evidence.novelty()` (`support < 0.5`) are placeholders → set from the known-vs-novel
  support distribution on real data.
- L2 `parity_check` (GPU) — proves `encode()` == `embed_shard.emb()` byte-for-byte; run when the card is free.
- `rerank()` — identity seam; the precision lever (#2→#1) drops in here (tighter re-score of top-K).
