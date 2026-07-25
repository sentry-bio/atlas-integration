# atlas-integration

The serving integration for **BiosphereAtlas v10.9** — a hyperbolic (Poincaré ball, κ≈1.2453)
taxonomic-placement kernel over a 234K-genome reference, served as a registration-into-a-shared-frame
API rather than a classifier.

The organizing principle: **serve the datum, not the model.** The index is the product; the encoder is a
disposable build-time chart; serving is a stateless, verify-gated read. A future encoder (v11) replaces this
one by rebuild → hot-swap, with zero serving-code change.

## Layers

- **L1 — map**: a self-describing index dir (`index.faiss` + `gids.npy` + `meta.json` + `calibration.json`),
  taxonomy shared by gid. Artifacts live on `/zfs_raid`; their sha256/encoder_id/GTDB-release are recorded in
  `docs/DATUM.md`, never committed as blobs.
- **L2 — `serving/atlas_encoder.py`**: tokenizer + `EncoderModule.encode` (byte-identical to the build path) +
  `effective_encoder_id` (sha256 of the *composed* v9+overlay weights, device-independent) + `ReadPolicy`
  (bounded deep-ensemble tiling, scale-routing).
- **L3 — `serving/atlas_evidence.py`**: pure `(neighbors, taxonomy) → Evidence`. Evidence ≠ verdict; full
  ladder never truncated; neighborhood first-class; frozen `scorer_v1` (γ=3, no-dedup, k=15); calibrated
  1-D margin confidence; one measurement, two readouts (placement + novelty).
- **L4 — `serving/atlas_serve.py`**: thin, verify-gated orchestrator. Two load gates — ENCODER-MATCH (live id
  == `meta.encoder_id`) and VERIFY (self-retrieval) — refuse to bind a mismatched/unverifiable artifact.
  Adaptive 2-band router (`<3kb→5kb`, `≥3kb→20kb`; no fused band — fusion measured net-negative). Novelty dial
  `SERVE_MIN_CONFIDENCE` (default 0.5; deploy sets 0.7). Hot-swap via `/admin/reload` (loads+gates fresh,
  swaps only on pass).

## Honest capability (family-bootstrap, in-distribution self-retrieval)

- Family placement ≈ **.84** raw / **~.92** label-corrected (reclassification + top-3 near-miss).
- Genus ≈ **.72** raw / **~.80** label-corrected. Genus is representation-limited (a v11 job), and the theory
  hypothesis is a fine-scale curvature mismatch — v10.9 uses one global κ≈1.25 (inter-domain) where the
  intra-domain scale wants κ=3–16.
- Novelty separates known from novel (in-ref conf .744 vs withheld-family .397); the dial trades
  novel-family false-commit against known recall (0.5→.28/.90 ; 0.7→.15/.76).
- Native placement latency ~2s p50 (CPU, 8-window ensemble). Scaling escape hatch: IVFFlat behind
  `FaissIndex` (unquantized, exactness preserved) or GPU serving — L1-local, encoder/L3 untouched.

**Caveat:** every number above is in-distribution self-retrieval on the reference catalog. External-sequence
validation (the make-or-break for "datum") is not yet done. See `docs/CLOSING_RUNBOOK.md` Phase 6.

## Layout

- `serving/` — the L1–L4 stack + unit tests (GPU-free, `python -m pytest serving/`).
- `harness/` — the validated result-producing scripts: `accept_v109.sh` (the re-runnable acceptance suite),
  `hotswap_drill.sh`, the five-lever sweep (`benchmark_adaptive`, `confirm_k`, `label_audit`, `align_rerank`,
  `novelty_dial`), calibration fitters, and `atlas-serve.service`.
- `docs/` — `DEPLOY_RUNBOOK.md`, `CLOSING_RUNBOOK.md` (the 8-phase program to close), `DATUM.md` (artifact
  provenance ledger).

## Run the acceptance suite

`bash harness/accept_v109.sh` re-checks config, numbers-of-record, the accuracy ceiling gap, novelty/degenerate
honesty, and the hot-swap drill against a running service. Re-run it against any v11 swap: rebuild index →
hot-swap → `accept_v109.sh`. (First harden it against a known-bad model — `CLOSING_RUNBOOK.md` Phase 4.)
