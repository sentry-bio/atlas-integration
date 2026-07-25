# CLOSING_RUNBOOK — the program to close v10.9 to a live, enduring, datum-shaped API

Ordered by dependency. Each phase: goal · steps · gate · rollback · what it closes. Invariants across all
phases: never disturb the live v9 container or other tenants on the inference box (deploy *beside*, never in
place); numpy stays 1.26.4 (2.x breaks the faiss ABI); heavy compute on Nexus, the inference box only *hosts*.

## Phase 0 — Freeze & commit the substrate
Commit this curated, secret-scanned integration repo in logical units; tag `v10.9-chart-1`. Index/checkpoint
blobs are **not** committed — their sha256/encoder_id/GTDB-release live in `docs/DATUM.md`, so the frame is
reconstructable-from-spec without bloat. **Closes:** reproducibility ledger.

## Phase 1 — Port & deploy-beside on the inference box
Read-only inventory (v9 `biosphere-v15-5` :8000 + CF tunnel healthy & untouched; disk/RAM/GPU; a dedicated
venv with numpy 1.26.4). Transfer the two index dirs + composed encoder + `build_pipeline/` from Nexus
`/zfs_raid`; verify sha256 vs `DATUM.md`. Launch on a NEW port (:8100) CPU, `STRICT_ENCODER=1`,
`SERVE_MIN_CONFIDENCE=0.7`. Load gates (encoder-match + verify) must pass. Run `accept_v109.sh` against :8100.
**Gate:** acceptance green on :8100, v9 provably untouched. **Rollback:** kill :8100.

## Phase 2 — Shadow, then flip
Shadow real traffic to :8100 alongside v9; compare placement agreement, novelty, latency. **Pre-registered
flip criterion:** agrees on high-confidence calls, disagreements are v10.9-better/honest-abstentions, native
p95 < SLO (else take Phase-8 scaling first). **← explicit human go →** flip the Cloudflare route v9→v10.9,
v9 warm. **Rollback:** re-point tunnel to v9 (one step). Persist: systemd + linger. **Closes:** live migration.

## Phase 3 — Earn the conformal novelty (the second readout)
Build `conformal.json` (per-family LOO isolation-score ECDFs, family→domain→global fallback,
provenance-stamped). Lift the pure `(neighbors)→isolation` score into L3. **Pre-registered discrimination
gate (may fail):** conformal typicality separates on-manifold-ambiguous from off-manifold-novel where margin
cannot (AUROC beats margin, CI clears). Ship only if it clears — else document the null and keep margin
abstention. **Closes:** novelty-true-to-form.

## Phase 4 — Harden the certifier
Meta-test `accept_v109.sh` against injected faults (mismatched encoder_id, shuffled taxonomy, blurred index,
permuted calibration) — each must be *caught* by its gate. Commit `accept_meta_test.sh`. A green scorecard now
means "properties hold," not "tests ran." **Closes:** the certifier-not-certified thread; makes v11 swap
self-verifying — this is what moves the calibration discipline out of the conversation and into the harness.

## Phase 5 — The datum contract
Add the `register` verb: response leads with **invariants** (geodesic distances to K nearest + accessions,
calibrated placement confidence, novelty readout) — never raw (r,θ) as physics. Provenance-stamp every
response `{datum_version, encoder_id, gtdb_release}` (the citability primitive). FASTA front door: real
gzipped/multi-contig/IUPAC/soft-masked input, end-to-end. Publish a minimal datum spec (metric, reference
points, curvature, calibration provenance) so the frame is reconstructable independent of the private encoder.

## Phase 6 — External validation & incumbent comparison (make-or-break for "datum")
Needs genuinely external sequences. Register never-seen genomes (a newer GTDB release's new taxa / environmental
MAGs) → stable coordinates + honest novelty on *true* novel input. Eka-silicon test: predict an absent region's
profile, check against a held-out external organism landing there. One incumbent comparison (vs GTDB-Tk /
sourmash) on a shared external set, naming the *specific* axis geometry wins. **Closes:** in-distribution-only.

## Phase 7 — Science backlog (separate ledger from the tool)
(1) Read the Zenodo derivation for one fact: *does h ever touch the geometry?* — decides n=2 real vs circular.
(2) Within-family curvature (independent κ_geo vs h_seq → back-solve n per family across κ=3–16): canonicity
check + genus-ceiling mechanism. (3) Reticulate negative control (recombination should read n>2 or fail to
embed) — the one experiment that can *falsify*. Publish tier-1 (hyperbolic frame canonical for branching
descent — theorem-backed) as foundation; tier-3 (n=2 universal) as a falsifiable conjecture *with* its
negative control. Never let the exciting claim's walk-back tar the safe one. Sets v11 = scale-adaptive curvature.

## Phase 8 — Durability & scaling close
Document (build only on SLO/scale demand) the IVFFlat/GPU escape hatch behind `FaissIndex` (L1-local). Confirm
the hot-swap path (encoder_id gate + Phase-4 meta-tested acceptance) is the *only* thing v11 needs. Record the
stewardship a public datum needs to outlive its origin (versioning cadence, who holds the reference points).

## Dependency spine
0 commit → 1 deploy-beside (gate: acceptance) → 2 shadow + human-go + flip (rollback armed). 3/4 buildable now
read-only on `/zfs_raid`, deploy via hot-swap once live. 6 needs external sequences. 7 gated on the derivation
read; sets v11. 8 build-on-demand, document now.
