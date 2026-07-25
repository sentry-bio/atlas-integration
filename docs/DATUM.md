# DATUM — artifact provenance ledger

The frame is reconstructable from spec; the blobs are not committed. This file is the binding record.

## Encoder (the chart)
- **encoder_id**: `sha256:3b1ab6ad57a31e97d004319933559c282fdeb8ae06a950092ea2980fe350c66c`
  — sha256 of the *composed* v9+overlay weights (device-independent). Stamped in each index `meta.json`;
  enforced by the L4 ENCODER-MATCH gate.
- Composition: v9 base (`v9_best.pt`) + curvature overlay (`v10_9_encoder.pt`). κ≈1.2453, emb_dim 129.

## Index tiers (the map) — on Nexus `/zfs_raid/SentryBio/`
| dir | scale | bp / ntok | vectors | genomes | type | encoder_id |
|---|---|---|---|---|---|---|
| `serve_index_v109_20kb/` | 20kb | 20000 / 4096 | 1,876,208 | 234,526 | FlatIP exact d=129 | 3b1ab6ad… |
| `serve_index_v109_5kb/`  | 5kb  | 5000 / 1000  | 1,876,208 | 234,526 | FlatIP exact d=129 | 3b1ab6ad… |

Each dir carries `index.faiss` (~968MB), `gids.npy`, `meta.json` (verify: domain-match 1.0, self-retrieval
1.0, passed), `calibration.json` (per-rank margin→P(correct); 5kb and 20kb copies are byte-identical).

## Taxonomy
- `atlas_index_full/300bp/genome_taxonomy_full.json` — list[234526] by gid. GTDB-derived; labels
  domain/family/genus/species (phylum/class/order are absent in this snapshot).

## Deploy-verified hashes (inference box, 2026-07-25)
Byte-identical to Nexus source, confirmed on transfer:
- `serve_index_v109_20kb/index.faiss`  sha256 `2eb738ae0ab7d5b126475039…`  (968123373 bytes)
- `serve_index_v109_5kb/index.faiss`   sha256 `2fc19a5f4131ae79ce58c7bf…`  (968123373 bytes)
- `v9_best.pt` (base)                   sha256 `f4bb9ad4c72addf479d20b85…`  (already on box as `best_v9_original.pt`)
- `v10_9_encoder.pt` (overlay)          sha256 `206a203d2eeea94ba06d26a9…`
- **Composed encoder_id VERIFIED on box under torch 2.13.0+cpu: `sha256:3b1ab6ad…` → MATCH** (152/152 keys, κ=1.2453).
  The coherence guarantee: the encoder hashes to the id that built the index, so ENCODER-MATCH passes and queries
  land in the exact space the index was built in.

## Inference-box deployment (INTERNAL, not public)
- Host: biosphereatlas (RTX 2060, 23GB RAM). `127.0.0.1:8100` (adaptive, CPU, `SERVE_MIN_CONFIDENCE=0.7`).
- `/fast/atlas_v109/`; indexes carry only the 4 serving essentials (`shards/` build intermediates excluded).
- systemd user unit `atlas-v109` (enabled + linger, Restart=on-failure). NOT in the Cloudflare tunnel — public
  `api.biosphereatlas.com` remains the v9/V15.5 product API. Flip held pending routing decision.

## Still TODO for the full datum contract (Phase 5, NOT yet executed)
- Provenance-stamp `/place` responses `{datum_version, encoder_id, gtdb_release}` (the citability string).
- `register` verb returning invariants (geodesic distances) not raw coordinates.
- GTDB release tag of the taxonomy snapshot.
