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

## To stamp at deploy (Phase 1)
- `sha256sum` of each `index.faiss` + `gids.npy`, recorded here post-transfer, verified on the inference box.
- GTDB release tag of the taxonomy snapshot.
