# Bundle provenance

Same policy as `docs/DATUM.md` for the serving index: **the bundle is a build artifact, never
committed as blobs.** This file records what the currently-deployed/last-known bundle *is*, so it
can be verified, re-fetched, or regenerated — the repo stays light, the provenance doesn't.

## Current bundle (as of last hand-off)

```
instrument.encoder_id   sha256:3b1ab6ad57a31e97d004319933559c282fdeb8ae06a950092ea2980fe350c66c
instrument.index_id     serve_index_v109_20kb
instrument.index_encoder v10.9
instrument.datum_version candidate-1.0
instrument.bundle_version v109-2026-08-11
instrument.space         raw-mean-129d
n                        234,526  (202,286 genomes + 32,240 organelles: 17,540 mitochondria,
                                   14,579 chloroplasts, 121 unclassified organelle)
K (neighbors per genome) 30
```

Lives on the inference box at `/fast/atlas_v109/viz_bin/` (host `100.86.142.125` at time of
writing). Fetch it with `pipeline/fetch_bundle.sh`, or regenerate from scratch with
`pipeline/build_bundle.py` against the same `serve_index_v109_20kb` (verify the encoder_id
matches — the viewer's own integrity check will refuse to render otherwise).

| file | bytes | contents |
|---|---:|---|
| `positions.bin` | 2,814,312 | Canonical PCA chart (float32 × 3 × n) |
| `positions_lda.bin` | 2,814,312 | Discriminant "Figure" chart (float32 × 3 × n) |
| `theta.bin` / `theta_r.bin` | 938,104 each | Datum θ chart: angle (radians) / advisory radius |
| `novelty.bin` | 938,104 | Distance-to-manifold, percentile-normalized [0,1] |
| `density.bin` | 938,104 | 15th-NN distance, log + percentile-normalized [0,1] |
| `domain.bin` | 234,526 | uint8 class code (see below) |
| `cell.bin` | 938,104 | Tessellation cell assignment (int32) |
| `family_idx.bin` / `genus_idx.bin` / `species_idx.bin` | 938,104 each | Dict-encoded taxonomy indices |
| `neighbors.bin` | 28,143,120 | int32 × 30 × n — exact kNN graph |
| `neighbor_dist_u8.bin` | 7,035,780 | uint8-quantized neighbor distances (÷255 × `dist_scale`) |
| `accessions.json` | 4,177,321 | Accession string per row |
| `meta.json` | ~3.8 MB | Manifest: counts, `pca_variance`, `families`/`genera`/`species` string tables, `instrument` block, `files` byte-size stamp |
| `figure_meta.json` | 611 | LDA chart medoid label positions |

Domain codes: `0` bacteria · `1` archaea · `2` eukaryota (nuclear) · `3` other/unknown ·
`4` mitochondrion · `5` chloroplast · `6` unclassified organelle.

## Why not append the organelles (a mistake made once, worth not repeating)

An earlier version of the pipeline treated the 32,240 organelle genomes as a *new* population and
appended freshly re-encoded vectors as new rows. This created near-duplicate phantom twins
sitting ~0.19 away from their true row — an artifact of window-sampling differing between the
original index build and a fresh encode pass, not a real biological signal. The fix, now the only
supported path (`build_bundle.py` stage 2): the organelles were **already** in the 234K reference
set, mislabeled `Eukaryota` under their host's family (a mitochondrion filed as its host plant's
family). Relabel the *existing* row's domain code from `organelle_manifest.csv`'s `group` column
by accession match. Same identity, corrected label, zero duplication.

## Known caveats carried by design, not fixed by this pipeline

- **Novelty conflates two things**: biologically-unusual and merely-under-sequenced. Both would
  read the same color. A stability-under-reference-subsampling split (drop 50% of the reference,
  recompute, see what moves) would separate them; not yet built. See `viz/README.md` roadmap.
- **θ chart resolution fades below family** (per the datum's own `CERTIFIED_RESOLUTION`) — the
  per-set-SVD backbone is fragile on biased/small samples. Fine at 234K representative genomes;
  do not trust it on a small or clade-biased subset without re-checking.
- **Radius is advisory everywhere except θ's certified axis is angle, not radius.** No chart in
  this bundle claims radius = a biological quantity except Morphospace, which explicitly declares
  radius = novelty (a measured proxy, not a biological law).

## 5M-genome scaling notes (not yet built, named so they aren't rediscovered the hard way)

1. **Explicit stable IDs.** Row-index identity (row *i* means the same organism across every
   file) is correct for a *frozen* reference but breaks the moment genomes are added — insertion
   reorders everything downstream. Needs an explicit accession-keyed identity before the
   reference set is allowed to grow incrementally.
2. **GPU-side chart morphing and picking.** The projection-morph tween and the hover raycaster
   are both O(n) JavaScript per frame — fine at 234K, dead at 5M. Move the morph to two position
   attributes + a shader mix uniform; move picking to GPU color-ID picking or a spatial index.
3. **Tessellation becomes the LOD tier, not a display toggle.** At 5M the exact 30-NN graph and
   the full point cloud are both too large to ship whole. The tessellation-cell structure already
   in the bundle (`cell.bin`) is the natural prototype/member two-tier cascade: ship cell
   aggregates for the global view, stream member-level detail (positions, neighbors) on focus —
   the same cascade named in the encoder-side "unified prototype+retrieval" design, converging
   independently on the rendering side.
