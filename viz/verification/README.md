# verification/

Receipts, in the same spirit as `ccs/verification/` and this repo's `docs/DATUM.md` — each script
here is a measurement that *decided* something about the viewer's design. None of them are part
of the production build (`pipeline/build_bundle.py`); they are the evidence trail for why the
bundle looks the way it does. Re-run one before you second-guess the design it gates.

| Script | Measures | Verdict it produced |
|---|---|---|
| `dimension_sweep.py` | Local/global fidelity vs. #PCA-dims kept | Embedding is ~20D; no 3D chart is locally faithful → ship multiple disclosed charts, never claim one is "correct" |
| `local_dimension_check.py` | Same, but on local neighborhoods (kNN balls, tessellation cells) | Local patches are ALSO ~12D → semantic zoom into a "faithful" local 3D chart doesn't work; constellation (exact edges) replaces it |
| `endosymbiosis_test.py` | Organelle → genome nearest-neighbor domain | Organelles cluster by HOST, not bacterial ancestor (co-divergence/barcoding signal); *Jakoba libera* is the one exception reaching toward Asgard archaea — the viewer's featured "deepest organelle" demo |
| `luca_projection_negative_result.py` | ρ(geodesic distance from a declared prokaryotic origin, SSU divergence) | ρ = 0.031 — a v9-style LUCA-radial chart is **not shippable**; do not build it without re-running this gate first |

## The rule these encode

**Measure before you build the seductive view.** The LUCA chart is the clearest example: the
proposal was elegant (a declared, versioned origin recovers v9's radial story as a theorem
instead of a fabricated composite score) and it still failed, because the finding it needed
(radius correlates with real divergence) wasn't true for *this* encoder. Running the measurement
first meant no button ever shipped a caption it couldn't back up.

If you add a chart, phenomenon, or interaction that makes an implicit empirical claim (X
correlates with Y, this axis means Z), write the measurement script first and put its receipt
here — pass or fail.
