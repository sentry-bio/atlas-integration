# viz/ — the Biosphere Atlas 3D viewer

An interactive, self-verifying 3D viewer for the v10.9 embedding: 234,526 genomes and organelles
in a Poincaré-ball point cloud, four disclosed projections, click-to-inspect true-neighbor
constellations, and a provenance stamp that makes every screenshot a citable measurement rather
than an illustration.

```
viz/
  viewer/index.html          the whole client — three.js, no build step, opens as a static file
  pipeline/build_bundle.py   the ONE script that produces the data bundle the viewer loads
  pipeline/fetch_bundle.sh   pull an already-built bundle from the inference box
  verification/              receipts — measurements that decided the design (read this before
                              changing a chart's semantics)
  docs/PROVENANCE.md         what's IN the current bundle, its instrument stamp, caveats, and
                              5M-scaling notes
```

## Quickstart

```bash
cd pipeline
./fetch_bundle.sh ../viewer         # pulls bundle files alongside index.html
cd ../viewer
python3 -m http.server 8000
open http://127.0.0.1:8000/index.html
```

No `npm install`, no build step. three.js loads from a CDN via an import map; everything else is
one HTML file plus the binary bundle sitting next to it.

## Regenerating the bundle from scratch

```bash
cd pipeline
pip install -r requirements.txt
python3 build_bundle.py --out /fast/atlas_v109/viz_bin \
    --index-dir /fast/atlas_v109/indexes/serve_index_v109_20kb
```

Needs, on the machine you run it on: the serve index (`index.faiss` + `gids.npy` + `meta.json`),
`genome_taxonomy_full.json`, the tessellation JSON, and the organelle manifest CSV. Paths default
to the inference-box layout; override with flags. Takes minutes on CPU, dominated by the exact
30-NN graph over 234K points.

## The design: four disclosed gauges over one embedding

This is the load-bearing idea, and it's worth stating precisely because it's what keeps the
viewer honest as it grows. **Every chart is a pure function of the same 129-dimensional
embedding** (computed once, in `build_bundle.py` stage 1 — every other script that used to
duplicate this reconstruction was consolidated away). No chart mutates the underlying data; each
one is a disclosed *viewpoint*, and the description panel states exactly what it guarantees and
what it doesn't:

| Chart | Guarantees | Costs / caveat |
|---|---|---|
| **Figure (discriminant)** | Rigid, undistorted projection — nothing moved toward its label | Viewpoint chosen to display domain separation; that choice is a disclosed degree of freedom |
| **Datum (θ · n=2)** | The canonical coordinate: angle θ anchored at *E. coli* K-12 = 0°, per `ccs/canonical_datum/datum.py`'s own conventions | Radius is advisory; θ resolution certified only phylum→family |
| **Morphospace (novelty depth)** | Radius = novelty (distance to the reference manifold), honestly labeled as such | Novelty conflates biologically-unusual with merely-under-sequenced (declared, not yet fixed — see `docs/PROVENANCE.md`) |
| **Canonical (PCA)** | The variance-maximal 3D shadow | Captures only ~32% of tangent-space variance; a baseline, not a claim |

A fifth chart was *measured and rejected* rather than built: a v9-style "LUCA-centered radial
divergence" view, which would have recovered v9's legibility as a literal geodesic-distance
theorem instead of a fabricated composite score. It failed its own pre-registered gate
(ρ = 0.031 against independent SSU divergence — see
`verification/luca_projection_negative_result.py`). The bundle and viewer contain zero trace of
it by design: **a chart doesn't ship until its measurement passes**, not the other way around.

## Why the underlying columns are trustworthy

Every genome is one row index, shared identically across every file — position, novelty, domain
label, kNN neighbors, taxonomy. That's the entire integrity model: if the files ever get
out of sync (partial upload, stale cache, wrong bundle version), row *i* stops meaning the same
organism in two files and every chart quietly lies. The viewer defends against this actively, not
just by convention:

- `meta.json` carries an **instrument stamp** — `encoder_id` (hash of the composed model weights
  that built the reference index), `datum_version`, `bundle_version`, and the byte size of every
  file — and the UI displays it (`instrument … · view <chart> · integrity ✓`).
- On load, the viewer **asserts** every binary's length against both the row count and the
  stamped size, and **refuses to render** (with the exact mismatch named) rather than draw a
  plausible-looking but scrambled atlas. This was deliberately tested: a truncated `novelty.bin`
  was correctly rejected before this was ever trusted in production.
- All bundle files are fetched with a `?v=<bundle_version>` cache key and `meta.json` itself is
  fetched with `cache: no-store` — a stale browser cache cannot silently mix two builds.

This is the same discipline as a placement receipt (see the CCS `PLACEMENT_RECEIPT.md` design in
the sibling `active-geometry/ccs` repo, if you have access to it): no orphan claims, every number
traceable to the instrument that produced it.

## Two self-validating findings worth knowing before you touch the constellation feature

Click any genome to see its true 30-nearest-neighbors in the full 129D space (exact, not
approximate — that's the point; see `docs/PROVENANCE.md` for what changes at 5M). Two demos are
featured on load because nobody told the model taxonomy and it found it anyway:

- ***Pan troglodytes*** (chimpanzee, `GCA_000002175.2`): 22 of 30 true neighbors are other great
  apes (*Homo*, *Pongo*); the remaining 8 reach across the family line to Old World monkeys
  (*Papio*, *Macaca*) — correct primate phylogeny, unsupervised.
- ***Jakoba libera*** mitochondrion (`NC_021127.1`), the most bacteria-like mitochondrial genome
  known: all 30 neighbors are Asgard/Woese archaea, not other eukaryotes — the one organelle
  whose signal is old enough to point toward deep ancestry instead of host co-divergence. See
  `verification/endosymbiosis_test.py` for the full finding this sits inside.

## Open roadmap

- κ dual-display and view-name-in-stamp are done (frame constant 5/4 vs. live fit 1.237, shown
  separately since they're different kinds of number).
- Not yet done: θ for a *query* sequence (not just reference genomes) — needs the backbone plane
  itself shipped as a datum artifact so a new point can be projected into it; the
  sampling-vs-biological novelty split (see `docs/PROVENANCE.md`); the 5M scaling items
  (explicit IDs, GPU-side morph/picking, tessellation-as-LOD) named in the same doc.
- Publish target: biosphereatlas.com (not yet deployed from this package — the last verified
  local build passed a deliberate abuse pre-flight: rapid projection-switching, mid-flight
  interrupts, corrupted-bundle rejection, all with zero errors).
