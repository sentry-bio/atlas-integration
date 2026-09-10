#!/usr/bin/env python3
"""
build_bundle.py — the ONE script that produces the viz bundle. Replaces four+ scratch scripts
(precompute_knn.py, relabel_organelles.py, lda_viewpoint.py, compute_theta.py, stamp_bundle.py)
that drifted across sessions because each re-implemented reconstruct/log_map independently.
Consolidation was named as a "5M readiness" debt in the kernel audit; this pays it down.

Runs on the inference box (needs the v10.9 serve index + taxonomy + tessellation + organelle
manifest). ~5-10 min on CPU for the kNN graph over 234K points; the rest is seconds.

Usage:
    python3 build_bundle.py --out /fast/atlas_v109/viz_bin [--stage all]

Stages (run in order; --stage N to resume from a cached .npy if a later stage fails):
    1 reconstruct   — genome embeddings, raw-mean 129D (the ONE space every downstream number lives in)
    2 relabel       — organelle domain codes (mito/chloro), IN PLACE — see docs/PROVENANCE.md
                       "why not append" for why this is not "add 32K new points"
    3 charts        — three projections: canonical PCA, discriminant Figure (LDA), datum theta
    4 novelty       — distance-to-manifold vs tessellation centers + cell assignment
    5 graph         — exact 30-NN graph in the raw-mean space + quantized distances
    6 density       — 15th-NN distance, log + percentile normalized
    7 taxonomy      — dict-encoded family/genus/species + accession list
    8 stamp         — instrument metadata (encoder_id, byte sizes) into meta.json

Every array is indexed by the SAME row i = SAME genome across every output file. That identity
is the whole kernel; do not subsample or reorder between stages.
"""
import argparse, csv, json, os, sys
from collections import Counter

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss required: pip install faiss-cpu (or faiss-gpu on a CUDA box)")

# ── frozen constants (match ccs/canonical_datum/datum.py) ──────────────────────────────────
KAPPA = 1.25                    # live-fit curvature; datum FRAME constant is 5/4 — display both, never average
K_NEIGHBORS = 30
DIST_SCALE = 0.6                # global distance scale for uint8 edge-distance quantization
PRIME_MERIDIAN = "GCF_000005845.2"   # E. coli K-12 -> theta = 0
CHIRALITY_ANCHOR = "GCF_000091665.1" # M. jannaschii -> theta > 0 (fixes handedness)
ORGANELLE_TYPE = {"mitochondrion": 4, "chloroplast": 5, "organelle": 6}
DOMAIN_CODE = {"bacteria": 0, "archaea": 1, "eukaryota": 2}

# ── default input paths (inference box layout) ─────────────────────────────────────────────
DEFAULTS = dict(
    index_dir="/fast/atlas_v109/indexes/serve_index_v109_20kb",
    taxonomy="/fast/atlas_v109/genome_taxonomy_full.json",
    tessellation="/fast/atlas_v109/tessellation_v109_serving.json",
    organelle_manifest="/fast/sentrybio/data/organelle_manifest.csv",
)


def log_map_origin(p, c):
    """Poincare-ball log map at the origin. p: (..., d) points inside the ball, curvature c."""
    sc = np.sqrt(c)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    n = np.clip(n, 1e-8, 1.0 / sc - 1e-4)
    return p * ((2.0 / sc) * np.arctanh(sc * n) / (n + 1e-8))


def poincare_norm(p, c):
    sc = np.sqrt(c)
    n = np.clip(np.linalg.norm(p, axis=-1), 0, 1.0 / sc - 1e-4)
    return (2.0 / sc) * np.arctanh(sc * n)


def stage1_reconstruct(args, cache):
    """Reconstruct per-genome embeddings by averaging their window vectors (raw-mean 129D).
    This is THE space: novelty, cells, the kNN graph, and every chart's tangent space all derive
    from it. Any downstream script computing its own reconstruction is a coherence bug waiting
    to happen — that duplication is exactly what this file exists to kill."""
    print("[1/8] reconstruct genome embeddings", file=sys.stderr)
    idx = faiss.read_index(f"{args.index_dir}/index.faiss")
    n_windows = idx.ntotal
    gids = np.load(f"{args.index_dir}/gids.npy").astype(np.int64)
    V = idx.reconstruct_n(0, n_windows).astype(np.float64)
    n_genomes = int(gids.max()) + 1
    sums = np.zeros((n_genomes, idx.d))
    np.add.at(sums, gids, V)
    counts = np.maximum(np.bincount(gids, minlength=n_genomes), 1).astype(np.float64)
    emb = sums / counts[:, None]
    del V, sums
    tax = json.load(open(args.taxonomy))
    print(f"   {n_genomes} genomes, dim={idx.d}", file=sys.stderr)
    cache.update(emb=emb, tax=tax, n=n_genomes, dim=idx.d)


def stage2_relabel_organelles(args, cache):
    """Organelles are already IN the 234K set, labeled Eukaryota under their HOST family — they
    are not a separate population to append. See docs/PROVENANCE.md 'why not append': an earlier
    version of this pipeline appended 32K freshly-encoded organelle vectors as new rows, which
    created near-duplicate phantom twins (distance ~0.19 from their true row) because window
    sampling differs between the original build and a fresh encode. Relabeling the EXISTING rows
    by accession is the correct fix — same identity, corrected domain code, zero duplication."""
    print("[2/8] relabel organelles in place", file=sys.stderr)
    kind = {}
    for r in csv.DictReader(open(args.organelle_manifest)):
        kind[r["accession"]] = r.get("group") or "organelle"
    tax, n = cache["tax"], cache["n"]
    dom = np.zeros(n, dtype=np.uint8)
    n_org = 0
    for j in range(n):
        acc = tax[j].get("accession")
        if acc in kind:
            dom[j] = ORGANELLE_TYPE.get(kind[acc], 6)
            n_org += 1
        else:
            dom[j] = DOMAIN_CODE.get((tax[j].get("domain") or "").lower(), 3)
    print(f"   relabeled {n_org} organelles: {dict(Counter(dom[dom >= 4].tolist()))}", file=sys.stderr)
    cache.update(dom=dom, n_organelles=n_org)


def stage3_charts(args, cache):
    """Three disclosed projections of the SAME embedding. Each is a pure function of columns
    already computed — no chart mutates the embedding, no chart's math leaks into another's."""
    emb, tax, dom, n = cache["emb"], cache["tax"], cache["dom"], cache["n"]
    tangent = log_map_origin(emb, KAPPA)

    print("[3a/8] canonical PCA chart", file=sys.stderr)
    mu = tangent.mean(0)
    Xc = tangent - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca_pos = Xc @ Vt[:3].T
    pca_pos = (pca_pos / (np.abs(pca_pos).max() + 1e-8) * 0.9).astype(np.float32)
    pca_variance = (S[:3] ** 2 / (S ** 2).sum()).tolist()

    print("[3b/8] discriminant Figure chart (LDA over 5 classes)", file=sys.stderr)
    classes = [0, 1, 2, 4, 5]  # bacteria/archaea/eukaryota/mito/chloro; skip 'other'(3)/generic-organelle(6)
    d = tangent.shape[1]
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for cl in classes:
        m = dom == cl
        n_cl = m.sum()
        if n_cl < 2:
            continue
        Xcl = tangent[m]
        mu_cl = Xcl.mean(0)
        Sw += (Xcl - mu_cl).T @ (Xcl - mu_cl)
        diff = (mu_cl - mu)[:, None]
        Sb += n_cl * (diff @ diff.T)
    Sw += np.eye(d) * 1e-3
    evals, evecs = np.linalg.eigh(np.linalg.solve(Sw, Sb))
    W = evecs[:, np.argsort(evals)[::-1][:3]].real
    lda_pos = Xc @ W
    lda_pos = (lda_pos / (np.abs(lda_pos).max() + 1e-8) * 0.9).astype(np.float32)
    # medoids for on-chart labels (real member genomes — centroids collapse under this projection)
    medoids = {}
    label_names = {0: "Bacteria", 1: "Archaea", 2: "Eukaryota", 4: "Mitochondria", 5: "Chloroplasts"}
    for cl, name in label_names.items():
        idxs = np.where(dom == cl)[0]
        if len(idxs) < 10:
            continue
        c = lda_pos[idxs].mean(0)
        med = idxs[np.argmin(np.linalg.norm(lda_pos[idxs] - c, axis=1))]
        medoids[name] = [float(lda_pos[med, 0]), float(lda_pos[med, 1]), float(lda_pos[med, 2])]

    print("[3c/8] datum theta chart (canonical n=2 disk)", file=sys.stderr)
    acc2i = {tax[j].get("accession"): j for j in range(n)}
    pm_i = acc2i.get(PRIME_MERIDIAN)
    ca_i = acc2i.get(CHIRALITY_ANCHOR)
    P2 = Xc @ Vt[:2].T  # backbone plane (reuses the PCA SVD — same tangent space, no re-fit)
    if pm_i is None:
        ec = [j for j in range(n) if (tax[j].get("species") or "") == "Escherichia coli"]
        p_ec = P2[ec].mean(0)
        t0 = np.arctan2(p_ec[1], p_ec[0])
        print(f"   prime meridian accession missing; fallback to {len(ec)}-genome E. coli medoid", file=sys.stderr)
    else:
        t0 = np.arctan2(P2[pm_i, 1], P2[pm_i, 0])
    theta = np.arctan2(P2[:, 1], P2[:, 0]) - t0
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    if ca_i is not None and theta[ca_i] < 0:
        theta = -theta
    r2 = np.linalg.norm(P2, axis=1)
    theta_r = (r2 / np.percentile(r2, 99)).clip(0, 1).astype(np.float32)
    print(f"   E. coli theta = {np.degrees(theta[pm_i] if pm_i is not None else 0):.2f}deg (should be 0)",
          file=sys.stderr)

    cache.update(
        pca_pos=pca_pos, pca_variance=pca_variance,
        lda_pos=lda_pos, medoids=medoids,
        theta=theta.astype(np.float32), theta_r=theta_r,
        tangent=tangent,
    )


def stage4_novelty(args, cache):
    """Novelty = distance to the nearest of 2000 tessellation centers, in the SAME raw-mean
    space as everything else. KNOWN CAVEAT (declared, not fixed by this pipeline): this
    conflates biologically-strange with merely-under-sequenced. See docs/PROVENANCE.md."""
    print("[4/8] novelty + cell assignment", file=sys.stderr)
    emb = cache["emb"].astype(np.float32)
    tess = json.load(open(args.tessellation))
    centers = np.array(tess["centers"], dtype=np.float32)
    ci = faiss.IndexFlatL2(centers.shape[1])
    ci.add(centers)
    Dc, Ic = ci.search(emb, 1)
    dist_manifold = np.sqrt(np.maximum(Dc[:, 0], 0))
    cell_id = Ic[:, 0].astype(np.int32)
    lo, hi = np.percentile(dist_manifold, [2, 98])
    novelty = np.clip((dist_manifold - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    cache.update(novelty=novelty, cell_id=cell_id)


def stage5_graph(args, cache):
    """Exact brute-force 30-NN in the raw-mean space. This IS the placement receipt's
    'attachment' evidence — not decoration, not approximate. At 5M genomes this stage is
    the one that needs a real ANN index (ntotal^2 stops being free); see docs/PROVENANCE.md
    '5M scaling notes'."""
    print("[5/8] exact kNN graph", file=sys.stderr)
    emb = cache["emb"].astype(np.float32)
    n = cache["n"]
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    NB = np.zeros((n, K_NEIGHBORS), dtype=np.int32)
    ND = np.zeros((n, K_NEIGHBORS), dtype=np.float32)
    batch = 8192
    for s in range(0, n, batch):
        e = min(s + batch, n)
        D, I = index.search(emb[s:e], K_NEIGHBORS + 1)
        for r in range(e - s):
            gi = s + r
            keep = I[r] != gi
            ni = I[r][keep][:K_NEIGHBORS]
            nd = np.sqrt(np.maximum(D[r][keep][:K_NEIGHBORS], 0))
            if len(ni) < K_NEIGHBORS:
                ni = np.pad(ni, (0, K_NEIGHBORS - len(ni)), constant_values=-1)
                nd = np.pad(nd, (0, K_NEIGHBORS - len(nd)))
            NB[gi] = ni
            ND[gi] = nd
        if s % (batch * 8) == 0:
            print(f"   {e}/{n}", file=sys.stderr)
    nd_u8 = np.clip(ND / DIST_SCALE * 255, 0, 255).astype(np.uint8)
    cache.update(neighbors=NB, neighbor_dist_u8=nd_u8, raw_neighbor_dist=ND)


def stage6_density(args, cache):
    """15th-NN distance -> log -> percentile-normalized. Reuses the graph from stage 5 (no
    second kNN pass) — this is exactly the kind of shared-computation discipline the
    consolidation buys: density used to be computed with its own separate FAISS pass."""
    print("[6/8] density from graph (no second kNN pass)", file=sys.stderr)
    ND = cache["raw_neighbor_dist"]
    d15 = ND[:, 14].copy()
    d15[d15 <= 0] = np.median(d15[d15 > 0])
    raw = -np.log(d15 + 1e-6)
    lo, hi = np.percentile(raw, [5, 95])
    density = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    cache.update(density=density)


def stage7_taxonomy(args, cache):
    print("[7/8] dict-encode taxonomy", file=sys.stderr)
    tax, n = cache["tax"], cache["n"]

    def field(j, k):
        return tax[j].get(k) or ""

    fam = [field(j, "family") for j in range(n)]
    gen = [field(j, "genus") for j in range(n)]
    sp = [field(j, "species") for j in range(n)]
    acc = [field(j, "accession") for j in range(n)]

    def encode(arr):
        uniq = sorted(set(arr))
        m = {s: i for i, s in enumerate(uniq)}
        return uniq, np.array([m[s] for s in arr], dtype=np.int32)

    fam_u, fam_i = encode(fam)
    gen_u, gen_i = encode(gen)
    sp_u, sp_i = encode(sp)
    cache.update(fam_u=fam_u, fam_i=fam_i, gen_u=gen_u, gen_i=gen_i, sp_u=sp_u, sp_i=sp_i, acc=acc)


def stage8_write(args, cache):
    """Write every binary + the stamped manifest. Byte layout is asserted by the viewer at
    load time (viz/viewer/index.html assertBundle/assertGraph) — get this wrong and the
    viewer refuses to render rather than silently drawing garbage."""
    print("[8/8] write bundle + instrument stamp", file=sys.stderr)
    out = args.out
    os.makedirs(out, exist_ok=True)

    cache["pca_pos"].tofile(f"{out}/positions.bin")
    cache["lda_pos"].tofile(f"{out}/positions_lda.bin")
    cache["theta"].tofile(f"{out}/theta.bin")
    cache["theta_r"].tofile(f"{out}/theta_r.bin")
    cache["novelty"].tofile(f"{out}/novelty.bin")
    cache["dom"].tofile(f"{out}/domain.bin")
    cache["cell_id"].tofile(f"{out}/cell.bin")
    cache["density"].tofile(f"{out}/density.bin")
    cache["fam_i"].tofile(f"{out}/family_idx.bin")
    cache["gen_i"].tofile(f"{out}/genus_idx.bin")
    cache["sp_i"].tofile(f"{out}/species_idx.bin")
    cache["neighbors"].tofile(f"{out}/neighbors.bin")
    cache["neighbor_dist_u8"].tofile(f"{out}/neighbor_dist_u8.bin")
    json.dump(cache["acc"], open(f"{out}/accessions.json", "w"))
    json.dump(
        {"medoids": cache["medoids"], "sep_note": "see verification/ for the LDA separability measurement"},
        open(f"{out}/figure_meta.json", "w"),
    )

    files = {}
    for f in os.listdir(out):
        if f.endswith((".bin",)):
            files[f] = os.path.getsize(f"{out}/{f}")

    from datetime import date

    meta = {
        "kappa": KAPPA,
        "n": cache["n"],
        "n_genomes": cache["n"] - cache["n_organelles"],
        "n_organelles": cache["n_organelles"],
        "K": K_NEIGHBORS,
        "dist_scale": DIST_SCALE,
        "pca_variance": cache["pca_variance"],
        "families": cache["fam_u"],
        "genera": cache["gen_u"],
        "species": cache["sp_u"],
        "counts": {
            "mitochondrion": int((cache["dom"] == 4).sum()),
            "chloroplast": int((cache["dom"] == 5).sum()),
            "organelle": int((cache["dom"] == 6).sum()),
        },
        "instrument": {
            "encoder_id": args.encoder_id or "UNSET — pass --encoder-id from index meta.json",
            "index_id": os.path.basename(args.index_dir.rstrip("/")),
            "index_encoder": "v10.9",
            "index_bp": 20000,
            "datum_version": "candidate-1.0",
            "bundle_version": f"v109-{date.today().isoformat()}",
            "built": date.today().isoformat(),
            "space": "raw-mean-129d",
        },
        "files": files,
    }
    json.dump(meta, open(f"{out}/meta.json", "w"))
    total = sum(files.values()) + os.path.getsize(f"{out}/accessions.json")
    print(f"DONE {out} — {total / 1e6:.1f} MB, {cache['n']} rows "
          f"({cache['n'] - cache['n_organelles']} genomes + {cache['n_organelles']} organelles)",
          file=sys.stderr)


STAGES = [
    (1, "reconstruct", stage1_reconstruct),
    (2, "relabel", stage2_relabel_organelles),
    (3, "charts", stage3_charts),
    (4, "novelty", stage4_novelty),
    (5, "graph", stage5_graph),
    (6, "density", stage6_density),
    (7, "taxonomy", stage7_taxonomy),
    (8, "stamp", stage8_write),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output bundle directory (e.g. /fast/atlas_v109/viz_bin)")
    p.add_argument("--index-dir", default=DEFAULTS["index_dir"])
    p.add_argument("--taxonomy", default=DEFAULTS["taxonomy"])
    p.add_argument("--tessellation", default=DEFAULTS["tessellation"])
    p.add_argument("--organelle-manifest", default=DEFAULTS["organelle_manifest"])
    p.add_argument("--encoder-id", default=None,
                   help="sha256:... from the serve index's meta.json — stamped into the bundle so the "
                        "viewer can refuse to render a mismatched build")
    args = p.parse_args()

    if args.encoder_id is None:
        try:
            idx_meta = json.load(open(f"{args.index_dir}/meta.json"))
            args.encoder_id = idx_meta.get("encoder_id")
            print(f"auto-detected encoder_id from index meta.json: {args.encoder_id}", file=sys.stderr)
        except Exception:
            print("WARNING: no --encoder-id and could not read it from index meta.json — "
                  "bundle will stamp a placeholder. The viewer's integrity check depends on this "
                  "being real; do not ship a bundle with an UNSET encoder_id.", file=sys.stderr)

    cache = {}
    for num, name, fn in STAGES:
        fn(args, cache)


if __name__ == "__main__":
    main()
