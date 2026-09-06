#!/usr/bin/env python3
"""
RECEIPT: the v10.9 embedding is genuinely ~20-dimensional; no 3D chart can be locally faithful.

This is the finding that killed the "just find a better 3D projection" instinct and redirected
the whole viz design toward MULTIPLE DISCLOSED charts (each honest about what it drops) instead
of chasing a single "correct" one. Sweeps local-neighborhood fidelity (kNN-Jaccard) and global
distance-correlation as a function of how many PCA dimensions of the tangent space are kept.

Result (last run, 234K genomes): 3D keeps only 22% of local neighbor structure; 20D keeps 85%.
Compare against the 129D-full-space numbers reported in the running viewer (57% family / 97%
domain via the exact kNN graph) — that gap IS the reason the viewer routes "truth" through
neighbor queries and cell/novelty color, never through 3D position.

Usage: python3 dimension_sweep.py --index-dir /fast/atlas_v109/indexes/serve_index_v109_20kb
"""
import argparse, sys
import numpy as np
import faiss

KAPPA = 1.25


def log_map_origin(p, c):
    sc = np.sqrt(c)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    n = np.clip(n, 1e-8, 1.0 / sc - 1e-4)
    return p * ((2.0 / sc) * np.arctanh(sc * n) / (n + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 5, 10, 20, 50])
    args = ap.parse_args()
    rng = np.random.RandomState(42)

    idx = faiss.read_index(f"{args.index_dir}/index.faiss")
    n = idx.ntotal
    gids = np.load(f"{args.index_dir}/gids.npy").astype(np.int64)
    V = idx.reconstruct_n(0, n).astype(np.float64)
    G = int(gids.max()) + 1
    sums = np.zeros((G, idx.d))
    np.add.at(sums, gids, V)
    cnt = np.maximum(np.bincount(gids, minlength=G), 1).astype(np.float64)
    emb = sums / cnt[:, None]
    del V, sums

    tan = log_map_origin(emb, KAPPA).astype(np.float32)
    si = rng.choice(G, min(args.sample, G), replace=False)
    Eref = tan[si]
    ir = faiss.IndexFlatL2(Eref.shape[1])
    ir.add(Eref)
    kk = 15
    _, nnRef = ir.search(Eref, kk + 1)
    ref_sets = [set(nnRef[i, 1:]) for i in range(len(Eref))]

    a = rng.randint(0, len(Eref), 40000)
    b = rng.randint(0, len(Eref), 40000)
    dfull = np.linalg.norm(Eref[a] - Eref[b], axis=1)

    Xc = tan - tan.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    print("cumulative variance:", {d: round(float(var[:d].sum()), 3) for d in args.dims})

    for d in args.dims:
        coords = (Xc @ Vt[:d].T).astype(np.float32)
        P = coords[si]
        ip = faiss.IndexFlatL2(d)
        ip.add(P)
        _, nnP = ip.search(P, kk + 1)
        jac = np.mean([len(ref_sets[i] & set(nnP[i, 1:])) / kk for i in range(len(P))])
        d3 = np.linalg.norm(P[a] - P[b], axis=1)
        gcorr = np.corrcoef(dfull, d3)[0, 1]
        print(f"dim={d:3d}  local(kNN-Jaccard)={jac:.3f}  global(dist-corr)={gcorr:.3f}")


if __name__ == "__main__":
    main()
