#!/usr/bin/env python3
"""
RECEIPT: local patches of the embedding are ALSO ~12-dimensional — semantic zoom ("bloom a
neighborhood into its own faithful 3D chart") does not rescue local fidelity the way manifold-
learning intuition suggests it should.

This killed a specific proposed feature (click-to-zoom into a locally-faithful re-projection)
before it was built. The replacement that shipped instead is the constellation: draw the TRUE
k-NN edges directly (exact adjacency, no re-projection needed) rather than trying to manufacture
a faithful local coordinate system.

Result (last run): both a 600-genome kNN ball and a tessellation-cell neighborhood need ~12
dimensions for 90% variance — local patches double 3D fidelity over the global case (0.22->0.45)
but never approach "faithful." The manifold is fat at every scale sampled.

Usage: python3 local_dimension_check.py --index-dir ... --tessellation ...
"""
import argparse, json, sys
from collections import defaultdict
import numpy as np
import faiss

KAPPA = 1.25


def log_map_origin(p, c):
    sc = np.sqrt(c)
    n = np.linalg.norm(p, axis=-1, keepdims=True)
    n = np.clip(n, 1e-8, 1.0 / sc - 1e-4)
    return p * ((2.0 / sc) * np.arctanh(sc * n) / (n + 1e-8))


def local_fidelity(tan, members, kk=15):
    X = tan[members]
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    var3 = float(var[:3].sum())
    dim90 = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
    P = (Xc @ Vt[:3].T).astype(np.float32)
    ie = faiss.IndexFlatL2(X.shape[1])
    ie.add(X)
    _, nnE = ie.search(X, kk + 1)
    ip = faiss.IndexFlatL2(3)
    ip.add(P)
    _, nnP = ip.search(P, kk + 1)
    jac = np.mean([len(set(nnE[i, 1:]) & set(nnP[i, 1:])) / kk for i in range(len(X))])
    return var3, dim90, float(jac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--tessellation", required=True)
    ap.add_argument("--budget", type=int, default=600, help="fixed-budget kNN patch size")
    ap.add_argument("--n-samples", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.RandomState(7)

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
    full = faiss.IndexFlatL2(tan.shape[1])
    full.add(tan)

    seeds = rng.choice(G, args.n_samples, replace=False)
    _, nbr = full.search(tan[seeds], args.budget)
    res = [local_fidelity(tan, nbr[i]) for i in range(len(seeds))]
    v = np.array(res)
    print(f"[A] fixed-budget kNN patches (n={args.budget}, {len(seeds)} samples)")
    print(f"    local-3D variance: {v[:,0].mean():.3f} (min {v[:,0].min():.3f})")
    print(f"    effective dim (90% var): median {np.median(v[:,1]):.0f}")
    print(f"    local kNN-Jaccard: {v[:,2].mean():.3f} (min {v[:,2].min():.3f})")

    tess = json.load(open(args.tessellation))
    C = np.array(tess["centers"], dtype=np.float32)
    cidx = faiss.IndexFlatL2(C.shape[1])
    cidx.add(C)
    _, cassign = cidx.search(tan, 1)
    cassign = cassign[:, 0]
    cells = defaultdict(list)
    for i, c in enumerate(cassign):
        cells[int(c)].append(i)
    big = [c for c, m in cells.items() if len(m) >= 120]
    pick = rng.choice(big, min(args.n_samples, len(big)), replace=False)
    resB = [local_fidelity(tan, np.array(cells[int(c)])) for c in pick]
    vB = np.array(resB)
    print(f"[B] tessellation cells ({len(pick)} samples)")
    print(f"    local-3D variance: {vB[:,0].mean():.3f} (min {vB[:,0].min():.3f})")
    print(f"    effective dim (90% var): median {np.median(vB[:,1]):.0f}")
    print(f"    local kNN-Jaccard: {vB[:,2].mean():.3f} (min {vB[:,2].min():.3f})")


if __name__ == "__main__":
    main()
