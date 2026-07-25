#!/usr/bin/env python3
"""
ship_calibration.py — fit + ship the MARGIN-ONLY calibration (Thread A verdict: support redundant).

Per-rank logistic margin -> P(rank-correct) fit on HELD-OUT in-ref queries (truth known at every rank).
Emits L3-format calibration.json ({rank:{margin_edges, support_edges:[-inf,inf], p:[[...]]}}) — 1D over margin,
single support bin (the degenerate 2D case L3 already handles). Writes into the 20kb (and 5kb) index dirs so
serving picks it up on /admin/reload. v1 — fit on the genus-sample in-ref set (~120 genomes); refit on a larger
held-out set for production, but this gives honest P(correct) instead of the current empty placeholder.

CPU. Run: CUDA_VISIBLE_DEVICES="" python ship_calibration.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_evidence import build_evidence, RANKS
OUT = "/zfs_raid/SentryBio/exp_v9_cascade"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
IDX_DIRS = ["/zfs_raid/SentryBio/serve_index_v109_20kb", "/zfs_raid/SentryBio/serve_index_v109_5kb"]
tax = json.load(open(FTAX)); KNN = 15; CAL_RANKS = ["domain", "phylum", "class", "order", "family", "genus"]
def lin(g): return {r: tax[int(g)].get(r) for r in RANKS}
def fit_lr(x, y, iters=1500, lr=0.5, l2=1e-3):
    mu, sd = x.mean(), x.std() + 1e-9; xs = (x - mu) / sd; Xb = np.c_[np.ones(len(xs)), xs]; w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w)); w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1]])
    return w, mu, sd
def prob(x, w, mu, sd): xs = (x - mu) / sd; return 1 / (1 + np.exp(-(w[0] + w[1] * xs)))

z = np.load(os.path.join(OUT, "genus_encoded_v10.9_20kb.npz"), allow_pickle=True)
Xr, gr = z["Xr"], z["gr"]; taxonomy = {int(g): lin(g) for g in np.unique(gr)}
def place(Vq):
    cos = Xr @ Vq.T; idxs = set()
    for w in range(Vq.shape[0]): idxs.update(np.argpartition(-cos[:, w], KNN)[:KNN].tolist())
    idxs = np.array(sorted(idxs)); wt = cos[idxs].max(1)
    return build_evidence([(int(gr[j]), float(wt[i])) for i, j in enumerate(idxs)], taxonomy, k_neighborhood=KNN)

# per-rank (margin, correct) from in-ref queries
per = {r: {"m": [], "c": []} for r in CAL_RANKS}
for g in np.unique(z["gi"]):
    ev = place(z["Xi"][z["gi"] == g]); truth = lin(g)
    for r in ev.ranks:
        if r.rank in per and not r.gap and truth.get(r.rank) is not None:
            per[r.rank]["m"].append(r.margin); per[r.rank]["c"].append(int(r.top == truth[r.rank]))

NB = 8; edges = np.linspace(0.0, 1.0, NB + 1); centers = (edges[:-1] + edges[1:]) / 2
table = {}
print("per-rank calibration (margin -> P(correct)):")
for r in CAL_RANKS:
    m = np.array(per[r]["m"], float); c = np.array(per[r]["c"], float)
    if len(m) < 8 or c.std() < 1e-6:                       # too few / all-correct -> identity-ish (p ~ acc)
        p_row = np.clip(np.full(NB, c.mean() if len(c) else 0.9), 0.02, 0.98)
    else:
        w, mu, sd = fit_lr(m, c); p_row = np.clip(prob(centers, w, mu, sd), 0.02, 0.98)
    table[r] = {"margin_edges": edges.tolist(), "support_edges": [float("-inf"), float("inf")], "p": [p_row.tolist()]}
    print(f"  {r:<8} n={len(m):3d} acc={c.mean():.2f}  P(correct) over margin bins: "
          + " ".join(f"{x:.2f}" for x in p_row))

for d in IDX_DIRS:
    if os.path.isdir(d):
        json.dump(table, open(os.path.join(d, "calibration.json"), "w"), indent=2)
        print(f"wrote {d}/calibration.json")
print("DONE — reload serving (/admin/reload) to pick it up.")
