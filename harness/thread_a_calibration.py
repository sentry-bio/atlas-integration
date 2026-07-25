#!/usr/bin/env python3
"""
thread_a_calibration.py — the free, decisive gate: is `support` a REAL, non-redundant second coordinate?

NOT "is support correlated with novelty" (we know it is, weakly) but the DECISION-relevant question:
"when margin has spoken, does knowing support change whether we should trust this placement?"

Label = TRUSTWORTHY-COMMIT: 1 iff in-ref AND family-vote correct; 0 otherwise (in-ref-wrong OR withheld-family).
Fit two calibrators on IDENTICAL family-grouped CV folds: logistic(margin) vs logistic(margin, support).
Primary metric = LOG-LOSS (calibration quality), plus AUC and the operating-curve TP@FC=0.10 (ties to the 0.483
baseline). Delta with FAMILY-BOOTSTRAP CI. Run on BOTH encoders → geometry-coherence check (support should help
v10.9 but NOT v9, since v9's support is flat — that ties the novelty-coordinate to the coarse-spreading story).

PRE-REGISTERED (null is a WELCOME outcome):
  2D beats margin, log-loss delta CI clears 0  -> support is a real 2nd coordinate; ship (margin,support); #4 bar=2D.
  2D ≈ margin (CI includes 0)                  -> support redundant per-query; RETRACT the claim; ship margin-only.
Deliverable: the winning calibrator fit on all data -> L3-format binned table (calibration.json-ready).

Zero new encoding beyond recomputing features from cached vectors. CPU. Run: CUDA_VISIBLE_DEVICES="" python thread_a_calibration.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_evidence import build_evidence, RANKS
OUT = "/zfs_raid/SentryBio/exp_v9_cascade"; CALIB_OUT = "/zfs_raid/SentryBio/radial_training"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
tax = json.load(open(FTAX)); rng = np.random.RandomState(0); KNN = 15; NBOOT = 2000; KFOLD = 5
def lin(g): return {r: tax[int(g)].get(r) for r in RANKS}
def load(enc):
    for nm in [f"genus_encoded_{enc}_20kb.npz", "genus_encoded_20kb.npz" if enc == "v9" else None]:
        if nm and os.path.exists(os.path.join(OUT, nm)): return np.load(os.path.join(OUT, nm), allow_pickle=True)
def place(Xr, gr, taxonomy, Vq):
    cos = Xr @ Vq.T; idxs = set()
    for w in range(Vq.shape[0]): idxs.update(np.argpartition(-cos[:, w], KNN)[:KNN].tolist())
    idxs = np.array(sorted(idxs)); wt = cos[idxs].max(1)
    return build_evidence([(int(gr[j]), float(wt[i])) for i, j in enumerate(idxs)], taxonomy, k_neighborhood=KNN)
def features(z):
    Xr, gr = z["Xr"], z["gr"]; taxonomy = {int(g): lin(g) for g in np.unique(gr)}; rows = []
    def fam_rank(ev): return next((r for r in ev.ranks if r.rank == "family"), None)
    for g in np.unique(z["gi"]):                                   # in-reference
        ev = place(Xr, gr, taxonomy, z["Xi"][z["gi"] == g]); fr = fam_rank(ev)
        if fr and not fr.gap: rows.append((fr.margin, ev.support, int(fr.top == lin(g)["family"]), str(lin(g)["family"]), 1))
    for g in np.unique(z["gf"]):                                   # withheld-family (any commit wrong → 0)
        ev = place(Xr, gr, taxonomy, z["Xf"][z["gf"] == g]); fr = fam_rank(ev)
        if fr and not fr.gap: rows.append((fr.margin, ev.support, 0, str(lin(g)["family"]), 0))
    X = np.array([[r[0], r[1]] for r in rows]); y = np.array([r[2] for r in rows])
    fam = np.array([r[3] for r in rows], object); tier = np.array([r[4] for r in rows])  # 1=in-ref, 0=withheld
    return X, y, fam, tier
# ── minimal logistic + metrics (no sklearn) ─────────────────────────────────────────────────────
def fit_lr(X, y, iters=1200, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd; Xb = np.c_[np.ones(len(Xs)), Xs]; w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w)); g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]; w -= lr * g
    return w, mu, sd
def prob(X, w, mu, sd):
    Xs = (X - mu) / sd; return 1 / (1 + np.exp(-np.c_[np.ones(len(Xs)), Xs] @ w))
def auc(p, y):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(len(p)); pos = y == 1
    a, b = int(pos.sum()), int((~pos).sum()); return float("nan") if a == 0 or b == 0 else (r[pos].sum() - a * (a - 1) / 2) / (a * b)
def logloss(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
def grouped_cv(X, y, fam, cols):
    ufam = np.array(sorted(set(fam))); rng.shuffle(ufam); folds = np.array_split(ufam, KFOLD); oof = np.zeros(len(y))
    for fo in folds:
        te = np.array([f in set(fo) for f in fam]); tr = ~te
        if tr.sum() < 10 or te.sum() == 0: continue
        w, mu, sd = fit_lr(X[tr][:, cols], y[tr]); oof[te] = prob(X[te][:, cols], w, mu, sd)
    return oof
def tp_at_fc(score, y, tier, fc=0.10):
    thr = np.quantile(score[tier == 0], 1 - fc)                    # threshold at FC on withheld
    inref = tier == 1; return float((score[inref & (y == 1)] >= thr).sum() / max(inref.sum(), 1))

for enc in ["v9", "v10.9"]:
    X, y, fam, tier = features(load(enc))
    print(f"\n===== {enc} =====  n={len(y)} ({int((tier==1).sum())} in-ref, {int((tier==0).sum())} withheld) pos-rate={y.mean():.2f}")
    m, s = X[:, 0], X[:, 1]
    lomask = m < np.median(m)
    print(f"  collinearity corr(margin,support): overall={np.corrcoef(m,s)[0,1]:+.3f}  low-margin={np.corrcoef(m[lomask],s[lomask])[0,1]:+.3f}")
    oof_m = grouped_cv(X, y, fam, [0]); oof_ms = grouped_cv(X, y, fam, [0, 1])
    print(f"  CV margin-only : AUC={auc(oof_m,y):.3f}  logloss={logloss(oof_m,y):.3f}  TP@FC=0.10={tp_at_fc(oof_m,y,tier):.3f}")
    print(f"  CV +support    : AUC={auc(oof_ms,y):.3f}  logloss={logloss(oof_ms,y):.3f}  TP@FC=0.10={tp_at_fc(oof_ms,y,tier):.3f}")
    # family-bootstrap the log-loss delta (margin - (margin+support)); positive = support HELPS
    ufam = np.array(sorted(set(fam))); d = []
    for _ in range(NBOOT):
        keep = set(rng.choice(ufam, len(ufam), replace=True)); mk = np.array([f in keep for f in fam])
        if mk.sum() < 10: continue
        d.append(logloss(oof_m[mk], y[mk]) - logloss(oof_ms[mk], y[mk]))
    d = np.array(d); lo, hi, mn = np.percentile(d, 2.5), np.percentile(d, 97.5), d.mean()
    verdict = "SUPPORT HELPS (2nd coordinate REAL)" if lo > 0 else ("support HURTS" if hi < 0 else "REDUNDANT (support ≈ margin per-query)")
    print(f"  Δlogloss (margin − 2D) = {mn:+.3f} [{lo:+.3f},{hi:+.3f}]  ->  {verdict}")
    # ship the winning calibrator (fit on all data) as L3-format binned table
    if lo > 0:                                                     # 2D wins -> ship (margin,support)
        w, mu, sd = fit_lr(X, y); me = np.linspace(0, 1, 9); se = np.linspace(s.min(), s.max(), 9)
        grid = np.array([[(me[i]+me[i+1])/2, (se[j]+se[j+1])/2] for j in range(8) for i in range(8)])
        P = prob(grid, w, mu, sd).reshape(8, 8)                    # [support_bin][margin_bin]
        table = {"family": {"margin_edges": me.tolist(), "support_edges": se.tolist(), "p": P.tolist()}}
        json.dump(table, open(os.path.join(CALIB_OUT, f"calibration_family_{enc}.json"), "w"))
        print(f"  -> shipped (margin,support) calibration_family_{enc}.json")
print("\nGEOMETRY-COHERENCE CHECK: does support help v10.9 but NOT v9? (support-coordinate = v10.9-specific property of its spreading)")
print("DONE")
