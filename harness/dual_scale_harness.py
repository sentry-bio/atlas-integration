#!/usr/bin/env python3
"""
dual_scale_harness.py — the decisive #4 test: does 5kb+20kb FUSION beat 20kb-only? Against the LIVE full indexes.

Queries BOTH full indexes (1.9M vec each) with family-masking to simulate in-ref vs withheld-family, on the SAME
query genomes (self-exclude = in-ref; family-exclude = withheld). Fusion = per-index RANK-NORMALIZE sims (the #1
trap: raw cosines differ across scales) THEN pool into L3. Compares, with the Thread-A rigor:
  1) operating curve: TP@FC=0.10 for 20kb-only vs fusion  (baseline to beat = 0.492 margin-only)
  2) cross-scale AGREEMENT (5kb-vote==20kb-vote) as a calibration feature — does it beat margin? (expect null:
     agreement is another density signal; but scale-STABILITY ≠ single-scale density, so test honestly)
PRE-REGISTERED: fusion helps TP@FC (CI clears baseline) OR agreement beats margin (CI clears 0) -> dual-scale
earns hosting. Else -> host 20kb-only; fusion buys smoothness not novelty (scope honestly).

CPU. ~10-15min. Run: CUDA_VISIBLE_DEVICES="" python dual_scale_harness.py
"""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts"); sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
from atlas_evidence import build_evidence, RANKS
V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20 = "/zfs_raid/SentryBio/serve_index_v109_20kb"; D5 = "/zfs_raid/SentryBio/serve_index_v109_5kb"
NQ = int(os.environ.get("NQ", "120")); KREF = 4; KNN = 15; OVER = 16
tax = json.load(open(FTAX)); rng = np.random.RandomState(1)
def lin(g): return {r: tax[int(g)].get(r) for r in RANKS}
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("loading indexes + encoder…")
i20 = faiss.read_index(os.path.join(D20, "index.faiss")); g20 = np.load(os.path.join(D20, "gids.npy"), mmap_mode="r")
i5 = faiss.read_index(os.path.join(D5, "index.faiss")); g5 = np.load(os.path.join(D5, "gids.npy"), mmap_mode="r")
enc = load_effective_encoder(V9, V109, device="cpu"); m = enc.model
fam_of = {int(g): lin(int(g)).get("family") for g in np.unique(np.asarray(g20))}
from collections import defaultdict
fam_gids = defaultdict(set)
for gid, f in fam_of.items(): fam_gids[f].add(gid)
taxonomy = {gid: lin(gid) for gid in fam_of}
log(f"indexes {i20.ntotal:,}/{i5.ntotal:,} vec; {len(fam_gids)} families")

import torch
def encode(gid, ntok):
    tk = np.load(os.path.join(TOK, tax[gid]["accession"] + ".npy"), mmap_mode="r"); rows = []
    for wi in range(min(KREF, tk.shape[0])):
        t = np.array(tk[wi, :ntok]).astype(np.int64)
        rows.append(np.pad(t, (0, ntok - len(t)), constant_values=PAD_ID) if len(t) < ntok else t)
    return enc.encode(rows)
def search(index, gids, qv, exclude):
    kf = KNN * OVER; D, I = index.search(np.asarray(qv, "float32"), kf); out = []
    for r in range(len(qv)):
        kept = 0
        for j in range(kf):
            wid = int(I[r, j])
            if wid < 0: continue
            gg = int(gids[wid])
            if gg in exclude: continue
            out.append((gg, float(D[r, j]))); kept += 1
            if kept >= KNN: break
    return out
def ranknorm(nb):
    if not nb: return []
    s = np.array([x[1] for x in nb]); rk = s.argsort().argsort() / max(len(s) - 1, 1)
    return [(nb[i][0], float(rk[i])) for i in range(len(nb))]
def famtop(ev):
    fr = next((r for r in ev.ranks if r.rank == "family"), None)
    return (fr.top, fr.margin, ev.support) if fr and not fr.gap else (None, 0.0, ev.support)

cands = [g for g in rng.permutation(len(tax))[:6000] if tax[g].get("family") and os.path.exists(os.path.join(TOK, tax[g]["accession"] + ".npy"))][:NQ]
log(f"{len(cands)} query genomes")
rec = []   # (tier, margin20, marginF, support20, agree, correct20, correctF)
for qi, g in enumerate(cands):
    q20, q5 = encode(g, 4096), encode(g, 1000); truth = lin(g)["family"]
    for tier, excl in [("in", {g}), ("wf", fam_gids[lin(g)["family"]])]:
        n20 = search(i20, g20, q20, excl); n5 = search(i5, g5, q5, excl)
        if not n20: continue
        f20, mar20, sup20 = famtop(build_evidence(n20, taxonomy, k_neighborhood=KNN))
        f5, _, _ = famtop(build_evidence(n5, taxonomy, k_neighborhood=KNN))
        fF, marF, _ = famtop(build_evidence(ranknorm(n20) + ranknorm(n5), taxonomy, k_neighborhood=KNN))
        c20 = int(f20 == truth) if tier == "in" else 0
        cF = int(fF == truth) if tier == "in" else 0
        rec.append((tier, mar20, marF, sup20, int(f20 == f5), c20, cF))
    if (qi + 1) % 30 == 0: log(f"  {qi+1}/{len(cands)}")

R = np.array([r[1:] for r in rec], float); tier = np.array([r[0] for r in rec], object)
mar20, marF, sup20, agree, c20, cF = R.T
inref, wf = tier == "in", tier == "wf"
def tp_at_fc(score, correct, fc=0.10):
    thr = np.quantile(score[wf], 1 - fc); return float((score[inref & (correct == 1)] >= thr).sum() / max(inref.sum(), 1))
print("\n===== DUAL-SCALE HARNESS (live full indexes, family-masked) =====")
print(f"  n queries: {int(inref.sum())} in-ref, {int(wf.sum())} withheld-family")
print(f"  20kb-only  TP@FC=0.10 = {tp_at_fc(mar20, c20):.3f}   (baseline to beat = 0.492)")
print(f"  FUSION     TP@FC=0.10 = {tp_at_fc(marF, cF):.3f}")
print(f"  cross-scale AGREEMENT rate: in-ref={agree[inref].mean():.2f}  withheld={agree[wf].mean():.2f}  "
      f"(separates? {'YES' if agree[inref].mean()-agree[wf].mean()>0.05 else 'no'})")
# does agreement add to margin? quick logistic delta (in-sample; directional)
def fit(Xc, y, it=1500, lr=0.5):
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9; Xs = (Xc - mu) / sd; Xb = np.c_[np.ones(len(Xs)), Xs]; w = np.zeros(Xb.shape[1])
    for _ in range(it): w -= lr * (Xb.T @ (1 / (1 + np.exp(-Xb @ w)) - y) / len(y))
    return lambda Z: 1 / (1 + np.exp(-np.c_[np.ones(len(Z)), (Z - mu) / sd] @ w))
def ll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
lab = (inref & (c20 == 1)).astype(float)                 # trustworthy-commit
Xm = mar20[:, None]; Xma = np.c_[mar20, agree]
pm, pma = fit(Xm, lab)(Xm), fit(Xma, lab)(Xma)
print(f"  agreement-adds-to-margin (in-sample logloss): margin {ll(pm,lab):.3f} vs margin+agree {ll(pma,lab):.3f}"
      f"  -> {'agreement helps' if ll(pma,lab)<ll(pm,lab)-0.003 else 'redundant/marginal'}")
np.savez(os.path.join("/zfs_raid/SentryBio/radial_training", "dual_scale.npz"), R=R, tier=tier)
print("DONE")
