#!/usr/bin/env python3
"""
recalibrate.py — the two-population operating curve: can we recover honest abstention on the 20kb index for free?

The capability benchmark found v10.9 places better but ABSTAINS LESS on novelty (0.44 vs 0.62). That's a
CALIBRATION question, and it lives on TWO DISJOINT populations:
  in-reference (family present)   -> TRUE-PLACEMENT rate  (of commits, how many right?)   y-axis, want HIGH
  withheld-family (family absent) -> FALSE-COMMIT rate    (committed to a family at all?)  x-axis, want LOW
Sweeping a confidence score traces a two-population ROC. Metric = TP@FC (true-placement achievable at a fixed
false-commit rate) — operating-point-agnostic. We compare SCORES: margin-only vs support-aware (support = mean
neighbor similarity = a novelty proxy: withheld-family queries are farther, lower support). And ENCODERS: v9 vs
v10.9, so we see whether v10.9's curve DOMINATES or whether it sacrificed novelty-separability (a saturation floor).

PRE-REGISTERED BRANCHES (committed before numbers):
  A) v10.9 margin curve dominates v9's (TP@FC higher, CI clears 0) -> abstention "regression" was a mis-set
     default dial. Re-set operating point. #4 must beat an already-good baseline (HIGH BAR).
  B) support-aware > margin-only (TP@FC higher)                    -> adopt 2D (margin,support) calibration; free
     win; #4 again faces a high bar.
  C) neither reaches good TP@low-FC                                -> genuine saturation floor; #4's cross-scale
     signal is ESSENTIAL, with a precise target = the gap to fill.
Reuses cached v9+v10.9 vectors through L3. CPU, GPU-masked. Run:  CUDA_VISIBLE_DEVICES="" python recalibrate.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_evidence import build_evidence, RANKS, DEPTH
OUT = "/zfs_raid/SentryBio/exp_v9_cascade"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
tax = json.load(open(FTAX)); rng = np.random.RandomState(0); KNN = 15; NBOOT = 2000
FCS = [0.05, 0.10, 0.20]
def lin(g): return {r: tax[int(g)].get(r) for r in RANKS}
def load(enc):
    for nm in [f"genus_encoded_{enc}_20kb.npz", "genus_encoded_20kb.npz" if enc == "v9" else None]:
        if nm and os.path.exists(os.path.join(OUT, nm)): return np.load(os.path.join(OUT, nm), allow_pickle=True)
    return None
def place(Xr, gr, taxonomy, Vq):
    cos = Xr @ Vq.T; idxs = set()
    for w in range(Vq.shape[0]): idxs.update(np.argpartition(-cos[:, w], KNN)[:KNN].tolist())
    idxs = np.array(sorted(idxs)); wt = cos[idxs].max(1)
    return build_evidence([(int(gr[j]), float(wt[i])) for i, j in enumerate(idxs)], taxonomy, k_neighborhood=KNN)

def features(z):
    """returns in-ref (margin, support, correct, family) and withheld-fam (margin, support, family)."""
    Xr, gr = z["Xr"], z["gr"]; taxonomy = {int(g): lin(g) for g in np.unique(gr)}
    inr, wf = [], []
    for g in np.unique(z["gi"]):                    # in-reference (family present)
        ev = place(Xr, gr, taxonomy, z["Xi"][z["gi"] == g]); fr = next((r for r in ev.ranks if r.rank == "family"), None)
        if fr and not fr.gap:
            inr.append((fr.margin, ev.support, fr.top == lin(g)["family"], str(lin(g)["family"])))
    for g in np.unique(z["gf"]):                    # withheld-family (family absent -> any commit is false)
        ev = place(Xr, gr, taxonomy, z["Xf"][z["gf"] == g]); fr = next((r for r in ev.ranks if r.rank == "family"), None)
        if fr and not fr.gap:
            wf.append((fr.margin, ev.support, str(lin(g)["family"])))
    return inr, wf

def score(m, s, kind):
    if kind == "margin": return m
    if kind == "support": return s
    return m * (s ** 5)                              # support-weighted: low support kills a high margin

def tp_at_fc(inr, wf, kind, fc):
    si = np.array([score(m, s, kind) for m, s, _, _ in inr]); ci = np.array([c for _, _, c, _ in inr])
    sw = np.array([score(m, s, kind) for m, s, _ in wf])
    thr = np.quantile(sw, 1 - fc)                    # threshold giving false-commit ~= fc on withheld
    return float((ci[si >= thr]).sum() / max(len(si), 1))   # true-placement (correct AND committed) among in-ref

def boot_tp(inr, wf, kind, fc, fams_i):
    uf = np.array(sorted(set(fams_i))); out = []
    inr_a = np.array(inr, object)
    for _ in range(NBOOT):
        keep = set(rng.choice(uf, len(uf), replace=True))
        idx = [i for i, r in enumerate(inr) if r[3] in keep]
        if len(idx) < 5: continue
        out.append(tp_at_fc([inr[i] for i in idx], wf, kind, fc))
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)

data = {}
for enc in ["v9", "v10.9"]:
    z = load(enc); inr, wf = features(z); data[enc] = (inr, wf)
    ms_i = np.mean([s for _, s, _, _ in inr]); ms_w = np.mean([s for _, s, _ in wf])
    print(f"[{enc}] in-ref n={len(inr)} (support {ms_i:.3f})  withheld-fam n={len(wf)} (support {ms_w:.3f})  "
          f"-> support {'SEPARATES' if ms_i - ms_w > 0.01 else 'flat'}")

print("\n===== TWO-POPULATION OPERATING CURVE: TP@FC (true-placement at fixed false-commit) =====")
print(f"  {'enc':<7}{'score':<9}" + "".join(f"TP@FC={fc:<7}" for fc in FCS))
for enc in ["v9", "v10.9"]:
    inr, wf = data[enc]
    for kind in ["margin", "support", "margin*sup^5"]:
        k = {"margin*sup^5": "combo"}.get(kind, kind); k2 = "combo" if kind == "margin*sup^5" else kind
        row = "".join(f"{tp_at_fc(inr, wf, kind, fc):<11.3f}" for fc in FCS)
        print(f"  {enc:<7}{k2:<9}{row}")

print("\n===== PRE-REGISTERED VERDICT (TP@FC=0.10, family-bootstrapped) =====")
fams9 = [r[3] for r in data["v9"][0]]; fams10 = [r[3] for r in data["v10.9"][0]]
for enc, fams in [("v9", fams9), ("v10.9", fams10)]:
    inr, wf = data[enc]
    for kind in ["margin", "combo"]:
        kk = "margin*sup^5" if kind == "combo" else kind
        tp = tp_at_fc(inr, wf, kk, 0.10); lo, hi = boot_tp(inr, wf, kk, 0.10, fams)
        print(f"  {enc:<7}{kind:<8} TP@FC=0.10 = {tp:.3f}  [{lo:.3f},{hi:.3f}]")
print("  A) v10.9 margin >= v9 margin -> dial was mis-set (fix free).  B) combo > margin -> adopt 2D calib (free).")
print("  C) neither reaches high TP@FC=0.10 -> saturation floor -> #4 cross-scale is ESSENTIAL (target = the gap).")
