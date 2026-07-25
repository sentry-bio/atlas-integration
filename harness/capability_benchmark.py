#!/usr/bin/env python3
"""
capability_benchmark.py — does v10.9 LIFT over v9 on real placement (not self-placement)? Scoped to 20kb.

Self-placement tested the plumbing. This tests capability + lift, on the SAME cached sampling for both encoders:
  PRIMARY  — HELD-OUT placement (in-ref queries: genome held out, family present): family & genus accuracy,
             v10.9 - v9 delta with FAMILY-BOOTSTRAP CI (effective n = families, not queries).
  HONEST   — WITHHELD-FAMILY queries (family absent): does it correctly ABSTAIN above family rather than
             confidently mis-place? (resolved_to at order-or-shallower = honest; deep wrong family = bad.)
Reuses L3 build_evidence. PRE-REGISTERED: lift is REAL only if the bootstrap CI clears 0. Report scoped to 20kb
(v10.9's strong regime); do NOT generalize to short reads.

CPU, GPU-masked. Run:  CUDA_VISIBLE_DEVICES="" python capability_benchmark.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_evidence import build_evidence, RANKS, DEPTH
OUT = "/zfs_raid/SentryBio/exp_v9_cascade"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
tax = json.load(open(FTAX)); rng = np.random.RandomState(0); KNN = 15; NBOOT = 2000
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

def eval_enc(z):
    Xr, gr = z["Xr"], z["gr"]; taxonomy = {int(g): lin(g) for g in np.unique(gr)}
    fi, gi_, oi = RANKS.index("family"), RANKS.index("genus"), RANKS.index("order")
    # held-out in-ref placement
    inrec = []                                   # (correct_fam, correct_gen, true_family)
    for g in np.unique(z["gi"]):
        Vq = z["Xi"][z["gi"] == g]; ev = place(Xr, gr, taxonomy, Vq)
        rk = {r.rank: r for r in ev.ranks}; truth = lin(g)
        inrec.append((rk.get("family", type("", (), {"top": None})).top == truth["family"],
                      rk.get("genus", type("", (), {"top": None})).top == truth["genus"], str(truth["family"])))
    # withheld-family honest abstention
    honest = 0; nfam = 0
    for g in np.unique(z["gf"]):
        Vq = z["Xf"][z["gf"] == g]; ev = place(Xr, gr, taxonomy, Vq); d = ev.decide(min_confidence=0.5)
        nfam += 1
        # honest = did NOT confidently resolve TO a (wrong) family or deeper
        honest += (d.resolved_to is None or DEPTH[d.resolved_to] < fi)
    return np.array([r[0] for r in inrec]), np.array([r[1] for r in inrec]), \
           np.array([r[2] for r in inrec], object), honest / max(nfam, 1)

z9, z10 = load("v9"), load("v10.9")
f9, g9, fam9, h9 = eval_enc(z9); f10, g10, fam10, h10 = eval_enc(z10)
assert list(fam9) == list(fam10), "sampling mismatch — not paired"  # same order (same seed/sampling)

def boot_delta(a9, a10, fams):
    uf = np.array(sorted(set(fams))); d = []
    for _ in range(NBOOT):
        keep = set(rng.choice(uf, len(uf), replace=True))
        m = np.array([x in keep for x in fams])
        if m.sum() < 5: continue
        d.append(a10[m].mean() - a9[m].mean())
    d = np.array(d); return a9.mean(), a10.mean(), (np.percentile(d, 2.5), np.percentile(d, 97.5), d.mean())

print("===== CAPABILITY BENCHMARK (v9 vs v10.9, held-out placement, 20kb) =====")
print(f"  n held-out queries = {len(f9)}   n families ~ {len(set(fam9))}")
for name, a9, a10 in [("family placement", f9, f10), ("genus placement", g9, g10)]:
    m9, m10, (lo, hi, mn) = boot_delta(a9, a10, fam9)
    sig = "SIG+" if lo > 0 else ("SIG-" if hi < 0 else "ns")
    print(f"  {name:<18} v9={m9:.3f}  v10.9={m10:.3f}  Δ={mn:+.3f} [{lo:+.3f},{hi:+.3f}]  {sig}")
print(f"  withheld-family HONEST-abstention rate:  v9={h9:.2f}  v10.9={h10:.2f}  (higher = correctly refuses to mis-place)")
print("  SCOPE: 20kb queries only (v10.9's strong regime). Lift is REAL iff CI clears 0. Do not generalize to short reads.")
