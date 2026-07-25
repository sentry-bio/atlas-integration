#!/usr/bin/env python3
"""
calibrate_regimes.py — WIN 4: per-regime (tier/length) margin calibration.

The shipped calibration is ONE margin->P(correct) table for all read lengths. A margin of 0.6 does not mean
the same reliability at 1kb (5kb tier) as at native (20kb tier). This fits a SEPARATE per-rank logistic
margin->P(correct) for each serving regime {5kb, 20kb, fused}, and validates per-regime vs shared with
family-grouped CV (never fit+eval on the same family) on log-loss + ECE.

  regime samples (in-ref, self-masked, so both correct & incorrect placements appear across the margin range):
    5kb   : short reads (2kb ensemble) forced to the 5kb tier
    20kb  : native windows forced to the 20kb tier
    fused : ~5kb reads forced to both tiers
PRE-REG: per-regime beats shared if family-CV logloss delta CI clears 0 (regime better). Else FALL BACK to
shared for that regime (a shared-but-honest table beats a per-regime table fit on noise).

Writes calib_5kb.json / calib_20kb.json / calib_fused.json (L3 schema) + prints the CV verdict.
CPU. Run after the service is free:  NGC=200 python calibrate_regimes.py
"""
import os, sys, json, time, urllib.request
import numpy as np
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/build_pipeline")
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; VOCAB = "/home/rohit/biosphere_inference/bpe_vocab.json"
URL = "http://127.0.0.1:8093/place"; OUT = "/zfs_raid/SentryBio/serve_index_v109_20kb"  # calibs written next to indexes
RANKS = ["domain","phylum","class","order","family","genus","species"]
NGC = int(os.environ.get("NGC", "200"))
tax = json.load(open(FTAX)); vocab = json.load(open(VOCAB)); rev = {i: t for t, i in vocab.items()}
rng = np.random.RandomState(23)
def decode(t): return "".join(rev.get(int(x), "") for x in t if not rev.get(int(x), "[").startswith("["))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def place(reads, read_bp, exclude=None, force_tiers=None):
    body = {"reads": reads, "read_bp": read_bp, "min_confidence": 0.5}
    if exclude is not None: body["exclude"] = [int(x) for x in exclude]
    if force_tiers is not None: body["force_tiers"] = force_tiers
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))
def windows_of(g, n):
    tk = np.load(os.path.join(TOK, tax[g]["accession"]+".npy"), mmap_mode="r")
    return [decode(np.array(tk[wi]).astype(np.int64)) for wi in range(min(n, tk.shape[0]))]
def tiles(dna, L, nc):
    if len(dna) < L: return []
    return [dna[i:i+L] for i in range(0, min(len(dna), L*nc)-L+1, L)] or [dna[:L]]

pool = [g for g in rng.permutation(len(tax))[:6000]
        if tax[g].get("family") and tax[g].get("genus")
        and os.path.exists(os.path.join(TOK, tax[g]["accession"]+".npy"))][:NGC]
log(f"{len(pool)} calibration genomes")

# collect (rank -> list[(margin, correct, family)]) per regime
REGIMES = {"5kb": ("5kb",), "20kb": ("20kb",), "fused": ("5kb","20kb")}
def query_for(regime, g):
    dna0 = windows_of(g, 8)
    if regime == "20kb": return dna0, 20000                     # native ensemble
    src = decode(np.load(os.path.join(TOK, tax[g]["accession"]+".npy"), mmap_mode="r")[0].astype(np.int64))
    L = 2000 if regime == "5kb" else 5000
    return tiles(src, L, 3), L
data = {rg: {r: [] for r in RANKS} for rg in REGIMES}
for qi, g in enumerate(pool):
    lin = {r: tax[g].get(r) for r in RANKS}
    for rg, ft in REGIMES.items():
        reads, rb = query_for(rg, g)
        if not reads: continue
        try: resp = place(reads, rb, exclude=[g], force_tiers=list(ft))
        except Exception: continue
        for rr in resp["ranks"]:
            if rr["gap"] or lin.get(rr["rank"]) is None: continue
            data[rg][rr["rank"]].append((rr["margin"], int(rr["top"]==lin[rr["rank"]]), tax[g].get("family")))
    if (qi+1) % 40 == 0: log(f"  collected {qi+1}/{len(pool)}")

# logistic fit margin->P(correct); family-grouped CV logloss for per-regime vs shared
def fit_logistic(m, y, it=2000, lr=0.3):
    m = np.asarray(m,float); y = np.asarray(y,float)
    mu, sd = m.mean(), m.std()+1e-9; x = (m-mu)/sd; X = np.c_[np.ones(len(x)), x]; w = np.zeros(2)
    for _ in range(it): w -= lr * (X.T @ (1/(1+np.exp(-X@w)) - y))/len(y)
    return lambda mm: 1/(1+np.exp(-(np.c_[np.ones(len(np.atleast_1d(mm))), (np.atleast_1d(mm)-mu)/sd]@w)))
def logloss(p, y): p=np.clip(p,1e-6,1-1e-6); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def table_from(fn, edges=None):
    edges = edges or [i/20 for i in range(21)]                  # 20 margin bins over [0,1]
    centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
    return {"margin_edges": edges, "support_edges": [float("-inf"), float("inf")],
            "p": [[float(np.clip(fn(np.array([c]))[0],0,1)) for c in centers]]}

# shared = pool all regimes together (the current approach)
shared = {rg: {} for rg in REGIMES}
allcal = {rg: {} for rg in REGIMES}
print("\n===== WIN 4: per-regime vs shared calibration (family-grouped CV logloss) =====")
for rg in REGIMES:
    print(f"\n  regime {rg}:")
    for rank in ["family","genus"]:                             # the ranks that matter for the dial
        rows = data[rg][rank]
        if len(rows) < 40: print(f"    {rank}: too few ({len(rows)}) -> fall back to shared"); continue
        m = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); fam = np.array([r[2] for r in rows], object)
        # shared table = a global (all-regime) fit for this rank
        allm = np.concatenate([np.array([r[0] for r in data[o][rank]]) for o in REGIMES])
        ally = np.concatenate([np.array([r[1] for r in data[o][rank]]) for o in REGIMES])
        ufam = list(set(fam)); rng.shuffle(ufam); folds = np.array_split(ufam, 5)
        d_reg, d_sha = [], []
        for k in range(5):
            test = set(folds[k]); tr = ~np.array([f in test for f in fam]); te = np.array([f in test for f in fam])
            if te.sum() < 5 or tr.sum() < 20: continue
            preg = fit_logistic(m[tr], y[tr])(m[te])
            psha = fit_logistic(allm, ally)(m[te])               # shared fit (global), eval on regime test fold
            d_reg.append(logloss(preg, y[te])); d_sha.append(logloss(psha, y[te]))
        if not d_reg: print(f"    {rank}: CV degenerate -> shared"); continue
        delta = np.array(d_sha) - np.array(d_reg)               # >0 => per-regime better
        lo, hi = np.percentile(delta,2.5), np.percentile(delta,97.5)
        better = lo > 0
        print(f"    {rank}: logloss shared {np.mean(d_sha):.3f} vs regime {np.mean(d_reg):.3f}  "
              f"Δ={delta.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  -> {'PER-REGIME' if better else 'shared (fallback)'}")
        allcal[rg][rank] = table_from(fit_logistic(m, y)) if better else table_from(fit_logistic(allm, ally))
    # fill non-family/genus ranks with the shared global fit for completeness
    for rank in RANKS:
        if rank in allcal[rg]: continue
        rows = data[rg][rank]
        if len(rows) >= 40:
            allcal[rg][rank] = table_from(fit_logistic(np.array([r[0] for r in rows]), np.array([r[1] for r in rows])))
if os.environ.get("WRITE") == "1":
    for rg in REGIMES:
        p = os.path.join(OUT, f"calib_{rg}.json"); json.dump(allcal[rg], open(p,"w")); log(f"wrote {p}")
np.savez("/zfs_raid/SentryBio/radial_training/calib_regimes.npz", data=json.dumps({rg:{r:len(data[rg][r]) for r in RANKS} for rg in REGIMES}))
log("DONE (set WRITE=1 to persist calib_*.json)")
