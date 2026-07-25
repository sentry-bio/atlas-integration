#!/usr/bin/env python3
"""
matched_contrast.py — TERMINAL decider: does the radial signal carry complexity BEYOND genome size?

Runs ONCE to conclude, not to refine. GTDB-corrected: match genus BASE name (strip "_L"/"_A" suffix) against
curated REDUCTIVE (parasites/symbionts) vs FREE-LIVING sets — a clean two-population contrast, not
reductive-vs-random. Size-match each reductive genome to a free-living genome of the same gene count (±band),
then a PAIRED SIGN TEST on the signal.

PRE-REGISTERED VERDICT (committed here, do not renegotiate after seeing the number):
  concordance in [0.35, 0.65]  -> radius is a SIZE axis. CLOSE the radius-as-complexity thread. Do NOT build Stage 1.
  concordance > 0.65 or < 0.35 -> BEYOND-SIZE structure exists. Thread stays open ONLY for composition-vs-complexity.
CAVEAT (asymmetric conclusiveness): reductive genomes are AT-biased, so a POSITIVE = "composition OR complexity";
a ~0.50 NEGATIVE is cleanly "pure size". If r0-prior separates but raw-spread doesn't -> it's composition, which
argues AGAINST a learned radial head and FOR an analytic overlay.
Need >=15 usable pairs; if fewer, band widens to 0.20 once, then reports n honestly.

CPU, GPU-masked. Run:  CUDA_VISIBLE_DEVICES="" python matched_contrast.py
"""
import os, sys, json, glob
import numpy as np, torch
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts")
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; EGGNOG = "/zfs_raid/SentryBio/eggnog/output"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
NTOK = 4096; NWIN = 8
# GTDB base names (matched via genus.split("_")[0]) — reductive parasites/symbionts, curated:
REDUCTIVE = {"Mycoplasma","Metamycoplasma","Mycoplasmoides","Mycoplasmopsis","Malacoplasma","Ureaplasma",
             "Buchnera","Rickettsia","Orientia","Wolbachia","Anaplasma","Ehrlichia","Neorickettsia",
             "Chlamydia","Chlamydophila","Coxiella","Bartonella","Borrelia","Borreliella"}
# free-living controls spanning small->large gene counts (so we can size-match across the range):
FREELIVING = {"Pelagibacter","Prochlorococcus","Synechococcus","Escherichia","Pseudomonas","Bacillus",
              "Vibrio","Staphylococcus","Bacteroides","Streptomyces","Caulobacter","Rhodobacter"}
def base(g): return str(g).split("_")[0] if g else ""
tax = json.load(open(FTAX)); rng = np.random.RandomState(0)
idx = {os.path.basename(f).split(".emapper")[0]: f for f in glob.glob(os.path.join(EGGNOG, "*", "*.emapper.annotations"))}
def genecount(acc):
    p = idx.get(acc)
    if not p: return None
    n = 0
    with open(p) as fh:
        for line in fh:
            if not line.startswith("#") and line.strip(): n += 1
    return n
def has_tok(g): return os.path.exists(os.path.join(TOK, tax[g]["accession"] + ".npy"))

red_g, free_g = [], []
for g in range(len(tax)):
    a = tax[g].get("accession"); b = base(tax[g].get("genus"))
    if not a or a not in idx or not has_tok(g): continue
    if b in REDUCTIVE: red_g.append(g)
    elif b in FREELIVING: free_g.append(g)
print(f"reductive: {len(red_g)}  free-living: {len(free_g)}")
if len(red_g) < 5 or len(free_g) < 10: sys.exit(f"too few in one group (red {len(red_g)}, free {len(free_g)})")
red_gc = {g: genecount(tax[g]["accession"]) for g in red_g}
free_gc = {g: genecount(tax[g]["accession"]) for g in free_g}
free_arr = np.array([[g, c] for g, c in free_gc.items() if c], float)

def make_pairs(band):
    pairs, used = [], set()
    for g, c in sorted(red_gc.items(), key=lambda kv: (kv[1] or 1e9)):
        if not c: continue
        cand = [x for x in free_arr[np.abs(free_arr[:, 1] - c) <= band * c] if int(x[0]) not in used]
        if not cand: continue
        best = min(cand, key=lambda x: abs(x[1] - c)); ctrl = int(best[0]); used.add(ctrl)
        pairs.append((g, ctrl, c, int(best[1])))
    return pairs
pairs = make_pairs(0.15)
if len(pairs) < 15: pairs = make_pairs(0.20)      # widen once, then accept
print(f"formed {len(pairs)} size-matched pairs" + (f"; reductive gene-count {min(p[2] for p in pairs):.0f}-{max(p[2] for p in pairs):.0f}" if pairs else ""))
if len(pairs) < 8: sys.exit(f"only {len(pairs)} pairs — underpowered; reductive genomes may lack size-matched free-living controls")

enc = load_effective_encoder(V9, V109, device="cpu"); m = enc.model
def signals(g):
    tk = np.load(os.path.join(TOK, tax[g]["accession"] + ".npy"), mmap_mode="r"); rows = []
    for wi in range(min(NWIN, tk.shape[0])):
        t = np.array(tk[wi, :NTOK]).astype(np.int64)
        rows.append(np.pad(t, (0, NTOK - len(t)), constant_values=PAD_ID) if len(t) < NTOK else t)
    bt = torch.from_numpy(np.stack(rows)).long()
    with torch.no_grad():
        r0, _ = m.radial_head.deterministic_prior(bt); z0 = m.encode_raw(bt).float().numpy()
    U = z0 / (np.linalg.norm(z0, axis=1, keepdims=True) + 1e-9); C = U @ U.T; n = len(U)
    raw = float((1 - C)[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0
    return float(r0.float().mean()), raw
S = {}
for i, (g, ctrl, _, _) in enumerate(pairs):
    for x in (g, ctrl):
        if x not in S: S[x] = signals(x)
    if (i + 1) % 15 == 0: print(f"  encoded {i+1}/{len(pairs)}")

def sign_test(k):
    conc = tot = 0
    for g, ctrl, _, _ in pairs:
        d = S[g][k] - S[ctrl][k]          # reductive - free-living; expect < 0 if signal>size (reductive simpler)
        if abs(d) < 1e-9: continue
        tot += 1; conc += (d < 0)
    return conc / max(tot, 1), tot
print("\n===== MATCHED-SIZE CONTRAST (paired sign test) — TERMINAL =====")
print(f"  n pairs = {len(pairs)}   (PRE-REGISTERED: 0.35-0.65 = size axis / CLOSE thread; else = beyond-size)")
for name, k in [("r0-prior  (reductive < free-living?)", 0), ("raw-spread (reductive < free-living?)", 1)]:
    c, n = sign_test(k)
    v = "SIZE AXIS (~0.5) → CLOSE" if 0.35 <= c <= 0.65 else "BEYOND-SIZE structure"
    print(f"  {name:<40} concordance={c:.2f} (n={n})  -> {v}")
print("  Caveat: 'beyond-size' may be composition (AT-bias), not complexity. r0-only separation => composition.")
