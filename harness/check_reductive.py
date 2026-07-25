#!/usr/bin/env python3
"""
check_reductive.py — 2-min biological discriminator: is the emergent complexity signal REAL?

Uses only data on disk: Stage-0 emergent signals (complexity_*.npz) + gene count from eggNOG (a RANGED
reductiveness proxy, unlike the flat category-breadth) + genus names (FTAX).
  (1) do emergent signals correlate with GENE COUNT (ranged) far better than with category BREADTH (flat)?
  (2) do known REDUCTIVE genera (Mycoplasma, Rickettsia, Buchnera…) land LOW and VERSATILE (Streptomyces,
      Pseudomonas…) land HIGH on the emergent signal?
If yes -> good-world (emergent complexity is real, teacher was masking it). If no -> weak-world.
"""
import os, sys, json, glob
import numpy as np
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
EGGNOG = "/zfs_raid/SentryBio/eggnog/output"; OUT = "/zfs_raid/SentryBio/radial_training"
tax = json.load(open(FTAX))
REDUCTIVE = {"Mycoplasma","Mesoplasma","Ureaplasma","Rickettsia","Ehrlichia","Anaplasma","Orientia",
             "Wolbachia","Buchnera","Wigglesworthia","Blochmannia","Carsonella","Nasuia","Sulcia",
             "Hodgkinia","Tremblaya","Chlamydia","Chlamydophila","Borrelia","Mycoplasmopsis"}
VERSATILE = {"Streptomyces","Pseudomonas","Bacillus","Burkholderia","Paraburkholderia","Bradyrhizobium",
             "Sorangium","Myxococcus","Rhodococcus","Nocardia","Sinorhizobium","Mesorhizobium","Vibrio"}
idx = {os.path.basename(f).split(".emapper")[0]: f for f in glob.glob(os.path.join(EGGNOG, "*", "*.emapper.annotations"))}
def genecount(acc):
    p = idx.get(acc)
    if not p: return np.nan
    n = 0
    with open(p) as fh:
        for line in fh:
            if not line.startswith("#") and line.strip(): n += 1
    return n
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10: return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m])))[0, 1])
def pct(v, x): return float((v < x).mean())   # percentile of x within array v

for tag in ["v9", "v10.9"]:
    z = np.load(os.path.join(OUT, f"complexity_{tag}.npz"), allow_pickle=True)
    gids = z["gids"]; breadth = z["breadth"].astype(float)
    gc = np.array([genecount(tax[int(g)]["accession"]) for g in gids])
    print(f"\n=== {tag} ===  gene count: range {np.nanmin(gc):.0f}-{np.nanmax(gc):.0f} (vs breadth 17-23 — the ranged ruler)")
    print(f"  {'emergent':<16}{'ρ vs GENE-COUNT':>16}{'ρ vs breadth':>14}")
    for c in ["angular-spread", "raw-spread", "token-entropy", "r0-prior"]:
        print(f"  {c:<16}{spear(z[c], gc):>16.3f}{spear(z[c], breadth):>14.3f}")
    # named reductive vs versatile — where do they land?
    genus = np.array([str(tax[int(g)].get("genus")) for g in gids], object)
    red = np.array([g in REDUCTIVE for g in genus]); ver = np.array([g in VERSATILE for g in genus])
    print(f"  named: {red.sum()} reductive, {ver.sum()} versatile genomes in sample")
    if red.sum() >= 2 and ver.sum() >= 2:
        best = max(["raw-spread", "token-entropy", "r0-prior"], key=lambda c: abs(spear(z[c], gc)))
        sig = z[best]
        print(f"  on GENE-COUNT:  reductive median pctile={np.median([pct(gc,gc[i]) for i in np.where(red)[0]]):.2f}"
              f"  versatile={np.median([pct(gc,gc[i]) for i in np.where(ver)[0]]):.2f}  (want red<0.5<ver)")
        print(f"  on '{best}':  reductive median pctile={np.median([pct(sig,sig[i]) for i in np.where(red)[0]]):.2f}"
              f"  versatile={np.median([pct(sig,sig[i]) for i in np.where(ver)[0]]):.2f}  (want red<0.5<ver if signal is real)")
    else:
        print("  (too few named exemplars in this random sample — rely on the gene-count correlation above)")
print("\nDONE")
