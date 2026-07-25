#!/usr/bin/env python3
"""
fathom_earn.py — the EARN-ITS-PLACE test for conformal fathom (novelty as distance-to-manifold).

Claim: conformal typicality separates ON-manifold-AMBIGUOUS (in-ref, vote split) from OFF-manifold-NOVEL
(withheld-family) queries WHERE MARGIN CANNOT. Margin is low for BOTH (both look "uncertain"); typicality
should be HIGH for ambiguous (near real family members) and LOW for novel (isolated from all families).

isolation(query, family F) = 1 - mean(top-k similarity to F's members among the neighbors)  [high = novel]
conformal null[F] = ECDF of member isolations (each F-member's isolation to its own family, leave-one-out)
typicality = p_value(iso) = (1 + #{null >= iso})/(n+1)   [low p = novel]

CALIBRATION: sample genomes -> each's isolation to its OWN family -> per-family ECDF (>=MINF members) + global.
TEST: ambiguous = in-ref self-excluded margin<0.6 ; novel = withheld-family. Compute margin + typicality for
each; AUROC(typicality) vs AUROC(margin) at separating ambiguous(on) from novel(off), full + low-margin-only.
PRE-REG: AUROC(typicality) > 0.65 AND > AUROC(margin)+0.05 on the low-margin population -> conformal EARNS it.
CPU. NG_CAL=800 NG_TEST=400 python fathom_earn.py
"""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0,"/home/rohit"); sys.path.insert(0,"/home/rohit/sentrybio/scripts"); sys.path.insert(0,"/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
from atlas_evidence import build_evidence, RANKS
from collections import defaultdict
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK="/zfs_raid/SentryBio/tokenized_4096"; FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG_CAL=int(os.environ.get("NG_CAL","800")); NG_TEST=int(os.environ.get("NG_TEST","400"))
NWIN=8; K=15; KISO=5; MINF=8; MARGIN_AMB=0.6
tax=json.load(open(FTAX)); rng=np.random.RandomState(29)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
log("loading 20kb index + encoder…")
i20=faiss.read_index(os.path.join(D20,"index.faiss")); g20=np.load(os.path.join(D20,"gids.npy"),mmap_mode="r")
enc=load_effective_encoder(V9,V109,device="cpu")
uniq=np.unique(np.asarray(g20)); taxonomy={int(g):{r:tax[int(g)].get(r) for r in RANKS} for g in uniq}
famg=defaultdict(list)
for g in uniq:
    f=tax[int(g)].get("family")
    if f: famg[f].append(int(g))
def rows(tk,n):
    o=[]
    for wi in range(min(n,tk.shape[0])):
        t=np.array(tk[wi,:4096]).astype(np.int64); o.append(np.pad(t,(0,max(0,4096-len(t))),constant_values=PAD_ID)[:4096])
    return o
def search(qv,k,excl):
    kf=k*24; D,I=i20.search(np.asarray(qv,"float32"),kf); out=[]
    for r in range(len(qv)):
        kept=0
        for j in range(kf):
            wid=int(I[r,j])
            if wid<0: continue
            gg=int(g20[wid])
            if gg in excl: continue
            out.append((gg,float(D[r,j]))); kept+=1
            if kept>=k: break
    return out
def isolation(nb, fam):
    sims=sorted([s for gid,s in nb if taxonomy[gid].get("family")==fam], reverse=True)[:KISO]
    return 1.0 - (float(np.mean(sims)) if sims else 0.0)   # no family members near -> isolation 1.0
def fam_margin(nb):
    ev=build_evidence(nb, taxonomy, k_neighborhood=K); fr=next((r for r in ev.ranks if r.rank=="family"),None)
    return (fr.top, fr.margin) if fr and not fr.gap else (None,0.0)
pool=[g for g in rng.permutation(len(tax))[:12000] if tax[g].get("family") and os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))]
def encode_g(g):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r"); return enc.encode(rows(tk,NWIN))

# ── CALIBRATION: per-family member isolation ECDFs ──────────────────────────────────────────
log(f"calibration on {NG_CAL} genomes…")
percal=defaultdict(list); allcal=[]
for i,g in enumerate(pool[:NG_CAL]):
    nb=search(encode_g(g),K,{g}); tf=tax[g]["family"]
    iso=isolation(nb,tf); percal[tf].append(iso); allcal.append(iso)
    if (i+1)%100==0: log(f"  cal {i+1}/{NG_CAL}")
gnull=np.sort(np.array(allcal))
def pval(iso, fam):
    arr = np.array(percal[fam]) if len(percal.get(fam,[]))>=MINF else gnull
    return (1.0 + int(np.count_nonzero(arr>=iso)))/(len(arr)+1.0)
log(f"calibration done: global null n={len(gnull)}, per-family ECDFs={sum(1 for f in percal if len(percal[f])>=MINF)}")

# ── TEST: ambiguous(in-ref) vs novel(withheld-family) ───────────────────────────────────────
log(f"test on {NG_TEST} genomes…")
amb=[]; nov=[]   # each: (margin, typicality)
for i,g in enumerate(pool[NG_CAL:NG_CAL+NG_TEST]):
    qv=encode_g(g); tf=tax[g]["family"]
    nb_in=search(qv,K,{g}); ft,mg=fam_margin(nb_in)
    if ft is not None and mg<MARGIN_AMB:                     # ambiguous = on-manifold but vote-split
        amb.append((mg, pval(isolation(nb_in,ft), ft)))
    nb_wf=search(qv,K,set(famg[tf])); fw,mw=fam_margin(nb_wf) # novel = its family withheld
    if fw is not None:
        nov.append((mw, pval(isolation(nb_wf,fw), fw)))
    if (i+1)%100==0: log(f"  test {i+1}/{NG_TEST}  amb={len(amb)} nov={len(nov)}")
amb=np.array(amb); nov=np.array(nov)
def auroc(pos, neg):   # P(pos > neg), Mann-Whitney
    if len(pos)==0 or len(neg)==0: return float("nan")
    allv=np.concatenate([pos,neg]); r=allv.argsort().argsort().astype(float)+1
    rp=r[:len(pos)].sum(); return (rp - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
print(f"\n===== FATHOM EARN-ITS-PLACE (ambiguous n={len(amb)}, novel n={len(nov)}) =====")
print(f"  FULL population:")
print(f"    AUROC(typicality) amb>nov = {auroc(amb[:,1], nov[:,1]):.3f}")
print(f"    AUROC(margin)     amb>nov = {auroc(amb[:,0], nov[:,0]):.3f}   (expect ~0.5: margin can't tell them apart)")
# low-margin population (the decisive test: where margin is uninformative)
la=amb[amb[:,0]<MARGIN_AMB]; ln=nov[nov[:,0]<MARGIN_AMB]
at=auroc(la[:,1], ln[:,1]); am=auroc(la[:,0], ln[:,0])
print(f"  LOW-MARGIN population (margin<{MARGIN_AMB}, amb={len(la)} nov={len(ln)}) — the decisive test:")
print(f"    AUROC(typicality) = {at:.3f}")
print(f"    AUROC(margin)     = {am:.3f}")
earn = at>0.65 and at>am+0.05
print(f"  mean typicality: ambiguous {amb[:,1].mean():.3f} vs novel {nov[:,1].mean():.3f} (sep {amb[:,1].mean()-nov[:,1].mean():+.3f})")
print(f"  VERDICT: {'SHIP conformal fathom (earns its place)' if earn else 'NULL — typicality does not beat margin; keep margin novelty'}")
np.savez("/zfs_raid/SentryBio/maximize/fathom_earn.npz", amb=amb, nov=nov, gnull=gnull)
log("DONE")
