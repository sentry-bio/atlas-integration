#!/usr/bin/env python3
"""LEVER 5 (novelty dial) + LEVER 4 (coverage diagnostic), in-process, 8-window ensemble, k=15.
L5: in-ref (self-mask) vs withheld-family (mask whole family) family-confidence; sweep min_confidence to find the
    operating point where withheld-family FALSE-COMMIT <= target, report the in-ref recall cost -> recommend default.
L4: does family miss-rate correlate with available window-count (are under-covered genomes placed worse)?"""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0,"/home/rohit"); sys.path.insert(0,"/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
from atlas_evidence import build_evidence, RANKS
from collections import defaultdict
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK="/zfs_raid/SentryBio/tokenized_4096"; FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG=int(os.environ.get("NG","150")); NWIN=8; K=15
tax=json.load(open(FTAX)); rng=np.random.RandomState(17)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
log("loading + family map…")
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
    D,I=i20.search(np.asarray(qv,"float32"),k*24); out=[]
    for r in range(len(qv)):
        kept=0
        for j in range(k*24):
            wid=int(I[r,j])
            if wid<0: continue
            gg=int(g20[wid])
            if gg in excl: continue
            out.append((gg,float(D[r,j]))); kept+=1
            if kept>=k: break
    return out
def fam(nb):
    ev=build_evidence(nb,taxonomy,k_neighborhood=K); fr=next((x for x in ev.ranks if x.rank=="family"),None)
    return (fr.top,fr.confidence) if fr and not fr.gap else (None,0.0)
cands=[g for g in rng.permutation(len(tax))[:8000] if tax[g].get("family") and os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))][:NG]
log(f"{len(cands)} genomes")
rec=[]  # (inref_conf, inref_hit, withheld_conf, nwin)
for qi,g in enumerate(cands):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r"); nwin=int(tk.shape[0]); tf=tax[g]["family"]
    qv=enc.encode(rows(tk,NWIN))
    it,ic=fam(search(qv,K,{g})); wt,wc=fam(search(qv,K,set(famg[tf])))
    rec.append((ic,int(it==tf),wc,nwin))
    if (qi+1)%25==0: log(f"  {qi+1}/{len(cands)}")
R=np.array([r[:3] for r in rec],float); nwin=np.array([r[3] for r in rec])
ic,ih,wc=R[:,0],R[:,1],R[:,2]
print(f"\n===== LEVER 5: NOVELTY DIAL (n={len(rec)}) =====")
print(f"  family-confidence: in-ref {ic.mean():.3f} vs withheld-family {wc.mean():.3f} (sep {ic.mean()-wc.mean():+.3f})")
print(f"  {'min_conf':>9}{'novel-commit(FC)':>18}{'known-recall':>14}")
for mc in [0.3,0.4,0.5,0.6,0.7,0.8]:
    fc=(wc>=mc).mean()                       # withheld families that still commit = false commit
    recall=((ic>=mc)&(ih==1)).sum()/max(ih.sum(),1)   # correct known placements kept
    print(f"  {mc:>9.2f}{fc:>18.3f}{recall:>14.3f}")
# recommend: lowest mc with FC<=0.15
rec_mc=next((mc for mc in [0.3,0.4,0.5,0.6,0.7,0.8] if (wc>=mc).mean()<=0.15), 0.8)
print(f"  RECOMMEND default min_confidence ~ {rec_mc} (FC<=0.15): novel-commit {(wc>=rec_mc).mean():.3f}, known-recall {((ic>=rec_mc)&(ih==1)).sum()/max(ih.sum(),1):.3f}")
print(f"\n===== LEVER 4: COVERAGE DIAGNOSTIC =====")
for lo,hi in [(0,4),(4,8),(8,24),(24,10**9)]:
    m=(nwin>=lo)&(nwin<hi)
    if m.sum()>=5:
        print(f"  windows [{lo},{hi}): n={int(m.sum())}  family-acc {ih[m].mean():.3f}  median_windows {int(np.median(nwin[m]))}")
corr=np.corrcoef(nwin, ih)[0,1] if len(set(nwin))>1 else float('nan')
print(f"  corr(window-count, family-hit) = {corr:+.3f}  -> {'coverage matters (rebuild worthwhile)' if corr>0.15 else 'coverage NOT the lever'}")
log("DONE")
