#!/usr/bin/env python3
"""LEVER 1 — confirm k=5 beats shipped k=15 at PRODUCTION config (8-window native ensemble) before re-freezing
the scorer. In-process, own encoder+20kb index, family-present self-mask, family-bootstrap CI. One search@k=50
per read, evaluate all k by top-k truncation (matches serving pool semantics)."""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0,"/home/rohit"); sys.path.insert(0,"/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
from atlas_evidence import build_evidence, RANKS
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK="/zfs_raid/SentryBio/tokenized_4096"; FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG=int(os.environ.get("NG","120")); NWIN=8; KMAX=50; KS=[5,10,15,25]
tax=json.load(open(FTAX)); rng=np.random.RandomState(7)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
log("loading…")
i20=faiss.read_index(os.path.join(D20,"index.faiss")); g20=np.load(os.path.join(D20,"gids.npy"),mmap_mode="r")
enc=load_effective_encoder(V9,V109,device="cpu")
taxonomy={int(g):{r:tax[int(g)].get(r) for r in RANKS} for g in np.unique(np.asarray(g20))}
def rows_native(tk,n):
    out=[]
    for wi in range(min(n,tk.shape[0])):
        t=np.array(tk[wi,:4096]).astype(np.int64)
        out.append(np.pad(t,(0,max(0,4096-len(t))),constant_values=PAD_ID)[:4096])
    return out
def call(nb,rank):
    ev=build_evidence(nb,taxonomy,k_neighborhood=15); r=next((x for x in ev.ranks if x.rank==rank),None)
    return r.top if r and not r.gap else None
cands=[g for g in rng.permutation(len(tax))[:6000] if tax[g].get("family") and tax[g].get("genus") and os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))][:NG]
log(f"{len(cands)} genomes")
recs=[]
for qi,g in enumerate(cands):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r")
    qv=enc.encode(rows_native(tk,NWIN))
    D,I=i20.search(np.asarray(qv,"float32"),KMAX*4)
    per_read=[]
    for r in range(len(qv)):
        lst=[]
        for j in range(KMAX*4):
            wid=int(I[r,j])
            if wid<0: continue
            gg=int(g20[wid])
            if gg==g: continue
            lst.append((gg,float(D[r,j])))
            if len(lst)>=KMAX: break
        per_read.append(lst)
    res={}
    for k in KS:
        pooled=[]
        for lst in per_read: pooled+=lst[:k]
        res[k]=(int(call(pooled,"family")==tax[g]["family"]), int(call(pooled,"genus")==tax[g]["genus"]))
    recs.append((tax[g]["family"],res))
    if (qi+1)%20==0: log(f"  {qi+1}/{len(cands)}")
fams=np.array([r[0] for r in recs],object); uf=list(set(fams))
def boot(vals):
    by={f:vals[fams==f] for f in uf}
    bs=[np.mean([by[uf[i]].mean() for i in rng.randint(0,len(uf),len(uf))]) for _ in range(3000)]
    return float(vals.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
print(f"\n===== LEVER 1: k-confirm at 8-window ensemble (n={len(recs)}) =====")
print(f"  {'k':>4}{'family':>24}{'genus':>24}")
for k in KS:
    fa=np.array([r[1][k][0] for r in recs]); ga=np.array([r[1][k][1] for r in recs])
    fm,flo,fhi=boot(fa); gm,glo,ghi=boot(ga)
    print(f"  {k:>4}   {fm:.3f} [{flo:.3f},{fhi:.3f}]   {gm:.3f} [{glo:.3f},{ghi:.3f}]")
g5=np.array([r[1][5][1] for r in recs]); g15=np.array([r[1][15][1] for r in recs])
f5=np.array([r[1][5][0] for r in recs]); f15=np.array([r[1][15][0] for r in recs])
by={f:((g5-g15)[fams==f],(f5-f15)[fams==f]) for f in uf}
gd=[np.mean([by[uf[i]][0].mean() for i in rng.randint(0,len(uf),len(uf))]) for _ in range(3000)]
fd=[np.mean([by[uf[i]][1].mean() for i in rng.randint(0,len(uf),len(uf))]) for _ in range(3000)]
print(f"\n  k5-k15 GENUS  Δ={(g5-g15).mean():+.3f} [{np.percentile(gd,2.5):+.3f},{np.percentile(gd,97.5):+.3f}]")
print(f"  k5-k15 FAMILY Δ={(f5-f15).mean():+.3f} [{np.percentile(fd,2.5):+.3f},{np.percentile(fd,97.5):+.3f}]")
win=(g5-g15).mean()>0 and (f5-f15).mean()>=-0.005
print(f"  VERDICT: k=5 -> {'CONFIRMED (commit)' if win else 'NOT confirmed (hold k=15)'}")
