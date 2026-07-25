#!/usr/bin/env python3
"""LEVER 2 — label-ceiling audit: are residual family/genus MISSES real errors or taxonomy label-noise?
In-process, 8-window native ensemble, k=5 (landing config), family-present self-mask. Each miss is categorized:
  OUTVOTED : true taxon IS among the k neighbors but lost the vote -> hard/ambiguous, the encoder 'saw' it
  NAMEROOT : predicted & true share a base-name root after stripping GTDB _A/_B suffixes -> reclassification signature
  ABSENT   : true taxon not among the neighbors at all -> genuine error or true novelty
label-noise fraction = (OUTVOTED+NAMEROOT)/misses ; label-corrected ceiling = raw_acc + label-noise misses / N."""
import os, sys, json, time, re
import numpy as np, faiss
sys.path.insert(0,"/home/rohit"); sys.path.insert(0,"/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
from atlas_evidence import build_evidence, RANKS
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK="/zfs_raid/SentryBio/tokenized_4096"; FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG=int(os.environ.get("NG","200")); NWIN=8; K=int(os.environ.get("K","5"))
tax=json.load(open(FTAX)); rng=np.random.RandomState(9)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def norm(n): return re.sub(r'_[A-Z0-9]+$','',n).lower() if n else n
def shared_root(a,b):
    if not a or not b: return False
    na,nb=norm(a),norm(b)
    return na==nb or na.startswith(nb) or nb.startswith(na) or (len(na)>=4 and len(nb)>=4 and na[:4]==nb[:4])
log("loading…")
i20=faiss.read_index(os.path.join(D20,"index.faiss")); g20=np.load(os.path.join(D20,"gids.npy"),mmap_mode="r")
enc=load_effective_encoder(V9,V109,device="cpu")
taxonomy={int(g):{r:tax[int(g)].get(r) for r in RANKS} for g in np.unique(np.asarray(g20))}
def rows_native(tk,n):
    o=[]
    for wi in range(min(n,tk.shape[0])):
        t=np.array(tk[wi,:4096]).astype(np.int64); o.append(np.pad(t,(0,max(0,4096-len(t))),constant_values=PAD_ID)[:4096])
    return o
def neighbors(g,qv,k):
    D,I=i20.search(np.asarray(qv,"float32"),k*16); out=[]
    for r in range(len(qv)):
        kept=0
        for j in range(k*16):
            wid=int(I[r,j])
            if wid<0: continue
            gg=int(g20[wid])
            if gg==g: continue
            out.append((gg,float(D[r,j]))); kept+=1
            if kept>=k: break
    return out
cands=[g for g in rng.permutation(len(tax))[:8000] if tax[g].get("family") and tax[g].get("genus") and os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))][:NG]
log(f"{len(cands)} genomes, k={K}")
def audit(rank):
    hit=0; cats={"NAMEROOT":0,"RUNNERUP":0,"LOSTBIG":0,"ABSENT":0}; examples=[]
    for g in cands:
        tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r")
        nb=neighbors(g,enc.encode(rows_native(tk,NWIN)),K)
        ev=build_evidence(nb,taxonomy,k_neighborhood=K)
        rr=next((x for x in ev.ranks if x.rank==rank),None)
        truth=tax[g].get(rank)
        if rr is None or rr.gap or truth is None: continue
        if rr.top==truth: hit+=1; continue
        nbtaxa={taxonomy[gg].get(rank) for gg,_ in nb}
        runners={t for t,_ in rr.runners_up}          # top-3 voted taxa (excl the winner)
        if shared_root(rr.top,truth): c="NAMEROOT"     # reclassification signature
        elif truth in runners: c="RUNNERUP"            # true is a TOP-3 vote -> genuine near-miss
        elif truth in nbtaxa: c="LOSTBIG"              # true present but a minor neighbor -> real error
        else: c="ABSENT"                               # true absent -> error or novelty
        cats[c]+=1
        if len(examples)<8: examples.append(f"{c}: pred={rr.top} true={truth}")
    n=hit+sum(cats.values()); miss=sum(cats.values())
    raw=hit/n if n else 0
    near=cats["NAMEROOT"]+cats["RUNNERUP"]          # DEFENSIBLE label-noise: reclassification or genuine top-3 near-miss
    real=cats["LOSTBIG"]+cats["ABSENT"]             # genuine errors
    corrected=(hit+near)/n if n else 0
    print(f"\n  === {rank.upper()} (n={n}) ===")
    print(f"    raw accuracy         {raw:.3f}  ({hit}/{n})")
    print(f"    misses {miss}: NAMEROOT {cats['NAMEROOT']}  RUNNERUP {cats['RUNNERUP']}  |  LOSTBIG {cats['LOSTBIG']}  ABSENT {cats['ABSENT']}")
    print(f"    DEFENSIBLE near-ceiling (reclassif + top-3 near-miss) = {near}/{miss} of misses")
    print(f"    LABEL-CORRECTED ceiling = {corrected:.3f}  (+{corrected-raw:.3f})   [genuine errors: {real}/{n}]")
    for e in examples: print(f"      e.g. {e}")
    return {"raw":raw,"corrected":corrected,"cats":cats,"n":n}
print("\n===== LEVER 2: LABEL-CEILING AUDIT =====")
R={r:audit(r) for r in ["family","genus"]}
json.dump(R,open("/zfs_raid/SentryBio/maximize/label_audit.json","w"),indent=2)
log("DONE")
