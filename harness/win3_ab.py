#!/usr/bin/env python3
"""
win3_ab.py — WIN 3: does conditioning genus on a confident family lift genus? (the rerank/scorer_v2 A/B)

In-process (own encoder + 20kb index), native 8-window ensemble, family-present (self-masked). For each genome
builds Evidence twice — v1 (independent ranks) vs v2 (genus/species re-voted over the confident family's
neighbors) — and compares genus accuracy with a FAMILY-GROUPED bootstrap on the paired delta.
PRE-REG: genus delta CI lower bound > 0 (v2 helps) AND family delta == 0 (structural) -> SHIP v2 as default.
         else -> genus is representation-limited, a v11 problem; leave v1 (honest null).
CPU. Run: PYTHONPATH=/home/rohit:/home/rohit/sentrybio/scripts NG=120 python win3_ab.py
"""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, Tokenizer
from atlas_evidence import build_evidence, RANKS
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
VOCAB="/home/rohit/biosphere_inference/bpe_vocab.json"; TOK="/zfs_raid/SentryBio/tokenized_4096"
FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"; D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG=int(os.environ.get("NG","120")); N_WIN=8; KNN=15; OVER=16
tax=json.load(open(FTAX)); vocab=json.load(open(VOCAB)); rev={i:t for t,i in vocab.items()}; tkz=Tokenizer(VOCAB)
rng=np.random.RandomState(31)
def decode(t): return "".join(rev.get(int(x),"") for x in t if not rev.get(int(x),"[").startswith("["))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
log("loading 20kb index + encoder…")
i20=faiss.read_index(os.path.join(D20,"index.faiss")); g20=np.load(os.path.join(D20,"gids.npy"),mmap_mode="r")
enc=load_effective_encoder(V9,V109,device="cpu")
taxonomy={int(g):{r:tax[int(g)].get(r) for r in RANKS} for g in np.unique(np.asarray(g20))}
def search(qv,gexcl):
    kf=KNN*OVER; D,I=i20.search(np.asarray(qv,"float32"),kf); out=[]
    for r in range(len(qv)):
        kept=0
        for j in range(kf):
            wid=int(I[r,j])
            if wid<0: continue
            gg=int(g20[wid])
            if gg==gexcl: continue
            out.append((gg,float(D[r,j]))); kept+=1
            if kept>=KNN: break
    return out
def gt(ev,rank):
    r=next((x for x in ev.ranks if x.rank==rank),None); return r.top if r and not r.gap else None
cands=[g for g in rng.permutation(len(tax))[:5000] if tax[g].get("family") and tax[g].get("genus") and os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))][:NG]
log(f"{len(cands)} genomes")
rows=[]; fired=0
for qi,g in enumerate(cands):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r"); tf=tax[g]["family"]; tg=tax[g]["genus"]
    rows_tok=[tkz.tokenize(decode(np.array(tk[wi]).astype(np.int64)),4096) for wi in range(min(N_WIN,tk.shape[0]))]
    qv=enc.encode(rows_tok); nb=search(qv,g)
    if not nb: continue
    v1=build_evidence(nb,taxonomy,k_neighborhood=KNN,condition_deep_on_family=False)
    v2=build_evidence(nb,taxonomy,k_neighborhood=KNN,condition_deep_on_family=True)
    famr=next(r for r in v1.ranks if r.rank=="family")
    fired+=int((not famr.gap) and famr.margin>=0.6)
    rows.append((tf, int(gt(v1,"genus")==tg), int(gt(v2,"genus")==tg),
                 int(gt(v1,"family")==tf), int(gt(v2,"family")==tf)))
    if (qi+1)%20==0: log(f"  {qi+1}/{len(cands)}  g_v1={np.mean([r[1] for r in rows]):.3f} g_v2={np.mean([r[2] for r in rows]):.3f}")
fams=np.array([r[0] for r in rows],object); gv1=np.array([r[1] for r in rows]); gv2=np.array([r[2] for r in rows])
fv1=np.array([r[3] for r in rows]); fv2=np.array([r[4] for r in rows])
uf=list(set(fams))
def famboot_delta(a,b):
    by={f:(a[fams==f],b[fams==f]) for f in uf}
    d=[np.mean([by[uf[i]][1].mean()-by[uf[i]][0].mean() for i in rng.randint(0,len(uf),len(uf))]) for _ in range(3000)]
    return float((b-a).mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))
gd,glo,ghi=famboot_delta(gv1,gv2)
print(f"\n===== WIN 3 A/B (n={len(rows)}, native 8-win ensemble, family-present) =====")
print(f"  conditioning fired (family margin>=0.6): {fired}/{len(rows)} = {fired/max(len(rows),1):.2f}")
print(f"  GENUS  v1={gv1.mean():.3f}  v2={gv2.mean():.3f}  Δ={gd:+.3f} [{glo:+.3f},{ghi:+.3f}]")
print(f"  FAMILY v1={fv1.mean():.3f}  v2={fv2.mean():.3f}  Δ={(fv2-fv1).mean():+.3f} (must be 0.000, structural)")
verdict = "SHIP v2 (condition genus on confident family)" if glo>0 and (fv2-fv1).mean()==0 else \
          ("family regressed - BUG" if (fv2-fv1).mean()!=0 else "NULL: genus is representation-limited (v11 job); keep v1")
print(f"  PRE-REG genus Δ CI clears 0 & family Δ==0 -> {verdict}")
