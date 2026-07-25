#!/usr/bin/env python3
"""LEVER 3 — alignment reranker: re-rank the top-K genome finalists by actual SEQUENCE IDENTITY (MinHash
bottom-s Jaccard on 21-mers) instead of cosine, then re-derive the genus vote. Brings information the geometry
discards. In-process, 8-window ensemble geometry search (k=15) to get finalists, MinHash on window-0 DNA.
Compares geometry-vote genus vs identity-reranked genus, family-bootstrap CI; also on the family-confident subset
(the 'right family, wrong leaf' regime). PRE-REG: genus Δ CI clears 0 -> ship rerank; else honest null."""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0,"/home/rohit"); sys.path.insert(0,"/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, Tokenizer, PAD_ID
from atlas_evidence import build_evidence, RANKS
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
VOCAB="/home/rohit/biosphere_inference/bpe_vocab.json"
TOK="/zfs_raid/SentryBio/tokenized_4096"; FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG=int(os.environ.get("NG","120")); NWIN=8; K=15; NFIN=10; KMER=21; SKETCH=500
tax=json.load(open(FTAX)); vocab=json.load(open(VOCAB)); rev={i:t for t,i in vocab.items()}
rng=np.random.RandomState(13)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def decode(t): return "".join(rev.get(int(x),"") for x in t if not rev.get(int(x),"[").startswith("["))
def w0(g):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r"); return decode(np.array(tk[0]).astype(np.int64))
def sketch(dna):
    if len(dna)<KMER: return np.array([],dtype=np.uint64)
    h=np.fromiter((hash(dna[i:i+KMER])&0xFFFFFFFFFFFFFFFF for i in range(0,len(dna)-KMER+1,3)),dtype=np.uint64)
    return np.unique(h)[:SKETCH*4] if h.size else h
def jac(a,b):
    if a.size==0 or b.size==0: return 0.0
    u=np.union1d(a,b)[:SKETCH]; sa=set(a.tolist()); sb=set(b.tolist())
    return sum(1 for x in u.tolist() if x in sa and x in sb)/max(len(u),1)
log("loading…")
i20=faiss.read_index(os.path.join(D20,"index.faiss")); g20=np.load(os.path.join(D20,"gids.npy"),mmap_mode="r")
enc=load_effective_encoder(V9,V109,device="cpu")
taxonomy={int(g):{r:tax[int(g)].get(r) for r in RANKS} for g in np.unique(np.asarray(g20))}
def rows(tk,n):
    o=[]
    for wi in range(min(n,tk.shape[0])):
        t=np.array(tk[wi,:4096]).astype(np.int64); o.append(np.pad(t,(0,max(0,4096-len(t))),constant_values=PAD_ID)[:4096])
    return o
def search(g,qv,k):
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
log(f"{len(cands)} genomes")
rows_rec=[]  # (family, geo_genus_hit, rerank_genus_hit, fam_confident, fam_hit)
for qi,g in enumerate(cands):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r")
    nb=search(g,enc.encode(rows(tk,NWIN)),K)
    ev=build_evidence(nb,taxonomy,k_neighborhood=K)
    fr=next((x for x in ev.ranks if x.rank=="family"),None); gr=next((x for x in ev.ranks if x.rank=="genus"),None)
    tf,tg=tax[g]["family"],tax[g]["genus"]
    geo_gen=gr.top if gr and not gr.gap else None
    fam_conf=bool(fr and not fr.gap and fr.margin>=0.6); fam_hit=int(fr and fr.top==tf)
    # top NFIN distinct finalists by best cosine
    best={}
    for gg,s in nb:
        if gg not in best or s>best[gg]: best[gg]=s
    fin=sorted(best.items(),key=lambda kv:kv[1],reverse=True)[:NFIN]
    qs=sketch(w0(g)); gvote={}
    for gg,_ in fin:
        try: idt=jac(qs,sketch(w0(gg)))
        except Exception: idt=0.0
        gen=taxonomy[gg].get("genus")
        if gen: gvote[gen]=gvote.get(gen,0.0)+idt**3
    rr_gen=max(gvote,key=gvote.get) if gvote else geo_gen
    rows_rec.append((tf,int(geo_gen==tg),int(rr_gen==tg),fam_conf,fam_hit))
    if (qi+1)%20==0: log(f"  {qi+1}/{len(cands)}")
fams=np.array([r[0] for r in rows_rec],object); uf=list(set(fams))
geo=np.array([r[1] for r in rows_rec]); rr=np.array([r[2] for r in rows_rec]); fc=np.array([r[3] for r in rows_rec])
def bootd(a,b,mask=None):
    idx=np.arange(len(a)) if mask is None else np.where(mask)[0]
    fm=fams[idx]; uu=list(set(fm)); by={f:(a[idx][fm==f],b[idx][fm==f]) for f in uu}
    d=[np.mean([by[uu[i]][1].mean()-by[uu[i]][0].mean() for i in rng.randint(0,len(uu),len(uu))]) for _ in range(3000)]
    return float((b[idx]-a[idx]).mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))
print(f"\n===== LEVER 3: ALIGNMENT RERANK (n={len(rows_rec)}) =====")
print(f"  genus geometry-vote {geo.mean():.3f}  identity-rerank {rr.mean():.3f}")
d,lo,hi=bootd(geo,rr); print(f"  ALL: genus Δ(rerank-geo) = {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
if fc.sum()>=10:
    d2,lo2,hi2=bootd(geo,rr,fc); print(f"  FAM-CONFIDENT subset (n={int(fc.sum())}): genus Δ = {d2:+.3f} [{lo2:+.3f},{hi2:+.3f}]")
print(f"  VERDICT: {'SHIP rerank' if lo>0 else 'NULL — identity rerank does not beat geometry'}")
log("DONE")
