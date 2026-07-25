#!/usr/bin/env python3
"""
win_ceiling.py — the decisive ACCURACY-JUSTICE test (spine #3): does the serving path return the encoder's
intrinsic best, or is the harness leaving recoverable accuracy on the table?

In-process (own encoder + 20kb index), family-present, self-masked, family-bootstrap CI.
  CEILING arm     : native tokens, ALL windows (cap 64), best-k   -> the most this encoder can do per genome
  PRODUCTION arm  : native tokens, 24-window cap (evenly sampled), k=15  -> what serving actually does
  gap = ceiling - production, per rank.  (decode->re-tokenize proven lossless in Win 1, so both arms use native
  tokens to isolate the two real serving levers: window-cap and k.)
Folded-in sweeps that localize any gap:
  N-window plateau : family/genus acc vs N in {1,2,4,8,16,24,64}   (does 24-cap cost anything? does it plateau?)
  k-sweep          : family/genus acc vs k in {5,10,15,25,50}       (is k=15 optimal?)
Reality link       : for a few genomes, production-arm (in-process) family/genus == live HTTP /place (same reads).

PRE-REG: |gap| < 0.02 with family-bootstrap CI including 0 -> HARNESS DOES THE ENCODER JUSTICE; residual is the
encoder's; STOP.  gap CI clears +0.02 -> real recoverable headroom; the sweeps say raise the cap or retune k.
CPU. Run: PYTHONPATH=/home/rohit:/home/rohit/sentrybio/scripts NG=80 python win_ceiling.py
"""
import os, sys, json, time, urllib.request
import numpy as np, faiss
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, Tokenizer, PAD_ID
from atlas_evidence import build_evidence, RANKS
V9="/home/rohit/v9_best.pt"; V109="/home/rohit/v10_curvature_field/v10_9_encoder.pt"
VOCAB="/home/rohit/biosphere_inference/bpe_vocab.json"; TOK="/zfs_raid/SentryBio/tokenized_4096"
FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"; D20="/zfs_raid/SentryBio/serve_index_v109_20kb"
NG=int(os.environ.get("NG","80")); MAXW=64; CAP=24
tax=json.load(open(FTAX)); vocab=json.load(open(VOCAB)); rev={i:t for t,i in vocab.items()}; tkz=Tokenizer(VOCAB)
rng=np.random.RandomState(41)
def decode(t): return "".join(rev.get(int(x),"") for x in t if not rev.get(int(x),"[").startswith("["))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
log("loading 20kb index + encoder…")
i20=faiss.read_index(os.path.join(D20,"index.faiss")); g20=np.load(os.path.join(D20,"gids.npy"),mmap_mode="r")
enc=load_effective_encoder(V9,V109,device="cpu")
taxonomy={int(g):{r:tax[int(g)].get(r) for r in RANKS} for g in np.unique(np.asarray(g20))}
def native_rows(tk, n):
    nw=min(n, tk.shape[0]); rows=[]
    for wi in range(nw):
        t=np.array(tk[wi,:4096]).astype(np.int64)
        rows.append(np.pad(t,(0,max(0,4096-len(t))),constant_values=PAD_ID)[:4096])
    return rows
def evenly(idxs, cap):
    if len(idxs)<=cap: return idxs
    sel=sorted({round(i*(len(idxs)-1)/(cap-1)) for i in range(cap)})
    return [idxs[i] for i in sel]
def search_vecs(qv, k, gexcl):
    kf=k*16; D,I=i20.search(np.asarray(qv,"float32"),kf); out=[]
    for r in range(len(qv)):
        kept=0
        for j in range(kf):
            wid=int(I[r,j])
            if wid<0: continue
            gg=int(g20[wid])
            if gg==gexcl: continue
            out.append((gg,float(D[r,j]))); kept+=1
            if kept>=k: break
    return out
def call(nb, rank):
    ev=build_evidence(nb, taxonomy, k_neighborhood=15); r=next((x for x in ev.ranks if x.rank==rank),None)
    return r.top if r and not r.gap else None

cands=[g for g in rng.permutation(len(tax))[:6000] if tax[g].get("family") and tax[g].get("genus")
       and os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))][:NG]
log(f"{len(cands)} genomes")
# precompute per-genome encoded native window vectors (up to MAXW), reuse across all sweeps
G=[]
for qi,g in enumerate(cands):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r")
    vecs=enc.encode(native_rows(tk, MAXW))                       # (nw, 129)
    G.append({"g":g,"tf":tax[g]["family"],"tg":tax[g]["genus"],"vecs":vecs,"nw":vecs.shape[0]})
    if (qi+1)%20==0: log(f"  encoded {qi+1}/{len(cands)}")

def acc_for(select_fn, k):
    fam=gen=n=0; famv=[]; fams=[]
    for e in G:
        idxs=select_fn(e["nw"])
        qv=e["vecs"][idxs]
        nb=search_vecs(qv, k, e["g"])
        f=call(nb,"family"); gn=call(nb,"genus")
        fam+=int(f==e["tf"]); gen+=int(gn==e["tg"]); n+=1
        famv.append(int(f==e["tf"])); fams.append(e["tf"])
    return fam/n, gen/n, np.array(famv), np.array(fams)

# --- N-window plateau (native tokens, k=15) ---
log("N-window plateau…")
plateau={}
for N in [1,2,4,8,16,24,64]:
    fa,ga,_,_=acc_for(lambda nw,N=N: list(range(min(N,nw))), 15)
    plateau[N]=(fa,ga)
# --- k-sweep (all windows, vary k) ---
log("k-sweep…")
ksweep={}
for k in [5,10,15,25,50]:
    fa,ga,_,_=acc_for(lambda nw: list(range(nw)), k)
    ksweep[k]=(fa,ga)
# --- ceiling (all windows, best k) vs production (24-cap evenly, k=15) ---
bestk=max(ksweep, key=lambda k: ksweep[k][0]+ksweep[k][1])
cfa,cga,cfam,fams=acc_for(lambda nw: list(range(nw)), bestk)             # ceiling
pfa,pga,pfam,_   =acc_for(lambda nw: evenly(list(range(nw)), CAP), 15)   # production
uf=list(set(fams))
def famboot_delta(a,b):
    by={f:(a[fams==f],b[fams==f]) for f in uf}
    d=[np.mean([by[uf[i]][0].mean()-by[uf[i]][1].mean() for i in rng.randint(0,len(uf),len(uf))]) for _ in range(3000)]
    return float((a-b).mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))
gfd,glo,ghi=famboot_delta(cfam,pfam)
big=[e for e in G if e["nw"]>CAP]

print("\n===== SPINE #3 — CEILING GAP (accuracy justice) =====")
print(f"  n={len(G)}  ({len(big)} genomes have >{CAP} windows, where the cap can bite)")
print(f"  N-window plateau (native, k=15):  " + "  ".join(f"N{N}:{plateau[N][0]:.3f}/{plateau[N][1]:.3f}" for N in plateau))
print(f"  k-sweep (all windows):            " + "  ".join(f"k{k}:{ksweep[k][0]:.3f}/{ksweep[k][1]:.3f}" for k in ksweep))
print(f"  best-k = {bestk}")
print(f"  CEILING  (all-win, k={bestk}):  family {cfa:.3f}  genus {cga:.3f}")
print(f"  PRODUCT  (24-cap,  k=15):     family {pfa:.3f}  genus {pga:.3f}")
print(f"  GAP family = {gfd:+.3f} [{glo:+.3f},{ghi:+.3f}]   genus = {cga-pga:+.3f}")
verdict = "HARNESS DOES JUSTICE (residual is the encoder; STOP)" if abs(gfd)<0.02 and glo<=0<=ghi else \
          "RECOVERABLE HEADROOM (raise cap / retune k per sweeps)"
print(f"  PRE-REG |gap|<0.02 & CI incl 0 -> {verdict}")

# --- reality link: in-process production arm == live HTTP /place, on 5 genomes ---
print("\n  reality link (in-process production == live HTTP /place):")
U="http://127.0.0.1:8093/place"
def http_place(reads):
    r=urllib.request.Request(U,data=json.dumps({"reads":reads,"read_bp":20000}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))
match=0
for e in G[:5]:
    tk=np.load(os.path.join(TOK,tax[e["g"]]["accession"]+".npy"),mmap_mode="r")
    idxs=evenly(list(range(min(MAXW,tk.shape[0]))),CAP)
    reads=[decode(np.array(tk[wi]).astype(np.int64)) for wi in idxs]
    o=http_place(reads)
    hf=next((x["top"] for x in o["ranks"] if x["rank"]=="family" and not x["gap"]),None)
    qv=e["vecs"][idxs]; ipf=call(search_vecs(qv,15,e["g"]),"family")
    ok=(hf==ipf); match+=int(ok)
    print(f"    g{e['g']}: in-proc={ipf}  http={hf}  {'match' if ok else 'DIFFER'}")
print(f"  reality link: {match}/5 match (validates the in-process replica == served path)")
np.savez("/zfs_raid/SentryBio/radial_training/ceiling_gap.npz", plateau=json.dumps({str(k):v for k,v in plateau.items()}), ksweep=json.dumps({str(k):v for k,v in ksweep.items()}))
log("DONE")
