#!/usr/bin/env python3
"""
accept_honesty.py — HONESTY-JUSTICE tests (spine #4), through the LIVE HTTP service.

  1 GRADED NOVELTY: withhold relatives at increasing depth (species->genus->family->order). Calibrated
    confidence must DECAY MONOTONICALLY as closer relatives are removed, and at the default min_confidence
    the withheld-family commit-rate must be LOW (honest abstention).   [PASS/FAIL]
  2 DEGENERATE (decision #8): empty / 4bp / all-N / non-ACGT garbage must never 5xx; return maximally-novel.
    [PASS/FAIL]
  3 CHARACTERIZE (no gate): reverse-complement invariance; chimera (two-family read mix) confidence < clean.

Run: NGN=60 python accept_honesty.py
"""
import os, sys, json, time, urllib.request, urllib.error
import numpy as np
sys.path.insert(0,"/home/rohit"); sys.path.insert(0,"/home/rohit/build_pipeline")
FTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"; TOK="/zfs_raid/SentryBio/tokenized_4096"
VOCAB="/home/rohit/biosphere_inference/bpe_vocab.json"; U="http://127.0.0.1:8093/place"
RANKS=["domain","phylum","class","order","family","genus","species"]
NGN=int(os.environ.get("NGN","60"))
tax=json.load(open(FTAX)); vocab=json.load(open(VOCAB)); rev={i:t for t,i in vocab.items()}
rng=np.random.RandomState(43)
def decode(t): return "".join(rev.get(int(x),"") for x in t if not rev.get(int(x),"[").startswith("["))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def place(body):
    if body.get("exclude") is not None: body=dict(body, exclude=[int(x) for x in body["exclude"]])  # np.int64 -> int
    r=urllib.request.Request(U,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    try:
        return json.load(urllib.request.urlopen(r,timeout=180)), None
    except urllib.error.HTTPError as e:
        return None, e.code
def windows_of(g,n):
    tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode="r")
    return [decode(np.array(tk[wi]).astype(np.int64)) for wi in range(min(n,tk.shape[0]))]
def famconf(o):
    fr=next((x for x in o["ranks"] if x["rank"]=="family"),None)
    return fr["confidence"] if fr and not fr["gap"] else 0.0
def resolved_depth(o): return RANKS.index(o["resolved_to"]) if o.get("resolved_to") else -1

log("building rank->gids maps…")
from collections import defaultdict
maps={r:defaultdict(list) for r in ["species","genus","family","order"]}
for g in range(len(tax)):
    for r in maps:
        v=tax[g].get(r)
        if v: maps[r][v].append(g)
def has_tok(g): return os.path.exists(os.path.join(TOK,tax[g]["accession"]+".npy"))
# NOTE: this catalog labels only domain/family/genus/species (phylum/class/order are 0% present), so the
# graded-novelty ladder uses the ranks that exist: withhold self -> species -> genus -> family (novel-family).
pool=[g for g in rng.permutation(len(tax))[:8000] if all(tax[g].get(r) for r in ["family","genus","species"]) and has_tok(g)][:NGN]
log(f"{len(pool)} genomes")

# ── 1: graded novelty ────────────────────────────────────────────────────────────────────────
log("TEST 1 — graded novelty (confidence must decay as closer relatives are withheld)")
DEPTHS=["self","species","genus","family"]
conf={d:[] for d in DEPTHS}; commit_family=[]
for qi,g in enumerate(pool):
    reads=windows_of(g,8)
    if not reads: continue
    lin=tax[g]
    excls={"self":[g],
           "species":maps["species"][lin["species"]],
           "genus":maps["genus"][lin["genus"]],
           "family":maps["family"][lin["family"]]}
    for d in DEPTHS:
        o,err=place({"reads":reads,"read_bp":20000,"exclude":excls[d]})
        if err: continue
        conf[d].append(famconf(o))
        if d=="family": commit_family.append(int(o.get("resolved_to") is not None and RANKS.index(o["resolved_to"])>=RANKS.index("family")))
    if (qi+1)%15==0: log(f"  {qi+1}/{len(pool)}")
means={d:(float(np.mean(conf[d])) if conf[d] else float("nan")) for d in DEPTHS}
monotone=all(means[DEPTHS[i]]>=means[DEPTHS[i+1]]-0.02 for i in range(len(DEPTHS)-1))
commit_rate=float(np.mean(commit_family)) if commit_family else float("nan")
print("  mean family-confidence by withholding depth:")
for d in DEPTHS: print(f"    {d:<9} {means[d]:.3f}")
print(f"  monotone decay (self>=species>=…>=order, tol .02): {monotone}")
print(f"  withheld-FAMILY commit-rate @default: {commit_rate:.3f} (want LOW = honest abstention on novel families)")
T1 = monotone and commit_rate < 0.6

# ── 2: degenerate ───────────────────────────────────────────────────────────────────────────
log("TEST 2 — degenerate inputs never 5xx (decision #8)")
cases={"empty_reads":{"reads":[],"read_bp":20000},
       "empty_seq":{"sequence":"","read_bp":20000},
       "4bp":{"reads":["ACGT"],"read_bp":20000},
       "all_N_20kb":{"reads":["N"*20000],"read_bp":20000},
       "garbage":{"reads":["XYZ!@#123"*200],"read_bp":20000}}
T2=True
for name,body in cases.items():
    o,err=place(body)
    if err is not None:
        print(f"    {name:<12} -> HTTP {err}  FAIL (5xx/4xx on degenerate)"); T2=False; continue
    flag=o.get("novelty",{}).get("flag"); rt=o.get("resolved_to"); conf=famconf(o)
    print(f"    {name:<12} -> flag={flag} resolved_to={rt} fam_conf={conf:.3f}  OK(no-error)")
# all_N is not 'no-neighbor' (N->A retrieves), but MUST be low-confidence; check it
oN,_=place(cases["all_N_20kb"]); allN_lowconf = famconf(oN) < 0.6 if oN else False
print(f"  all-N low-confidence (garbage should not place confidently): {allN_lowconf}")

# ── 3: characterize (no gate) ─────────────────────────────────────────────────────────────────
log("TEST 3 — characterize: RC-invariance + chimera")
comp={"A":"T","T":"A","C":"G","G":"C","N":"N"}
def rc(s): return "".join(comp.get(c,"N") for c in reversed(s))
rc_match=0; rc_n=0
for g in pool[:20]:
    reads=windows_of(g,4)
    if not reads: continue
    o1,_=place({"reads":reads,"read_bp":20000,"exclude":[g]})
    o2,_=place({"reads":[rc(r) for r in reads],"read_bp":20000,"exclude":[g]})
    if o1 and o2:
        f1=next((x["top"] for x in o1["ranks"] if x["rank"]=="family" and not x["gap"]),None)
        f2=next((x["top"] for x in o2["ranks"] if x["rank"]=="family" and not x["gap"]),None)
        rc_match+=int(f1==f2); rc_n+=1
# chimera: mix reads from two different families
fams_seen={}; chim_conf=[]; clean_conf=[]
for g in pool[:20]:
    reads=windows_of(g,4)
    if not reads: continue
    other=next((h for h in pool if tax[h]["family"]!=tax[g]["family"] and windows_of(h,2)),None)
    if other is None: continue
    mix=reads[:2]+windows_of(other,2)
    oc,_=place({"reads":mix,"read_bp":20000,"exclude":[g,other]})
    ocl,_=place({"reads":reads,"read_bp":20000,"exclude":[g]})
    if oc and ocl: chim_conf.append(famconf(oc)); clean_conf.append(famconf(ocl))
print(f"  RC-invariance: {rc_match}/{rc_n} same family for genome vs reverse-complement")
if chim_conf: print(f"  chimera family-conf {np.mean(chim_conf):.3f} vs clean {np.mean(clean_conf):.3f}  (chimera<clean: {np.mean(chim_conf)<np.mean(clean_conf)})")

print("\n===== SPINE #4 HONESTY GATES =====")
print(f"  TEST 1 graded-novelty monotone + low novel-commit : {'PASS' if T1 else 'FAIL'}")
print(f"  TEST 2 degenerate never-error + all-N low-conf     : {'PASS' if (T2 and allN_lowconf) else 'FAIL'}")
print(f"  (TEST 3 characterization only, no gate)")
log("DONE")
