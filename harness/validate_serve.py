#!/usr/bin/env python3
"""
validate_serve.py — end-to-end L3 validation over the live /place API.

Decodes known genomes' tokenized windows back to DNA (BPE is losslessly decodable), POSTs to /place, and
checks the returned placement against ground-truth taxonomy. Exercises the full stack: HTTP -> tokenize ->
encode -> FAISS search (1.9M vec) -> L3 evidence -> decide. Self-placement should nail family/genus.
"""
import os, sys, json, urllib.request
import numpy as np
VOCAB = "/home/rohit/biosphere_inference/bpe_vocab.json"
TOK = "/zfs_raid/SentryBio/tokenized_4096"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
URL = "http://127.0.0.1:8091/place"; NTOK = 4096
vocab = json.load(open(VOCAB)); rev = {i: t for t, i in vocab.items()}
tax = json.load(open(FTAX)); rng = np.random.RandomState(7)
def decode(tokens):
    return "".join(rev.get(int(t), "") for t in tokens if not rev.get(int(t), "[").startswith("["))
def place(dna):
    body = json.dumps({"reads": [dna], "read_bp": 20000, "min_confidence": 0.5}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

# pick a few genomes across domains that have windows
cands = [g for g in rng.permutation(len(tax))[:4000]
         if tax[g].get("family") and os.path.exists(os.path.join(TOK, tax[g]["accession"] + ".npy"))][:8]
print(f"validating {len(cands)} genomes over live /place …\n")
fam_ok = gen_ok = 0
for g in cands:
    tk = np.load(os.path.join(TOK, tax[g]["accession"] + ".npy"), mmap_mode="r")
    dna = decode(np.array(tk[0, :NTOK]))
    try:
        r = place(dna)
    except Exception as e:
        print(f"  gid {g}: ERROR {e}"); continue
    ranks = {x["rank"]: x for x in r.get("ranks", [])}
    truth_f, truth_g = tax[g].get("family"), tax[g].get("genus")
    got_f = ranks.get("family", {}).get("top"); got_g = ranks.get("genus", {}).get("top")
    fam_ok += (got_f == truth_f); gen_ok += (got_g == truth_g)
    print(f"  {tax[g]['accession']:<18} truth={truth_f}/{truth_g}")
    print(f"     -> call={r.get('call')!r}  support={r.get('support')}  novelty={r.get('novelty',{}).get('flag')}  lat={r.get('latency_ms')}ms")
    print(f"        family {'OK' if got_f==truth_f else 'MISS(%s)'%got_f}  genus {'OK' if got_g==truth_g else 'MISS(%s)'%got_g}  nbrs={len(r.get('neighborhood',[]))}")
print(f"\n===== L3 END-TO-END: family {fam_ok}/{len(cands)}  genus {gen_ok}/{len(cands)} =====")
