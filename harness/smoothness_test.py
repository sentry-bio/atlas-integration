#!/usr/bin/env python3
"""
smoothness_test.py — the test the 5kb index was BUILT for: does a genuine SHORT query place better against the
richness-appropriate reference tier? Three compositional arms across a query-length sweep.

Un-foolable construction (fixes last turn's error): decode window→DNA, cut a CONTIGUOUS L-bp fragment, re-BPE
the FRAGMENT ALONE, pad to each index's NTOK, encode. The encoder sees only L bp of real sequence. At L≤5kb both
arms see the FULL fragment → the only variable is the reference tier (the clean routing question); at long L the
5kb arm is info-limited (truncated), so long L should favor 20kb.

Arms per fragment:
  20kb-only : frag→tokens@4096 → 20kb index → L3
  5kb-only  : frag→tokens@1000 → 5kb index  → L3
  FUSED     : both neighbor sets, per-index rank-normalized, pooled → L3
Family-present (exclude only the query genome) → this is ROUTING/placement, not novelty (novelty is settled).
Averaged over genomes × random cut positions.

PRE-REGISTERED:
  crossover (5kb-only > 20kb-only at short L) -> route short→5kb; the 5kb index EARNS hosting.
  FUSED > max(arms) at any L                  -> fusion adds for placement (not just native/novelty).
  neither                                     -> host 20kb-only; 5kb genuinely unneeded; concede.

CPU, ~30min. Run: CUDA_VISIBLE_DEVICES="" python smoothness_test.py
"""
import os, sys, json, time
import numpy as np, faiss
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts"); sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, Tokenizer
from atlas_evidence import build_evidence, RANKS
V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
VOCAB = "/home/rohit/biosphere_inference/bpe_vocab.json"; TOK = "/zfs_raid/SentryBio/tokenized_4096"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
D20 = "/zfs_raid/SentryBio/serve_index_v109_20kb"; D5 = "/zfs_raid/SentryBio/serve_index_v109_5kb"
NG = int(os.environ.get("NG", "60")); NCUT = 3; KNN = 15; OVER = 4
LENGTHS = [1000, 2000, 3000, 5000, 10000, None]                    # None = native (full window)
tax = json.load(open(FTAX)); rng = np.random.RandomState(3)
vocab = json.load(open(VOCAB)); rev = {i: t for t, i in vocab.items()}; tkz = Tokenizer(VOCAB)
def lin(g): return {r: tax[int(g)].get(r) for r in RANKS}
def decode(tokens): return "".join(rev.get(int(t), "") for t in tokens if not rev.get(int(t), "[").startswith("["))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("loading indexes + encoder…")
i20 = faiss.read_index(os.path.join(D20, "index.faiss")); g20 = np.load(os.path.join(D20, "gids.npy"), mmap_mode="r")
i5 = faiss.read_index(os.path.join(D5, "index.faiss")); g5 = np.load(os.path.join(D5, "gids.npy"), mmap_mode="r")
enc = load_effective_encoder(V9, V109, device="cpu")
taxonomy = {int(g): lin(int(g)) for g in np.unique(np.asarray(g20))}
def qvec(dna, ntok): return enc.encode([tkz.tokenize(dna, ntok)])
def search(index, gids, qv, gexcl):
    kf = KNN * OVER; D, I = index.search(np.asarray(qv, "float32"), kf); out = []
    for j in range(kf):
        wid = int(I[0, j])
        if wid < 0: continue
        gg = int(gids[wid])
        if gg == gexcl: continue
        out.append((gg, float(D[0, j])));
        if len(out) >= KNN: break
    return out
def ranknorm(nb):
    if not nb: return []
    s = np.array([x[1] for x in nb]); rk = s.argsort().argsort() / max(len(s) - 1, 1)
    return [(nb[i][0], float(rk[i])) for i in range(len(nb))]
def calls(nb):
    ev = build_evidence(nb, taxonomy, k_neighborhood=KNN); r = {x.rank: x for x in ev.ranks}
    fr, gr = r.get("family"), r.get("genus")
    return (fr.top if fr and not fr.gap else None, gr.top if gr and not gr.gap else None)

cands = [g for g in rng.permutation(len(tax))[:5000] if tax[g].get("family") and tax[g].get("genus")
         and os.path.exists(os.path.join(TOK, tax[g]["accession"] + ".npy"))][:NG]
log(f"{len(cands)} query genomes")
# accumulators: acc[length_key][arm] = [fam_hits, gen_hits, n]
LK = [str(L) if L else "native" for L in LENGTHS]; ARMS = ["20kb", "5kb", "fused"]
acc = {lk: {a: [0, 0, 0] for a in ARMS} for lk in LK}
for qi, g in enumerate(cands):
    tk = np.load(os.path.join(TOK, tax[g]["accession"] + ".npy"), mmap_mode="r")
    full = np.array(tk[0]).astype(np.int64); dna = decode(full); tf, tg = lin(g)["family"], lin(g)["genus"]
    for L, lk in zip(LENGTHS, LK):
        frags = []
        if L is None: frags = [None]                              # native: use full window tokens directly
        elif len(dna) >= L: frags = [rng.randint(0, len(dna) - L) for _ in range(NCUT)]
        for pos in frags:
            if L is None:
                q20 = enc.encode([full[:4096].tolist() + [3] * max(0, 4096 - len(full))][:1] if False else [np.pad(full[:4096],(0,max(0,4096-len(full))),constant_values=3).tolist()])
                q5 = enc.encode([np.pad(full[:1000],(0,max(0,1000-len(full))),constant_values=3).tolist()])
            else:
                frag = dna[pos:pos + L]; q20 = qvec(frag, 4096); q5 = qvec(frag, 1000)
            n20 = search(i20, g20, q20, g); n5 = search(i5, g5, q5, g)
            if not n20 and not n5: continue
            for arm, nb in [("20kb", n20), ("5kb", n5), ("fused", ranknorm(n20) + ranknorm(n5))]:
                f, gn = calls(nb); a = acc[lk][arm]
                a[0] += int(f == tf); a[1] += int(gn == tg); a[2] += 1
    if (qi + 1) % 15 == 0: log(f"  {qi+1}/{len(cands)}")

print("\n===== SMOOTHNESS / SCALE-ROUTING (family-present, avg over genomes×cuts) =====")
print(f"  {'length':<8}" + "".join(f"{a+'-fam':>12}" for a in ARMS) + "   best-fam")
for lk in LK:
    row = ""; fams = {}
    for a in ARMS:
        h, gh, n = acc[lk][a]; fa = h / n if n else float("nan"); fams[a] = fa; row += f"{fa:>12.3f}"
    best = max(fams, key=fams.get)
    print(f"  {lk:<8}{row}   {best} ({fams[best]:.3f})")
print("\n  genus:")
print(f"  {'length':<8}" + "".join(f"{a+'-gen':>12}" for a in ARMS))
for lk in LK:
    print(f"  {lk:<8}" + "".join(f"{acc[lk][a][1]/max(acc[lk][a][2],1):>12.3f}" for a in ARMS))
print("\n  ROUTING TABLE (best family arm per length) + does FUSED ever beat both single arms?")
for lk in LK:
    fa = {a: acc[lk][a][0]/max(acc[lk][a][2],1) for a in ARMS}
    fused_wins = fa["fused"] > max(fa["20kb"], fa["5kb"]) + 0.005
    print(f"    {lk:<8} best={max(fa,key=fa.get)}  fused-beats-both={fused_wins}")
np.savez("/zfs_raid/SentryBio/radial_training/smoothness.npz", acc=json.dumps(acc))
print("DONE")
