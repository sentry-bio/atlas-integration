#!/usr/bin/env python3
"""
atlas_harness.py — the placement harness: SCALE-ROUTE -> DEEP VOTE-ENSEMBLE -> FATHOM honest-depth.

  ┌─ COHERENCE NOTE (serving refactor, 2026-07-17) ──────────────────────────────────────────────┐
  │ SUPERSEDED by the L2/L3/L4 stack.  The SCORING here (`descend`) now DELEGATES to atlas_evidence │
  │ (L3) so there is exactly ONE scorer — no second, drifting implementation.  This file survives   │
  │ as: (a) the legacy AtlasHarness/serve.py entry path, and (b) the L1 index backends (InRAMIndex, │
  │ FaissFullIndex).  NEW serving code: atlas_serve.py (L4) -> atlas_encoder (L2) -> atlas_evidence  │
  │ (L3).  Do not add scoring logic here; add it to atlas_evidence and let this shim inherit it.     │
  └─────────────────────────────────────────────────────────────────────────────────────────────────┘


Every design choice here is empirically specified by the ensemble-depth measurement (2026-07-14):
  - VOTING aggregation.  Karcher-pool converged to voting at 2kb (0.60 vs 0.60); voting is simpler +
    outlier-robust, so the harness votes. (My pooling bet was measured wrong.)
  - DEEP ensemble where the curve climbs.  2kb novel-species genus: 0.51@N=8 -> 0.60@N=32, still rising.
    So mid-length reads get a deep ensemble budget; the +9 points live in depth, not cleverness.
  - SCALE-ROUTING.  Depth pays only where reads are VARIANCE-limited (~1-5kb).  At 300bp it's the
    information floor (flat ~0.10 -> more reads useless); at 20kb reads are already rich (flat).  So the
    router spends deep-ensemble budget only in the middle band.
  - FATHOM honest-depth.  At short reads genus is at the Bayes floor, so the descent must return
    "family + novel_at=genus", NEVER a guessed genus.  The margin gate is calibrated per read length
    (looser at short reads to avoid under-claiming family; the genus rank still self-limits by noise).

One call: harness.place(reads) -> Placement (lineage-to-supported-depth + confidence + novel_at + support).
Encoder-agnostic (v9/v10.6/...) and backend-agnostic (in-RAM exact / Annoy / FAISS).
"""
from __future__ import annotations
import os, sys, json, collections, numpy as np
from dataclasses import dataclass, field

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
DEPTH = {r: i for i, r in enumerate(RANKS)}

def _as_exclude_set(exclude):
    """None -> empty set; int -> {int}; iterable -> set.  Lets the benchmark mask a whole
    taxon (e.g. all same-species gids) to simulate novelty over the full 234K index."""
    if exclude is None: return set()
    if isinstance(exclude, (int, np.integer)): return {int(exclude)}
    return set(int(g) for g in exclude)

# ── read-length policy (the measured curve, encoded) ──────────────────────────────────────────
# (read_bp threshold, n_tok, ensemble_target_N, margin_min).  Chosen by the ensemble-depth test:
#   ~300bp : info FLOOR  -> shallow ensemble is enough (deep won't help); loosen margin so family still
#            resolves, but genus self-limits by noise -> honest novel_at=genus.
#   ~1-5kb : VARIANCE-limited -> DEEP ensemble (the +9-pt band); mid margin.
#   >=20kb : reads already rich -> shallow ensemble; strict margin (we can trust deep claims).
@dataclass
class ScaleBand:
    name: str; read_bp: int; n_tok: int; ensemble_N: int; margin_min: float; gamma: float = 10.0
# Calibrated on DEPLOYED regimes (push_harness.py/confirm_config.py, 2026-07-14): dedup OFF + top-k=15
# neighbors/read + soft sharpen sim^gamma(3) + light margin(0.03). Deep-ensemble @2kb: in-ref 0.95,
# novel-assembly 0.62, novel-species 0.46 genus (family +~0.08), depth compounds, ~5-6% honest novel
# abstention. (Prior k20/g6/m.05 = 0.57/0.39; k15/g3 is the frontier for the novel tiers.)
SCALE_BANDS = [
    ScaleBand("300bp",  300,   60,   16, 0.08, 3.0),    # info floor: abstain more (honest)
    ScaleBand("1kbp",  1000,  200,   48, 0.03, 3.0),
    ScaleBand("2kbp",  2000,  400,   48, 0.03, 3.0),    # ONT eDNA sweet spot (calibrated on this band)
    ScaleBand("5kbp",  5000, 1000,   48, 0.03, 3.0),
    ScaleBand("20kbp",20000, 4000,   12, 0.05, 3.0),
]
def route(read_bp: int) -> ScaleBand:
    band = SCALE_BANDS[0]
    for b in SCALE_BANDS:
        if read_bp >= b.read_bp: band = b
    return band

# ── the honest output ─────────────────────────────────────────────────────────────────────────
@dataclass
class Placement:
    lineage: list                 # [(rank, taxon, margin), ...] to the supported depth
    resolved_to: str | None       # deepest rank the evidence supports
    novel_at: str | None          # first rank the evidence could NOT support (== honest novelty)
    confidence: float             # vote margin at resolved_to
    support: float                # mean retrieval similarity (coverage proxy; low -> "uncovered")
    n_reads: int
    scale: str
    def call(self) -> str:
        if not self.lineage: return "unplaced"
        r, t, *_ = self.lineage[-1]
        nov = f", novel at {self.novel_at}" if self.novel_at else ""
        return f"{t} ({r}){nov}"
    def __repr__(self):
        path = " > ".join(f"{t}" for _, t, *_ in self.lineage) or "(none)"
        nov = f" | novel_at={self.novel_at}" if self.novel_at else ""
        return f"[{path}] conf={self.confidence:.2f} support={self.support:.2f}{nov} ({self.n_reads}rd,{self.scale})"

# ── the descent: consensus per rank, confidence-weighted, honest stop ────────────────────────
def _calib_conf(margin, c):
    """calibrated P(correct) for a margin, from the fitted per-rank curve (edges -> p)."""
    edges, p = c.get("edges", []), c.get("p", [])
    if not p: return margin
    for i in range(len(edges) - 1):
        if margin <= edges[i + 1]: return p[i]
    return p[-1]

def descend(neighbors, taxonomy_of, margin_min, n_reads, scale, gamma=3.0, dedup=False, calib=None):
    """neighbors: pooled list[(gid, sim)] across the deep read-ensemble (sim = retrieval confidence).
    Consensus descent: accept each rank while the sharpened top taxon holds a margin; stop where the
    evidence runs out.  resolved_to = placement; novel_at = honest novelty flag.
    gamma: sharpen votes as sim**gamma so the nearest (self / true-relative) matches dominate the crowd
    of weakly-similar distractors — calibrated to recover in-reference accuracy (tune_aggregation.py).
    dedup: OFF by default — collapsing per-genome was found to BURY the correct genome's strong
    self-matches under the distractor crowd (in-ref 0.87 -> 0.37); keep every read's vote."""
    # DEPRECATED shim -> atlas_evidence (L3) is now the ONE scorer.  Kept so legacy AtlasHarness.place /
    # serve.py keep working while there is a single scoring implementation (the coherence fix: no second,
    # drifting scorer).  When calib is None this is behavior-IDENTICAL to the historical descend — same
    # sim**gamma independent vote, same support = mean(sim), same gap-skip, same break-at-first-fail — the
    # per-rank vote math simply lives in atlas_evidence.score_ranks now.  The operating point is a single
    # dial (min_confidence) rather than per-rank baked taus.  New code should call:
    #     atlas_evidence.build_evidence(neighbors, taxonomy).decide(min_confidence)
    from atlas_evidence import build_evidence, DEPTH as _D
    if dedup:                                              # legacy option; frozen scorer_v1 is dedup-OFF
        best = {}
        for gid, s in neighbors:
            if gid not in best or s > best[gid]: best[gid] = s
        neighbors = list(best.items())
    # only forward L3-shaped calibration (margin_edges); the legacy {tau,edges,p} schema is superseded.
    l3calib = calib if (calib and all("margin_edges" in v for v in calib.values())) else None
    ev = build_evidence(neighbors, taxonomy_of, n_reads=n_reads, gamma=gamma, calib=l3calib)
    d = ev.decide(min_confidence=margin_min)
    depth = _D[d.resolved_to] if d.resolved_to else -1
    lineage = [(r.rank, r.top, r.confidence) for r in ev.ranks if not r.gap and _D[r.rank] <= depth]
    conf = lineage[-1][2] if lineage else 0.0
    return Placement(lineage, d.resolved_to, d.novel_at, conf, ev.support, n_reads, scale)

# ── the harness ────────────────────────────────────────────────────────────────────────────────
class AtlasHarness:
    def __init__(self, encoder, index, taxonomy, k=15, calibration=None):   # k=15 neighbors/read
        """encoder: reads_tok -> unit angular embeddings. index: .search(cloud,k)->[(gid,sim),...] per read.
        taxonomy: gid -> {rank: taxon}. calibration: path to calibration.json (conformal gate; optional)."""
        self.encoder = encoder; self.index = index; self.taxonomy = taxonomy; self.k = k
        self.calib = None
        if calibration and os.path.exists(calibration):
            self.calib = json.load(open(calibration))

    def place(self, read_toks, read_bp, exclude=None) -> Placement:
        band = route(read_bp)
        toks = read_toks[: band.ensemble_N]                # deep only where the curve says it pays
        cloud = self.encoder(toks)                          # (n_reads, d) unit embeddings
        neigh = self.index.search_cloud(cloud, self.k, exclude=exclude)   # pooled vote-ensemble
        return descend(neigh, self.taxonomy, band.margin_min, len(cloud), band.name, gamma=band.gamma, calib=self.calib)

# ── exact in-RAM retrieval backend (selftest / small deployments) ────────────────────────────
class InRAMIndex:
    def __init__(self, X, gids):
        import faiss
        self.idx = faiss.IndexFlatIP(X.shape[1]); self.idx.add(X.astype("float32"))
        self.gids = np.asarray(gids)
    def search_cloud(self, cloud, k=20, exclude=None):
        ex = _as_exclude_set(exclude)
        D, I = self.idx.search(cloud.astype("float32"), k); out = []
        for r in range(len(cloud)):
            for j in range(k):
                nid = I[r, j]
                if nid < 0: continue
                g = int(self.gids[nid])
                if g in ex: continue
                out.append((g, float(D[r, j])))
        return out


# ── full 234K IVFPQ backend (the production cloud: 779M window vectors, 39.8GB) ────────────────
class FaissFullIndex:
    """Wraps the merged IVFPQ index over all 234,526 genomes.  Window vector i -> gids[i] (genome).
    Over-fetches k*oversample windows/read then keeps the top-k distinct-neighbor hits after masking
    excluded gids, so taxon-masked novelty (same-species / same-genome held out) still returns k."""
    def __init__(self, index_path, gids_path, nprobe=24, oversample=4):
        import faiss
        self.idx = faiss.read_index(index_path); self.idx.nprobe = nprobe
        self.gids = np.load(gids_path, mmap_mode="r"); self.oversample = oversample
    def search_cloud(self, cloud, k=20, exclude=None):
        ex = _as_exclude_set(exclude)
        kf = k * (self.oversample if ex else 1)
        D, I = self.idx.search(cloud.astype("float32"), kf); out = []
        for r in range(len(cloud)):
            kept = 0
            for j in range(kf):
                wid = int(I[r, j])
                if wid < 0: continue
                g = int(self.gids[wid])
                if g in ex: continue
                out.append((g, float(D[r, j]))); kept += 1
                if kept >= k: break
        return out


if __name__ == "__main__":
    # selftest: full harness on a small in-RAM index, multiple read lengths -> honest Placements
    import time, torch
    sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts")
    TOK="/zfs_raid/SentryBio/tokenized_4096"; VOCAB="/home/rohit/biosphere_inference/bpe_vocab.json"
    CKPT=os.environ.get("CKPT","/home/rohit/v10_curvature_field/v10_6_encoder.pt"); BASE="/home/rohit/v9_best.pt"
    FULLTAX="/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"; rng=np.random.RandomState(0)
    vocab=json.load(open(VOCAB)); rev={int(i):t for t,i in vocab.items()}; by_len={}
    for t,i in vocab.items():
        if not t.startswith("["): by_len.setdefault(len(t),{})[t]=i
    mtl=max(by_len)
    def decode(t): return "".join(p for p in (rev.get(int(x),"") for x in t) if p and p[0] in "ACGT")
    def bpe(s,nt):
        s=s.upper().replace("N","A"); o=[]; i=0
        while i<len(s) and len(o)<nt:
            for L in range(min(mtl,len(s)-i),0,-1):
                if L in by_len and s[i:i+L] in by_len[L]: o.append(by_len[L][s[i:i+L]]); i+=L; break
            else: o.append(vocab.get(s[i],0)); i+=1
        while len(o)<nt: o.append(3)
        return o[:nt]
    from model_v15_5 import load_v15_5_model
    m=load_v15_5_model(BASE,device="cuda")
    if CKPT!=BASE:
        s=torch.load(CKPT,map_location="cpu",weights_only=False); s=s.get("model",s)
        cur=m.state_dict(); m.load_state_dict({k:v for k,v in s.items() if k in cur and cur[k].shape==v.shape},strict=False)
    m.eval()
    def enc(toks):
        bt=torch.from_numpy(np.asarray(toks,dtype=np.int64)).cuda()
        with torch.no_grad(): z=m.encode_angular_only(bt)
        z=z.float(); z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-10); return z.cpu().numpy().astype("float32")
    tax=json.load(open(FULLTAX)); gc=collections.defaultdict(list)
    for g,t in enumerate(tax):
        if t.get("domain")=="Bacteria" and t.get("genus") and os.path.exists(os.path.join(TOK,str(t.get("accession"))+".npy")):
            gc[t["genus"]].append(g)
    genera=[k for k,v in gc.items() if len(v)>=4]; rng.shuffle(genera); genera=genera[:12]
    genomes=[g for k in genera for g in gc[k][:5]]
    def reads(g,nt,bp,n):
        tk=np.load(os.path.join(TOK,tax[g]["accession"]+".npy"),mmap_mode='r'); out=[]
        for wi in range(min(3,tk.shape[0])):
            d=decode(np.array(tk[wi]))
            for st in range(0,len(d)-bp+1,bp): out.append(bpe(d[st:st+bp],nt))
            if len(out)>=n: break
        rng.shuffle(out); return out[:n]
    print(f"[selftest] {len(genera)} genera, {len(genomes)} genomes; building index at 2kb (n_tok=400)")
    X=[]; gids=[]
    for g in genomes:
        for v in enc(reads(g,400,2000,30)): X.append(v); gids.append(g)
    harness=AtlasHarness(encoder=enc, index=InRAMIndex(np.array(X,"float32"),gids),
                         taxonomy={g:tax[g] for g in range(len(tax))}, k=20)
    print("[selftest] placing genomes (same-genome excluded, honest descent) at 300bp / 2kb / 5kb:\n")
    for g in genomes[:6]:
        truth=f"{tax[g].get('family')}/{tax[g].get('genus')}"
        for nt,bp in [(60,300),(400,2000)]:
            rd=reads(g,nt,bp,64)
            pl=harness.place(rd, bp, exclude=g)
            print(f"  {bp:>5}bp  truth {truth:<34} -> {pl}")
        print()
