#!/usr/bin/env python3
"""
stage0_complexity.py — is functional complexity EMERGENT, and which signal captures it?

Decides the training target BEFORE any training.  For a validation subset (genomes with eggNOG):
  TEACHER  : COG functional-category BREADTH (distinct categories across the genome's genes) — ground truth.
  EMERGENT candidates (no annotation), per genome from N frozen-encoded windows:
    (a) angular window spread   — mean pairwise (1-cos) of z_ang dirs   [expected WEAK: windows share lineage]
    (b) raw window spread       — same on encode_raw (z0)               [functional content the angle discards?]
    (c) token k-mer richness    — distinct-token entropy of the windows  [pure compositional]
    (d) model's own r0 prior    — deterministic_prior(tokens)            [what the head already computes]
Correlate each vs COG breadth (Spearman) -> pick the emergent signal that tracks complexity.  That winner
becomes the Stage-1 training target; if none tracks, fall back to COG breadth itself (annotation-bound).
Also emits the ANALYTIC radius (chosen complexity -> responsive ball range) = a navigable morphospace with
ZERO training, zero angular risk (Stage 0 deliverable).

CPU-ok, GPU-masked. Run:  CUDA_VISIBLE_DEVICES="" python stage0_complexity.py
"""
import os, sys, json, glob, time
import numpy as np, torch
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts")
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; EGGNOG = "/zfs_raid/SentryBio/eggnog/output"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
OUT = "/zfs_raid/SentryBio/radial_training"; os.makedirs(OUT, exist_ok=True)
NTOK = 4096; NWIN = int(os.environ.get("NWIN", "16")); NGEN = int(os.environ.get("NGEN", "400"))
R_LO, R_HI = 0.6, 3.9                       # responsive ball-radius range (tanh not yet saturated) — see design
DROP_CATS = set("S-")                        # S=unknown function, -=none: exclude from "breadth"
tax = json.load(open(FTAX)); rng = np.random.RandomState(0)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── COG breadth teacher ────────────────────────────────────────────────────────────────────────
def cog_index():
    idx = {}
    for f in glob.glob(os.path.join(EGGNOG, "*", "*.emapper.annotations")):
        acc = os.path.basename(f).split(".emapper")[0]          # GCF_000006745.1
        idx[acc] = f
    return idx
def cog_breadth(path):
    cats = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"): continue
            col = line.split("\t")
            if len(col) > 6:
                for ch in col[6].strip():                        # COG_category, e.g. "JKL"
                    if ch not in DROP_CATS: cats.add(ch)
    return len(cats)

# ── emergent candidates ────────────────────────────────────────────────────────────────────────
def token_entropy(tk):                                           # distinct-token Shannon entropy (non-PAD)
    nz = tk[tk != PAD_ID]
    if len(nz) == 0: return 0.0
    c = np.bincount(nz, minlength=4096).astype(float); p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())
def spread(V):                                                   # mean pairwise (1 - cos) of unit dirs
    U = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    C = U @ U.T; n = len(U)
    return float((1 - C)[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10: return float("nan")
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m])); return float(np.corrcoef(rx, ry)[0, 1])

# ── sample genomes with BOTH tokenized windows AND eggNOG ────────────────────────────────────────
log("indexing eggNOG annotations…"); COG = cog_index(); log(f"  {len(COG)} annotated genomes")
cand = [g for g in range(len(tax)) if tax[g].get("accession") in COG
        and os.path.exists(os.path.join(TOK, tax[g]["accession"] + ".npy"))]
rng.shuffle(cand); cand = cand[:NGEN]
log(f"{len(cand)} genomes with tokens+eggNOG sampled")

def run(tag, overlay):
    enc = load_effective_encoder(V9, overlay, device="cpu"); m = enc.model
    breadth, s_ang, s_raw, s_tok, s_r0, dom = [], [], [], [], [], []
    for gi, g in enumerate(cand):
        acc = tax[g]["accession"]
        tk = np.load(os.path.join(TOK, acc + ".npy"), mmap_mode="r")
        W = min(NWIN, tk.shape[0]); rows = []
        for wi in range(W):
            t = np.array(tk[wi, :NTOK]).astype(np.int64)
            rows.append(np.pad(t, (0, NTOK - len(t)), constant_values=PAD_ID) if len(t) < NTOK else t)
        rows = np.stack(rows); bt = torch.from_numpy(rows).long()
        with torch.no_grad():
            z0 = m.encode_raw(bt).float().numpy()                # raw embedding (pre lineage-projection)
            zang = m.encode_angular_only(bt).float().numpy()     # lineage direction
            r0, _ = m.radial_head.deterministic_prior(bt); r0 = r0.float().numpy()
        breadth.append(cog_breadth(COG[acc]))
        s_ang.append(spread(zang)); s_raw.append(spread(z0))
        s_tok.append(np.mean([token_entropy(r) for r in rows])); s_r0.append(float(r0.mean()))
        dom.append(tax[g].get("domain"))
        if (gi + 1) % 100 == 0: log(f"  {tag}: {gi+1}/{len(cand)}")
    breadth = np.array(breadth, float)
    cands = {"angular-spread": s_ang, "raw-spread": s_raw, "token-entropy": s_tok, "r0-prior": s_r0}
    print(f"\n=== {tag} === COG breadth: mean={breadth.mean():.1f} range {breadth.min():.0f}-{breadth.max():.0f}")
    print("  emergent candidate            Spearman(vs COG breadth)")
    best = (None, -1)
    for name, v in cands.items():
        r = spear(v, breadth); print(f"    {name:<24} {r:+.3f}")
        if abs(r) > best[1]: best = (name, abs(r), name, v, r)
    print(f"  >>> WINNER: {best[0]} (|rho|={best[1]:.3f}) -> {'EMERGENT viable' if best[1] > 0.4 else 'WEAK — fall back to COG breadth as target'}")
    # analytic radius from the chosen complexity signal (winner if viable, else COG breadth)
    comp = np.array(best[3], float) if best[1] > 0.4 else breadth
    cn = (comp - comp.min()) / (comp.ptp() + 1e-9)
    target_r = R_LO + (R_HI - R_LO) * cn
    np.savez(os.path.join(OUT, f"complexity_{tag}.npz"),
             gids=np.array(cand), breadth=breadth, target_r=target_r, comp=comp,
             signal=("emergent:" + best[0]) if best[1] > 0.4 else "cog_breadth",
             **{k: np.array(v) for k, v in cands.items()})
    log(f"  saved complexity + analytic target_r -> {OUT}/complexity_{tag}.npz  (signal={best[0] if best[1]>0.4 else 'cog_breadth'})")

run("v9", None); run("v10.9", V109)
print("\nDONE — Stage 0: chosen target + analytic radius written; feed target_r into stage1.")
