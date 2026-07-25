#!/usr/bin/env python3
"""
stage1_radial.py — retarget the radial head onto the complexity radius. FROZEN TRUNK.

Trains only {radial_head} so r_h(genome) tracks the Stage-0 complexity target, remapped into the RESPONSIVE
ball range (off the tanh-saturated boundary).  The head's delta_net already detaches z_ang internally, and
we freeze the trunk, so `encode_angular_only` (the served ANGULAR index) is provably bit-identical before/after
— asserted below.  Only `encode()` (full ball = direction x new radius) changes.

Loss:  Huber(r_h, target_r)  [place in responsive range, ordered by complexity]
     + pairwise-rank         [monotonicity guarantee]
     + spread floor          [never collapse back to a shell]

Run (GPU when free, else CPU):  python stage1_radial.py    # honors CUDA_VISIBLE_DEVICES
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts")
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; OUT = "/zfs_raid/SentryBio/radial_training"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
NTOK = 4096; NWIN = int(os.environ.get("NWIN", "8")); EPOCHS = int(os.environ.get("EPOCHS", "300"))
LR = float(os.environ.get("LR", "1e-3")); MARGIN = 0.2; SD_FLOOR = 0.5; L_RANK, L_SPREAD = 0.3, 0.3
DEV = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "") else "cpu"
tax = json.load(open(FTAX))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── target from Stage 0 ─────────────────────────────────────────────────────────────────────────
z0 = np.load(os.path.join(OUT, "complexity_v10.9.npz"), allow_pickle=True)
gids, target_r, signal = z0["gids"], z0["target_r"].astype("float32"), str(z0["signal"])
log(f"target signal={signal}; {len(gids)} genomes, target_r range {target_r.min():.2f}..{target_r.max():.2f}")

# ── encode windows -> (tokens, target) samples; z_ang computed frozen ────────────────────────────
enc = load_effective_encoder(V9, V109, device=DEV); m = enc.model
for p in m.parameters(): p.requires_grad = False
for p in m.radial_head.parameters(): p.requires_grad = True          # ONLY the head trains
def rows_for(g):
    tk = np.load(os.path.join(TOK, tax[int(g)]["accession"] + ".npy"), mmap_mode="r")
    out = []
    for wi in range(min(NWIN, tk.shape[0])):
        t = np.array(tk[wi, :NTOK]).astype(np.int64)
        out.append(np.pad(t, (0, NTOK - len(t)), constant_values=PAD_ID) if len(t) < NTOK else t)
    return out
toks, tgt = [], []
for g, tr in zip(gids, target_r):
    for r in rows_for(g): toks.append(r); tgt.append(tr)
toks = torch.from_numpy(np.stack(toks)).long(); tgt = torch.tensor(tgt, dtype=torch.float32, device=DEV)
log(f"{len(toks)} window samples on {DEV}")

# angular-index preservation probe (must be identical after training)
probe = toks[:32].to(DEV)
with torch.no_grad(): ang_before = m.encode_angular_only(probe).clone()

opt = torch.optim.Adam(m.radial_head.parameters(), lr=LR)
BS = 64
for ep in range(EPOCHS):
    perm = torch.randperm(len(toks)); tot = 0.0
    for i in range(0, len(toks), BS):
        idx = perm[i:i+BS]; bt = toks[idx].to(DEV); ty = tgt[idx]
        with torch.no_grad():                                        # frozen trunk -> z_ang
            za = m.ode_flow(m.encode_raw(bt), m.live_kappa)[0]
        r_h, _ = m.radial_head(za, bt)                               # head trainable (detaches z_ang inside)
        l_place = F.smooth_l1_loss(r_h, ty)
        d = r_h.unsqueeze(0) - r_h.unsqueeze(1); dt = ty.unsqueeze(0) - ty.unsqueeze(1)
        l_rank = F.relu(MARGIN - torch.sign(dt) * d)[dt.abs() > 1e-3].mean() if (dt.abs() > 1e-3).any() else r_h.sum()*0
        l_spread = F.relu(SD_FLOOR - r_h.std())
        loss = l_place + L_RANK * l_rank + L_SPREAD * l_spread
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    if ep % 25 == 0 or ep == EPOCHS - 1:
        with torch.no_grad():
            rr = m.radial_head(m.ode_flow(m.encode_raw(toks[:256].to(DEV)), m.live_kappa)[0], toks[:256].to(DEV))[0]
        log(f"ep{ep:3d} loss={tot:.3f}  r_h mean={rr.mean():.2f} sd={rr.std():.2f} range {rr.min():.2f}-{rr.max():.2f}")

# ── PROVE the angular index is untouched ─────────────────────────────────────────────────────────
with torch.no_grad(): ang_after = m.encode_angular_only(probe)
drift = (ang_before - ang_after).abs().max().item()
log(f"ANGULAR-INDEX DRIFT max|Δ| = {drift:.2e}  -> {'PRESERVED (index untouched)' if drift < 1e-6 else 'DRIFT! bug'}")
assert drift < 1e-6, "angular index changed — trunk was not frozen correctly"

os.makedirs(OUT, exist_ok=True)
torch.save({"model": m.state_dict(), "signal": signal, "note": "v10.9 + Stage1 complexity-radius; angle unchanged"},
           os.path.join(OUT, "encoder_stage1.pt"))
log(f"saved -> {OUT}/encoder_stage1.pt ; DONE. (Gate next: r_h vs COG breadth on held-out; reduction-inward check.)")
