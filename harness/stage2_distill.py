#!/usr/bin/env python3
"""
stage2_distill.py — ONLY IF Stage 1's radius is too weak. Unfreeze the trunk, but DISTILL the angle to stay put.

Rationale: if the frozen-trunk head can't express complexity well enough, let gradients into the trunk — but
penalize every genome's angular DIRECTION for moving from where v10.9 put it.  LAMBDA is the compromise dial
you set consciously: high -> angle ~frozen (approaches Stage 1); low -> radius free to drag the trunk.
Sweep LAMBDA, measure angular drift + placement regression on the gates, pick the knee.

Loss:  L_complexity(r_h)  +  LAMBDA * mean(1 - cos(z_ang_new, z_ang_ref))     [z_ang_ref = frozen v10.9]

Run:  LAMBDA=1.0 python stage2_distill.py    (try several LAMBDA; higher = safer angle)
"""
import os, sys, json, time, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/sentrybio/scripts")
sys.path.insert(0, "/home/rohit/build_pipeline")
from atlas_encoder import load_effective_encoder, PAD_ID
V9 = "/home/rohit/v9_best.pt"; V109 = "/home/rohit/v10_curvature_field/v10_9_encoder.pt"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; OUT = "/zfs_raid/SentryBio/radial_training"
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
NTOK = 4096; NWIN = int(os.environ.get("NWIN", "6")); EPOCHS = int(os.environ.get("EPOCHS", "60"))
LR = float(os.environ.get("LR", "2e-5")); LAMBDA = float(os.environ.get("LAMBDA", "1.0"))
DEV = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "") else "cpu"
tax = json.load(open(FTAX))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

z0 = np.load(os.path.join(OUT, "complexity_v10.9.npz"), allow_pickle=True)
gids, target_r = z0["gids"], z0["target_r"].astype("float32")

enc = load_effective_encoder(V9, V109, device=DEV); m = enc.model            # TRAINABLE
ref = load_effective_encoder(V9, V109, device=DEV).model                     # FROZEN reference (v10.9 angle)
for p in ref.parameters(): p.requires_grad = False; ref.eval()
for p in m.parameters(): p.requires_grad = True                              # trunk + head both train now

def rows_for(g):
    tk = np.load(os.path.join(TOK, tax[int(g)]["accession"] + ".npy"), mmap_mode="r"); out = []
    for wi in range(min(NWIN, tk.shape[0])):
        t = np.array(tk[wi, :NTOK]).astype(np.int64)
        out.append(np.pad(t, (0, NTOK - len(t)), constant_values=PAD_ID) if len(t) < NTOK else t)
    return out
toks, tgt = [], []
for g, tr in zip(gids, target_r):
    for r in rows_for(g): toks.append(r); tgt.append(tr)
toks = torch.from_numpy(np.stack(toks)).long(); tgt = torch.tensor(tgt, dtype=torch.float32, device=DEV)
log(f"{len(toks)} samples on {DEV}; LAMBDA(angle-preserve)={LAMBDA}")

def zang(model, bt): return model.ode_flow(model.encode_raw(bt), model.live_kappa)[0]
opt = torch.optim.Adam(m.parameters(), lr=LR); BS = 32
for ep in range(EPOCHS):
    perm = torch.randperm(len(toks)); tot = tp = td = 0.0
    for i in range(0, len(toks), BS):
        idx = perm[i:i+BS]; bt = toks[idx].to(DEV); ty = tgt[idx]
        za = zang(m, bt)
        with torch.no_grad(): za_ref = zang(ref, bt)
        r_h, _ = m.radial_head(za, bt)
        l_comp = F.smooth_l1_loss(r_h, ty) + F.relu(0.5 - r_h.std())
        cos = F.cosine_similarity(za.float(), za_ref.float(), dim=-1)
        l_pres = (1 - cos).mean()
        loss = l_comp + LAMBDA * l_pres
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); tp += float(l_comp); td += float(l_pres)
    if ep % 10 == 0 or ep == EPOCHS - 1:
        log(f"ep{ep:2d} loss={tot:.2f} complexity={tp:.2f} angle-drift={td:.4f}")

# ── measure the compromise: angular drift + radius spread ────────────────────────────────────────
with torch.no_grad():
    pb = toks[:512].to(DEV)
    drift = (1 - F.cosine_similarity(zang(m, pb).float(), zang(ref, pb).float(), dim=-1)).mean().item()
    rr = m.radial_head(zang(m, pb), pb)[0]
log(f"COMPROMISE @ LAMBDA={LAMBDA}: mean angular drift(1-cos)={drift:.4f}  radius sd={rr.std():.2f}")
log("  -> re-run validate_geometry.py (T4 placement, T2) on encoder_stage2 to price the drift, then pick LAMBDA.")
torch.save({"model": m.state_dict(), "lambda": LAMBDA, "angular_drift": drift},
           os.path.join(OUT, f"encoder_stage2_lam{LAMBDA}.pt"))
log(f"saved -> {OUT}/encoder_stage2_lam{LAMBDA}.pt ; DONE")
