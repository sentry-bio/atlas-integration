#!/usr/bin/env python3
"""
benchmark_adaptive.py — the four-axis benchmark of the LIVE adaptive v10.9 serving harness (port 8093).

Everything drives the real HTTP serving path (tokenize->encode->search->fuse->L3->calibrate->JSON), so the
numbers are what production would return. Genuine DNA fragments (decode a window -> cut L bp -> re-encode),
family-grouped, self/family masked via the /place `exclude` affordance, single-tier arms via `force_tiers`.

AXES
  1 SCALE-ROUTING CURVE   family+genus accuracy vs read length, 3 arms {routed, 5kb-only, 20kb-only},
                          ensemble of non-overlapping tiles (the deploy recipe), family-present (mask self).
                          PRE-REG: routed >= max(single arms) at every length; crossover (5kb>20kb at short L).
  2 OPERATING CURVE       native 20kb, in-ref vs withheld-family; TP@FC at FC in {0.05,0.10,0.20} on the
                          calibrated family confidence. Baseline to beat (shipped margin) = 0.492 @FC0.10.
                          Plus the DIAL: commit-rate on novel vs resolved-depth on known across min_confidence.
  3 v10.9 SERVED LIFT     served native family/genus accuracy (family-bootstrap CI) vs the ESTABLISHED v9
                          baseline (capability_benchmark: family 0.808, genus 0.575). Confirms the lift
                          survives the full serving stack (v9 not re-run; cited).
  4 LATENCY               p50/p95/p99 per routing regime through HTTP (short=8x1kb ensemble, crossover=4x5kb,
                          native=1x20kb) — the realistic serving payloads.

Run: NG=40 NG2=120 python benchmark_adaptive.py    (CPU; ~30-50min at NG=40)
"""
import os, sys, json, time, urllib.request
import numpy as np
sys.path.insert(0, "/home/rohit"); sys.path.insert(0, "/home/rohit/build_pipeline")
FTAX = "/zfs_raid/SentryBio/atlas_index_full/300bp/genome_taxonomy_full.json"
TOK = "/zfs_raid/SentryBio/tokenized_4096"; VOCAB = "/home/rohit/biosphere_inference/bpe_vocab.json"
URL = "http://127.0.0.1:8093/place"
NG = int(os.environ.get("NG", "40"))          # genomes for axis 1 (× lengths × arms — the expensive one)
NG2 = int(os.environ.get("NG2", "120"))       # genomes for axes 2/3 (native only, cheaper)
NLAT = int(os.environ.get("NLAT", "40"))      # queries per regime for axis 4
KREF = 15                                     # k neighbors (matches service)
tax = json.load(open(FTAX)); vocab = json.load(open(VOCAB)); rev = {i: t for t, i in vocab.items()}
rng = np.random.RandomState(11)
RANKS = ["domain","phylum","class","order","family","genus","species"]
def decode(tokens): return "".join(rev.get(int(t), "") for t in tokens if not rev.get(int(t), "[").startswith("["))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def place(reads, read_bp, exclude=None, force_tiers=None, min_confidence=0.5):
    body = {"reads": reads, "read_bp": read_bp, "min_confidence": min_confidence}
    if exclude is not None: body["exclude"] = [int(x) for x in exclude]
    if force_tiers is not None: body["force_tiers"] = force_tiers
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))
def rank_of(resp, rank):
    r = next((x for x in resp["ranks"] if x["rank"] == rank), None)
    return r
def famcall(resp): r = rank_of(resp, "family"); return r["top"] if r and not r["gap"] else None
def gencall(resp): r = rank_of(resp, "genus");  return r["top"] if r and not r["gap"] else None

# ── build the query pool + family index ─────────────────────────────────────────────────────────
log("indexing families over the 234K catalog…")
from collections import defaultdict
fam_gids = defaultdict(list)
for g in range(len(tax)):
    f = tax[g].get("family")
    if f: fam_gids[f].append(g)
def has_tok(g): return os.path.exists(os.path.join(TOK, tax[g]["accession"] + ".npy"))
pool = [g for g in rng.permutation(len(tax))[:8000]
        if tax[g].get("family") and tax[g].get("genus") and has_tok(g)]
log(f"{len(pool)} usable query genomes in pool")
def dna_of(g):
    tk = np.load(os.path.join(TOK, tax[g]["accession"] + ".npy"), mmap_mode="r")
    return decode(np.array(tk[0]).astype(np.int64))
def windows_of(g, n):
    """Up to n distinct 20kb windows (decoded DNA) for the deep-ensemble native test — the deploy recipe."""
    tk = np.load(os.path.join(TOK, tax[g]["accession"] + ".npy"), mmap_mode="r")
    nw = min(n, tk.shape[0])
    return [decode(np.array(tk[wi]).astype(np.int64)) for wi in range(nw)]
def tiles(dna, L, ncut):
    if L is None: return [dna]                                   # native = full window
    if len(dna) < L: return []
    return [dna[i:i+L] for i in range(0, min(len(dna), L*ncut) - L + 1, L)] or [dna[:L]]

RESULTS = {}

# ════ AXIS 1 — SCALE-ROUTING CURVE ════════════════════════════════════════════════════════════
def axis1():
    log("AXIS 1 — scale-routing curve (routed vs 5kb-only vs 20kb-only, ensemble, family-present)")
    LENGTHS = [1000, 2000, 3000, 5000, 10000, None]; LK = [str(L) if L else "native" for L in LENGTHS]
    ARMS = {"routed": None, "5kb": ["5kb"], "20kb": ["20kb"]}
    acc = {lk: {a: [0,0,0] for a in ARMS} for lk in LK}         # [fam_hits, gen_hits, n]
    cands = pool[:NG]
    for qi, g in enumerate(cands):
        dna = dna_of(g); tf, tg = tax[g]["family"], tax[g]["genus"]
        for L, lk in zip(LENGTHS, LK):
            reads = tiles(dna, L, 3)
            if not reads: continue
            rb = L if L else 20000
            for a, ft in ARMS.items():
                try: r = place(reads, rb, exclude=[g], force_tiers=ft)
                except Exception as e: log(f"  err g{g} L{lk} {a}: {e}"); continue
                cell = acc[lk][a]
                cell[0] += int(famcall(r) == tf); cell[1] += int(gencall(r) == tg); cell[2] += 1
        if (qi+1) % 10 == 0: log(f"  axis1 {qi+1}/{len(cands)}")
    RESULTS["axis1"] = acc
    print("\n  FAMILY accuracy (rows=read length, cols=arm):")
    print(f"    {'length':<9}" + "".join(f"{a:>10}" for a in ARMS) + "     verdict")
    for lk in LK:
        fa = {a: acc[lk][a][0]/max(acc[lk][a][2],1) for a in ARMS}
        best = max(fa, key=fa.get)
        routed_ok = fa["routed"] >= max(fa["5kb"], fa["20kb"]) - 0.005
        print(f"    {lk:<9}" + "".join(f"{fa[a]:>10.3f}" for a in ARMS) +
              f"   best={best} routed>=arms:{routed_ok}")
    print("\n  GENUS accuracy:")
    print(f"    {'length':<9}" + "".join(f"{a:>10}" for a in ARMS))
    for lk in LK:
        ga = {a: acc[lk][a][1]/max(acc[lk][a][2],1) for a in ARMS}
        print(f"    {lk:<9}" + "".join(f"{ga[a]:>10.3f}" for a in ARMS))

# ════ AXIS 2 — OPERATING CURVE (dial) ═════════════════════════════════════════════════════════
def axis2():
    log("AXIS 2 — operating curve: in-ref vs withheld-family @ native 20kb")
    cands = pool[NG:NG+NG2]
    inref, withheld = [], []     # (family_confidence, correct)
    dial = {mc: {"novel_commit": [0,0], "known_depth": []} for mc in [0.3,0.5,0.7,0.9]}
    for qi, g in enumerate(cands):
        tf = tax[g]["family"]
        reads = windows_of(g, 8)                                 # native 8-window ensemble = production payload
        if not reads: continue
        # in-ref: mask only self
        try: ri = place(reads, 20000, exclude=[g])
        except Exception as e: log(f"  err in g{g}: {e}"); continue
        fr = rank_of(ri, "family")
        if fr and not fr["gap"]:
            inref.append((fr["confidence"], int(fr["top"] == tf)))
        # withheld-family: mask the whole family (novel-organism sim)
        try: rw = place(reads, 20000, exclude=fam_gids[tf])
        except Exception as e: log(f"  err wf g{g}: {e}"); continue
        fw = rank_of(rw, "family")
        if fw and not fw["gap"]:
            withheld.append((fw["confidence"], 0))               # any commit here is a FALSE commit
        # dial: at each min_confidence, does novel abstain & known resolve deep?
        for mc in dial:
            di = place(reads, 20000, exclude=fam_gids[tf], min_confidence=mc)   # novel input
            dk = place(reads, 20000, exclude=[g], min_confidence=mc)            # known input
            dial[mc]["novel_commit"][1] += 1
            dial[mc]["novel_commit"][0] += int(di["resolved_to"] is not None)
            dial[mc]["known_depth"].append(RANKS.index(dk["resolved_to"]) if dk["resolved_to"] else -1)
        if (qi+1) % 20 == 0: log(f"  axis2 {qi+1}/{len(cands)}")
    ic = np.array([c for c,_ in inref]); iok = np.array([o for _,o in inref])
    wc = np.array([c for c,_ in withheld])
    def tp_at_fc(fc):
        if len(wc)==0 or len(ic)==0: return float("nan")
        thr = np.quantile(wc, 1-fc)
        return float(((ic >= thr) & (iok == 1)).sum() / len(ic))
    RESULTS["axis2"] = {"tp_fc": {fc: tp_at_fc(fc) for fc in [0.05,0.10,0.20]},
                        "conf_inref_mean": float(ic.mean()) if len(ic) else None,
                        "conf_withheld_mean": float(wc.mean()) if len(wc) else None,
                        "n_inref": len(ic), "n_withheld": len(wc)}
    print(f"\n  n: {len(ic)} in-ref, {len(wc)} withheld-family")
    print(f"  calibrated family confidence  in-ref={ic.mean():.3f}  withheld={wc.mean():.3f}  (sep={ic.mean()-wc.mean():+.3f})")
    print(f"  TP@FC:  0.05={tp_at_fc(0.05):.3f}   0.10={tp_at_fc(0.10):.3f}   0.20={tp_at_fc(0.20):.3f}   (baseline @0.10 = 0.492)")
    print("  DIAL (min_confidence -> novel commit-rate should DROP, known depth should shorten):")
    for mc in sorted(dial):
        nc = dial[mc]["novel_commit"]; kd = [d for d in dial[mc]["known_depth"] if d>=0]
        print(f"    mc={mc}:  novel-commit={nc[0]/max(nc[1],1):.3f}   known-median-depth={RANKS[int(np.median(kd))] if kd else 'n/a'}")
    RESULTS["axis2"]["dial"] = {str(mc): {"novel_commit": dial[mc]["novel_commit"][0]/max(dial[mc]["novel_commit"][1],1)} for mc in dial}

# ════ AXIS 3 — v10.9 SERVED LIFT: single-window vs multi-window ensemble (WIN 1) ═══════════════
def axis3():
    N = int(os.environ.get("N_WIN", "8"))
    log(f"AXIS 3 — served native accuracy: single-window vs {N}-window ensemble (paired, family-bootstrap CI)")
    cands = pool[NG:NG+NG2]
    rows = []   # (family, fam1, gen1, famN, genN, nwin)
    nwins = []
    for qi, g in enumerate(cands):
        tf, tg = tax[g]["family"], tax[g]["genus"]
        wins = windows_of(g, N)
        if not wins: continue
        try:
            r1 = place([wins[0]], 20000, exclude=[g])           # single window (the axis-3 confound)
            rN = place(wins, 20000, exclude=[g])                # N-window deep ensemble (the deploy recipe)
        except Exception as e: log(f"  err g{g}: {e}"); continue
        rows.append((tf, int(famcall(r1)==tf), int(gencall(r1)==tg),
                     int(famcall(rN)==tf), int(gencall(rN)==tg))); nwins.append(len(wins))
        if (qi+1) % 20 == 0: log(f"  axis3 {qi+1}/{len(cands)}")
    fams = np.array([x[0] for x in rows], object)
    cols = {k: np.array([x[i] for x in rows]) for k, i in [("f1",1),("g1",2),("fN",3),("gN",4)]}
    uf = list(set(fams))
    def famboot(vals):                                          # family-grouped bootstrap
        by = {f: vals[fams==f] for f in uf}
        bs = [np.mean([by[uf[i]].mean() for i in rng.randint(0, len(uf), len(uf))]) for _ in range(2000)]
        return float(np.mean(vals)), float(np.quantile(bs,0.025)), float(np.quantile(bs,0.975))
    f1,f1lo,f1hi = famboot(cols["f1"]); fN,fNlo,fNhi = famboot(cols["fN"])
    g1,g1lo,g1hi = famboot(cols["g1"]); gN,gNlo,gNhi = famboot(cols["gN"])
    med_w = int(np.median(nwins)) if nwins else 0
    RESULTS["axis3"] = {"single":{"family":[f1,f1lo,f1hi],"genus":[g1,g1lo,g1hi]},
                        "ensemble":{"family":[fN,fNlo,fNhi],"genus":[gN,gNlo,gNhi]},
                        "N":N, "median_windows_used":med_w, "n":len(rows),
                        "v9_baseline":{"family":0.808,"genus":0.575}, "capability_multiwindow":{"family":0.908,"genus":0.700}}
    print(f"\n  n={len(rows)}, median windows used={med_w} (cap N={N}); family-bootstrap 95% CI:")
    print(f"    {'':<10}{'single-window':<26}{f'{N}-window ensemble':<26}{'ensemble lift':<14}")
    print(f"    family    {f1:.3f} [{f1lo:.3f},{f1hi:.3f}]     {fN:.3f} [{fNlo:.3f},{fNhi:.3f}]     {fN-f1:+.3f}")
    print(f"    genus     {g1:.3f} [{g1lo:.3f},{g1hi:.3f}]     {gN:.3f} [{gNlo:.3f},{gNhi:.3f}]     {gN-g1:+.3f}")
    print(f"  PRE-REG: ensemble family in [.85,.92] (reproduce capability .908) -> {'PASS' if 0.85<=fN<=0.92 else 'CHECK'}")
    print(f"  vs v9 baseline .808/.575: ensemble family lift {fN-0.808:+.3f}, genus lift {gN-0.575:+.3f}")

# ════ AXIS 4 — LATENCY ════════════════════════════════════════════════════════════════════════
def axis4():
    log("AXIS 4 — latency per PRODUCTION regime (HTTP, real payloads incl. native 8-window ensemble)")
    cands = pool[:NLAT]
    # settled-config regimes: short reads -> 5kb tier; native genome -> 20kb tier with 8-window ensemble.
    regimes = {"short_1kb(5kb,8reads)": (1000, 8), "mid_3kb(20kb,4reads)": (3000, 4), "native(20kb,8win-ensemble)": (20000, 8)}
    lat = {k: [] for k in regimes}; tiers = {}
    for g in cands:
        for name,(rb,nc) in regimes.items():
            reads = windows_of(g, nc) if rb >= 20000 else (tiles(dna_of(g), rb, nc)[:nc])   # native = real ensemble payload
            if not reads: continue
            try: r = place(reads, rb, exclude=[g])
            except Exception as e: continue
            lat[name].append(r["latency_ms"]); tiers[name] = r["scale"]["tier_used"]
    RESULTS["axis4"] = {}
    print(f"\n  {'regime':<24}{'tier':<10}{'n':>4}{'p50':>8}{'p95':>8}{'p99':>8}  (ms)")
    for name in regimes:
        a = np.array(lat[name])
        if len(a)==0: continue
        p50,p95,p99 = np.percentile(a,[50,95,99])
        RESULTS["axis4"][name] = {"tier": tiers.get(name), "n": len(a), "p50": float(p50), "p95": float(p95), "p99": float(p99)}
        print(f"  {name:<24}{tiers.get(name,''):<10}{len(a):>4}{p50:>8.1f}{p95:>8.1f}{p99:>8.1f}")

# ════ WIN 2 — CROSSOVER BAND: measure the fused arm at the 3kb boundary ═══════════════════════
def win2():
    """The pre-registered fusion-band test. At the crossover lengths, is FUSED >= max(single arms)? If yes,
    widen the fused band DOWN (fusion never-worst) instead of moving the hard step. Family-present, ensemble."""
    log("WIN 2 — crossover fusion sweep (5kb / 20kb / fused)")
    LENGTHS = [int(x) for x in os.environ.get("W2_LENGTHS", "2000,2500,3000,3500,4000").split(",")]
    LK = [str(L) for L in LENGTHS]
    ARMS = {"5kb": ["5kb"], "20kb": ["20kb"], "fused": ["5kb", "20kb"]}
    acc = {lk: {a: [0,0,0] for a in ARMS} for lk in LK}
    cands = pool[:int(os.environ.get("NG_W2", "100"))]
    for qi, g in enumerate(cands):
        dna = dna_of(g); tf, tg = tax[g]["family"], tax[g]["genus"]
        for L, lk in zip(LENGTHS, LK):
            reads = tiles(dna, L, 3)
            if not reads: continue
            for a, ft in ARMS.items():
                try: r = place(reads, L, exclude=[g], force_tiers=ft)
                except Exception as e: continue
                cell = acc[lk][a]
                cell[0] += int(famcall(r)==tf); cell[1] += int(gencall(r)==tg); cell[2] += 1
        if (qi+1) % 20 == 0: log(f"  win2 {qi+1}/{len(cands)}")
    RESULTS["win2"] = acc
    print(f"\n  FAMILY (n≈{acc[LK[0]]['fused'][2]}):  fused >= max(arms) - 0.005 ?")
    print(f"    {'length':<9}{'5kb':>10}{'20kb':>10}{'fused':>10}   fused-wins/ties")
    for lk in LK:
        fa = {a: acc[lk][a][0]/max(acc[lk][a][2],1) for a in ARMS}
        ok = fa["fused"] >= max(fa["5kb"], fa["20kb"]) - 0.005
        print(f"    {lk:<9}{fa['5kb']:>10.3f}{fa['20kb']:>10.3f}{fa['fused']:>10.3f}   {ok}")
    print("  GENUS:")
    print(f"    {'length':<9}{'5kb':>10}{'20kb':>10}{'fused':>10}")
    for lk in LK:
        ga = {a: acc[lk][a][1]/max(acc[lk][a][2],1) for a in ARMS}
        print(f"    {lk:<9}{ga['5kb']:>10.3f}{ga['20kb']:>10.3f}{ga['fused']:>10.3f}")
    fused_ever_wins = any(acc[lk]["fused"][0] > max(acc[lk]["5kb"][0], acc[lk]["20kb"][0]) + 0.5 for lk in LK)  # in hit-counts
    print(f"  VERDICT: does fused beat both single arms at ANY length in {LK}? -> "
          f"{'YES (revisit fusion)' if fused_ever_wins else 'NO — fusion stays dead, single-tier routing holds'}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all","1"): axis1()
    if which in ("all","2"): axis2()
    if which in ("all","3"): axis3()
    if which in ("all","4"): axis4()
    if which == "win2": win2()
    json.dump(RESULTS, open(os.environ.get("OUTJSON","/zfs_raid/SentryBio/radial_training/benchmark_settled.json"),"w"), indent=2)
    log(f"DONE in {(time.time()-t0)/60:.1f} min -> benchmark_adaptive.json")
