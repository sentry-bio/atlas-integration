#!/usr/bin/env python3
"""
RECEIPT (NEGATIVE RESULT — do not resurrect without re-running this first): a v9-style
"LUCA-centered radial divergence" chart cannot be honestly recovered in the v10.9 geometry.

The proposal: declare a versioned origin ("luca-proxy-1.0" = Karcher mean of a prokaryote
sample), chart radius = EXACT geodesic distance from that origin (a real theorem, not a
fabricated composite score like v9's "k-mer entropy + COG + SSU"), and print
rho(radius, independent SSU divergence) on the chart so the story's truth-content is measured,
not asserted.

Discipline: compute rho BEFORE writing the chart or its caption. This script is that
measurement. Result (last run, 234K genomes vs gtdb_molecular_divergence.json ssu_identity,
n=52,787 matched): rho = 0.031 (p=7.2e-13 — real but negligible effect size). The geodesic radii
are also DEGENERATE: median radius is IDENTICAL across bacteria/archaea/eukaryota/mito/chloro
(8.758), because v10.9 is angular-only (embeddings hug the Poincare ball boundary, norms
~0.83-1.0 against ball radius 0.894) — every interior point is saturated-far from every other
point near the boundary. The datum's "radius: advisory" caveat now has a number: ~0.03.

VERDICT: do not ship this chart. The encoder simply does not encode divergence radially. This
is not a bug to fix — it is why the angular structure (constellations, theta, domain separation)
works as well as it does; the radial coordinate was not spent on faking a hierarchy.

If you want to try again: a Euclidean tangent-frame distance (rather than exact geodesic) might
avoid the boundary-saturation degeneracy, but it carries a WEAKER guarantee ("tangent-frame
distance", not "geodesic distance") and must pass this same rho gate before any chart ships.
"""
import argparse, csv, json, sys
import numpy as np
import faiss

try:
    from scipy.stats import spearmanr
except ImportError:
    sys.exit("pip install scipy")

C = 1.25
SC = np.sqrt(C)
RMAX = 1.0 / SC - 1e-4


def clipball(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    f = np.minimum(1.0, RMAX / np.maximum(n, 1e-12))
    return x * f


def madd(a, b):
    ab = (a * b).sum(-1, keepdims=True)
    a2 = (a * a).sum(-1, keepdims=True)
    b2 = (b * b).sum(-1, keepdims=True)
    num = (1 + 2 * C * ab + C * b2) * a + (1 - C * a2) * b
    den = 1 + 2 * C * ab + (C ** 2) * a2 * b2
    return num / np.maximum(den, 1e-12)


def logmap(o, x):
    lam = 2.0 / (1 - C * (o * o).sum())
    w = madd(-o, x)
    wn = np.linalg.norm(w, axis=-1, keepdims=True).clip(1e-12, RMAX)
    return (2.0 / (SC * lam)) * np.arctanh(SC * wn) * (w / wn)


def expmap(o, v):
    lam = 2.0 / (1 - C * (o * o).sum())
    vn = np.linalg.norm(v).clip(1e-12)
    return madd(o, np.tanh(SC * lam * vn / 2.0) * v / (SC * vn))


def gdist(o, x):
    w = madd(-o, x)
    wn = np.linalg.norm(w, axis=-1).clip(0, RMAX)
    return (2.0 / SC) * np.arctanh(SC * wn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--domain-bin", required=True, help="domain.bin from a built bundle (uint8 codes)")
    ap.add_argument("--ssu-divergence", required=True, help="gtdb_molecular_divergence.json")
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    idx = faiss.read_index(f"{args.index_dir}/index.faiss")
    n = idx.ntotal
    gids = np.load(f"{args.index_dir}/gids.npy").astype(np.int64)
    V = idx.reconstruct_n(0, n).astype(np.float64)
    G = int(gids.max()) + 1
    sums = np.zeros((G, idx.d))
    np.add.at(sums, gids, V)
    cnt = np.maximum(np.bincount(gids, minlength=G), 1).astype(np.float64)
    emb = clipball(sums / cnt[:, None])
    del V, sums
    tax = json.load(open(args.taxonomy))
    dom = np.fromfile(args.domain_bin, dtype=np.uint8)

    print("Karcher mean of prokaryotes (luca-proxy-1.0)", file=sys.stderr)
    prok = np.where((dom == 0) | (dom == 1))[0]
    samp = rng.choice(prok, min(30000, len(prok)), replace=False)
    o = emb[samp[:2000]].mean(0)
    o = clipball(o[None])[0]
    for it in range(12):
        v = logmap(o[None], emb[samp]).mean(0)
        o = clipball(expmap(o, 0.7 * v)[None])[0]
        if it % 4 == 3:
            print(f"   iter{it} |step|={np.linalg.norm(v):.5f} (should shrink toward 0 — if it "
                  f"doesn't, the origin has nowhere sensible to converge)", file=sys.stderr)

    d = gdist(o[None], emb)
    div = json.load(open(args.ssu_divergence))
    xs, ys = [], []
    for j in range(G):
        a = tax[j].get("accession")
        r = div.get(a)
        if r and r.get("ssu_identity") is not None:
            xs.append(d[j])
            ys.append(100.0 - r["ssu_identity"])
    xs, ys = np.array(xs), np.array(ys)
    rho, p = spearmanr(xs, ys)
    print(f"\nmatched {len(xs)} genomes")
    print(f"rho(geodesic radius from luca-proxy, SSU divergence) = {rho:.3f} (p={p:.1e})")
    for dm, nm in [(0, "bac"), (1, "arc"), (2, "euk"), (4, "mito"), (5, "chl")]:
        print(f"   {nm} radius p50 {np.median(d[dom == dm]):.3f}  <- IDENTICAL across classes = degenerate")
    print("\nGATE: rho < 0.3 => do not ship. Re-run this before ever wiring a LUCA chart into the viewer.")


if __name__ == "__main__":
    main()
