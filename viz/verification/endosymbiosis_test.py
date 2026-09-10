#!/usr/bin/env python3
"""
RECEIPT: organelle genomes cluster by HOST phylogeny, not by bacterial ancestor — the model
rediscovered why DNA barcoding works, not endosymbiotic ancestry. One documented exception
(Jakoba libera mitochondrion -> Asgard archaea) is the interesting edge case, not the rule.

The romantic hypothesis going in was "mitochondria should sit near Alphaproteobacteria,
chloroplasts near Cyanobacteria" (endosymbiotic ancestry, encoded structurally). REFUTED: 100%
of organelles have a eukaryote as their nearest genome neighbor; a fish mitochondrion's nearest
neighbors are other fish. This makes sense in hindsight — a modern organelle's overwhelming
sequence signal is co-divergence with its host over ~1-2 billion years, not residual similarity
to a bacterial ancestor that diverged that long ago. The exception - Jakoba libera, one of the
most bacteria-like, least-reduced mitochondrial genomes known - is exactly where a signal from
deeper ancestry should still be legible, and it neighbors Heimdallarchaeota/Woesearchaeota
(Asgard/Woese archaea) rather than other eukaryotes. That genome is the viewer's featured
"deepest organelle" demo.

Usage: python3 endosymbiosis_test.py --index-dir ... --organelle-emb organelle_emb.npz
       (organelle_emb.npz is produced by encoding organelle genomes through the SAME encoder
        that built the index — see docs/PROVENANCE.md if you need to regenerate it; this is
        NOT part of build_bundle.py because organelles turned out to already be in the index,
        see stage2_relabel_organelles's docstring)
"""
import argparse, json, sys
from collections import Counter
import numpy as np
import faiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--organelle-emb", required=True, help=".npz with keys emb (N,129), accs (N,)")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    idx = faiss.read_index(f"{args.index_dir}/index.faiss")
    n = idx.ntotal
    gids = np.load(f"{args.index_dir}/gids.npy").astype(np.int64)
    V = idx.reconstruct_n(0, n).astype(np.float64)
    G = int(gids.max()) + 1
    sums = np.zeros((G, idx.d))
    np.add.at(sums, gids, V)
    cnt = np.maximum(np.bincount(gids, minlength=G), 1).astype(np.float64)
    gemb = (sums / cnt[:, None]).astype(np.float32)
    del V, sums
    tax = json.load(open(args.taxonomy))

    d = np.load(args.organelle_emb, allow_pickle=True)
    oemb = d["emb"].astype(np.float32)
    oacc = d["accs"]
    n_o = len(oemb)

    gi = faiss.IndexFlatL2(gemb.shape[1])
    gi.add(gemb)
    D1, I1 = gi.search(oemb, args.k)
    top1_dom, pooled_dom = [], []
    for i in range(n_o):
        for k in range(args.k):
            j = int(I1[i, k])
            dom = tax[j].get("domain") or "?"
            pooled_dom.append(dom)
            if k == 0:
                top1_dom.append(dom)
    print("== ORGANELLE -> GENOME neighbors ==")
    print("top-1 domain:", Counter(top1_dom).most_common(6))
    print("pooled domain (k={}):".format(args.k), Counter(pooled_dom).most_common(6))

    print("\n== EXEMPLARS ==")
    rs = np.random.RandomState(1).choice(n_o, 5, replace=False)
    for i in rs:
        ns = [f"{tax[int(I1[i,k])].get('species') or tax[int(I1[i,k])].get('genus')}"
              f"(d={np.sqrt(D1[i,k]):.2f})" for k in range(3)]
        print(f"  {oacc[i]}: " + " | ".join(ns))


if __name__ == "__main__":
    main()
