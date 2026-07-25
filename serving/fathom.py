#!/usr/bin/env python3
"""
Fathom — the V0/V1 spine.  ONE calibrated descent, on ONE cloud, in ONE manifold.
Where it stops IS the classification, the confidence, AND the novelty.  (atlas-place + atlas-confidence
+ atlas-novelty = one traversal, three fields.)  To *fathom* = measure the depth AND comprehend.

STEP 1 (this file): the pure descent + conformal machinery + result object. No embeddings, no GPU — the
neighborhood is passed in as (genome_id, distance) candidates, so the *logic* is unit-testable in isolation.
Later steps supply the neighborhood from the real index (encoders/coverage/index = neighborhood fidelity).

  ┌─ COHERENCE NOTE (serving refactor, 2026-07-17) ──────────────────────────────────────────────┐
  │ The SERVING scorer is now atlas_evidence (L3) — this file is NOT a third serving path.  Fathom's │
  │ lasting role is CALIBRATION-FIT machinery: `ConformalNull` produces the per-rank null p-values    │
  │ from which calibration.json is fit, and atlas_evidence._calibrated_confidence CONSUMES that table │
  │ at serve time.  So the conformal idea lives on inside L3's (margin, support) calibration; it is    │
  │ pulled INTO the calibration path rather than run as a parallel descent.  Fathom.descend remains an │
  │ isolation-based experiment; do not wire it into serving — build_evidence(...).decide(...) is truth.│
  └─────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

# ── conformal per-rank null (distribution-free) ─────────────────────────────────────────────
class ConformalNull:
    """Null distribution of the ISOLATION score for KNOWN organisms scored at their TRUE rank.
    p_value(s) = fraction of calibration scores >= s  ->  high p = query is as-close-or-closer than a
    genuine member (belongs);  low p = query is an outlier (novel below this rank)."""
    def __init__(self, scores):
        self.scores = np.sort(np.asarray(scores, dtype=float))
    def p_value(self, s: float) -> float:
        n = len(self.scores)
        if n == 0: return 1.0
        return (1.0 + float(np.count_nonzero(self.scores >= s))) / (n + 1.0)

# ── result object: the three fields of the one traversal ────────────────────────────────────
@dataclass
class Fathoming:
    lineage: list          # [(rank, taxon, p_value, margin, isolation), ...] up to the stop
    resolved_to: str|None  # deepest accepted rank (classification)
    novel_at: str|None     # rank BELOW resolved_to where descent halted  (novelty == depth honesty)
    confidence: float      # p_value at resolved_to
    support: float         # local reference density; LOW -> "uncovered", not "novel"
    n_reads: int
    trace: list = None     # per-rank [(rank, top, margin, iso, p, decision)] — why it stopped
    def uncovered(self, support_floor: float) -> bool:
        return self.support < support_floor
    def call(self) -> str:
        if not self.lineage: return "unplaced"
        r, t, *_ = self.lineage[-1]
        return f"{t} ({r})"

# ── the engine ──────────────────────────────────────────────────────────────────────────────
class Fathom:
    def __init__(self, taxonomy_of, nulls, alpha=0.05, margin_min=0.5, k=15, support_floor=0.0):
        self.taxonomy_of = taxonomy_of   # genome_id -> {rank: taxon}
        self.nulls = nulls               # {rank: ConformalNull}
        self.alpha = alpha; self.margin_min = margin_min; self.k = k; self.support_floor = support_floor

    def _rank_readout(self, neighbors, rank):
        """(top_taxon, vote_margin, isolation) at a rank from proximity-weighted neighbor votes."""
        w = defaultdict(float); dists = defaultdict(list)
        for gid, d in neighbors:
            t = self.taxonomy_of.get(gid, {}).get(rank)
            if not t: continue
            w[t] += 1.0 / (1e-6 + d); dists[t].append(d)
        if not w: return None
        top = max(w, key=w.get)
        margin = w[top] / sum(w.values())
        isolation = float(np.mean(sorted(dists[top])[: self.k]))   # closeness to the top-taxon members
        return top, margin, isolation

    def descend(self, neighbors, n_reads=1, support=None) -> Fathoming:
        """neighbors: list[(genome_id, geodesic_distance)]. The one traversal."""
        neighbors = sorted(neighbors, key=lambda x: x[1])
        if support is None:
            near = [d for _, d in neighbors[: self.k]]
            support = 1.0 / (1e-6 + float(np.mean(near))) if near else 0.0
        lineage = []; resolved = None; novel = None; trace = []
        for rank in RANKS:
            r = self._rank_readout(neighbors, rank)
            if r is None:
                # taxonomy GAP at this rank (e.g. GTDB c__Unknown) -> skip, not a boundary
                trace.append((rank, None, 0.0, None, None, "skip-gap")); continue
            top, margin, iso = r
            p = self.nulls[rank].p_value(iso) if rank in self.nulls else 1.0
            if p >= self.alpha and margin >= self.margin_min:
                lineage.append((rank, top, p, margin, iso)); resolved = rank
                trace.append((rank, top, margin, iso, p, "accept"))
            else:
                why = "low-margin" if margin < self.margin_min else "conformal-reject"
                trace.append((rank, top, margin, iso, p, why)); novel = rank; break
        conf = lineage[-1][2] if lineage else 0.0
        return Fathoming(lineage, resolved, novel, conf, support, n_reads, trace)
