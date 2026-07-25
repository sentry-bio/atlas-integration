#!/usr/bin/env python3
"""
atlas_evidence.py — Layer 3: the pure evidence function.

The refactor's elegant core.  Everything the model KNOWS about a query is assembled here as *evidence*;
a *verdict* is a VIEW over that evidence at a chosen operating point.  The previous instantiation
(atlas_harness.descend / fathom.Fathom.descend) conflated the two — it computed the evidence and rendered
one verdict in the same pass, baking the operating point in and truncating the ladder at compute time.
This module separates them.  Seven design decisions, each traceable to a measured result:

  #1  EVIDENCE vs DECISION.  build_evidence() computes the per-rank calibrated distribution ONCE;
      decide() is a cheap pure view that thresholds it.  Re-decide at any precision/recall point with
      ZERO recompute — because the P-R data says the product is the whole curve (the dial), not a point.
  #2  FULL LADDER, never truncated.  score_ranks() computes confidence at EVERY rank (domain..species)
      and never `break`s.  The cut is a presentation choice in decide(), not a computation choice.
      (recall is index-bound, precision is threshold-bound — the evidence must span the whole curve.)
  #3  NEIGHBORHOOD is first-class.  The K nearest known genomes + distances ARE the triage signal
      ("map, not classifier"); the taxonomic vote is a summary of them.  Returned, not discarded.
  #4  SUPPORT-CONDITIONED calibration.  Confidence is looked up over (margin, support), not margin
      alone — because conformal was only MARGINALLY valid (a novel-species call at 0.6 is less reliable
      than an in-ref call at 0.6), and support is the regime proxy.  Schema is 2D now; a 1D table is the
      single-support-bin degenerate case, so we can populate it 1D today and 2D once the 20kb index exists.
  #5  PER-RANK INDEPENDENT calls, presented as a TABLE not a path.  Independent voting beat
      hierarchical-constrained descent by ~9pt genus (constraint propagates high-rank errors down), so we
      keep it — but at low confidence the ranks needn't nest, and that honest disagreement is information a
      triage user wants to SEE, not a bug to hide.  decide(require_nesting=...) offers the path view when asked.
  #6  SCORER FROZEN as v1.  Aggregation is at its measured tuning frontier (dedup/hierarchical/hubness/
      per-read all declined).  So the winners (sim**3 sharpen, no dedup, top-k neighbors) are a named,
      versioned unit — not an env-knob surface.  A future idea becomes scorer_v2, A/B'd on the frozen grid.
  #8  DEGENERATE states are EVIDENCE, not exceptions.  Empty neighborhood / all-gap / too-few-reads are
      the most interesting triage answers ("nothing near this -> maximally novel"), returned as a normal
      Evidence with support=0, never a raised error.

PURE: no torch, no faiss, no GPU, no I/O.  The neighborhood is passed in as (gid, similarity) — the SEARCH
(L1) and ENCODE (L2) live elsewhere.  So the logic is unit-testable in isolation (see test_atlas_evidence.py),
backend-agnostic (exact / Annoy / FAISS), and encoder-agnostic (v9 / v10.9 / ...).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import math

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
DEPTH = {r: i for i, r in enumerate(RANKS)}

SCORER_VERSION = "v1"           # frozen aggregation (decision #6); bump only via a measured A/B win
GAMMA = 3.0                     # sim**gamma sharpen — calibrated on deployed regimes (tune_aggregation.py)
RUNNERS_UP = 3                  # how many alternatives to surface per rank


# ── data structures ───────────────────────────────────────────────────────────────────────────
@dataclass
class Neighbor:
    """One known genome near the query.  The raw triage signal (decision #3)."""
    gid: int
    similarity: float               # cosine, or 1 - d^2/2 from angular distance
    lineage: dict = field(default_factory=dict)   # rank -> taxon (denormalized for the response)
    accession: str | None = None

    def as_dict(self):
        # geodesic_distance = the angular (gauge-free) metric invariant, arccos(cosine). This is the
        # datum's physical quantity — a distance between two organisms in the tree of life — as opposed to
        # raw coordinates (which are gauge). similarity kept for backward-compat.
        s = max(-1.0, min(1.0, self.similarity))
        return {"gid": self.gid, "similarity": round(self.similarity, 4),
                "geodesic_distance": round(math.acos(s), 4),
                "accession": self.accession, "lineage": self.lineage}


@dataclass
class RankReadout:
    """Independent evidence statement at ONE rank (decision #5).  Ranks need not nest."""
    rank: str
    top: str | None                 # None => taxonomy GAP here (no vote), NOT a boundary
    confidence: float               # calibrated P(correct) over (margin, support); raw margin if uncalibrated
    margin: float                   # sharpened vote share of the top taxon (evidence strength)
    support: float                  # neighborhood density (same across ranks — the regime/novelty proxy)
    runners_up: list = field(default_factory=list)   # [(taxon, margin), ...] alternatives
    n_taxa: int = 0                 # distinct taxa that voted (evidence breadth)
    gap: bool = False               # True => rank absent from the reference taxonomy (skip, not novelty)

    def as_dict(self):
        return {"rank": self.rank, "top": self.top, "confidence": round(self.confidence, 4),
                "margin": round(self.margin, 4), "support": round(self.support, 4),
                "runners_up": [(t, round(m, 4)) for t, m in self.runners_up],
                "n_taxa": self.n_taxa, "gap": self.gap}


@dataclass
class Decision:
    """A VIEW over the evidence at a chosen operating point (decision #1).  Cheap, pure, re-derivable."""
    call: str                       # human-readable placement at the operating point
    resolved_to: str | None         # deepest rank the evidence supports at this threshold
    novel_at: str | None            # first rank the evidence could NOT support (== honest novelty)
    operating_point: dict           # the dial setting that produced this view

    def as_dict(self):
        return {"call": self.call, "resolved_to": self.resolved_to,
                "novel_at": self.novel_at, "operating_point": self.operating_point}


@dataclass
class Evidence:
    """Everything the model knows about a query.  Computed ONCE; verdicts are views over it."""
    ranks: list                     # FULL ladder, always len(RANKS) (decision #2)
    neighborhood: list              # top-K Neighbor, nearest first (decision #3)
    support: float                  # novelty axis (decision #4)
    n_reads: int
    scorer_version: str = SCORER_VERSION

    def novelty(self) -> dict:
        """The explicit novelty axis (decision #4).  Regime read straight off support + neighborhood."""
        if not self.neighborhood:
            flag = "no-neighbor"        # nothing near this -> maximally novel (decision #8)
        elif self.support < 0.5:
            flag = "novel-tail"         # neighbors exist but distant -> the tool's target regime
        else:
            flag = "known"
        return {"support": round(self.support, 4), "flag": flag}

    def decide(self, min_confidence: float = 0.5, require_nesting: bool = False) -> Decision:
        """Threshold the evidence at an operating point.  The dial (decision #1).

        min_confidence : accept a rank only while calibrated confidence >= this.  Higher -> precision;
                         lower -> recall.  Sweeping it traces our whole P-R curve from ONE evidence object.
        require_nesting: False (default) = independent per-rank calls (the +9pt winner, decision #5);
                         True = the classic nesting 'path' view — stop at the first rank that fails OR
                         breaks lineage with the accepted parent.
        """
        resolved = None
        novel = None
        parent_taxa = {}                # rank -> accepted taxon, for the nesting view
        for r in self.ranks:
            if r.gap:
                continue                # taxonomy gap: skip, never a boundary (decision: gap != novelty)
            supported = r.confidence >= min_confidence
            if require_nesting and supported and resolved is not None:
                # nesting view: the accepted top must be consistent with what we accepted above.
                # (independent scoring can disagree across ranks; the path view refuses the break.)
                supported = _nests(r, parent_taxa, self.neighborhood)
            if supported:
                resolved = r.rank
                parent_taxa[r.rank] = r.top
            else:
                novel = r.rank
                break
        if resolved is None:
            call = "unplaceable — maximally novel" if not self.neighborhood else \
                   f"below-threshold at {novel or RANKS[0]}"
        else:
            top = next(x.top for x in self.ranks if x.rank == resolved)
            nov = f", novel at {novel}" if novel else ""
            call = f"{top} ({resolved}){nov}"
        return Decision(call, resolved, novel,
                        {"min_confidence": min_confidence, "require_nesting": require_nesting})

    def as_dict(self, min_confidence: float = 0.5):
        """Full serializable payload — the additive superset the API returns (docs' §7).
        Top-level default-view aliases preserve backward compat with the old {call,resolved_to,...}."""
        d = self.decide(min_confidence)
        return {
            "scorer_version": self.scorer_version,
            "n_reads": self.n_reads,
            "ranks": [r.as_dict() for r in self.ranks],
            "neighborhood": [n.as_dict() for n in self.neighborhood],
            "novelty": self.novelty(),
            "decision": d.as_dict(),
            # backward-compatible default-view aliases (don't break existing benchmark harnesses):
            "call": d.call, "resolved_to": d.resolved_to, "novel_at": d.novel_at,
            "confidence": next((r.confidence for r in self.ranks if r.rank == d.resolved_to), 0.0),
            "support": round(self.support, 4),
        }


# ── the frozen scorer (v1) ──────────────────────────────────────────────────────────────────────
def _score_one_rank(rank, neighbors, taxonomy_of, gamma, support, calib) -> RankReadout:
    """One rank's sharpened, confidence-weighted vote over `neighbors` (the frozen-v1 aggregation kernel).
    Factored out so a conditioned rank (decision #5b) can re-vote over a neighbor SUBSET with identical math."""
    w = defaultdict(float)
    for gid, s in neighbors:
        t = taxonomy_of.get(gid, {}).get(rank)
        if t:
            w[t] += max(s, 0.0) ** gamma              # sharpened, confidence-weighted, NO dedup (decision #6)
    if not w:
        return RankReadout(rank, None, 0.0, 0.0, support, [], 0, gap=True)
    total = sum(w.values())
    ordered = sorted(w.items(), key=lambda kv: kv[1], reverse=True)
    top, top_w = ordered[0]
    margin = top_w / total if total > 0 else 0.0
    runners = [(t, wv / total) for t, wv in ordered[1:1 + RUNNERS_UP]]
    conf = _calibrated_confidence(margin, support, calib.get(rank) if calib else None)
    return RankReadout(rank, top, conf, margin, support, runners, len(w), gap=False)


def score_ranks(neighbors, taxonomy_of, gamma: float = GAMMA, calib: dict | None = None,
                condition_deep_on_family: bool = False, cond_thresh: float = 0.6) -> list:
    """Per-rank INDEPENDENT sharpened vote over the pooled neighbor set.  Computes EVERY rank; no break.

    neighbors  : list[(gid, similarity)] pooled across the read-ensemble (sim = retrieval confidence).
    taxonomy_of: gid -> {rank: taxon}.
    Returns a full ladder of RankReadout (len == len(RANKS)), gaps marked, never truncated (decision #2).

    condition_deep_on_family (decision #5b, scorer_v2, default OFF = frozen v1): when family is CONFIDENT
    (margin >= cond_thresh), re-vote genus & species over ONLY the neighbors in that family — removing
    cross-family leaf contamination ("right family, wrong leaf").  Family itself is NEVER touched, so this
    can only affect the deep ranks (the 'family no-regress' guarantee is structural, not empirical).
    """
    support = _support(neighbors)
    out = [_score_one_rank(r, neighbors, taxonomy_of, gamma, support, calib) for r in RANKS]
    if condition_deep_on_family:
        fam = out[DEPTH["family"]]
        if not fam.gap and fam.margin >= cond_thresh:
            sub = [(g, s) for g, s in neighbors if taxonomy_of.get(g, {}).get("family") == fam.top]
            if sub:
                for rank in ("genus", "species"):     # support stays the FULL-neighborhood value (novelty proxy)
                    out[DEPTH[rank]] = _score_one_rank(rank, sub, taxonomy_of, gamma, support, calib)
    return out


def build_evidence(neighbors, taxonomy_of, n_reads: int = 1, k_neighborhood: int = 20,
                   gamma: float = GAMMA, calib: dict | None = None,
                   accession_of=None, condition_deep_on_family: bool = False) -> Evidence:
    """Assemble the full Evidence from a retrieved neighborhood.  The one pure call (L3 entry point).

    Degenerate inputs (empty neighbors) return a valid all-gap Evidence with support=0 — never raise
    (decision #8).  The rerank seam (decision #6b) lives OUTSIDE this function: search -> [rerank] -> here.
    condition_deep_on_family (default OFF) forwards to the scorer_v2 genus/species conditioning (decision #5b).
    """
    neighbors = list(neighbors)
    ranks = score_ranks(neighbors, taxonomy_of, gamma=gamma, calib=calib,
                        condition_deep_on_family=condition_deep_on_family)
    support = _support(neighbors)
    hood = _neighborhood(neighbors, taxonomy_of, k_neighborhood, accession_of)
    return Evidence(ranks=ranks, neighborhood=hood, support=support, n_reads=n_reads)


# ── the rerank seam (decision #6b) — identity today, a precision lever later ────────────────────
def rerank(neighbors, query_vecs=None):
    """search -> [rerank] -> score.  Identity passthrough now; the named slot where a finer second-pass
    (more reads vs the top-K, an alignment, a different metric) drops in without restructuring the flow.
    Kept a no-op until coverage lands and the residual 'right neighborhood, wrong leaf' gap is measured."""
    return neighbors


# ── cross-tier fusion (adaptive-resolution serving) ─────────────────────────────────────────────
def rank_normalize(neighbors):
    """Replace each neighbor's similarity with its rank-percentile [0,1] WITHIN this set.  Cosines from
    different index tiers are NOT comparable (richer references run higher) — pooling raw sims lets one tier
    dominate the vote regardless of correctness (the measured #1 fusion trap).  Rank-normalizing first makes
    them comparable.  Pure-python; order-preserving; empty-safe."""
    n = len(neighbors)
    if n == 0: return []
    order = sorted(range(n), key=lambda i: neighbors[i][1])       # ascending by similarity
    rank = [0.0] * n
    for r, i in enumerate(order): rank[i] = r / max(n - 1, 1)
    return [(neighbors[i][0], rank[i]) for i in range(n)]


def fuse(*neighbor_sets):
    """Cross-tier fusion for the scale-crossover band: rank-normalize EACH tier's neighbors, then pool.
    A single (or only-nonempty) set is returned raw — no normalization needed when there's nothing to compare."""
    sets = [ns for ns in neighbor_sets if ns]
    if len(sets) <= 1: return sets[0] if sets else []
    out = []
    for ns in sets: out += rank_normalize(ns)
    return out


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────
def _support(neighbors) -> float:
    """Neighborhood density = mean similarity (the regime/novelty proxy, decision #4).  0 if empty."""
    if not neighbors:
        return 0.0
    return sum(s for _, s in neighbors) / len(neighbors)


def _neighborhood(neighbors, taxonomy_of, k, accession_of):
    """Top-K DISTINCT genomes by best similarity — DISPLAY dedup only.  Orthogonal to the voting dedup
    bug: voting keeps every read's vote (decision #6), the map shows each landmark once."""
    best = {}
    for gid, s in neighbors:
        if gid not in best or s > best[gid]:
            best[gid] = s
    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
    acc = accession_of or {}
    return [Neighbor(gid, s, taxonomy_of.get(gid, {}), acc.get(gid)) for gid, s in ordered]


def _calibrated_confidence(margin, support, rank_calib) -> float:
    """(margin, support) -> P(correct) via a binned lookup (decision #4).  Falls back to raw margin when
    no calibration is supplied.  A 1D (margin-only) table is the single-support-bin special case, so the
    same code path serves today's 1D calibration and tomorrow's 2D one."""
    if not rank_calib:
        return margin
    m_edges = rank_calib["margin_edges"]
    s_edges = rank_calib.get("support_edges", [float("-inf"), float("inf")])   # 1D default: one support bin
    p = rank_calib["p"]                     # shape [len(s_edges)-1][len(m_edges)-1]
    si = _bin(support, s_edges)
    mi = _bin(margin, m_edges)
    return float(p[si][mi])


def _bin(x, edges) -> int:
    """Index i with edges[i] <= x < edges[i+1], clamped to [0, nbins-1].  Values below edges[0] -> 0."""
    for i in range(len(edges) - 1):
        if x < edges[i + 1]:
            return i
    return len(edges) - 2


def _nests(readout, parent_taxa, neighborhood) -> bool:
    """Does this rank's top taxon sit UNDER the accepted parent? (path view only, decision #5).
    Checks the reference: is there any neighbor whose lineage has both the accepted parent taxa AND this
    rank's top?  If the reference never co-occurs them, the ranks disagree -> not nested."""
    for n in neighborhood:
        lin = n.lineage
        if lin.get(readout.rank) == readout.top and all(lin.get(r) == t for r, t in parent_taxa.items()):
            return True
    return False
