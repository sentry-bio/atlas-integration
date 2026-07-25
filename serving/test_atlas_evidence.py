#!/usr/bin/env python3
"""
test_atlas_evidence.py — golden tests that DEFINE scorer_v1 and lock the refactor's invariants.

Pure: synthetic neighbors, no GPU / index / torch / faiss.  These are the contract — any future
scorer_v2 must keep the invariants (ladder length, evidence/decision split, gap!=novelty, degenerate
states) and beat v1 on the frozen accuracy grid, or it is not shipped.

Run:  python test_atlas_evidence.py     (plain asserts; also pytest-discoverable)
"""
from __future__ import annotations
import atlas_evidence as ae
from atlas_evidence import build_evidence, score_ranks, RANKS

# ── a tiny synthetic reference: gid -> lineage ─────────────────────────────────────────────────
# Two clean lineages under one domain, plus a distractor family, plus a gid with a mid-rank gap.
TAX = {
    1: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "genus": "Lactobacillus", "species": "L. acidophilus"},
    2: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "genus": "Lactobacillus", "species": "L. gasseri"},
    3: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "genus": "Pediococcus", "species": "P. acidilactici"},
    4: {"domain": "Bacteria", "phylum": "Proteobacteria", "class": "Gamma", "order": "Enterobacterales",
        "family": "Enterobacteriaceae", "genus": "Escherichia", "species": "E. coli"},
    # gid 5: a genome whose reference taxonomy is missing 'genus' (GTDB g__ gap) — must be SKIPPED, not a boundary
    5: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "species": "L. mystery"},
}


def _passed(name):
    print(f"  ok  {name}")


def test_full_ladder_never_truncated():
    """Decision #2: score_ranks always returns a readout for EVERY rank, even below threshold."""
    ev = build_evidence([(1, 0.9), (2, 0.85)], TAX)
    assert len(ev.ranks) == len(RANKS)
    assert [r.rank for r in ev.ranks] == RANKS
    _passed("full ladder is always len(RANKS), in order")


def test_unanimous_clade_high_confidence():
    """Strong agreement -> high margin deep, known regime, resolves to species."""
    ev = build_evidence([(1, 0.95), (2, 0.93)], TAX, n_reads=2)  # both Lactobacillus
    d = ev.decide(min_confidence=0.5)
    assert d.resolved_to == "species", d       # two close L. species split ~50/50 but still clear the 0.5 gate
    assert d.novel_at is None
    assert ev.novelty()["flag"] == "known"
    # unanimous THROUGH genus (both Lactobacillus); species splits (two different L. species) -> margin < 1
    assert all(r.margin == 1.0 for r in ev.ranks if not r.gap and r.rank != "species")
    assert next(r for r in ev.ranks if r.rank == "species").margin < 1.0
    _passed("unanimous clade -> resolves to species, unanimous through genus, known regime")


def test_split_at_genus_is_honest():
    """Family agrees, genus splits -> honest 'family, novel at genus' (decision #5)."""
    # 2 Lactobacillus + 1 Pediococcus: family Lactobacillaceae unanimous, genus splits ~2:1 sharpened.
    ev = build_evidence([(1, 0.9), (2, 0.9), (3, 0.9)], TAX)
    fam = next(r for r in ev.ranks if r.rank == "family")
    gen = next(r for r in ev.ranks if r.rank == "genus")
    assert fam.margin == 1.0
    assert gen.margin < 1.0 and gen.top == "Lactobacillus"  # sharpened vote still leads, but not unanimous
    # at a strict operating point the genus is NOT supported -> novelty surfaces there
    d = ev.decide(min_confidence=0.9)
    assert d.resolved_to == "family" and d.novel_at == "genus", d
    _passed("split genus -> family resolved, novel_at=genus at strict threshold")


def test_taxonomy_gap_is_skipped_not_a_boundary():
    """Decision: a missing rank in the reference (gid 5 has no genus) is a GAP, not novelty.
    Descent must pass THROUGH it to species, not stop at it."""
    ev = build_evidence([(5, 0.95)], TAX)     # only the gap genome
    gen = next(r for r in ev.ranks if r.rank == "genus")
    assert gen.gap is True and gen.top is None
    d = ev.decide(min_confidence=0.5)
    # genus is a gap (skipped), species is present -> resolves to species despite the genus hole
    assert d.resolved_to == "species", d
    _passed("taxonomy gap is skipped, descent continues past it")


def test_empty_neighborhood_is_evidence_not_exception():
    """Decision #8: no neighbors -> valid Evidence, support 0, 'no-neighbor', maximally novel. No raise."""
    ev = build_evidence([], TAX)
    assert len(ev.ranks) == len(RANKS)
    assert all(r.gap for r in ev.ranks)
    assert ev.support == 0.0
    assert ev.novelty()["flag"] == "no-neighbor"
    d = ev.decide()
    assert d.resolved_to is None and "maximally novel" in d.call
    _passed("empty neighborhood -> maximally novel evidence, no exception")


def test_one_evidence_two_operating_points():
    """Decision #1: the SAME evidence object re-decides at different thresholds with zero recompute."""
    ev = build_evidence([(1, 0.9), (2, 0.9), (3, 0.9)], TAX)   # family unanimous, genus split
    loose = ev.decide(min_confidence=0.3)   # recall-leaning
    strict = ev.decide(min_confidence=0.9)  # precision-leaning
    assert DEPTH(loose.resolved_to) > DEPTH(strict.resolved_to), (loose, strict)
    # crucially: same evidence, two verdicts — the ladder was not recomputed
    assert loose.operating_point != strict.operating_point
    _passed("one evidence, two operating points -> deeper call when looser (the dial)")


def test_neighborhood_is_first_class_and_display_deduped():
    """Decision #3: nearest known genomes returned; #6: display dedup (best sim per gid) but voting keeps dups."""
    # gid 1 appears in three reads; neighborhood shows it ONCE at its best sim.
    neighbors = [(1, 0.7), (1, 0.95), (1, 0.8), (2, 0.6)]
    ev = build_evidence(neighbors, TAX, k_neighborhood=5)
    gids = [n.gid for n in ev.neighborhood]
    assert gids.count(1) == 1, "neighborhood display-dedups genomes"
    assert ev.neighborhood[0].gid == 1 and abs(ev.neighborhood[0].similarity - 0.95) < 1e-9
    assert ev.neighborhood[0].lineage["genus"] == "Lactobacillus"  # lineage denormalized onto the landmark
    # but the VOTE saw all three copies of gid 1 (support reflects the pooled set, not the deduped one)
    assert abs(ev.support - (0.7 + 0.95 + 0.8 + 0.6) / 4) < 1e-9
    _passed("neighborhood first-class + display-deduped; voting keeps every read")


def test_nesting_view_refuses_incoherent_path():
    """Decision #5: require_nesting=True gives the classic path view — it stops where independent ranks
    disagree, rather than reporting a family and a genus that don't nest under it."""
    # The real ±9pt mechanism: Lactobacillaceae mass is SPLIT across two genera (Lactobacillus,
    # Pediococcus) so it wins FAMILY on summed mass, but a CONCENTRATED rival (Escherichia) wins GENUS.
    # sim**3 weights: gid1,gid3 -> 0.4 each (Lacto family = 0.8); gid4 -> 0.7 (Entero family = 0.7).
    # genus: Lactobacillus 0.4, Pediococcus 0.4, Escherichia 0.7 -> Escherichia wins. Incoherent by design.
    neighbors = [(1, 0.4 ** (1 / 3)), (3, 0.4 ** (1 / 3)), (4, 0.7 ** (1 / 3))]
    ev = build_evidence(neighbors, TAX)
    fam = next(r for r in ev.ranks if r.rank == "family")
    gen = next(r for r in ev.ranks if r.rank == "genus")
    assert fam.top == "Lactobacillaceae", fam.top     # split family wins on summed mass
    assert gen.top == "Escherichia", gen.top          # concentrated rival wins genus -> INCOHERENT
    d_indep = ev.decide(min_confidence=0.0, require_nesting=False)
    d_path = ev.decide(min_confidence=0.0, require_nesting=True)
    # independent view descends past the incoherence (surfacing both, the +9pt behavior);
    # path view REFUSES the non-nesting genus and stops at family.
    assert d_indep.resolved_to == "species"
    assert d_path.resolved_to == "family", d_path
    _passed("nesting view refuses incoherent family/genus; independent view surfaces both")


def test_calibration_1d_and_2d_lookup():
    """Decision #4: (margin, support) binned lookup; 1D table = single-support-bin special case."""
    # 1D: margin-only table, one support bin spanning all support.
    calib_1d = {"genus": {"margin_edges": [0.0, 0.5, 1.01], "p": [[0.2, 0.8]]}}
    ev = build_evidence([(1, 0.9), (2, 0.9), (3, 0.9)], TAX, calib=calib_1d)
    gen = next(r for r in ev.ranks if r.rank == "genus")
    # genus margin (Lactobacillus 2:1 sharpened) is > 0.5 -> high-margin bin -> p 0.8
    assert abs(gen.confidence - 0.8) < 1e-9, gen.confidence
    # 2D: same margin, but LOW support routes to a discounted p (novel-tail penalty).
    calib_2d = {"genus": {"margin_edges": [0.0, 0.5, 1.01],
                          "support_edges": [0.0, 0.5, 1.01],
                          "p": [[0.1, 0.4],    # low support row: discounted
                                [0.2, 0.8]]}}  # high support row
    # low-support neighbors (all sim 0.3) -> support 0.3 -> low row; high margin col.
    ev_lo = build_evidence([(1, 0.3), (2, 0.3), (3, 0.3)], TAX, calib=calib_2d)
    gen_lo = next(r for r in ev_lo.ranks if r.rank == "genus")
    assert abs(gen_lo.confidence - 0.4) < 1e-9, gen_lo.confidence
    _passed("calibration: 1D margin lookup + 2D (margin,support) discount for low support")


# tiny helper so the depth-comparison reads cleanly in tests
def DEPTH(rank):
    return ae.DEPTH.get(rank, -1)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"atlas_evidence golden tests (scorer {ae.SCORER_VERSION}) — {len(tests)} cases")
    for t in tests:
        t()
    print(f"\nALL {len(tests)} PASSED — scorer_{ae.SCORER_VERSION} contract holds.")


if __name__ == "__main__":
    main()


def test_conditioning_off_is_frozen_v1():
    """scorer_v2 default OFF: build_evidence must be byte-identical to frozen v1 (no behavior drift)."""
    nb = [(1, 0.9), (2, 0.85), (3, 0.8), (4, 0.7)]
    a = build_evidence(nb, TAX)
    b = build_evidence(nb, TAX, condition_deep_on_family=False)
    assert [r.as_dict() for r in a.ranks] == [r.as_dict() for r in b.ranks]
    _passed("condition_deep_on_family=False == frozen v1 (no drift)")


def test_conditioning_prunes_cross_family_genus():
    """scorer_v2 ON: with a confident family, a genus vote leaking from ANOTHER family is removed,
    while the family readout itself is untouched (structural no-regress)."""
    # 3 Lactobacillaceae (2 Lactobacillus, 1 Pediococcus) + a strong Enterobacteriaceae/Escherichia distractor.
    # v1 genus vote sees Escherichia competing; conditioning on the confident family drops it.
    nb = [(1, 0.95), (2, 0.95), (3, 0.9), (4, 0.92)]
    v1 = build_evidence(nb, TAX, condition_deep_on_family=False)
    v2 = build_evidence(nb, TAX, condition_deep_on_family=True, )
    fam_v1 = next(r for r in v1.ranks if r.rank == "family")
    fam_v2 = next(r for r in v2.ranks if r.rank == "family")
    assert fam_v1.as_dict() == fam_v2.as_dict()                  # family NEVER touched (structural guarantee)
    gen_v2 = next(r for r in v2.ranks if r.rank == "genus")
    # after conditioning on Lactobacillaceae, Escherichia can no longer be the genus top
    assert gen_v2.top in ("Lactobacillus", "Pediococcus"), gen_v2.top
    # and the surviving genus taxa are all within the confident family
    assert all(t in ("Lactobacillus", "Pediococcus") for t, _ in gen_v2.runners_up)
    _passed("conditioning prunes cross-family genus leakage; family untouched")


def test_conditioning_noop_when_family_uncertain():
    """If family is NOT confident (margin < cond_thresh), conditioning must not fire -> == v1."""
    # two families ~50/50 at family rank -> low family margin -> no conditioning
    nb = [(1, 0.9), (4, 0.9)]                                    # Lactobacillaceae vs Enterobacteriaceae, tied
    v1 = build_evidence(nb, TAX, condition_deep_on_family=False)
    v2 = build_evidence(nb, TAX, condition_deep_on_family=True)
    assert [r.as_dict() for r in v1.ranks] == [r.as_dict() for r in v2.ranks]
    _passed("conditioning is a no-op when family is uncertain (== v1)")
