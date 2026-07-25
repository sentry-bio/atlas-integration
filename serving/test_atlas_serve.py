#!/usr/bin/env python3
"""
test_atlas_serve.py — L4 orchestration tests with FAKE L1/L2 (no torch, faiss, GPU, or fastapi).

Proves the wiring (reads -> tokens -> vecs -> neighbors -> evidence -> decision at an operating point),
the degenerate path (#8), the meta-schema adapter, and the ENCODER-MATCH gate logic.  The GPU/index-bound
gates (verify_gate, load_service) are exercised on Nexus, not here.

Run:  python test_atlas_serve.py
"""
from __future__ import annotations
from atlas_serve import AtlasService, AdaptiveService, Router, assert_encoder_matches, meta_bp_ntok
from atlas_encoder import ReadPolicy
from atlas_evidence import RANKS

RANK_I = {r: i for i, r in enumerate(RANKS)}      # rank name -> its slot in the full ladder

TAX = {
    1: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "genus": "Lactobacillus", "species": "L. acidophilus", "accession": "GCA_1"},
    2: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "genus": "Lactobacillus", "species": "L. gasseri", "accession": "GCA_2"},
    3: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Lactobacillales",
        "family": "Lactobacillaceae", "genus": "Pediococcus", "species": "P. acidilactici", "accession": "GCA_3"},
    4: {"domain": "Bacteria", "phylum": "Firmicutes", "class": "Bacilli", "order": "Bacillales",
        "family": "Bacillaceae", "genus": "Bacillus", "species": "B. subtilis", "accession": "GCA_4"},
}


class FakeTokenizer:
    def tokenize(self, dna, ntok):
        return [5] * ntok                       # content irrelevant; the fake index ignores vectors


class FakeEncoder:
    encoder_id = "sha256:fake-encoder"
    def encode(self, rows):
        return list(rows)                        # stand-in "vectors"; one per read


class FakeIndex:
    """Returns a canned neighborhood, ignoring the query vectors (the point is to test WIRING)."""
    def __init__(self, neighbors):
        self._n = neighbors
    def all_gids(self):
        return list({g for g, _ in self._n})
    def search(self, qvecs, k, exclude=None):
        ex = set() if exclude is None else set(exclude if hasattr(exclude, "__iter__") else [exclude])
        # one copy of the canned set per read (mimics pooling across the ensemble)
        out = []
        for _ in range(max(1, len(qvecs))):
            out += [(g, s) for g, s in self._n if g not in ex][:k]
        return out


def _passed(name): print(f"  ok  {name}")


def _service(neighbors):
    return AtlasService(FakeTokenizer(), FakeEncoder(), FakeIndex(neighbors), TAX,
                        ReadPolicy(index_bp=2000, index_ntok=400), k=15)


def test_full_wire_place():
    """reads -> tokens -> vecs -> neighbors -> evidence -> decision.  Unanimous-family clade resolves."""
    svc = _service([(1, 0.9), (2, 0.9), (3, 0.9)])   # family Lactobacillaceae unanimous, genus splits
    out = svc.place(reads=["ACGT" * 600], read_bp=2000, min_confidence=0.9)
    assert out["scorer_version"] == "v1"
    assert len(out["ranks"]) == 7                    # full ladder (decision #2) surfaces through L4
    assert out["resolved_to"] == "family" and out["novel_at"] == "genus", out["decision"]
    assert out["neighborhood"][0]["lineage"]["family"] == "Lactobacillaceae"   # neighborhood #3
    _passed("full place() wire: reads -> evidence -> decision, family resolved / novel_at genus")


def test_operating_point_dial_through_service():
    """Decision #1: the min_confidence knob traverses the P-R curve at the API boundary."""
    svc = _service([(1, 0.9), (2, 0.9), (3, 0.9)])
    strict = svc.place(reads=["ACGT" * 600], read_bp=2000, min_confidence=0.9)
    loose = svc.place(reads=["ACGT" * 600], read_bp=2000, min_confidence=0.3)
    from atlas_evidence import DEPTH
    assert DEPTH[loose["resolved_to"]] > DEPTH[strict["resolved_to"]], (loose["resolved_to"], strict["resolved_to"])
    _passed("operating-point dial works through the service (looser -> deeper call)")


def test_no_reads_is_evidence_not_error():
    """Decision #8: too-short / empty input returns maximally-novel evidence, not an exception/400."""
    svc = _service([(1, 0.9)])
    out = svc.place(reads=["ACGT"], read_bp=2000)    # 4bp << bp//2 -> dropped -> no frags
    assert out["novelty"]["flag"] == "no-neighbor"
    assert out["resolved_to"] is None and "maximally novel" in out["call"]
    _passed("no usable reads -> maximally-novel evidence, no exception")


def test_encoder_match_gate():
    """ENCODER-MATCH: equal ids pass; mismatch raises; missing id warns (lenient) / raises (strict)."""
    enc = FakeEncoder()
    assert assert_encoder_matches(enc, {"encoder_id": "sha256:fake-encoder"}, log=lambda *_: None) is True
    try:
        assert_encoder_matches(enc, {"encoder_id": "sha256:OTHER"}, log=lambda *_: None); assert False
    except RuntimeError as e:
        assert "MISMATCH" in str(e)
    # legacy (no encoder_id): lenient warns and returns False; strict raises
    assert assert_encoder_matches(enc, {}, strict=False, log=lambda *_: None) is False
    try:
        assert_encoder_matches(enc, {}, strict=True, log=lambda *_: None); assert False
    except RuntimeError as e:
        assert "unstamped" in str(e)
    _passed("encoder-match gate: pass / mismatch-raise / legacy-warn / strict-raise")


def test_meta_schema_adapter():
    """L4 reads BOTH meta schemas: interim (bp/ntok) and full-tree build (scale)."""
    assert meta_bp_ntok({"bp": 2000, "ntok": 400}) == (2000, 400)
    assert meta_bp_ntok({"scale": "20kb"}) == (20000, 4096)
    assert meta_bp_ntok({"scale": "300bp"}) == (300, 60)
    try:
        meta_bp_ntok({"encoder": "v10.9"}); assert False
    except ValueError:
        pass
    _passed("meta adapter handles bp/ntok schema, scale schema, and rejects unknown")


# ── adaptive-resolution service (Router + AdaptiveService) ──────────────────────────────────────
def _adaptive(n5, n20, bands=None):
    """Two-tier fake: 5kb tier returns n5, 20kb tier returns n20 (each a canned neighborhood)."""
    tiers = {
        "5kb":  {"index": FakeIndex(n5),  "ntok": 1000, "bp": 5000,  "meta": {"scale": "5kb"}},
        "20kb": {"index": FakeIndex(n20), "ntok": 4096, "bp": 20000, "meta": {"scale": "20kb"}},
    }
    router = Router(bands) if bands else Router.default(tiers.keys())
    return AdaptiveService(FakeTokenizer(), FakeEncoder(), tiers, TAX, router,
                           ReadPolicy(index_bp=20000, index_ntok=4096), k=15)


LONG_READ = "ACGT" * 6000    # 24kb — survives truncation to any tier bp in reads_from


def test_router_bands():
    """Length -> single best tier: <3kb->5kb, >=3kb->20kb (fusion is a measured net-negative, no fused band)."""
    r = Router.default(["5kb", "20kb"])
    assert r.route(1000) == ["5kb"] and r.route(2999) == ["5kb"]      # short reads -> 5kb tier
    assert r.route(3000) == ["20kb"] and r.route(5000) == ["20kb"] and r.route(20000) == ["20kb"]
    assert Router.default(["20kb"]).route(300) == ["20kb"]           # degenerate single-tier
    _passed("router: <3kb->5kb / >=3kb->20kb (no fused band) / single-tier passthrough")


def test_adaptive_short_routes_5kb_only():
    """A short read hits the 5kb tier alone — 20kb is not consulted, no cross-scale block."""
    svc = _adaptive(n5=[(1, 0.9), (2, 0.9), (3, 0.9)], n20=[(4, 0.9)])   # tiers would DISAGREE if both ran
    out = svc.place(reads=[LONG_READ], read_bp=1000, min_confidence=0.5)
    assert out["scale"]["tier_used"] == "5kb", out["scale"]
    assert "cross_scale" not in out                                  # single tier -> no diagnostic
    assert out["ranks"][RANK_I["family"]]["top"] == "Lactobacillaceae"   # answered by 5kb only
    _passed("short read routes to 5kb-only (20kb untouched, no cross_scale)")


def test_adaptive_long_routes_20kb_only():
    """A native-length read hits the 20kb tier alone."""
    svc = _adaptive(n5=[(1, 0.9)], n20=[(4, 0.9)])
    out = svc.place(reads=[LONG_READ], read_bp=20000, min_confidence=0.5)
    assert out["scale"]["tier_used"] == "20kb", out["scale"]
    assert "cross_scale" not in out
    assert out["ranks"][RANK_I["family"]]["top"] == "Bacillaceae"    # answered by 20kb only
    _passed("native read routes to 20kb-only")


def test_fusion_mechanism_fuses_and_emits_diagnostic():
    """The fuse() machinery (now eval-only via force_tiers) pools BOTH tiers and surfaces cross_scale.agreed."""
    # both tiers vote the SAME family -> agreed True
    svc = _adaptive(n5=[(1, 0.9), (2, 0.8)], n20=[(2, 0.7), (3, 0.6)])
    out = svc.place(reads=[LONG_READ], read_bp=20000, force_tiers=["5kb", "20kb"], min_confidence=0.5)
    assert out["scale"]["tier_used"] == "5kb+20kb", out["scale"]
    assert out["cross_scale"]["agreed"] is True, out["cross_scale"]
    assert set(out["cross_scale"]["per_tier_family"].values()) == {"Lactobacillaceae"}
    _passed("forced fusion pools both tiers + emits cross_scale.agreed=True on agreement")


def test_fusion_mechanism_disagreement_flagged():
    """When the forced tiers disagree on family, the diagnostic reports it (agreed=False) — surfaced, not hidden."""
    svc = _adaptive(n5=[(1, 0.9), (2, 0.9)], n20=[(4, 0.9)])          # Lactobacillaceae vs Bacillaceae
    out = svc.place(reads=[LONG_READ], read_bp=20000, force_tiers=["5kb", "20kb"], min_confidence=0.5)
    assert out["scale"]["tier_used"] == "5kb+20kb"
    assert out["cross_scale"]["agreed"] is False, out["cross_scale"]
    assert set(out["cross_scale"]["per_tier_family"].values()) == {"Lactobacillaceae", "Bacillaceae"}
    _passed("forced-fusion disagreement -> cross_scale.agreed=False (diagnostic surfaced)")


def test_adaptive_force_tiers_pins_arm():
    """The eval-only force_tiers knob overrides the router (needed for the routed-vs-single-arm benchmark)."""
    svc = _adaptive(n5=[(1, 0.9), (2, 0.9)], n20=[(4, 0.9)])
    # a native read would normally route 20kb-only; force it to 5kb instead
    out = svc.place(reads=[LONG_READ], read_bp=20000, force_tiers=["5kb"], min_confidence=0.5)
    assert out["scale"]["tier_used"] == "5kb", out["scale"]
    assert out["ranks"][RANK_I["family"]]["top"] == "Lactobacillaceae"   # answered by the pinned 5kb tier
    # unknown tier names are ignored; falling back to the valid one
    out2 = svc.place(reads=[LONG_READ], read_bp=1000, force_tiers=["20kb", "bogus"], min_confidence=0.5)
    assert out2["scale"]["tier_used"] == "20kb"
    _passed("force_tiers pins the arm (overrides router; filters unknown names)")


def test_adaptive_no_reads_is_evidence_not_error():
    """Decision #8 holds through the adaptive path too: empty -> maximally-novel, not an exception."""
    svc = _adaptive(n5=[(1, 0.9)], n20=[(1, 0.9)])
    out = svc.place(reads=["ACGT"], read_bp=5000)                    # 4bp dropped -> no frags
    assert out["novelty"]["flag"] == "no-neighbor" and out["resolved_to"] is None
    _passed("adaptive no-usable-reads -> maximally-novel evidence, no exception")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"atlas_serve L4 orchestration tests — {len(tests)} cases (GPU/index gates run on Nexus)")
    for t in tests:
        t()
    print(f"\nALL {len(tests)} PASSED — L4 wiring + gates + meta-adapter contract holds.")


if __name__ == "__main__":
    main()
