"""Strategy arms: BM25 search, filters, dense determinism, RRF fusion."""

from pathlib import Path

import pytest
from conftest import material_row, write_materials

from bench.strategies import (
    Bm25Strategy,
    RrfStrategy,
    build_strategy,
    config_hash,
    strategy_config,
)

ROWS = [
    material_row(101, "Ocean Waves at Sunset", tags=("ocean", "waves", "sunset")),
    material_row(102, "Ocean Waves Slow Motion", tags=("ocean", "waves", "slow motion")),
    material_row(103, "City Night Skyline Timelapse", tags=("city", "night", "timelapse")),
    material_row(104, "Forest Morning Fog", tags=("forest", "fog")),
    material_row(
        105,
        "Portrait Interview Talking",
        tags=("interview",),
        orientation="portrait",
        duration=12,
        width=2160,
    ),
    material_row(
        106,
        "Ocean Aerial Coastline",
        tags=("ocean", "aerial"),
        duration=90,
    ),
]


def raw_texts(materials):
    """The raw-text corpus (title/description/tags) for these fixtures."""
    return {doc_id: materials.rows[doc_id].embed_text for doc_id in materials.ids}


@pytest.fixture()
def materials(bench_root: Path):
    return write_materials(bench_root, "v1", ROWS)


def _brute_force_top(model, materials, query, k, hard=None):
    """Reference scoring straight from the seam, no inverted index."""
    qw = model.query_weights(query)
    scored = {}
    for doc_id in materials.ids:
        if hard and not materials.passes_filters(materials.rows[doc_id], hard):
            continue
        dw = model.doc_weights(materials.rows[doc_id].embed_text)
        scored[doc_id] = sum(w * dw.get(t, 0.0) for t, w in qw.items())
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def test_bm25_search_matches_brute_force(materials):
    strategy = Bm25Strategy("bm25-query", materials, idf_side="query", texts=raw_texts(materials))
    result = strategy.search("ocean waves", k=3, hard={})
    expected = _brute_force_top(strategy.model, materials, "ocean waves", 3)
    assert [doc_id for doc_id, _ in result.hits] == [doc_id for doc_id, _ in expected]


def test_bm25_retrieves_the_matching_doc_first(materials):
    strategy = Bm25Strategy("bm25-query", materials, idf_side="query", texts=raw_texts(materials))
    result = strategy.search("city night skyline", k=2, hard={})
    assert result.hits[0][0] == 103


def test_bm25_both_vs_query_is_a_real_ab_arm(materials):
    both = Bm25Strategy("bm25-both", materials, idf_side="both", texts=raw_texts(materials))
    fixed = Bm25Strategy("bm25-query", materials, idf_side="query", texts=raw_texts(materials))
    query = "ocean slow motion waves"
    both_ranking = [doc_id for doc_id, _ in both.search(query, k=4, hard={}).hits]
    fixed_ranking = [doc_id for doc_id, _ in fixed.search(query, k=4, hard={}).hits]
    # both arms retrieve, but the idf^2 weighting is a different ranking function
    assert both_ranking and fixed_ranking


def test_bm25_respects_hard_filters(materials):
    strategy = Bm25Strategy("bm25-query", materials, idf_side="query", texts=raw_texts(materials))
    portrait = strategy.search("ocean waves", k=6, hard={"orientation": "portrait"})
    assert [doc_id for doc_id, _ in portrait.hits] == []  # nothing portrait + ocean

    long_ = strategy.search("ocean", k=6, hard={"duration_min": 60})
    assert [doc_id for doc_id, _ in long_.hits] == [106]

    medium = strategy.search("ocean", k=6, hard={"duration_max": 60})
    assert {doc_id for doc_id, _ in medium.hits} == {101, 102}

    uhd = strategy.search("interview", k=6, hard={"min_width": 3840})
    assert [doc_id for doc_id, _ in uhd.hits] == []


def test_bm25_cjk_query_hits_nothing(materials):
    strategy = Bm25Strategy("bm25-query", materials, idf_side="query", texts=raw_texts(materials))
    result = strategy.search("慢动作海浪", k=5, hard={})
    assert result.hits == []


def test_bm25_tie_break_by_doc_id(tmp_path):
    twin_rows = [
        material_row(201, "Identical Content", tags=("same", "words")),
        material_row(200, "Identical Content", tags=("same", "words")),
    ]
    twin_materials = write_materials(tmp_path, "twins", twin_rows)
    strategy = Bm25Strategy(
        "bm25-query", twin_materials, idf_side="query", texts=raw_texts(twin_materials)
    )
    result = strategy.search("identical content same words", k=2, hard={})
    assert [doc_id for doc_id, _ in result.hits] == [200, 201]


def test_hash_dense_deterministic_and_filtered(materials):
    from bench.strategies import build_hash_dense

    first = build_hash_dense(materials, raw_texts(materials))
    second = build_hash_dense(materials, raw_texts(materials))
    q1 = first.search("ocean waves", k=3, hard={})
    q2 = second.search("ocean waves", k=3, hard={})
    assert [doc_id for doc_id, _ in q1.hits] == [doc_id for doc_id, _ in q2.hits]
    assert q1.hits[0][0] in (101, 102)

    portrait = first.search("interview talking", k=6, hard={"orientation": "portrait"})
    assert [doc_id for doc_id, _ in portrait.hits] == [105]


class _FixedArm:
    def __init__(self, hits):
        self.spec_id = "fixed"
        self._hits = hits

    def search(self, query_text, k, hard):
        from bench.strategies import SearchResult

        return SearchResult(hits=self._hits[:k], search_ms=0.0, embed_ms=0.0)


def test_rrf_fusion_math():
    arm_a = _FixedArm([(1, 9.0), (2, 8.0)])
    arm_b = _FixedArm([(2, 5.0), (3, 4.0)])
    rrf = RrfStrategy("rrf", [arm_a, arm_b])
    result = rrf.search("anything", k=3, hard={})
    expected = {
        1: 1 / 61,
        2: 1 / 61 + 1 / 62,
        3: 1 / 62,
    }
    assert [doc_id for doc_id, _ in result.hits] == [2, 1, 3]
    for doc_id, score in result.hits:
        assert score == pytest.approx(expected[doc_id])


def test_rrf_tie_break_by_doc_id():
    arm = _FixedArm([(5, 1.0), (4, 1.0)])
    rrf = RrfStrategy("rrf", [arm])
    # same arm weight -> rrf score differs by rank, so 5 (rank 1) beats 4
    result = rrf.search("x", k=2, hard={})
    assert [doc_id for doc_id, _ in result.hits] == [5, 4]
    twin = RrfStrategy("rrf", [_FixedArm([(4, 1.0)]), _FixedArm([(5, 1.0)])])
    tied = twin.search("x", k=2, hard={})
    # both docs fuse to 1/61 + 1/61 -> equal scores -> doc_id ascending
    assert [doc_id for doc_id, _ in tied.hits] == [4, 5]


def test_build_strategy_rejects_unknown(materials, bench_root):
    with pytest.raises(SystemExit):
        build_strategy("nope", materials, bench_root, None)


def test_build_strategy_needs_endpoint_for_dense(materials, bench_root):
    with pytest.raises(SystemExit, match="VPC_EMBED"):
        build_strategy("dense", materials, bench_root, None)
    with pytest.raises(SystemExit, match="VPC_EMBED"):
        build_strategy("rrf", materials, bench_root, None)


def test_strategy_config_and_hash_stable():
    spec = strategy_config("bm25-query", None)
    assert spec["idf_side"] == "query"
    assert spec["k1"] == 1.2
    assert config_hash(spec) == config_hash(strategy_config("bm25-query", None))
    assert config_hash(spec) != config_hash(strategy_config("bm25-both", None))
    rrf_spec = strategy_config("rrf", None)
    assert rrf_spec["rrf_k"] == 60
    assert rrf_spec["arms"] == ["bm25-query", "dense"]
