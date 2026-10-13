"""Text-source ablation variants: title / title-desc / DF-filtered tag corpora."""

import pytest
from conftest import material_row, write_materials

from bench import strategies


def _fixture(bench_root):
    """100 docs; tag DF engineered around the 5% / 10% thresholds.

    - `common`   in 15 docs (15%) -> dropped by df5 and df10
    - `middle`   in 10 docs (10%) -> dropped by df5 and df10 (>= threshold)
    - `edge`      in  5 docs (5%)  -> dropped by df5, kept by df10
    - `specific`  in  3 docs (3%)  -> kept by both (real content)
    - `uniqueN`   in  1 doc  (1%)  -> kept by both
    """
    rows = []
    for i in range(1, 101):
        tags = [f"unique{i}"]
        if i <= 15:
            tags.append("common")
        if i <= 10:
            tags.append("middle")
        if i <= 5:
            tags.append("edge")
        if i <= 3:
            tags.append("specific")
        rows.append(material_row(i, f"Clip {i}", tags=tuple(tags), description=f"Desc {i}."))
    return write_materials(bench_root, "v1", rows)


def test_raw_title_is_title_only(bench_root):
    materials = _fixture(bench_root)
    texts = strategies.resolve_texts(materials, None, "raw-title")
    assert texts[1] == "Clip 1"
    assert "Desc" not in texts[1] and "unique" not in texts[1]


def test_raw_title_desc_excludes_tags(bench_root):
    materials = _fixture(bench_root)
    texts = strategies.resolve_texts(materials, None, "raw-title-desc")
    assert texts[1] == "Clip 1 — Desc 1."
    assert "unique" not in texts[1]


def test_df10_drops_high_frequency_tags(bench_root):
    materials = _fixture(bench_root)
    texts = strategies.resolve_texts(materials, None, "raw-tags-df10")
    first = texts[1]
    assert "common" not in first and "middle" not in first  # 15% and 10% dropped
    assert "edge" in first and "specific" in first  # 5% and 3% survive the 10% cut
    assert "unique1" in first


def test_df5_drops_down_to_five_percent(bench_root):
    materials = _fixture(bench_root)
    texts = strategies.resolve_texts(materials, None, "raw-tags-df5")
    first = texts[1]
    assert "common" not in first and "middle" not in first and "edge" not in first
    assert "specific" in first and "unique1" in first  # 3% and 1% survive


def test_generic_tag_drop_is_deterministic(bench_root):
    materials = _fixture(bench_root)
    assert strategies._generic_tags(materials, 0.10) == frozenset({"common", "middle"})
    assert strategies._generic_tags(materials, 0.05) == frozenset({"common", "middle", "edge"})
    assert strategies._generic_tags(materials, 0.10) == strategies._generic_tags(materials, 0.10)


def test_unknown_text_source_rejected(bench_root):
    materials = _fixture(bench_root)
    with pytest.raises(SystemExit):
        strategies.resolve_texts(materials, None, "subtitles")


def test_corpus_label_covers_variants():
    assert strategies.corpus_label({"strategy": "dense"}) == "raw"
    assert strategies.corpus_label({"text_source": "raw-title"}) == "raw-title"
    assert strategies.corpus_label({"text_source": "raw-tags-df5"}) == "raw-tags-df5"
    assert (
        strategies.corpus_label({"text_source": "caption", "captions_model": "gemma-4-26b-a4b-it"})
        == "caption:gemma-4-26b-a4b-it"
    )


def test_variant_specs_hash_differently():
    base = {"strategy": "dense", "model": "bge", "dim": None}
    hashes = {
        source: strategies.config_hash({**base, "text_source": source})
        for source in ("raw", "raw-title", "raw-title-desc", "raw-tags-df5", "raw-tags-df10")
    }
    assert len(set(hashes.values())) == len(hashes)
