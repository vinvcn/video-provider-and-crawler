"""Unit tests for the embedding seam (pure logic, no I/O)."""

import pytest

from store.embedding import EmbeddingSpec, HashEmbeddingProvider, build_embed_text


def test_spec_rejects_bad_values():
    with pytest.raises(ValueError):
        EmbeddingSpec(model_id="x", dim=8, distance="hamming")
    with pytest.raises(ValueError):
        EmbeddingSpec(model_id="x", dim=8, max_batch=0)
    with pytest.raises(ValueError):
        EmbeddingSpec(model_id="x", dim=8, supports=frozenset())
    with pytest.raises(ValueError):
        EmbeddingSpec(model_id="x", dim=8, supports=frozenset({"nope"}))
    with pytest.raises(ValueError):
        EmbeddingSpec(model_id="x", dim=0, supports=frozenset({"dense_text"}))


def test_sparse_only_spec_needs_no_dim():
    spec = EmbeddingSpec(model_id="bm25", dim=0, supports=frozenset({"sparse_text"}))
    assert spec.dim == 0


def test_build_embed_text_joins_and_normalises():
    row = {
        "title": "  Aerial   View ",
        "description": "Night city",
        "tags": ["city", "night", ""],
    }
    assert build_embed_text(row) == "Aerial View — Night city — city, night"


def test_build_embed_text_handles_empty_row():
    assert build_embed_text({}) == ""
    assert build_embed_text({"title": None, "description": None, "tags": []}) == ""


def test_hash_provider_is_deterministic_and_unit_length():
    provider = HashEmbeddingProvider()
    first = provider.dense_text(["city night lights"])
    second = provider.dense_text(["city night lights"])
    assert first == second
    assert len(first[0]) == provider.spec.dim
    norm = sum(value * value for value in first[0]) ** 0.5
    assert norm == pytest.approx(1.0)


def test_hash_provider_sparse_counts_repeat_tokens():
    provider = HashEmbeddingProvider()
    [sparse] = provider.sparse_text(["dog dog cat"])
    assert len(sparse) == 2
    assert sorted(sparse.values()) == [1.0, 2.0]


def test_hash_provider_enforces_batch_cap():
    provider = HashEmbeddingProvider()
    with pytest.raises(ValueError):
        provider.dense_text(["a"] * (provider.spec.max_batch + 1))
