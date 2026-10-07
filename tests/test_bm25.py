"""Unit tests for the local BM25 sparse provider (pure logic, no I/O)."""

import json

import pytest

from store.bm25 import Bm25Model, Bm25SparseProvider
from store.embedding import tokenize

CORPUS = [
    "city night lights",
    "city day traffic",
    "quiet forest morning",
]


def test_tokenize_is_lowercase_alphanumeric():
    assert tokenize("City-Night, lights!") == ["city", "night", "lights"]


def test_fit_collects_stats():
    model = Bm25Model().fit(CORPUS)
    assert model.n_docs == 3
    assert model.total_len == 9
    assert model.avg_len == pytest.approx(3.0)
    assert model.df["city"] == 2
    assert model.df["forest"] == 1


def test_term_ids_are_deterministic_and_ranked_by_frequency():
    first = Bm25Model().fit(CORPUS).term_ids
    second = Bm25Model().fit(CORPUS).term_ids
    assert first == second
    assert first["city"] == 1  # most frequent term gets the lowest id


def test_idf_rewards_rare_terms():
    model = Bm25Model().fit(CORPUS)
    assert model.idf("forest") > model.idf("city")
    assert model.idf("city") > 0


def test_doc_weights_skip_unseen_terms_and_stay_positive():
    model = Bm25Model().fit(CORPUS)
    weights = model.doc_weights("city zebra")
    assert model.term_ids["city"] in weights
    assert all(value > 0 for value in weights.values())
    assert "zebra" not in model.term_ids


def test_doc_weights_normalise_by_length():
    model = Bm25Model().fit(CORPUS)
    short = model.doc_weights("city")
    # same term frequency, longer document -> smaller weight
    long = model.doc_weights("city quiet forest morning")
    term = model.term_ids["city"]
    assert short[term] > long[term]


def test_query_weights_use_idf_per_distinct_term():
    model = Bm25Model().fit(CORPUS)
    weights = model.query_weights("city city forest nope")
    assert set(weights) == {model.term_ids["city"], model.term_ids["forest"]}
    assert weights[model.term_ids["forest"]] > weights[model.term_ids["city"]]


def test_provider_is_sparse_only():
    provider = Bm25SparseProvider(model=Bm25Model().fit(CORPUS))
    [weights] = provider.sparse_text(["city"])
    assert weights
    with pytest.raises(NotImplementedError):
        provider.dense_text(["city"])
    with pytest.raises(NotImplementedError):
        provider.dense_image([b"x"])


def test_save_and_load_round_trip(tmp_path):
    model = Bm25Model().fit(CORPUS)
    path = tmp_path / "bm25.json"
    model.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["n_docs"] == 3
    loaded = Bm25Model.load(path)
    assert loaded.term_ids == model.term_ids
    assert loaded.df == model.df
    assert loaded.doc_weights("city night") == model.doc_weights("city night")
