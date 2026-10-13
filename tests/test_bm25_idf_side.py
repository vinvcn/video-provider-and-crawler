"""BM25 idf_side seam: legacy idf^2 vs textbook BM25 equivalence (D4)."""

from collections import Counter

import pytest

from store.bm25 import Bm25Model
from store.embedding import tokenize

CORPUS = [
    "city night lights skyline",
    "city day traffic people",
    "quiet forest morning fog",
    "ocean waves slow motion",
]


def _textbook_score(model: Bm25Model, query: str, doc: str) -> float:
    """Hand-rolled canonical BM25 (idf * saturated tf), independent of the seam."""
    tokens = tokenize(doc)
    tf = Counter(tokens)
    dl = len(tokens)
    norm = model.k1 * (1.0 - model.b + model.b * dl / model.avg_len)
    score = 0.0
    for term in set(tokenize(query)):
        if term not in tf:
            continue
        score += model.idf(term) * (tf[term] * (model.k1 + 1.0)) / (tf[term] + norm)
    return score


@pytest.mark.parametrize("idf_side", ["query", "doc"])
def test_dot_product_equals_textbook_bm25(idf_side):
    model = Bm25Model(idf_side=idf_side).fit(CORPUS)
    query = "city lights night"
    for doc in CORPUS:
        dw = model.doc_weights(doc)
        qw = model.query_weights(query)
        dot = sum(w * dw.get(term_id, 0.0) for term_id, w in qw.items())
        assert dot == pytest.approx(_textbook_score(model, query, doc))


def test_legacy_both_side_is_idf_squared():
    model = Bm25Model(idf_side="both").fit(CORPUS)
    query = "city lights night"
    doc = CORPUS[0]
    dw = model.doc_weights(doc)
    qw = model.query_weights(query)
    dot = sum(w * dw.get(term_id, 0.0) for term_id, w in qw.items())
    # hand-rolled idf^2 * saturated tf: what the legacy dot product actually is
    tokens = tokenize(doc)
    tf = Counter(tokens)
    norm = model.k1 * (1.0 - model.b + model.b * len(tokens) / model.avg_len)
    manual = 0.0
    for term in set(tokenize(query)):
        if term not in tf:
            continue
        manual += model.idf(term) ** 2 * (tf[term] * (model.k1 + 1.0)) / (tf[term] + norm)
    assert dot == pytest.approx(manual)
    # and the query side really carries idf too (the second factor)
    assert qw[model.term_ids["city"]] == pytest.approx(model.idf("city"))


def test_query_weights_doc_mode_is_one():
    model = Bm25Model(idf_side="doc").fit(CORPUS)
    weights = model.query_weights("city lights")
    assert weights and all(value == 1.0 for value in weights.values())


def test_doc_weights_query_mode_has_no_idf():
    model = Bm25Model(idf_side="query").fit(CORPUS)
    weights = model.doc_weights("city lights")
    tokens = tokenize("city lights")
    tf = Counter(tokens)
    norm = model.k1 * (1.0 - model.b + model.b * len(tokens) / model.avg_len)
    for term_id, weight in weights.items():
        term = next(t for t, i in model.term_ids.items() if i == term_id)
        expected = (tf[term] * (model.k1 + 1.0)) / (tf[term] + norm)
        assert weight == pytest.approx(expected)


def test_invalid_idf_side_is_rejected():
    with pytest.raises(ValueError, match="idf_side"):
        Bm25Model(idf_side="sometimes")


def test_save_load_preserves_idf_side(tmp_path):
    model = Bm25Model(idf_side="query").fit(CORPUS)
    path = tmp_path / "bm25.json"
    model.save(path)
    loaded = Bm25Model.load(path)
    assert loaded.idf_side == "query"
    assert loaded.doc_weights("city lights") == model.doc_weights("city lights")
