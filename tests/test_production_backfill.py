"""Unit tests for the embedding backfill selection SQL and remote provider."""

import httpx
import pytest

from store.db import embed_select_sql
from store.remote_embedding import EmbeddingEndpointError, RemoteEmbeddingProvider


def test_select_dense_only_tracks_embedding_model():
    sql, params = embed_select_sql(dense=True, sparse=False)
    assert "embedding_model IS DISTINCT FROM %s" in sql
    assert "sparse_model" not in sql
    assert "btrim(COALESCE(title, ''))" in sql
    assert params == ["model", "limit"]


def test_select_sparse_only_tracks_sparse_model():
    sql, params = embed_select_sql(dense=False, sparse=True)
    assert "sparse_model IS DISTINCT FROM %s" in sql
    assert "embedding_model" not in sql
    assert params == ["model", "limit"]


def test_select_both_arms_or_the_predicates():
    sql, params = embed_select_sql(dense=True, sparse=True)
    assert "OR" in sql
    assert sql.count("%s") == 3  # two model params + limit
    assert params == ["model", "model", "limit"]


def test_select_shard_restricts_by_hash():
    sql, _ = embed_select_sql(dense=True, sparse=False, shard=0, shards=3)
    assert "pexels_id % 3 = 0" in sql


def test_select_rejects_bad_shard():
    with pytest.raises(ValueError):
        embed_select_sql(dense=True, sparse=False, shard=3, shards=3)
    with pytest.raises(ValueError):
        embed_select_sql(dense=False, sparse=False)


def _mock_transport(responses):
    """Responses is a list of (status, json_body); consumed per request."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = state["calls"]
        state["calls"] += 1
        status, body = responses[min(index, len(responses) - 1)]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), state


def _provider(transport):
    return RemoteEmbeddingProvider(
        base_url="https://embed.example/v1",
        api_key="test-key",
        model="BAAI/bge-m3",
        dim=4,
        batch=2,
        retries=2,
        transport=transport,
    )


def test_remote_provider_returns_vectors_in_order():
    transport, _ = _mock_transport(
        [
            (
                200,
                {
                    "data": [
                        {"index": 1, "embedding": [0.2, 0.2, 0.2, 0.2]},
                        {"index": 0, "embedding": [0.1, 0.1, 0.1, 0.1]},
                    ],
                    "usage": {"total_tokens": 9},
                },
            )
        ]
    )
    provider = _provider(transport)
    vectors = provider.dense_text(["a", "b"])
    assert vectors == [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]]
    assert provider.total_tokens == 9
    assert provider.spec.model_id == "BAAI/bge-m3@4"
    assert provider.spec.max_batch == 2


def test_remote_provider_retries_429_then_succeeds():
    transport, state = _mock_transport(
        [
            (429, {"error": "rate"}),
            (500, {"error": "boom"}),
            (200, {"data": [{"index": 0, "embedding": [0.5, 0.5, 0.5, 0.5]}], "usage": {}}),
        ]
    )
    provider = _provider(transport)
    assert provider.dense_text(["a"]) == [[0.5, 0.5, 0.5, 0.5]]
    assert state["calls"] == 3


def test_remote_provider_raises_on_dimension_mismatch():
    transport, _ = _mock_transport(
        [(200, {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "usage": {}})]
    )
    provider = _provider(transport)
    with pytest.raises(EmbeddingEndpointError, match="expected 4"):
        provider.dense_text(["a"])


def test_remote_provider_bubbles_hard_rejections():
    transport, _ = _mock_transport([(400, {"error": "bad model"})])
    provider = _provider(transport)
    with pytest.raises(EmbeddingEndpointError, match="rejected the request"):
        provider.dense_text(["a"])


def test_bm25_provider_registry_loads_fitted_model(tmp_path, monkeypatch):
    from store.bm25 import Bm25Model
    from store.embedding import get_provider

    path = tmp_path / "bm25-corpus.json"
    Bm25Model(idf_side="query").fit(["ocean waves sunset", "city night timelapse"]).save(path)
    monkeypatch.setenv("VPC_BM25_MODEL", str(path))

    provider = get_provider("bm25")
    assert provider.spec.model_id == "local-bm25-query-v1"
    weights = provider.sparse_text(["ocean waves"])
    assert all(isinstance(weight, dict) for weight in weights)
    assert weights[0]
