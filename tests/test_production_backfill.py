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
    assert "mod(pexels_id, 3) = 0" in sql


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


# --- async write path (thread + queue mechanics, fake connections) ---


class _FakeCursor:
    def __init__(self, conn: "_FakeWriterConn") -> None:
        self._conn = conn

    def executemany(self, sql: str, params: list[tuple]) -> None:
        if self._conn.fail_on_write:
            raise RuntimeError("db write exploded")
        self._conn.batches.append(list(params))

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeWriterConn:
    def __init__(self, fail_on_write: bool = False) -> None:
        self.gucs: list[str] = []
        self.batches: list[list[tuple]] = []
        self.commits = 0
        self.closed = False
        self.fail_on_write = fail_on_write

    def execute(self, sql: str) -> None:
        self.gucs.append(" ".join(sql.split()))

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakeDenseProvider:
    """Dense-only provider: one 2-dim vector per input text, batch of 2."""

    from store.embedding import EmbeddingSpec  # noqa: PLC0415

    spec = EmbeddingSpec(
        model_id="fake-dense@2", dim=2, max_batch=2, supports=frozenset({"dense_text"})
    )

    def dense_text(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def _rows(n: int) -> list[tuple]:
    return [(100 + i, f"title {i}", None, ["tag"]) for i in range(n)]


def test_async_write_flushes_ordered_batches_on_background_connection():
    from store.db import _embed_rows_async

    writer_conn = _FakeWriterConn()
    embedded = _embed_rows_async(
        _rows(5),
        _FakeDenseProvider(),
        dsn="fake-dsn",
        model="fake-dense@2",
        flush_rows=2,
        queue_depth=2,
        connect=lambda dsn: writer_conn,
    )
    assert embedded == 5
    # GUCs tuned for a resumable bulk backfill.
    assert "SET synchronous_commit = off" in writer_conn.gucs
    assert "SET wal_compression = on" in writer_conn.gucs
    # 5 rows flushed as ceil(5/2) buffers, committed on the writer connection.
    ids = [params[-1] for batch in writer_conn.batches for params in batch]
    assert ids == [100, 101, 102, 103, 104]
    assert writer_conn.commits == len(writer_conn.batches) == 3
    assert writer_conn.closed


def test_async_write_surfaces_writer_errors():
    from store.db import _embed_rows_async

    writer_conn = _FakeWriterConn(fail_on_write=True)
    try:
        _embed_rows_async(
            _rows(4),
            _FakeDenseProvider(),
            dsn="fake-dsn",
            model="fake-dense@2",
            flush_rows=2,
            connect=lambda dsn: writer_conn,
        )
        raised = False
    except RuntimeError as exc:
        raised = "db write exploded" in str(exc)
    assert raised
    assert writer_conn.closed
