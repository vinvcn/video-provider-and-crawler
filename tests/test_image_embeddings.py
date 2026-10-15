"""Unit tests for the lazy image-embedding store and its service endpoints.

No DB and no provider: `db.connect` and `embed_image` are faked per test, the
thumbnail dir points at tmp_path, and the FastAPI endpoints run through the
TestClient with `get_or_embed` patched. What these tests pin down:

- a hit answers from the store without touching the file or the provider;
- a miss reads the local file, embeds once, upserts and reports `cached=False`;
- the in-flight re-check turns a racing second call into a hit;
- the endpoints map errors the way the MPT gate expects (404/502) and the
  thumbnail route serves the mounted file.
"""

from __future__ import annotations

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from service import app as service_app  # noqa: E402
from store import image_embeddings  # noqa: E402

VEC = [0.25, -0.5, 0.75]


class _FakeConn:
    """Connection double: scripted lookups plus recorded upserts."""

    def __init__(self, *, hits: dict[str, list[float]] | None = None) -> None:
        self.hits = dict(hits or {})
        self.upserts: list[tuple[str, str, int, list[float]]] = []

    def execute(self, sql: str, params: tuple | None = None):  # noqa: ANN001
        class _Row:
            def __init__(self, value: Any) -> None:
                self._value = value

            def fetchone(self) -> tuple | None:
                return (self._value,) if self._value is not None else None

        if sql.startswith("SELECT embedding"):
            key = params[0]
            return _Row(self.hits.get(key))
        if sql.startswith("SELECT thumbnail_url"):
            return _Row("https://images.pexels.com/t/1.jpeg")
        return _Row(None)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None

    def record_upsert(self, key: str, model: str, dim: int, vector: list[float]) -> None:
        self.upserts.append((key, model, dim, vector))
        self.hits[key] = vector


@pytest.fixture()
def settings(monkeypatch):
    monkeypatch.setenv("VPC_IMAGE_EMBED_MODEL", "test-model")
    monkeypatch.setenv("VPC_IMAGE_EMBED_API_KEY", "test-key")
    return image_embeddings.embed_settings()


def test_hit_never_reads_file_or_provider(monkeypatch, settings):
    conn = _FakeConn(hits={image_embeddings.pexels_key(7): VEC})
    monkeypatch.setattr(image_embeddings.db, "connect", lambda: conn)
    reads: list[int] = []

    def _fail_read(pexels_id: int):
        reads.append(pexels_id)
        raise AssertionError("a hit must not read the file")

    monkeypatch.setattr(image_embeddings, "image_bytes", _fail_read)

    result = image_embeddings.get_or_embed(7)

    assert result == {
        "key": "pexels:7",
        "model": "test-model",
        "dim": 3,
        "vector": VEC,
        "cached": True,
        "image": "none",
    }
    assert reads == []


def test_miss_embeds_local_file_once(monkeypatch, tmp_path, settings):
    conn = _FakeConn()
    monkeypatch.setattr(image_embeddings.db, "connect", lambda: conn)
    monkeypatch.setattr(
        image_embeddings, "thumbnail_path", lambda pexels_id: tmp_path / f"{pexels_id}.jpg"
    )
    (tmp_path / "7.jpg").write_bytes(b"jpeg-bytes")

    calls: list[bytes] = []

    def _fake_embed(payload: bytes, settings):
        calls.append(payload)
        return VEC, {"total_tokens": 66}

    monkeypatch.setattr(image_embeddings, "embed_image", _fake_embed)
    original_execute = conn.execute

    def _execute(sql: str, params: tuple | None = None):
        if sql.strip().startswith("INSERT INTO image_embeddings"):
            conn.record_upsert(
                params[0],
                params[1],
                params[3],
                [float(x) for x in params[4].strip("[]").split(",")],
            )
            return None
        return original_execute(sql, params)

    conn.execute = _execute  # type: ignore[method-assign]

    result = image_embeddings.get_or_embed(7)

    assert result["cached"] is False
    assert result["image"] == "local"
    assert result["dim"] == 3
    assert calls == [b"jpeg-bytes"]
    assert conn.hits == {"pexels:7": VEC}


def test_second_lookup_wins_after_the_race(monkeypatch, settings):
    first = _FakeConn()  # the lookup outside the lock misses
    second = _FakeConn(hits={image_embeddings.pexels_key(7): VEC})  # row appeared meanwhile
    conns = [first, second]
    monkeypatch.setattr(image_embeddings.db, "connect", lambda: conns.pop(0))
    embeds: list[bytes] = []
    monkeypatch.setattr(
        image_embeddings,
        "embed_image",
        lambda payload, settings: embeds.append(payload) or (VEC, {}),
    )

    result = image_embeddings.get_or_embed(7)

    assert result["cached"] is True
    assert result["vector"] == VEC
    assert embeds == []  # the loser never embeds


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(
        image_embeddings, "thumbnail_path", lambda pexels_id: tmp_path / f"{pexels_id}.jpg"
    )
    return TestClient(service_app.app)


def test_thumbnail_endpoint_serves_local_file(client, tmp_path):
    (tmp_path / "11922250.jpg").write_bytes(b"jpeg-bytes")

    response = client.get("/thumbnail/11922250")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content == b"jpeg-bytes"


def test_thumbnail_endpoint_404_when_file_absent(client):
    assert client.get("/thumbnail/999999").status_code == 404


def test_embed_endpoint_shapes_and_errors(client, monkeypatch):
    def _fake(pexels_id: int):
        return {
            "key": f"pexels:{pexels_id}",
            "model": "test-model",
            "dim": 3,
            "vector": VEC,
            "cached": False,
            "image": "local",
        }

    monkeypatch.setattr(image_embeddings, "get_or_embed", _fake)

    response = client.post("/embed", json={"asset_id": "8775004"})

    assert response.status_code == 200
    body = response.json()
    assert body["key"] == "pexels:8775004"
    assert body["dim"] == 3
    assert body["cached"] is False

    assert client.post("/embed", json={"asset_id": "not-a-number"}).status_code == 400

    def _missing(pexels_id: int):
        raise FileNotFoundError("no local thumbnail and no thumbnail_url: pexels 1")

    monkeypatch.setattr(image_embeddings, "get_or_embed", _missing)
    assert client.post("/embed", json={"asset_id": "1"}).status_code == 404

    def _provider_down(pexels_id: int):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(image_embeddings, "get_or_embed", _provider_down)
    assert client.post("/embed", json={"asset_id": "1"}).status_code == 502
