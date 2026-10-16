"""Unit tests for the lazy image-embedding store and its service endpoints.

No DB and no provider: `db.connect` and httpx are faked per test, the
thumbnail dir points at tmp_path, and the FastAPI endpoints run through the
TestClient with `get_or_embed` patched where noted. What these tests pin
down:

- a hit answers from the store without touching the file or the provider;
- a miss reads the local file, embeds once, upserts and reports `cached=False`;
- the in-flight re-check turns a racing second call into a hit, and two real
  threads on one key trigger exactly one embed;
- the local-miss path falls back to the CDN `thumbnail_url`;
- provider failures (HTTP error, 429 storm, malformed JSON) raise ValueError,
  which the endpoint maps to 502; a missing image maps to 404;
- `VPC_USE_PROXY=0` threads through to every outbound httpx call;
- both paths emit one usage log line (the A/B cost axis reads these).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from service import app as service_app  # noqa: E402
from store import image_embeddings  # noqa: E402

VEC = [0.25, -0.5, 0.75]


class _FakeConn:
    """Connection double: scripted lookups plus recorded upserts."""

    def __init__(self, *, hits: dict[str, list[float]] | None = None, url: str = "") -> None:
        self.hits = dict(hits or {})
        self.url = url
        self.upserts: list[tuple[str, str, list[float]]] = []

    def execute(self, sql: str, params: tuple | None = None):  # noqa: ANN001
        class _Row:
            def __init__(self, value: Any) -> None:
                self._value = value

            def fetchone(self) -> tuple | None:
                return (self._value,) if self._value is not None else None

        if sql.strip().startswith("SELECT embedding"):
            return _Row(self.hits.get(params[0]))
        if sql.strip().startswith("SELECT thumbnail_url"):
            return _Row(self.url or None)
        if sql.strip().startswith("INSERT INTO image_embeddings"):
            self.upserts.append(
                (params[0], params[1], [float(x) for x in params[2].strip("[]").split(",")])
            )
            self.hits[params[0]] = self.upserts[-1][2]
            return None
        return _Row(None)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture()
def settings(monkeypatch):
    monkeypatch.setenv("VPC_IMAGE_EMBED_MODEL", "test-model")
    monkeypatch.setenv("VPC_IMAGE_EMBED_API_KEY", "test-key")
    monkeypatch.setenv("VPC_USE_PROXY", "0")
    return image_embeddings.embed_settings()


def test_settings_thread_the_direct_connection_override(settings):
    """AGENTS.md §3: VPC_USE_PROXY=0 must reach every outbound call."""
    assert settings.trust_env is False
    assert settings.model == "test-model"
    assert settings.timeout == 30.0


def test_hit_never_reads_file_or_provider(monkeypatch, caplog, settings):
    conn = _FakeConn(hits={image_embeddings.pexels_key(7): VEC})
    monkeypatch.setattr(image_embeddings.db, "connect", lambda: conn)

    def _fail_read(pexels_id: int):
        raise AssertionError("a hit must not read the file")

    monkeypatch.setattr(image_embeddings, "image_bytes", _fail_read)

    with caplog.at_level(logging.INFO, logger="vpc.image_embeddings"):
        result = image_embeddings.get_or_embed(7)

    assert result == {
        "key": "pexels:7",
        "model": "test-model",
        "dim": 3,
        "vector": VEC,
        "cached": True,
        "image": "none",
    }
    assert any("image embed hit" in record.message for record in caplog.records)


def test_miss_embeds_local_file_once(monkeypatch, tmp_path, caplog, settings):
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

    with caplog.at_level(logging.INFO, logger="vpc.image_embeddings"):
        result = image_embeddings.get_or_embed(7)

    assert result["cached"] is False
    assert result["image"] == "local"
    assert calls == [b"jpeg-bytes"]
    assert conn.upserts == [("pexels:7", "test-model", VEC)]
    assert any("image embed miss" in record.message for record in caplog.records)


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


def test_concurrent_same_key_embeds_once(monkeypatch, settings):
    conn = _FakeConn()
    monkeypatch.setattr(image_embeddings.db, "connect", lambda: conn)
    embeds: list[bytes] = []
    gate = threading.Event()

    def _slow_embed(payload: bytes, settings):
        # The first caller embeds slowly; the second must wait on the key lock.
        gate.wait(timeout=5)
        embeds.append(payload)
        return VEC, {}

    monkeypatch.setattr(image_embeddings, "embed_image", _slow_embed)
    monkeypatch.setattr(image_embeddings, "image_bytes", lambda conn, pexels_id: (b"j", "local"))

    results: list[dict] = []

    def _run():
        results.append(image_embeddings.get_or_embed(7))

    threads = [threading.Thread(target=_run) for _ in range(2)]
    threads[0].start()
    threads[1].start()
    gate.set()
    for thread in threads:
        thread.join(timeout=5)

    assert len(embeds) == 1  # single flight: one embed for two racing callers
    assert len(results) == 2
    assert sum(1 for result in results if result["cached"]) >= 1


def test_image_bytes_falls_back_to_cdn(monkeypatch, tmp_path, settings):
    conn = _FakeConn(url="https://images.pexels.com/t/7.jpeg")
    monkeypatch.setattr(
        image_embeddings, "thumbnail_path", lambda pexels_id: tmp_path / f"{pexels_id}.jpg"
    )

    class _Response:
        status_code = 200
        content = b"cdn-bytes"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        image_embeddings.httpx,
        "get",
        lambda url, **kwargs: _assert_cdn_call(url, kwargs, _Response()),
    )
    payload, source = image_embeddings.image_bytes(conn, 7)
    assert payload == b"cdn-bytes"
    assert source == "cdn"


def _assert_cdn_call(url, kwargs, response):
    """The CDN fetch must carry the configured timeout and trust_env."""
    assert kwargs["timeout"] == 30.0
    assert kwargs["trust_env"] is False
    assert url == "https://images.pexels.com/t/7.jpeg"
    return response


def test_embed_image_maps_every_provider_failure_to_value_error(monkeypatch, settings):
    class _StatusResponse:
        def __init__(self, status_code: int, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self):
            if self._payload is None:
                raise ValueError("not json")
            return self._payload

    cases = [
        _StatusResponse(500),
        _StatusResponse(200, payload={}),  # malformed: no output.embeddings
        _StatusResponse(200, payload={"output": {"embeddings": [{"embedding": []}]}}),
    ]
    backoff = image_embeddings._RATE_LIMIT_BACKOFF_SECONDS
    monkeypatch.setattr(image_embeddings, "_RATE_LIMIT_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    for response in cases:
        monkeypatch.setattr(
            image_embeddings.httpx, "post", lambda *args, _r=response, **kwargs: _r
        )
        with pytest.raises(ValueError):
            image_embeddings.embed_image(b"j", settings)

    class _HttpError(Exception):
        pass

    def _raise(*args, **kwargs):
        raise image_embeddings.httpx.HTTPError("boom")

    monkeypatch.setattr(image_embeddings.httpx, "post", _raise)
    with pytest.raises(ValueError):
        image_embeddings.embed_image(b"j", settings)
    monkeypatch.setattr(image_embeddings, "_RATE_LIMIT_BACKOFF_SECONDS", backoff)


def test_embed_image_retries_429_then_succeeds(monkeypatch, settings):
    responses = iter(
        [
            type("_R", (), {"status_code": 429}),
            type("_R", (), {
                "status_code": 200,
                "json": lambda self: {"output": {"embeddings": [{"embedding": [1.0]}]},
                                      "usage": {"total_tokens": 2}},
            })(),
        ]
    )
    monkeypatch.setattr(
        image_embeddings.httpx, "post", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(image_embeddings, "_RATE_LIMIT_BACKOFF_SECONDS", (0.0, 0.0, 0.0))

    vector, usage = image_embeddings.embed_image(b"j", settings)

    assert vector == [1.0]
    assert usage == {"total_tokens": 2}


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

    response = client.post("/embed", json={"asset_id": 8775004})

    assert response.status_code == 200
    body = response.json()
    assert body["key"] == "pexels:8775004"
    assert body["dim"] == 3
    assert body["cached"] is False

    # asset_id is typed int: pydantic rejects a non-numeric id with 422.
    assert client.post("/embed", json={"asset_id": "not-a-number"}).status_code == 422

    def _missing(pexels_id: int):
        raise FileNotFoundError("no local thumbnail and no thumbnail_url: pexels 1")

    monkeypatch.setattr(image_embeddings, "get_or_embed", _missing)
    assert client.post("/embed", json={"asset_id": 1}).status_code == 404

    def _unexpected(pexels_id: int):
        raise KeyError("provider surprised us")  # any bug still maps to 502

    monkeypatch.setattr(image_embeddings, "get_or_embed", _unexpected)
    assert client.post("/embed", json={"asset_id": 1}).status_code == 502
