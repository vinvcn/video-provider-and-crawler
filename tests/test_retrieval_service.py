"""Unit tests for the shared retrieval core (store/search.py) and service glue."""

import pytest

from store.search import (
    _hydrate_hits,
    hit_url_set,
    pexels_page_url,
    query_vector,
    resolve_query,
)


class _FakeProvider:
    """Counts dense_text calls to observe the query-vector cache."""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = 0

    def dense_text(self, texts):
        self.calls += 1
        return [[0.1, 0.2] for _ in texts]

    def close(self):
        self.closed += 1


def _fake_hit() -> dict:
    """A hit that satisfies the SearchResponse model."""
    return {
        "pexels_id": 1,
        "rrf": 0.02,
        "url": "https://www.pexels.com/video/x-1/",
        "title": "Waves",
        "duration": 10,
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "orientation": "landscape",
        "thumbnail_path": None,
        "thumbnail_url": "https://images.pexels.com/t/1.jpeg",
        "video_files": [{"link": "https://videos.pexels.com/video-files/1/a.mp4"}],
        "vector1024": [0.0],
    }


def _fake_result(q: str) -> dict:
    return {
        "query": q,
        "hits": [_fake_hit()],
        "arms": {"dense": 1, "sparse": 0},
        "latency_ms": {"embed_api": 1.0, "dense": 1.0, "sparse": 0.0},
    }


@pytest.fixture()
def fake_provider(monkeypatch):
    from store import search as core

    fake = _FakeProvider()
    # patch get_provider (not _provider) so the real singleton path is exercised
    monkeypatch.setattr(core, "get_provider", lambda name: fake)
    core.reset_caches()
    yield fake
    core.reset_caches()


def test_resolve_query_prefers_coarse():
    assert resolve_query(["ocean", "waves"], "sunset over sea") == "sunset over sea"
    assert resolve_query(["ocean", " waves ", None, ""], "") == "ocean waves"
    assert resolve_query(None, "") == ""
    assert resolve_query(["  "], "  ") == ""


def test_pexels_page_url():
    assert (
        pexels_page_url("sunset-over-mountains", 123)
        == "https://www.pexels.com/video/sunset-over-mountains-123/"
    )
    # missing slug still yields a stable, resolvable URL
    assert pexels_page_url(None, 7).endswith("/video-7/")


def test_hit_url_set_collects_every_url():
    hit = {
        "url": "https://www.pexels.com/video/x-1/",
        "thumbnail_url": "https://images.pexels.com/t/1.jpeg",
        "video_files": [
            {"link": "https://videos.pexels.com/video-files/1/a.mp4"},
            {"link": ""},
            "not-a-dict",
        ],
    }
    assert hit_url_set(hit) == {
        "https://www.pexels.com/video/x-1/",
        "https://images.pexels.com/t/1.jpeg",
        "https://videos.pexels.com/video-files/1/a.mp4",
    }


def test_query_vector_caches_per_text(fake_provider):
    from store import search as core

    first = query_vector("bge-m3", "sunset")
    second = query_vector("bge-m3", "sunset")
    assert first == second == [0.1, 0.2]
    assert fake_provider.calls == 1  # second call came from the cache

    key = "bge-m3\x00sunset"
    stamp, vector = core._QUERY_CACHE[key]
    core._QUERY_CACHE[key] = (stamp - core._QUERY_CACHE_TTL - 1, vector)
    query_vector("bge-m3", "sunset")
    assert fake_provider.calls == 2  # expired entry recomputed


def test_reset_caches_closes_providers_and_drops_queries(fake_provider):
    from store import search as core

    query_vector("bge-m3", "sunset")
    assert core._QUERY_CACHE
    assert "bge-m3" in core._PROVIDERS  # the real singleton pooled the fake
    core.reset_caches()
    assert not core._QUERY_CACHE
    assert not core._PROVIDERS
    assert fake_provider.closed == 1


def test_search_endpoint_maps_contract(monkeypatch):
    """POST /search: terms->q, exclude_urls->set, vector included for dedup."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import service.app as app_module

    captured = {}

    def fake_search_library(q, **kwargs):
        captured["q"] = q
        captured.update(kwargs)
        return _fake_result(q)

    monkeypatch.setattr(app_module, "search_library", fake_search_library)
    client = TestClient(app_module.app)

    resp = client.post(
        "/search",
        json={
            "terms": ["ocean", "waves"],
            "k": 5,
            "exclude_urls": ["https://videos.pexels.com/video-files/1/a.mp4"],
        },
    )
    assert resp.status_code == 200
    assert captured["q"] == "ocean waves"
    assert captured["k"] == 5
    assert captured["include_vector"] is True
    assert captured["exclude_urls"] == {"https://videos.pexels.com/video-files/1/a.mp4"}
    assert resp.json()["hits"][0]["vector1024"] == [0.0]

    # coarse_query overrides terms
    client.post("/search", json={"terms": ["x"], "coarse_query": "stormy sea"})
    assert captured["q"] == "stormy sea"

    # out-of-range k is rejected by validation
    assert client.post("/search", json={"k": 0}).status_code == 422


def test_query_log_written(tmp_path, monkeypatch):
    """Each lookup appends one JSONL line; disabled by VPC_QUERY_LOG=0."""
    pytest.importorskip("fastapi")
    import json as json_module

    from fastapi.testclient import TestClient

    import service.app as app_module

    monkeypatch.setattr(app_module, "search_library", lambda q, **kw: _fake_result(q))
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setenv("VPC_QUERY_LOG_PATH", str(log_path))
    monkeypatch.setenv("VPC_QUERY_LOG", "1")  # conftest disables it suite-wide
    client = TestClient(app_module.app)

    assert client.post("/search", json={"terms": ["ocean"], "k": 1}).status_code == 200
    line = json_module.loads(log_path.read_text(encoding="utf-8").strip())
    assert line["q"] == "ocean"
    assert line["k"] == 1
    assert line["top"] == [1]
    assert line["arms"] == {"dense": 1, "sparse": 0}

    monkeypatch.setenv("VPC_QUERY_LOG", "0")
    client.post("/search", json={"terms": ["ocean"], "k": 1})
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_hydrate_hits_maps_creator_and_parses_vector():
    """行组装契约：creator 来自 user_name 列；显式位置解析向量文本。"""
    row = (
        7,
        "slug-x",
        "Title",
        12,
        1920,
        1080,
        29.97,
        "landscape",
        None,
        "https://images.example/7.jpeg",
        [{"link": "https://videos.example/7.mp4", "width": 1920, "height": 1080}],
        "Creator Name",
        "[0.5, 0.25]",
    )
    hits = _hydrate_hits([row], {7: 0.016}, [7], include_vector=True, exclude_urls=None)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["creator"] == "Creator Name"
    assert hit["vector1024"] == [0.5, 0.25]
    assert hit["url"] == "https://www.pexels.com/video/slug-x-7/"
    assert hit["rrf"] == 0.016


def test_hydrate_hits_drops_excluded_urls():
    """exclude_urls 命中页面/缩略图/档位任一 URL 时整行丢弃。"""
    row = (
        7,
        "slug-x",
        "Title",
        12,
        1920,
        1080,
        29.97,
        "landscape",
        None,
        "https://images.example/7.jpeg",
        [{"link": "https://videos.example/7.mp4", "width": 1920, "height": 1080}],
        None,
    )
    hits = _hydrate_hits(
        [row],
        {7: 0.016},
        [7],
        include_vector=False,
        exclude_urls={"https://videos.example/7.mp4"},
    )

    assert hits == []


def test_openapi_schema_and_docs_page():
    """The Swagger surface self-documents the full contract."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import service.app as app_module

    client = TestClient(app_module.app)
    assert client.get("/docs").status_code == 200

    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/search", "/healthz"}
    assert spec["paths"]["/search"]["post"]["tags"] == ["search"]

    schemas = spec["components"]["schemas"]
    hit = schemas["SearchHit"]["properties"]
    assert {"pexels_id", "url", "video_files", "vector1024"} <= set(hit)
    request_schema = schemas["SearchRequest"]
    assert "examples" in request_schema  # clickable sample payloads in Swagger UI
    assert "exclude_urls" in request_schema["properties"]
    assert schemas["SearchRequest"]["properties"]["k"]["maximum"] == 100
