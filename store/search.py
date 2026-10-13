"""Hybrid retrieval core shared by the `vpc search` CLI and the HTTP service.

One search = hard-filter SQL -> dense HNSW arm + sparse BM25 arm -> RRF (k=60)
-> metadata fetch. The dense query embedding is cached per (model, text) so
repeated creator terms skip the embedding API (~400 ms each).

The sparse arm's ORDER BY must textually match the expression index
(``sparse_embedding::sparsevec(SPARSE_DIM)`` — interpolated from
``store.embedding.SPARSE_DIM``) or the planner seq-scans; same
for the GUC: ``SET`` does not accept bind parameters.
"""

from __future__ import annotations

import time
from typing import Any

from store import db
from store.embedding import SPARSE_DIM, get_provider, to_sparsevec_literal, to_vector_literal

RRF_K = 60
PAGE_URL = "https://www.pexels.com/video/{slug}-{pexels_id}/"

_QUERY_CACHE: dict[str, tuple[float, list[float]]] = {}
_QUERY_CACHE_TTL = 600.0
_QUERY_CACHE_MAX = 512
_PROVIDERS: dict[str, Any] = {}


def reset_caches() -> None:
    """Drop cached providers and query vectors (tests)."""
    for provider in _PROVIDERS.values():
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    _PROVIDERS.clear()
    _QUERY_CACHE.clear()


def _provider(name: str) -> Any:
    """Process-wide provider singleton: keeps one httpx pool per model."""
    if name not in _PROVIDERS:
        _PROVIDERS[name] = get_provider(name)
    return _PROVIDERS[name]


def query_vector(model: str, text: str) -> list[float]:
    """Dense query embedding with a TTL cache keyed by (model, text)."""
    key = f"{model}\x00{text}"
    now = time.time()
    cached = _QUERY_CACHE.get(key)
    if cached and now - cached[0] < _QUERY_CACHE_TTL:
        return cached[1]
    vector = _provider(model).dense_text([text])[0]
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:  # simple bound; no LRU needed
        oldest = min(_QUERY_CACHE, key=lambda k: _QUERY_CACHE[k][0])
        del _QUERY_CACHE[oldest]
    _QUERY_CACHE[key] = (now, vector)
    return vector


def resolve_query(terms: list[str] | None, coarse_query: str | None) -> str:
    """The MPT contract sends terms[] and/or a coarse_query; one text wins."""
    coarse = (coarse_query or "").strip()
    if coarse:
        return coarse
    return " ".join(term.strip() for term in (terms or []) if term and term.strip())


def pexels_page_url(slug: str | None, pexels_id: int) -> str:
    """Canonical page URL for a video (the MPT `url` field)."""
    return PAGE_URL.format(slug=slug or "video", pexels_id=pexels_id)


def hit_url_set(hit: dict[str, Any]) -> set[str]:
    """Every URL a hit can be known by, for exclude_urls matching."""
    urls = {hit["url"], hit.get("thumbnail_url") or ""}
    for file in hit.get("video_files") or []:
        link = file.get("link") if isinstance(file, dict) else None
        if link:
            urls.add(link)
    return {url for url in urls if url}


def _hydrate_hits(
    rows: list[tuple[Any, ...]],
    scores: dict[int, float],
    top_ids: list[int],
    *,
    include_vector: bool,
    exclude_urls: set[str] | None,
) -> list[dict[str, Any]]:
    """Assemble DB rows into hit dicts, in RRF order.

    Column order matches the metadata SELECT in `search_library`
    (pexels_id, slug, title, duration, width, height, fps, orientation,
    thumbnail_path, thumbnail_url, video_files, user_name[, embedding]).
    psycopg returns the unregistered `vector` type as its "[a,b,...]"
    literal text; parse it without a new dependency. Rows whose page,
    thumbnail or rendition URL matches `exclude_urls` are dropped.
    """
    by_id = {row[0]: row for row in rows}
    hits: list[dict[str, Any]] = []
    for pid in top_ids:
        row = by_id.get(pid)
        if row is None:
            continue
        hit = {
            "pexels_id": pid,
            "rrf": round(scores[pid], 6),
            "url": pexels_page_url(row[1], pid),
            "title": row[2],
            "duration": row[3],
            "width": row[4],
            "height": row[5],
            "fps": float(row[6]) if row[6] is not None else None,
            "orientation": row[7],
            "thumbnail_path": row[8],
            "thumbnail_url": row[9],
            "video_files": row[10],
            "creator": row[11],
        }
        if include_vector:
            raw = row[12]
            if isinstance(raw, str):
                hit["vector1024"] = [float(x) for x in raw.strip("[]").split(",")]
            else:
                hit["vector1024"] = list(raw) if raw is not None else None
        if exclude_urls and hit_url_set(hit) & exclude_urls:
            continue
        hits.append(hit)
    return hits


def _hard_filter_sql(
    orientation: str | None,
    min_duration: int | None,
    max_duration: int | None,
    min_width: int | None,
    license: str | None,
) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if orientation:
        where.append("orientation = %s")
        params.append(orientation)
    if min_duration is not None:
        where.append("duration >= %s")
        params.append(min_duration)
    if max_duration is not None:
        where.append("duration <= %s")
        params.append(max_duration)
    if min_width is not None:
        where.append("width >= %s")
        params.append(min_width)
    if license:
        where.append("license = %s")
        params.append(license)
    return where, params


def search_library(
    q: str,
    *,
    k: int = 30,
    depth: int = 100,
    ef_search: int = 100,
    orientation: str | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_width: int | None = None,
    license: str | None = None,
    dense_model: str = "bge-m3",
    dsn: str | None = None,
    include_vector: bool = False,
    exclude_urls: set[str] | None = None,
) -> dict[str, Any]:
    """Hybrid search over the indexed library; returns ranked hits.

    `exclude_urls` drops hits whose page/CDN URLs the caller already used
    (the MPT dedup contract), fetching extra candidates so k still fills.
    `include_vector` returns the row's 1024-dim vector for downstream
    near-duplicate gating.

    Callers must run `crawler.config.load_settings()` first so `.env`
    reaches the provider and DSN (the CLI and service both do).
    """
    from store.bm25 import Bm25Model

    # query_vector warms the dense provider singleton; BM25 stays process-wide.
    bm25: Bm25Model = _provider("bm25").model  # type: ignore[assignment]
    conn = db.connect(dsn)
    try:
        conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
        hard, hard_params = _hard_filter_sql(
            orientation, min_duration, max_duration, min_width, license
        )
        hard_sql = (" AND ".join(hard) + " AND ") if hard else ""

        start = time.perf_counter()
        vector_literal = to_vector_literal(query_vector(dense_model, q))
        api_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        rows = conn.execute(
            f"""
            SELECT pexels_id FROM stock_videos
            WHERE {hard_sql}embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (*hard_params, vector_literal, depth),
        ).fetchall()
        dense_hits = [row[0] for row in rows]
        dense_ms = (time.perf_counter() - start) * 1000

        sparse_ms = 0.0
        sparse_hits: list[int] = []
        weights = bm25.query_weights(q)
        if weights:
            start = time.perf_counter()
            rows = conn.execute(
                f"""
                SELECT pexels_id FROM stock_videos
                WHERE {hard_sql}sparse_embedding IS NOT NULL
                ORDER BY sparse_embedding::sparsevec({SPARSE_DIM}) <#> %s::sparsevec
                LIMIT %s
                """,
                (*hard_params, to_sparsevec_literal(weights), depth),
            ).fetchall()
            sparse_hits = [row[0] for row in rows]
            sparse_ms = (time.perf_counter() - start) * 1000

        scores: dict[int, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, pexels_id in enumerate(hits, start=1):
                scores[pexels_id] = scores.get(pexels_id, 0.0) + 1.0 / (RRF_K + rank)
        margin = min(len(exclude_urls), k) if exclude_urls else 0
        top_ids = sorted(scores, key=lambda pid: (-scores[pid], pid))[: k + margin]

        hits: list[dict[str, Any]] = []
        if top_ids:
            vector_select = ", embedding" if include_vector else ""
            rows = conn.execute(
                f"""
                SELECT pexels_id, slug, title, duration, width, height, fps,
                       orientation, thumbnail_path, thumbnail_url, video_files,
                       user_name
                       {vector_select}
                FROM stock_videos WHERE pexels_id = ANY(%s)
                """,
                (top_ids,),
            ).fetchall()
            hits = _hydrate_hits(
                rows,
                scores,
                top_ids,
                include_vector=include_vector,
                exclude_urls=exclude_urls,
            )
        return {
            "query": q,
            "hits": hits[:k],
            "arms": {"dense": len(dense_hits), "sparse": len(sparse_hits)},
            "latency_ms": {
                "embed_api": round(api_ms, 1),
                "dense": round(dense_ms, 1),
                "sparse": round(sparse_ms, 1),
            },
        }
    finally:
        conn.close()
