"""HTTP retrieval service — the Phase 2 product surface.

Contract (docs/handoff-2026-10-07.md §5): `POST /search` takes the caller's
terms (and/or a coarse query) plus structured filters and already-used URLs,
and returns top-k hits with page URL, dimensions, thumbnail, every video
rendition and the row's 1024-dim vector — everything the MoneyPrinterTurbo
funnel (coarse rank -> precise rank -> VLM walk -> dedup gate) needs.

Self-documenting: FastAPI serves the OpenAPI schema at /openapi.json with
Swagger UI at /docs and ReDoc at /redoc.

Fail-open is the *caller's* contract: this service answers errors with a
plain 5xx so the consumer can degrade to its default video source; it never
hangs. `GET /healthz` reports index coverage for readiness probes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from crawler.config import REPO_ROOT, load_settings
from store import db
from store.embedding import SPARSE_DIM, to_sparsevec_literal, to_vector_literal
from store.remote_embedding import DEFAULT_DIM
from store.search import resolve_query, search_library

logger = logging.getLogger("vpc.retrieval")

load_settings()  # .env -> environ once, before any provider/DSN read


def _warm_indexes() -> None:
    """Touch both HNSW graphs once so the day's first real query avoids cold I/O.

    A tiny uniform vector and a single-term sparsevec still traverse the
    graph; failures are logged and ignored (warmup must never block startup).
    """
    try:
        dense_literal = to_vector_literal([0.01] * DEFAULT_DIM)
        sparse_literal = to_sparsevec_literal({1: 1.0})
        conn = db.connect()
        try:
            conn.execute(
                "SELECT pexels_id FROM stock_videos WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT 1",
                (dense_literal,),
            ).fetchall()
            conn.execute(
                "SELECT pexels_id FROM stock_videos WHERE sparse_embedding IS NOT NULL "
                f"ORDER BY sparse_embedding::sparsevec({SPARSE_DIM}) <#> %s::sparsevec "
                "LIMIT 1",
                (sparse_literal,),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:  # warmup is best-effort; never block startup
        logger.debug("index warmup skipped: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warm_indexes()
    yield


app = FastAPI(
    title="vpc-retrieval",
    version="1.0.0",
    description=(
        "Hybrid retrieval over the full Pexels video library (665k rows): "
        "bge-m3 dense HNSW + BM25 sparse HNSW, fused with RRF (k=60). "
        "Repeated query texts hit an embedding cache; excluded URLs are "
        "dropped and backfilled so k still fills."
    ),
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    """One funnel lookup; `terms` and `coarse_query` are alternatives."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "terms": ["ocean waves", "slow motion"],
                    "orientation": "landscape",
                    "min_duration": 5,
                    "k": 3,
                    "exclude_urls": ["https://videos.pexels.com/video-files/1/a.mp4"],
                },
                {
                    "coarse_query": "雨夜城市霓虹街道",
                    "orientation": "portrait",
                    "k": 5,
                },
            ]
        }
    )

    terms: list[str] = Field(
        default_factory=list, description="Query segments, joined by spaces."
    )
    coarse_query: str = Field(
        "", description="Whole-sentence query; takes priority over `terms` when non-empty."
    )
    orientation: str | None = Field(None, description="landscape | portrait | square")
    min_duration: int | None = Field(None, description="Minimum clip duration, seconds.")
    max_duration: int | None = Field(None, description="Maximum clip duration, seconds.")
    min_width: int | None = Field(None, description="Minimum width, pixels (3840 = 4K).")
    license: str | None = Field(None, description="License filter, e.g. 'Pexels'.")
    k: int = Field(30, ge=1, le=100, description="Hits to return.")
    exclude_urls: list[str] = Field(
        default_factory=list,
        description="URLs the caller already used; a hit matching any of them "
        "(page, thumbnail or video_files link) is dropped and backfilled "
        "from the next candidate.",
    )


class SearchHit(BaseModel):
    """One ranked clip with everything the MPT funnel needs."""

    pexels_id: int = Field(..., description="Pexels video id.")
    rrf: float = Field(..., description="Reciprocal-rank-fusion score, higher is better.")
    url: str = Field(..., description="Pexels page URL.")
    title: str | None = None
    duration: int | None = Field(None, description="Clip duration, seconds.")
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    orientation: str | None = None
    thumbnail_path: str | None = Field(None, description="Local thumbnail path.")
    thumbnail_url: str | None = None
    video_files: list[dict[str, Any]] = Field(
        ..., description="Every rendition: link, quality, width/height, fps."
    )
    creator: str | None = Field(None, description="Uploader/creator name when known.")
    vector1024: list[float] | None = Field(
        None, description="Row embedding for the caller's dedup gate."
    )


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    arms: dict[str, int] = Field(..., description="Candidates each arm returned (depth 100).")
    latency_ms: dict[str, float] = Field(
        ..., description="Per-stage milliseconds: embed_api / dense / sparse."
    )


class HealthStatus(BaseModel):
    status: str
    total: int
    dense: int
    sparse: int


def _log_query(request: SearchRequest, result: dict[str, Any]) -> None:
    """Append one JSONL line per lookup; logging never fails a search.

    Feeds future relevance tuning (what creators actually search, which
    latency stage dominates). Disable with VPC_QUERY_LOG=0; redirect the
    file with VPC_QUERY_LOG_PATH.
    """
    if os.environ.get("VPC_QUERY_LOG", "1") != "1":
        return
    path = Path(
        os.environ.get("VPC_QUERY_LOG_PATH", str(REPO_ROOT / "storage" / "logs" / "queries.jsonl"))
    )
    try:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "q": result["query"],
            "k": request.k,
            "orientation": request.orientation,
            "min_duration": request.min_duration,
            "exclude": len(request.exclude_urls),
            "arms": result["arms"],
            "latency_ms": result["latency_ms"],
            "top": [hit["pexels_id"] for hit in result["hits"][:10]],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # the query log must never break search
        logger.debug("query log write skipped: %s", exc)


@app.get(
    "/healthz", response_model=HealthStatus, tags=["health"], summary="Index coverage counters"
)
def healthz() -> HealthStatus:
    """Total rows and dense/sparse coverage; 200 means the service is usable."""
    conn = db.connect()
    try:
        row = conn.execute(
            """
            SELECT count(*) AS total,
                   count(embedding) AS dense,
                   count(sparse_embedding) AS sparse
            FROM stock_videos
            """
        ).fetchone()
    finally:
        conn.close()
    return {"status": "ok", "total": row[0], "dense": row[1], "sparse": row[2]}


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Hybrid top-k retrieval (dense + sparse + RRF)",
)
def search(request: SearchRequest) -> dict[str, Any]:
    """Top-k hybrid hits; vectors included for the dedup gate."""
    q = resolve_query(request.terms, request.coarse_query)
    result = search_library(
        q,
        k=request.k,
        orientation=request.orientation,
        min_duration=request.min_duration,
        max_duration=request.max_duration,
        min_width=request.min_width,
        license=request.license,
        include_vector=True,
        exclude_urls=set(request.exclude_urls) or None,
    )
    _log_query(request, result)
    return result
