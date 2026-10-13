"""HTTP retrieval service — the Phase 2 product surface.

Contract (docs/handoff-2026-10-07.md §5): `POST /search` takes the caller's
terms (and/or a coarse query) plus structured filters and already-used URLs,
and returns top-k hits with page URL, dimensions, thumbnail, every video
rendition and the row's 1024-dim vector — everything the MoneyPrinterTurbo
funnel (coarse rank -> precise rank -> VLM walk -> dedup gate) needs.

Fail-open is the *caller's* contract: this service answers errors with a
plain 5xx so the consumer can degrade to its default video source; it never
hangs. `GET /healthz` reports index coverage for readiness probes.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from crawler.config import load_settings
from store import db
from store.search import resolve_query, search_library

app = FastAPI(title="vpc-retrieval", version="1")
load_settings()  # .env -> environ once, before any provider/DSN read


class SearchRequest(BaseModel):
    """One funnel lookup; `terms` and `coarse_query` are alternatives."""

    terms: list[str] = Field(default_factory=list)
    coarse_query: str = ""
    orientation: str | None = None
    min_duration: int | None = None
    max_duration: int | None = None
    min_width: int | None = None
    license: str | None = None
    k: int = Field(default=30, ge=1, le=100)
    exclude_urls: list[str] = Field(default_factory=list)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Index coverage counters; 200 = service usable."""
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
    return {
        "status": "ok",
        "total": row[0],
        "dense": row[1],
        "sparse": row[2],
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """Top-k hybrid hits; vectors included for the dedup gate."""
    q = resolve_query(request.terms, request.coarse_query)
    return search_library(
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
