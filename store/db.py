"""Postgres access for the stock library."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

DEFAULT_DSN = os.environ.get("VPC_DB_DSN", "postgresql://postgres:postgres@127.0.0.1:15433/stock")

_UPSERT_SQL = """
INSERT INTO stock_videos (
    pexels_id, slug, title, description, tags, duration, width, height, fps,
    aspect_ratio, orientation, license, user_name, created_at, video_files,
    thumbnail_url, download_link, source, raw
) VALUES (
    %(pexels_id)s, %(slug)s, %(title)s, %(description)s, %(tags)s, %(duration)s,
    %(width)s, %(height)s, %(fps)s, %(aspect_ratio)s, %(orientation)s,
    %(license)s, %(user_name)s, %(created_at)s, %(video_files)s,
    %(thumbnail_url)s, %(download_link)s, %(source)s, %(raw)s
)
ON CONFLICT (pexels_id) DO UPDATE SET
    slug = EXCLUDED.slug,
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    tags = EXCLUDED.tags,
    duration = EXCLUDED.duration,
    width = EXCLUDED.width,
    height = EXCLUDED.height,
    fps = EXCLUDED.fps,
    aspect_ratio = EXCLUDED.aspect_ratio,
    orientation = EXCLUDED.orientation,
    license = EXCLUDED.license,
    user_name = EXCLUDED.user_name,
    created_at = EXCLUDED.created_at,
    video_files = EXCLUDED.video_files,
    thumbnail_url = EXCLUDED.thumbnail_url,
    download_link = EXCLUDED.download_link,
    source = EXCLUDED.source,
    raw = EXCLUDED.raw,
    updated_at = NOW()
"""


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a new connection; caller manages commit/close."""
    return psycopg.connect(dsn or DEFAULT_DSN)


_JSONB_KEYS = ("video_files", "raw")


def _wrap_jsonb(row: dict[str, Any]) -> dict[str, Any]:
    """Wrap dict/list values headed for jsonb columns so psycopg can adapt them."""
    prepared = dict(row)
    for key in _JSONB_KEYS:
        value = prepared.get(key)
        if isinstance(value, (dict, list)):
            prepared[key] = Jsonb(value)
    return prepared


def upsert_videos(conn: psycopg.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """Idempotently upsert normalized rows (keyed by pexels_id); returns count."""
    batch = list(rows)
    if not batch:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, [_wrap_jsonb(row) for row in batch])
    conn.commit()
    return len(batch)


_CATALOG_TABLES = {"videos": "catalog_videos", "queries": "catalog_queries"}

def count_catalog(conn: psycopg.Connection, kind: str) -> int:
    """Row count of the catalog table backing a sitemap kind (videos|queries)."""
    table = _CATALOG_TABLES[kind]
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def upsert_catalog_videos(
    conn: psycopg.Connection, rows: Iterable[tuple[int, str | None, str | None]]
) -> int:
    """Idempotently upsert video-sitemap (pexels_id, slug, lastmod) rows."""
    batch = list(rows)
    if not batch:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO catalog_videos (pexels_id, slug, lastmod)
            VALUES (%s, %s, %s)
            ON CONFLICT (pexels_id) DO UPDATE SET
                slug = EXCLUDED.slug,
                lastmod = EXCLUDED.lastmod,
                last_seen_at = NOW()
            """,
            batch,
        )
    conn.commit()
    return len(batch)


def upsert_catalog_queries(conn: psycopg.Connection, rows: Iterable[tuple[str, str | None]]) -> int:
    """Idempotently upsert query-sitemap (term, lastmod) rows."""
    batch = list(rows)
    if not batch:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO catalog_queries (term, lastmod)
            VALUES (%s, %s)
            ON CONFLICT (term) DO UPDATE SET
                lastmod = EXCLUDED.lastmod,
                last_seen_at = NOW()
            """,
            batch,
        )
    conn.commit()
    return len(batch)


_EMBED_SELECT_SQL = """
SELECT pexels_id, title, description, tags
FROM stock_videos
WHERE embedding_model IS DISTINCT FROM %s
  AND (
      COALESCE(title, '') <> ''
      OR COALESCE(description, '') <> ''
      OR cardinality(tags) > 0
  )
ORDER BY pexels_id
LIMIT %s
"""

_EMBED_UPDATE_SQL = """
UPDATE stock_videos SET
    embed_text = %s,
    embedding = %s::vector,
    embedding_model = %s,
    embedded_at = NOW(),
    sparse_embedding = %s::sparsevec,
    sparse_model = %s
WHERE pexels_id = %s
"""


def embed_pending(
    conn: psycopg.Connection, provider: Any, limit: int = 1000
) -> dict[str, Any]:
    """Backfill embedding columns for rows not yet embedded with `provider`.

    Vectors travel as pgvector text literals cast in SQL, so no extra database
    adapter dependency is required. Rows with empty canonical text are excluded
    by the selection filter, so they are never re-selected.
    """
    from store.embedding import build_embed_text, to_sparsevec_literal, to_vector_literal

    rows = conn.execute(_EMBED_SELECT_SQL, (provider.spec.model_id, limit)).fetchall()
    if not rows:
        return {"embedded": 0, "model": provider.spec.model_id}

    dense_ok = "dense_text" in provider.spec.supports
    sparse_ok = "sparse_text" in provider.spec.supports
    step = max(1, provider.spec.max_batch)
    embedded = 0
    for start in range(0, len(rows), step):
        chunk = rows[start : start + step]
        texts = [
            build_embed_text({"title": row[1], "description": row[2], "tags": row[3]})
            for row in chunk
        ]
        dense = provider.dense_text(texts) if dense_ok else [None] * len(chunk)
        sparse = provider.sparse_text(texts) if sparse_ok else [None] * len(chunk)
        with conn.cursor() as cur:
            for row, text, vector, weights in zip(chunk, texts, dense, sparse, strict=True):
                cur.execute(
                    _EMBED_UPDATE_SQL,
                    (
                        text,
                        to_vector_literal(vector) if vector is not None else None,
                        provider.spec.model_id if vector is not None else None,
                        to_sparsevec_literal(weights) if weights else None,
                        provider.spec.model_id if weights else None,
                        row[0],
                    ),
                )
        conn.commit()
        embedded += len(chunk)
    return {"embedded": embedded, "model": provider.spec.model_id}


def stats(conn: psycopg.Connection) -> dict[str, Any]:
    """Library counters for `vpc status`."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE thumbnail_path IS NOT NULL) AS with_thumb,
            COALESCE(
                SUM(CASE WHEN video_files @> '[{"quality":"uhd"}]' THEN 1 ELSE 0 END), 0
            ) AS with_uhd,
            COALESCE(SUM(CASE WHEN cardinality(tags) = 0 THEN 1 ELSE 0 END), 0) AS no_tags
        FROM stock_videos
        """
    ).fetchone()
    keys = ["total", "with_thumb", "with_uhd", "no_tags"]
    return dict(zip(keys, row or (0, 0, 0, 0), strict=True))
