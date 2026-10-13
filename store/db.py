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
WHERE stock_videos.raw IS DISTINCT FROM EXCLUDED.raw
"""
# The WHERE guard makes replayed, byte-identical records a no-op: no heap
# rewrite, no index churn, no updated_at bump. `raw` holds the full source
# payload, so it is a faithful change detector for every derived column.


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


_EMBED_TEXT_FILTER_SQL = """(
    btrim(COALESCE(title, '')) <> ''
    OR btrim(COALESCE(description, '')) <> ''
    OR cardinality(tags) > 0
)"""

_EMBED_UPDATE_SQL = """
UPDATE stock_videos SET
    embed_text = %s,
    embedding = COALESCE(%s::vector, embedding),
    embedding_model = COALESCE(%s, embedding_model),
    sparse_embedding = COALESCE(%s::sparsevec, sparse_embedding),
    sparse_model = COALESCE(%s, sparse_model),
    embedded_at = NOW()
WHERE pexels_id = %s
"""


def embed_select_sql(
    dense: bool, sparse: bool, shard: int | None = None, shards: int = 3
) -> tuple[str, list[str]]:
    """(sql, param_names) for the rows a provider still owes vectors for.

    Selection follows the provider's modalities: a dense-only provider tracks
    `embedding_model`, a sparse-only one tracks `sparse_model`, so the two
    production backfills never re-select (or clobber) each other. `shard`
    restricts to `pexels_id % shards == shard` for incremental index builds.
    """
    owed: list[str] = []
    params: list[str] = []
    if dense:
        owed.append("embedding_model IS DISTINCT FROM %s")
        params.append("model")
    if sparse:
        owed.append("sparse_model IS DISTINCT FROM %s")
        params.append("model")
    if not owed:
        raise ValueError("provider supports neither dense_text nor sparse_text")
    sql = f"""
SELECT pexels_id, title, description, tags
FROM stock_videos
WHERE ({" OR ".join(owed)})
  AND {_EMBED_TEXT_FILTER_SQL}"""
    if shard is not None:
        if shards < 1:
            raise ValueError("shards must be positive")
        if not 0 <= shard < shards:
            raise ValueError(f"shard {shard} out of range 0..{shards - 1}")
        # mod() instead of %: psycopg would treat a literal '%' as a placeholder.
        sql += f"\n  AND mod(pexels_id, {shards}) = {shard}"
    sql += "\nORDER BY pexels_id\nLIMIT %s"
    params.append("limit")
    return sql, params


def _embed_row_params(
    row: Any,
    text: str,
    vector: list[float] | None,
    weights: dict[int, float] | None,
    model: str,
    dense_ok: bool,
    sparse_ok: bool,
) -> tuple:
    """Bound parameters for one _EMBED_UPDATE_SQL row write."""
    from store.embedding import to_sparsevec_literal, to_vector_literal

    return (
        text,
        to_vector_literal(vector) if vector is not None else None,
        model if dense_ok else None,
        to_sparsevec_literal(weights) if weights else None,
        model if sparse_ok else None,
        row[0],
    )


def _embed_rows_legacy(rows: list[Any], provider: Any, conn: psycopg.Connection, model: str) -> int:
    """Original write path: per-row execute on the caller's connection."""
    from store.embedding import build_embed_text

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
        live = [index for index, text in enumerate(texts) if text]
        dense: list[Any] = [None] * len(chunk)
        sparse: list[Any] = [None] * len(chunk)
        if dense_ok and live:
            vectors = provider.dense_text([texts[index] for index in live])
            for index, vector in zip(live, vectors, strict=True):
                dense[index] = vector
        if sparse_ok and live:
            weights = provider.sparse_text([texts[index] for index in live])
            for index, weight in zip(live, weights, strict=True):
                sparse[index] = weight
        with conn.cursor() as cur:
            for row, text, vector, weights in zip(chunk, texts, dense, sparse, strict=True):
                cur.execute(
                    _EMBED_UPDATE_SQL,
                    _embed_row_params(row, text, vector, weights, model, dense_ok, sparse_ok),
                )
        conn.commit()
        embedded += len(chunk)
    return embedded


def _embed_rows_async(
    rows: list[Any],
    provider: Any,
    dsn: str,
    model: str,
    flush_rows: int = 1000,
    queue_depth: int = 2,
    connect: Any = None,
) -> int:
    """Batched write path: executemany on a background connection.

    The embedding API (the slow part) runs in the calling thread while a
    writer thread drains buffers of `flush_rows` parameter tuples via
    executemany on its own connection, so DB writes hide behind API latency.
    The writer session disables synchronous_commit (a crash loses at most the
    un-committed buffers; the selection predicate re-picks those rows) and
    turns on wal_compression to shrink full-page WAL images.
    """
    import queue as _queue
    import threading

    from store.embedding import build_embed_text

    connect = connect or psycopg.connect
    dense_ok = "dense_text" in provider.spec.supports
    sparse_ok = "sparse_text" in provider.spec.supports
    writer_conn = connect(dsn)
    writer_conn.execute("SET synchronous_commit = off")
    writer_conn.execute("SET wal_compression = on")
    buffers: _queue.Queue = _queue.Queue(maxsize=queue_depth)
    errors: list[BaseException] = []

    def writer() -> None:
        while True:
            item = buffers.get()
            if item is None:
                return
            if errors:
                continue  # keep draining so the producer never deadlocks
            try:
                with writer_conn.cursor() as cur:
                    cur.executemany(_EMBED_UPDATE_SQL, item)
                writer_conn.commit()
            except BaseException as exc:  # surfaced by the producer below
                errors.append(exc)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()

    step = max(1, provider.spec.max_batch)
    pending: list[tuple] = []
    embedded = 0
    try:
        for start in range(0, len(rows), step):
            chunk = rows[start : start + step]
            texts = [
                build_embed_text({"title": row[1], "description": row[2], "tags": row[3]})
                for row in chunk
            ]
            live = [index for index, text in enumerate(texts) if text]
            dense: list[Any] = [None] * len(chunk)
            sparse: list[Any] = [None] * len(chunk)
            if dense_ok and live:
                vectors = provider.dense_text([texts[index] for index in live])
                for index, vector in zip(live, vectors, strict=True):
                    dense[index] = vector
            if sparse_ok and live:
                weights = provider.sparse_text([texts[index] for index in live])
                for index, weight in zip(live, weights, strict=True):
                    sparse[index] = weight
            for row, text, vector, weights in zip(chunk, texts, dense, sparse, strict=True):
                pending.append(
                    _embed_row_params(row, text, vector, weights, model, dense_ok, sparse_ok)
                )
            if len(pending) >= flush_rows:
                buffers.put(pending[:flush_rows])
                pending = pending[flush_rows:]
                if errors:
                    raise errors[0]
            embedded += len(chunk)
        if pending:
            buffers.put(pending)
        buffers.put(None)
        thread.join(timeout=600)
        if errors:
            raise errors[0]
    finally:
        buffers.put(None)  # release the writer even on failure
        thread.join(timeout=60)
        writer_conn.close()
    return embedded


def embed_pending(
    conn: psycopg.Connection,
    provider: Any,
    limit: int = 1000,
    shard: int | None = None,
    shards: int = 3,
    write: str = "legacy",
    dsn: str | None = None,
) -> dict[str, Any]:
    """Backfill embedding columns for rows not yet embedded with `provider`.

    `write="legacy"` keeps the original per-row path (A/B baseline);
    `write="async"` batches executemany writes on a background connection so
    DB work hides behind embedding API latency (`dsn` required). Both paths
    are resumable: un-committed rows simply get re-selected on the next run.
    """
    from store.embedding import build_embed_text  # noqa: F401  (kept for callers)

    dense_ok = "dense_text" in provider.spec.supports
    sparse_ok = "sparse_text" in provider.spec.supports
    sql, _params = embed_select_sql(dense_ok, sparse_ok, shard=shard, shards=shards)
    model = provider.spec.model_id

    model_params = [model] * (int(dense_ok) + int(sparse_ok))
    rows = conn.execute(sql, (*model_params, limit)).fetchall()
    if not rows:
        return {"embedded": 0, "model": model}

    if write == "async":
        if not dsn:
            raise ValueError("write=async needs dsn for the writer connection")
        embedded = _embed_rows_async(rows, provider, dsn, model)
    else:
        embedded = _embed_rows_legacy(rows, provider, conn, model)
    return {"embedded": embedded, "model": model}


_CATALOG_GAP_ORDER = {
    "lastmod": "c.lastmod DESC NULLS LAST, c.pexels_id DESC",
    "id": "c.pexels_id DESC",
    "random": "random()",
}


def catalog_gap(
    conn: psycopg.Connection, limit: int = 500, order: str = "lastmod"
) -> list[tuple[int, str | None]]:
    """Catalog IDs still missing from stock_videos (gap-fill worklist)."""
    clause = _CATALOG_GAP_ORDER.get(order)
    if clause is None:
        raise ValueError(f"unknown order: {order}")
    sql = f"""
        SELECT c.pexels_id, c.slug
        FROM catalog_videos c
        WHERE c.missing_since IS NULL
          AND NOT EXISTS (SELECT 1 FROM stock_videos s WHERE s.pexels_id = c.pexels_id)
        ORDER BY {clause}
        LIMIT %s
    """
    return [(int(row[0]), row[1]) for row in conn.execute(sql, (limit,)).fetchall()]


def mark_catalog_missing(conn: psycopg.Connection, pexels_ids: Iterable[int]) -> int:
    """Flag catalog rows whose video is gone (deleted/unlisted); returns newly marked."""
    ids = list(pexels_ids)
    if not ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE catalog_videos
            SET missing_since = NOW()
            WHERE pexels_id = ANY(%s) AND missing_since IS NULL
            """,
            (ids,),
        )
        marked = cur.rowcount
    conn.commit()
    return marked


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
