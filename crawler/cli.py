"""vpc command line: crawl driver + ingest + ops."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from crawler import browser, nextdata, sitemap
from crawler.config import load_settings
from crawler.pexels_v3 import (
    build_search_url,
    build_seed_url,
    normalize_video,
    request_headers,
)
from crawler.spool import (
    iter_records,
    load_ingest_marks,
    record_attributes,
    record_missing_id,
    save_ingest_marks,
    term_slug,
)
from store import db, embedding, migrate, thumbnails

try:  # bench is a pure addition; the crawl CLI stays functional without it
    from bench import cli as bench_cli
except ImportError:  # pragma: no cover - bench needs numpy (main dependency)
    bench_cli = None  # type: ignore[assignment]


def _state_file(kind: str) -> Path:
    settings = load_settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    return settings.state_dir / f"{kind}.json"


def cmd_migrate(args: argparse.Namespace) -> int:
    migrate.main(args.dsn)
    return 0


def cmd_fetch_seed(args: argparse.Namespace) -> int:
    settings = load_settings()
    kind = "seed-head" if args.head else "seed"
    spec = {
        "mode": "seed_chain",
        "base_url": build_seed_url(None),
        "headers": request_headers(settings.pexels_secret),
        "out_dir": str(settings.spool_dir / kind),
        "state_file": str(_state_file(kind)),
        "max_pages": args.pages,
        "pace_ms": args.pace_ms,
        "head": bool(args.head),
        "until": args.until,
    }
    return browser.run_fetch(spec, attempts=args.retries)


def cmd_fetch_search(args: argparse.Namespace) -> int:
    settings = load_settings()
    terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    if not terms:
        print("no terms given")
        return 2
    targets = []
    for term in terms:
        for page in range(1, args.pages_per_term + 1):
            targets.append(
                {
                    "key": f"search-{term_slug(term)}-p{page}",
                    "term": term,
                    "url": build_search_url(term, page),
                }
            )
    spec = {
        "mode": "search_list",
        "targets": targets,
        "headers": request_headers(settings.pexels_secret),
        "out_dir": str(settings.spool_dir / "search"),
        "state_file": str(_state_file("search")),
        "pool": args.pool,
        "pace_ms": args.pace_ms,
    }
    return browser.run_fetch(spec, attempts=args.retries)


def cmd_fetch_ids(args: argparse.Namespace) -> int:
    settings = load_settings()
    conn = db.connect(settings.db_dsn)
    try:
        rows = db.catalog_gap(conn, limit=args.limit, order=args.order)
    finally:
        conn.close()
    if not rows:
        print(json.dumps({"targets": 0, "note": "catalog gap is empty"}, ensure_ascii=False))
        return 0
    targets = [
        {"key": f"id-{pexels_id}", "id": pexels_id, "slug": slug or ""} for pexels_id, slug in rows
    ]
    spec = {
        "mode": "id_list",
        "targets": targets,
        "url_template": nextdata.DATA_URL_TEMPLATE,
        "out_dir": str(settings.spool_dir / "ids"),
        "state_file": str(_state_file("ids")),
        "pool": args.pool,
        "pace_ms": args.pace_ms,
    }
    return browser.run_fetch(spec, attempts=args.retries)


def cmd_fetch_sitemaps(args: argparse.Namespace) -> int:
    settings = load_settings()
    conn = db.connect(settings.db_dsn)
    try:
        kinds = ["videos", "queries"] if args.kind == "all" else [args.kind]
        summary: dict[str, dict[str, int]] = {}
        for kind in kinds:
            before = db.count_catalog(conn, kind)
            shards = 0
            entries = 0
            for name, rows in sitemap.iter_shards(
                kind,
                settings.spool_dir,
                use_proxy=settings.use_proxy,
                pace_s=args.pace_ms / 1000.0,
                limit_shards=args.limit_shards,
            ):
                if kind == "videos":
                    db.upsert_catalog_videos(
                        conn, [(row.pexels_id, row.slug, row.lastmod) for row in rows]
                    )
                else:
                    db.upsert_catalog_queries(conn, rows)
                shards += 1
                entries += len(rows)
                print(f"SHARD {name} n={len(rows)}")
            after = db.count_catalog(conn, kind)
            summary[kind] = {
                "shards": shards,
                "entries": entries,
                "new": after - before,
                "total": after,
            }
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        conn.close()
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    seen = 0
    upserted = 0
    skipped = 0
    conn = db.connect(settings.db_dsn)
    try:
        marks = {} if args.full else load_ingest_marks()
        run_started_ns = time.time_ns()
        touched: set[str] = set()
        pending: list[dict] = []
        missing: list[int] = []
        missing_marks = 0
        for record in iter_records(args.kind, since_ns=None if args.full else marks):
            seen += 1
            touched.add(Path(record["_path"]).parent.name)
            gone_id = record_missing_id(record)
            if gone_id is not None:
                missing.append(gone_id)
            attributes = record_attributes(record)
            if not attributes:
                skipped += 1
            else:
                for attrs in attributes:
                    pending.append(normalize_video(attrs))
            if len(pending) >= args.batch:
                upserted += db.upsert_videos(conn, pending)
                pending = []
            if len(missing) >= args.batch:
                missing_marks += db.mark_catalog_missing(conn, missing)
                missing = []
        upserted += db.upsert_videos(conn, pending)
        missing_marks += db.mark_catalog_missing(conn, missing)
        if not args.full:
            # Floors advance only after a completed pass, so a crash mid-run
            # replays the unprocessed files on the next run (upserts are idempotent).
            for kind_name in touched:
                marks[kind_name] = run_started_ns
            save_ingest_marks(marks)
        result: dict[str, object] = {
            "records": seen,
            "upserted": upserted,
            "skipped": skipped,
            "catalog_missing_marked": missing_marks,
            "mode": "full" if args.full else "incremental",
            "kinds": sorted(touched),
        }
        if not args.skip_thumbnails:
            result["thumbnails"] = thumbnails.download_pending(
                conn,
                settings.thumbnail_dir,
                limit=args.thumb_limit,
                concurrency=args.thumb_concurrency,
                use_proxy=settings.use_proxy,
            )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        conn.close()
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    settings = load_settings()
    provider = embedding.get_provider(args.model)
    conn = db.connect(settings.db_dsn)
    try:
        result = db.embed_pending(
            conn,
            provider,
            limit=args.limit,
            shard=args.shard,
            shards=args.shards,
            write=args.write,
            dsn=settings.db_dsn,
        )
        result["shard"] = args.shard
        result["shards"] = args.shards
        result["write"] = args.write
        print(json.dumps(result, ensure_ascii=False))
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        conn.close()
    return 0


def cmd_bm25_fit(args: argparse.Namespace) -> int:
    from pathlib import Path

    from store.bm25 import Bm25Model
    from store.embedding import build_embed_text

    settings = load_settings()
    model = Bm25Model(idf_side=args.idf_side)
    conn = db.connect(settings.db_dsn)
    try:
        fitted = 0
        # Server-side cursor: streams 665k rows without holding them in RAM.
        with conn.cursor(name="vpc_bm25_fit") as cur:
            cur.itersize = 5000
            cur.execute("SELECT title, description, tags FROM stock_videos")
            while True:
                rows = cur.fetchmany(cur.itersize)
                if not rows:
                    break
                model.fit(
                    build_embed_text({"title": t, "description": d, "tags": g}) for t, d, g in rows
                )
                fitted += len(rows)
        out = Path(args.out) if args.out else settings.state_dir / "bm25-corpus.json"
        model.save(out)
        print(
            json.dumps(
                {
                    "fitted": fitted,
                    "idf_side": model.idf_side,
                    "terms": len(model.term_ids),
                    "out": str(out),
                },
                ensure_ascii=False,
            )
        )
    finally:
        conn.close()
    return 0


def _search_hard_filter(args: argparse.Namespace) -> tuple[list[str], list[object]]:
    """WHERE fragments + params for the structured (hard) constraints."""
    where: list[str] = []
    params: list[object] = []
    if args.orientation:
        where.append("orientation = %s")
        params.append(args.orientation)
    if args.min_duration is not None:
        where.append("duration >= %s")
        params.append(args.min_duration)
    if args.max_duration is not None:
        where.append("duration <= %s")
        params.append(args.max_duration)
    if args.min_width is not None:
        where.append("width >= %s")
        params.append(args.min_width)
    if args.license:
        where.append("license = %s")
        params.append(args.license)
    return where, params


def cmd_search(args: argparse.Namespace) -> int:
    from store.bm25 import Bm25Model
    from store.embedding import to_sparsevec_literal, to_vector_literal

    settings = load_settings()
    dense_provider = embedding.get_provider(args.dense_model)
    sparse_provider = embedding.get_provider("bm25")
    bm25: Bm25Model = sparse_provider.model  # type: ignore[assignment]
    conn = db.connect(settings.db_dsn)
    try:
        # GUC assignments do not accept bind parameters; ef_search is an int.
        conn.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")
        conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
        hard, hard_params = _search_hard_filter(args)
        hard_sql = (" AND ".join(hard) + " AND ") if hard else ""

        dense_ms = 0.0
        sparse_ms = 0.0
        dense_hits: list[int] = []
        sparse_hits: list[int] = []

        import time as _time

        start = _time.perf_counter()
        query_vector = dense_provider.dense_text([args.q])[0]
        dense_ms = (_time.perf_counter() - start) * 1000

        start = _time.perf_counter()
        rows = conn.execute(
            f"""
            SELECT pexels_id FROM stock_videos
            WHERE {hard_sql}embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (*hard_params, to_vector_literal(query_vector), args.depth),
        ).fetchall()
        dense_hits = [row[0] for row in rows]
        dense_ms += (_time.perf_counter() - start) * 1000

        weights = bm25.query_weights(args.q)
        if weights:
            start = _time.perf_counter()
            # The ORDER BY expression must textually match the expression
            # index (sparse_embedding::sparsevec(1000000000)) so the planner
            # uses it; a bare column reference would seq-scan.
            rows = conn.execute(
                f"""
                SELECT pexels_id FROM stock_videos
                WHERE {hard_sql}sparse_embedding IS NOT NULL
                ORDER BY sparse_embedding::sparsevec(1000000000) <#> %s::sparsevec
                LIMIT %s
                """,
                (*hard_params, to_sparsevec_literal(weights), args.depth),
            ).fetchall()
            sparse_hits = [row[0] for row in rows]
            sparse_ms += (_time.perf_counter() - start) * 1000

        # RRF (k=60): rank-position voting, no score calibration needed.
        rrf_k = 60
        scores: dict[int, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, pexels_id in enumerate(hits, start=1):
                scores[pexels_id] = scores.get(pexels_id, 0.0) + 1.0 / (rrf_k + rank)
        top_ids = sorted(scores, key=lambda pid: (-scores[pid], pid))[: args.k]
        if not top_ids:
            print(
                json.dumps(
                    {
                        "query": args.q,
                        "hits": [],
                        "arms": {"dense": len(dense_hits), "sparse": len(sparse_hits)},
                    }
                )
            )
            return 0

        rows = conn.execute(
            """
            SELECT pexels_id, title, duration, width, height, orientation,
                   thumbnail_path, thumbnail_url
            FROM stock_videos WHERE pexels_id = ANY(%s)
            """,
            (top_ids,),
        ).fetchall()
        by_id = {row[0]: row for row in rows}
        hits = [
            {
                "pexels_id": pid,
                "rrf": round(scores[pid], 6),
                "title": by_id[pid][1],
                "duration": by_id[pid][2],
                "width": by_id[pid][3],
                "height": by_id[pid][4],
                "orientation": by_id[pid][5],
                "thumbnail_path": by_id[pid][6],
                "thumbnail_url": by_id[pid][7],
            }
            for pid in top_ids
            if pid in by_id
        ]
        print(
            json.dumps(
                {
                    "query": args.q,
                    "hits": hits,
                    "arms": {"dense": len(dense_hits), "sparse": len(sparse_hits)},
                    "latency_ms": {"dense": round(dense_ms, 1), "sparse": round(sparse_ms, 1)},
                },
                ensure_ascii=False,
            )
        )
    finally:
        close = getattr(dense_provider, "close", None)
        if callable(close):
            close()
        conn.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = load_settings()
    conn = db.connect(settings.db_dsn)
    try:
        print(json.dumps(db.stats(conn), ensure_ascii=False, indent=2))
    finally:
        conn.close()
    spool_files = list(settings.spool_dir.rglob("*.json")) if settings.spool_dir.exists() else []
    print(f"spool files: {len(spool_files)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vpc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="apply forward-only SQL migrations")
    p.add_argument("--dsn")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("fetch-seed", help="crawl the popular-feed seed chain")
    p.add_argument("--pages", type=int, default=100)
    p.add_argument("--pace-ms", type=int, default=1500)
    p.add_argument("--until", help="stop once the chain cursor reaches this ISO timestamp")
    p.add_argument(
        "--head",
        action="store_true",
        help="start from the live feed head and track the head marker (daily incremental mode)",
    )
    p.add_argument(
        "--retries", type=int, default=1, help="retry aborted runs while progress is made"
    )
    p.set_defaults(func=cmd_fetch_seed)

    p = sub.add_parser("fetch-search", help="crawl keyword search pages")
    p.add_argument("--terms", required=True, help="comma-separated English terms")
    p.add_argument("--pages-per-term", type=int, default=2)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--pace-ms", type=int, default=1500)
    p.add_argument(
        "--retries", type=int, default=1, help="retry aborted runs while progress is made"
    )
    p.set_defaults(func=cmd_fetch_search)

    p = sub.add_parser("fetch-sitemaps", help="harvest catalog IDs and the search-query universe")
    p.add_argument("--kind", choices=["videos", "queries", "all"], default="all")
    p.add_argument("--pace-ms", type=int, default=1000)
    p.add_argument("--limit-shards", type=int)
    p.set_defaults(func=cmd_fetch_sitemaps)

    p = sub.add_parser("fetch-ids", help="per-video metadata via the Next data route (gap fill)")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--order", choices=["lastmod", "id", "random"], default="lastmod")
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--pace-ms", type=int, default=1000)
    p.add_argument(
        "--retries", type=int, default=1, help="retry aborted runs while progress is made"
    )
    p.set_defaults(func=cmd_fetch_ids)

    p = sub.add_parser("ingest", help="spool -> normalize -> upsert -> thumbnails")
    p.add_argument("--kind", choices=["seed", "search", "ids"], default=None)
    p.add_argument(
        "--full",
        action="store_true",
        help="replay the entire spool, ignoring ingest watermarks (state/ingest.json)",
    )
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--skip-thumbnails", action="store_true")
    p.add_argument("--thumb-limit", type=int, default=2000)
    p.add_argument(
        "--thumb-concurrency",
        type=int,
        default=8,
        help="parallel thumbnail downloads (latency-bound: 8 is ~2.6/s, 96 ~30/s)",
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("embed", help="backfill embedding columns (Phase 2 seam)")
    p.add_argument(
        "--model",
        default="hash",
        help="embedding provider (hash | bge-m3 | bm25)",
    )
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument(
        "--shard",
        type=int,
        help="only rows with pexels_id %% shards == shard (incremental index builds)",
    )
    p.add_argument("--shards", type=int, default=3, help="shard count for --shard")
    p.add_argument(
        "--write",
        choices=["legacy", "async"],
        default="legacy",
        help="write path: legacy per-row UPDATEs | async batched executemany "
        "(background connection, synchronous_commit=off, wal_compression=on)",
    )
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("bm25-fit", help="fit BM25 corpus statistics for the sparse arm")
    p.add_argument(
        "--idf-side",
        choices=["both", "query", "doc"],
        default="query",
        help="where the idf factor lives (query = textbook BM25, the eval winner)",
    )
    p.add_argument("--out", help="model output path (default storage/state/bm25-corpus.json)")
    p.set_defaults(func=cmd_bm25_fit)

    p = sub.add_parser("search", help="hybrid retrieval over the indexed library")
    p.add_argument("--q", required=True, help="natural-language query")
    p.add_argument("--k", type=int, default=30, help="results to return")
    p.add_argument("--depth", type=int, default=100, help="per-arm retrieval depth")
    p.add_argument("--ef-search", type=int, default=100, help="hnsw.ef_search (recall dial)")
    p.add_argument("--dense-model", default="bge-m3", help="dense query-side provider")
    p.add_argument("--orientation", choices=["landscape", "portrait", "square"])
    p.add_argument("--min-duration", type=int)
    p.add_argument("--max-duration", type=int)
    p.add_argument("--min-width", type=int)
    p.add_argument("--license")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("status", help="library + spool counters")
    p.set_defaults(func=cmd_status)

    if bench_cli is not None:
        bench_cli.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
