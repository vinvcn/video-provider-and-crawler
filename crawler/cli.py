"""vpc command line: crawl driver + ingest + ops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crawler import browser, nextdata, sitemap
from crawler.config import load_settings
from crawler.pexels_v3 import (
    build_search_url,
    build_seed_url,
    normalize_video,
    request_headers,
)
from crawler.spool import iter_records, record_attributes, term_slug
from store import db, embedding, migrate, thumbnails


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
        {"key": f"id-{pexels_id}", "id": pexels_id, "slug": slug or ""}
        for pexels_id, slug in rows
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
        pending: list[dict] = []
        for record in iter_records(args.kind):
            seen += 1
            attributes = record_attributes(record)
            if not attributes:
                skipped += 1
                continue
            for attrs in attributes:
                pending.append(normalize_video(attrs))
            if len(pending) >= args.batch:
                upserted += db.upsert_videos(conn, pending)
                pending = []
        upserted += db.upsert_videos(conn, pending)
        result: dict[str, object] = {
            "records": seen,
            "upserted": upserted,
            "skipped": skipped,
        }
        if not args.skip_thumbnails:
            result["thumbnails"] = thumbnails.download_pending(
                conn,
                settings.thumbnail_dir,
                limit=args.thumb_limit,
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
        print(json.dumps(db.embed_pending(conn, provider, limit=args.limit), ensure_ascii=False))
    finally:
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
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--skip-thumbnails", action="store_true")
    p.add_argument("--thumb-limit", type=int, default=2000)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("embed", help="backfill embedding columns (Phase 2 seam)")
    p.add_argument("--model", default="hash", help="embedding provider (hash = local dev provider)")
    p.add_argument("--limit", type=int, default=1000)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("status", help="library + spool counters")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
