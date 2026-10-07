"""vpc command line: crawl driver + ingest + ops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crawler import browser
from crawler.config import load_settings
from crawler.pexels_v3 import (
    build_search_url,
    build_seed_url,
    request_headers,
)
from crawler.spool import iter_records, term_slug
from store import db, migrate, thumbnails


def _state_file(kind: str) -> Path:
    settings = load_settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    return settings.state_dir / f"{kind}.json"


def cmd_migrate(args: argparse.Namespace) -> int:
    migrate.main(args.dsn)
    return 0


def cmd_fetch_seed(args: argparse.Namespace) -> int:
    settings = load_settings()
    spec = {
        "mode": "seed_chain",
        "base_url": build_seed_url(None),
        "headers": request_headers(settings.pexels_secret),
        "out_dir": str(settings.spool_dir / "seed"),
        "state_file": str(_state_file("seed")),
        "max_pages": args.pages,
        "pace_ms": args.pace_ms,
    }
    return browser.run_fetch(spec)


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
    return browser.run_fetch(spec)


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
            body = record.get("body")
            items = body.get("data") if isinstance(body, dict) else None
            if not isinstance(items, list):
                skipped += 1
                continue
            for item in items:
                attrs = item.get("attributes") if isinstance(item, dict) else None
                if not isinstance(attrs, dict) or "id" not in attrs:
                    skipped += 1
                    continue
                from crawler.pexels_v3 import normalize_video

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
    p.set_defaults(func=cmd_fetch_seed)

    p = sub.add_parser("fetch-search", help="crawl keyword search pages")
    p.add_argument("--terms", required=True, help="comma-separated English terms")
    p.add_argument("--pages-per-term", type=int, default=2)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--pace-ms", type=int, default=1500)
    p.set_defaults(func=cmd_fetch_search)

    p = sub.add_parser("ingest", help="spool -> normalize -> upsert -> thumbnails")
    p.add_argument("--kind", choices=["seed", "search"], default=None)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--skip-thumbnails", action="store_true")
    p.add_argument("--thumb-limit", type=int, default=2000)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("status", help="library + spool counters")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
