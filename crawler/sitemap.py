"""Harvest Pexels sitemap shards: the video catalog (coverage oracle) and the
search-query universe.

The indexes under /sitemaps/ are whitelisted in robots.txt and served without
the Cloudflare challenge that guards normal pages. Each index lists .gz shards;
raw shards are spooled to storage/spool/sitemap/ before parsing so every
harvest can be replayed and audited.
"""

from __future__ import annotations

import gzip
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

SITEMAP_BASE = "https://www.pexels.com/sitemaps/en-US"
VIDEO_INDEX_URL = f"{SITEMAP_BASE}/video-sitemap.xml.gz"
QUERY_INDEX_URL = f"{SITEMAP_BASE}/video-search-queries-sitemap.xml.gz"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
)

_SITEMAP_BLOCK_RE = re.compile(r"<sitemap>(.*?)</sitemap>", re.S)
_URL_BLOCK_RE = re.compile(r"<url>(.*?)</url>", re.S)
_LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
_LASTMOD_RE = re.compile(r"<lastmod>([^<]+)</lastmod>")
_VIDEO_RE = re.compile(r"/video/(?P<slug>[^/]+?)-(?P<pid>\d+)/?$")
_QUERY_RE = re.compile(r"/search/videos/(?P<term>[^/]+)/?$")


@dataclass(frozen=True)
class Shard:
    """One shard listed in a sitemap index."""

    loc: str
    lastmod: str | None


@dataclass(frozen=True)
class CatalogVideo:
    """One video entry from the video sitemap."""

    pexels_id: int
    slug: str | None
    lastmod: str | None


def _gunzip(data: bytes) -> bytes:
    """Decompress gzip bytes; pass other payloads through unchanged."""
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _entries(xml_text: str, block_re: re.Pattern[str]) -> Iterator[tuple[str, str | None]]:
    for block in block_re.finditer(xml_text):
        body = block.group(1)
        loc = _LOC_RE.search(body)
        if not loc:
            continue
        lastmod = _LASTMOD_RE.search(body)
        yield loc.group(1), (lastmod.group(1) if lastmod else None)


def parse_index(xml_text: str) -> list[Shard]:
    """Parse a sitemap index into its ordered shard list."""
    return [
        Shard(loc=loc, lastmod=lastmod) for loc, lastmod in _entries(xml_text, _SITEMAP_BLOCK_RE)
    ]


def parse_video_shard(xml_text: str) -> list[CatalogVideo]:
    """Parse a video shard into catalog entries; non-video URLs are skipped."""
    rows: list[CatalogVideo] = []
    for loc, lastmod in _entries(xml_text, _URL_BLOCK_RE):
        match = _VIDEO_RE.search(loc)
        if not match:
            continue
        rows.append(
            CatalogVideo(
                pexels_id=int(match.group("pid")),
                slug=match.group("slug"),
                lastmod=lastmod,
            )
        )
    return rows


def parse_query_shard(xml_text: str) -> list[tuple[str, str | None]]:
    """Parse a search-query shard into (term, lastmod) pairs; terms URL-decoded."""
    rows: list[tuple[str, str | None]] = []
    for loc, lastmod in _entries(xml_text, _URL_BLOCK_RE):
        match = _QUERY_RE.search(loc)
        if not match:
            continue
        term = unquote(match.group("term")).strip()
        if term:
            rows.append((term, lastmod))
    return rows


def _fetch(client: httpx.Client, url: str, attempts: int = 2) -> bytes:
    """GET `url` with one retry; raises the last httpx error when all fail."""
    last: httpx.HTTPError | None = None
    for _ in range(attempts):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(2.0)
    raise last if last is not None else RuntimeError("unreachable")


def iter_shards(
    kind: str,
    spool_dir: Path,
    *,
    use_proxy: bool = True,
    pace_s: float = 1.0,
    limit_shards: int | None = None,
) -> Iterator[tuple[str, list[Any]]]:
    """Download each shard of `kind`, spool its raw bytes, yield (name, entries).

    A shard that keeps failing is skipped with a SHARD-FAIL note; the harvest
    can simply be re-run (upserts are idempotent).
    """
    if kind not in ("videos", "queries"):
        raise ValueError(f"unknown sitemap kind: {kind}")
    index_url = VIDEO_INDEX_URL if kind == "videos" else QUERY_INDEX_URL
    parse = parse_video_shard if kind == "videos" else parse_query_shard
    out_dir = spool_dir / "sitemap"
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        timeout=120.0,
        follow_redirects=True,
        headers={"User-Agent": _UA},
        trust_env=use_proxy,
    ) as client:
        index_xml = _gunzip(_fetch(client, index_url)).decode("utf-8", "replace")
        for number, shard in enumerate(parse_index(index_xml), start=1):
            if limit_shards is not None and number > limit_shards:
                break
            try:
                raw = _fetch(client, shard.loc)
            except httpx.HTTPError as exc:
                print(f"SHARD-FAIL {kind}-{number:02d} {exc}")
                continue
            suffix = ".xml.gz" if raw[:2] == b"\x1f\x8b" else ".xml"
            path = out_dir / f"{kind}-{number:02d}{suffix}"
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(raw)
            tmp.replace(path)
            yield f"{kind}-{number:02d}", parse(_gunzip(raw).decode("utf-8", "replace"))
            time.sleep(pace_s)
