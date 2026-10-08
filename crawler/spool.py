"""Spool: raw JSON responses on disk (replayable, auditable).

The browser-side script (crawler/browser_scripts/fetch_batch.py) writes the
same record shape directly; this module provides the shared conventions plus
read/write helpers for tests and the ingest path.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from crawler.config import load_settings

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def term_slug(term: str, max_len: int = 60) -> str:
    """Filesystem-safe deterministic slug for a search term."""
    slug = _SLUG_RE.sub("-", str(term).lower()).strip("-")
    return slug[:max_len] or "term"


def write_record(kind: str, key: str, url: str, payload: dict[str, Any]) -> Path:
    """Atomically write one spool record; returns its path."""
    directory = load_settings().spool_dir / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    tmp = path.with_suffix(".json.tmp")
    record = {"key": key, "url": url, "fetched_at": time.time(), **payload}
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def record_attributes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Attribute objects inside one spool record, for every payload shape.

    List payloads (`body.data[].attributes`, from feed and search) yield their
    items; per-id payloads (`attributes`, from the Next data route) yield one.
    """
    single = record.get("attributes")
    if isinstance(single, dict) and "id" in single:
        return [single]
    body = record.get("body")
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        attributes = item.get("attributes") if isinstance(item, dict) else None
        if isinstance(attributes, dict) and "id" in attributes:
            out.append(attributes)
    return out


def iter_records(kind: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield spool records (for `ingest`); file-name order within each kind."""
    spool_dir = load_settings().spool_dir
    if kind:
        kinds = [kind]
    elif spool_dir.exists():
        kinds = sorted(p.name for p in spool_dir.iterdir() if p.is_dir())
    else:
        kinds = []
    for k in kinds:
        for path in sorted((spool_dir / k).glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            record["_path"] = str(path)
            yield record
