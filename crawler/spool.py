"""Spool: raw JSON responses on disk (replayable, auditable).

The browser-side script (crawler/browser_scripts/fetch_batch.py) writes the
same record shape directly; this module provides the shared conventions plus
read/write helpers for tests and the ingest path.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from crawler.config import load_settings

_SLUG_RE = re.compile(r"[^a-z0-9]+")
INGEST_MARKS_FILE = "ingest.json"


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


_MISSING_KEY_RE = re.compile(r"^id-(\d+)$")


def record_missing_id(record: dict[str, Any]) -> int | None:
    """Pexels id for an id-kind record that says the video is gone.

    Only the data route's `missing_medium` redirect proves a video is gone; a
    bare 404 can also mean a rotated buildId, which once mislabeled thousands
    of live videos (2026-10-10 incident). Records with attributes are successes.
    """
    if "attributes" in record:
        return None
    match = _MISSING_KEY_RE.match(str(record.get("key") or ""))
    if not match:
        return None
    if "missing_medium" not in str(record.get("error") or ""):
        return None
    return int(match.group(1))


def iter_records(
    kind: str | None = None,
    since_ns: Mapping[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield spool records (for `ingest`); file-name order within each kind.

    `since_ns` maps a spool kind to an mtime floor in nanoseconds: files that
    have not changed since that floor are skipped, so a run only replays what
    is new (or was rewritten). `None` replays everything.
    """
    spool_dir = load_settings().spool_dir
    if kind:
        kinds = [kind]
    elif spool_dir.exists():
        kinds = sorted(p.name for p in spool_dir.iterdir() if p.is_dir())
    else:
        kinds = []
    for k in kinds:
        floor = since_ns.get(k, 0) if since_ns is not None else 0
        for path in sorted((spool_dir / k).glob("*.json")):
            try:
                if since_ns is not None and path.stat().st_mtime_ns <= floor:
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            record["_path"] = str(path)
            yield record


def load_ingest_marks() -> dict[str, int]:
    """Per-kind spool consumption floors (mtime_ns); empty if never written."""
    path = load_settings().state_dir / INGEST_MARKS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if isinstance(v, int)}


def save_ingest_marks(marks: Mapping[str, int]) -> None:
    """Atomically persist ingest floors (delete the file to force a replay)."""
    state_dir = load_settings().state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / INGEST_MARKS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(marks), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
