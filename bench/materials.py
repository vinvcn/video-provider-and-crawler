"""Materials set: frozen, deterministic, stratified sample of stock_videos.

The materials set is one of the two fixed poles of every benchmark comparison
("materials stay fixed, strategies vary"). It is a snapshot: rows are copied
out of stock_videos into a JSONL artifact with a content hash, so later crawl
activity can never change what a run saw. See docs/bench-harness-design-2026-10-09.md §2.

Sampling rules (all pure functions, unit-tested):
- population  = rows with non-empty text (title OR description OR tags);
- stratum     = orientation (landscape/portrait/square/unknown) x 4K (uhd/sd)
                x duration bucket (lt5 / b5_15 / b16_60 / gt60 / unknown);
- quota       = min(100, rows) floor per non-empty cell, the rest distributed
                proportionally (largest-remainder), never above cell capacity;
- selection   = within a cell, ascending blake2b(pexels_id) order (ties by id);
- v2 superset = each cell records its hash cutoff; re-running the same rule on
                a grown population with "hash <= cutoff" yields a superset of v1.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bench import util
from store.embedding import build_embed_text

RULE_VERSION = "materials-v1"
STRATUM_FLOOR = 100
UHD_WIDTH = 3840

ORIENTATIONS = ("landscape", "portrait", "square", "unknown")
UHD_KEYS = ("uhd", "sd")
DURATION_BUCKETS = ("lt5", "b5_15", "b16_60", "gt60", "unknown")

# Hard-filterable columns kept in the snapshot (evaluated in-memory by strategies).
_ROW_FIELDS = (
    "pexels_id",
    "title",
    "description",
    "tags",
    "duration",
    "width",
    "height",
    "orientation",
    "license",
)

_POPULATION_SQL = """
SELECT pexels_id,
       COALESCE(orientation, 'unknown'),
       CASE WHEN COALESCE(width, 0) >= 3840 THEN 'uhd' ELSE 'sd' END,
       CASE
           WHEN duration IS NULL THEN 'unknown'
           WHEN duration < 5 THEN 'lt5'
           WHEN duration <= 15 THEN 'b5_15'
           WHEN duration <= 60 THEN 'b16_60'
           ELSE 'gt60'
       END
FROM stock_videos
WHERE COALESCE(title, '') <> ''
   OR COALESCE(description, '') <> ''
   OR cardinality(tags) > 0
"""

_ROW_SQL = """
SELECT pexels_id, title, description, tags, duration, width, height, orientation, license
FROM stock_videos
WHERE pexels_id = ANY(%s)
ORDER BY pexels_id
"""


def stratum_key(
    orientation: str | None, width: int | None, duration: int | None
) -> tuple[str, str, str]:
    """(orientation, uhd, duration-bucket) cell key for one row."""
    orient = orientation if orientation in ORIENTATIONS else "unknown"
    uhd = "uhd" if (width or 0) >= UHD_WIDTH else "sd"
    if duration is None:
        bucket = "unknown"
    elif duration < 5:
        bucket = "lt5"
    elif duration <= 15:
        bucket = "b5_15"
    elif duration <= 60:
        bucket = "b16_60"
    else:
        bucket = "gt60"
    return (orient, uhd, bucket)


def _largest_remainder(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Split `total` exactly across keys proportionally to weights (ties by key)."""
    if not weights or total <= 0:
        return {key: 0 for key in weights}
    mass = sum(weights.values())
    if mass <= 0:  # degenerate: split evenly
        weights = {key: 1.0 for key in weights}
        mass = float(len(weights))
    exact = {key: total * weight / mass for key, weight in weights.items()}
    shares = {key: int(value) for key, value in exact.items()}
    leftover = total - sum(shares.values())
    by_fraction = sorted(exact, key=lambda k: (-(exact[k] - shares[k]), k))
    for key in by_fraction[:leftover]:
        shares[key] += 1
    return shares


def compute_quotas(cell_rows: Mapping[tuple[str, str, str], int], n_total: int, floor: int) -> dict:
    """Per-cell quotas: floor min(floor, rows) + proportional remainder.

    Capacity-capped cells hand their excess back, which is redistributed over
    cells that still have headroom (largest-remainder, deterministic).
    Returns {cell_key: quota}.
    """
    cells = {key: int(rows) for key, rows in cell_rows.items() if rows > 0}
    if not cells or n_total <= 0:
        return {key: 0 for key in cell_rows}

    reserves = {key: min(floor, rows) for key, rows in cells.items()}
    reserve_total = sum(reserves.values())
    if reserve_total >= n_total:
        scaled = _largest_remainder({str(key): value for key, value in reserves.items()}, n_total)
        return {key: scaled.get(str(key), 0) for key in cells}

    shares = _largest_remainder(
        {str(key): rows for key, rows in cells.items()}, n_total - reserve_total
    )
    quotas = {key: reserves[key] + shares.get(str(key), 0) for key in cells}

    # Redistribute excess from cells over their own capacity.
    for _ in range(len(quotas) + 1):
        excess = sum(max(0, quotas[key] - cells[key]) for key in quotas)
        open_cells = {
            str(key): cells[key] - quotas[key] for key in quotas if quotas[key] < cells[key]
        }
        if excess <= 0 or not open_cells:
            break
        for key in quotas:
            quotas[key] = min(quotas[key], cells[key])
        top_up = _largest_remainder(
            {key: room for key, room in open_cells.items() if room > 0},
            min(excess, sum(room for room in open_cells.values() if room > 0)),
        )
        if not top_up:
            break
        for key in quotas:
            quotas[key] += top_up.get(str(key), 0)
    for key in quotas:  # final clamp
        quotas[key] = min(quotas[key], cells[key])
    return quotas


@dataclass(frozen=True)
class MaterialRow:
    """One materials row (snapshot of the filterable + embeddable fields)."""

    doc_id: int
    title: str
    description: str
    tags: tuple[str, ...]
    duration: int | None
    width: int | None
    height: int | None
    orientation: str | None
    license: str | None
    embed_text: str
    doc_hash: str

    def to_json(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation,
            "license": self.license,
            "embed_text": self.embed_text,
            "doc_hash": self.doc_hash,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> MaterialRow:
        return cls(
            doc_id=int(data["doc_id"]),
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            tags=tuple(str(tag) for tag in (data.get("tags") or [])),
            duration=(int(data["duration"]) if data.get("duration") is not None else None),
            width=(int(data["width"]) if data.get("width") is not None else None),
            height=(int(data["height"]) if data.get("height") is not None else None),
            orientation=(str(data["orientation"]) if data.get("orientation") is not None else None),
            license=(str(data["license"]) if data.get("license") is not None else None),
            embed_text=str(data.get("embed_text") or ""),
            doc_hash=str(data.get("doc_hash") or ""),
        )


def build_materials(dsn: str | None, n_total: int, version: str, root: Path) -> dict:
    """Sample + freeze the materials snapshot; returns the manifest.

    Only ever reads stock_videos. The output directory is written once per
    version; an existing version is never silently overwritten.
    """
    import psycopg

    from store.db import DEFAULT_DSN

    out_dir = root / "materials" / version
    rows_path = out_dir / "rows.jsonl"
    manifest_path = out_dir / "manifest.json"
    if rows_path.exists():
        raise SystemExit(f"materials version {version!r} already exists: {rows_path}")

    with psycopg.connect(dsn or DEFAULT_DSN) as conn:
        population: list[tuple[int, str, str, str]] = conn.execute(_POPULATION_SQL).fetchall()
        cell_rows: dict[tuple[str, str, str], list[tuple[bytes, int]]] = {}
        for pexels_id, orient, uhd, bucket in population:
            key = (str(orient), str(uhd), str(bucket))
            cell_rows.setdefault(key, []).append((util.hash_key(pexels_id), int(pexels_id)))
        quotas = compute_quotas(
            {key: len(rows) for key, rows in cell_rows.items()}, n_total, STRATUM_FLOOR
        )

        selected: list[int] = []
        cells_report: list[dict] = []
        for key in sorted(cell_rows):
            candidates = sorted(cell_rows[key])
            quota = quotas.get(key, 0)
            picked = candidates[:quota]
            selected.extend(doc_id for _, doc_id in picked)
            cutoff = {"hash": picked[-1][0].hex(), "doc_id": picked[-1][1]} if picked else None
            cells_report.append(
                {
                    "cell": {"orientation": key[0], "uhd": key[1], "duration": key[2]},
                    "population": len(candidates),
                    "quota": quota,
                    "sampled": len(picked),
                    "cutoff": cutoff,
                }
            )

        full_rows = conn.execute(_ROW_SQL, (selected,)).fetchall()

    rows = sorted((_row_from_db(row) for row in full_rows), key=lambda row: row.doc_id)
    _validate_sample(rows, selected, quotas)

    manifest = {
        "version": version,
        "rule": RULE_VERSION,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_target": n_total,
        "n": len(rows),
        "population_total": len(population),
        "population_filter": "non-empty title/description/tags",
        "stratum_floor": STRATUM_FLOOR,
        "cells": cells_report,
        "tags_stats": _tags_stats(rows),
        "content_hash": util.content_hash([row.to_json() for row in rows]),
        "code": util.code_version(),
    }
    util.write_jsonl(rows_path, (row.to_json() for row in rows))
    util.write_json(manifest_path, manifest)
    return manifest


def _row_from_db(row: Sequence[object]) -> MaterialRow:
    pexels_id, title, description, tags, duration, width, height, orientation, license = row
    base = {
        "pexels_id": int(pexels_id),
        "title": title or "",
        "description": description or "",
        "tags": [str(tag) for tag in (tags or [])],
        "duration": duration,
        "width": width,
        "height": height,
        "orientation": orientation,
        "license": license,
    }
    embed_text = build_embed_text(base)
    doc_hash = util.content_hash({key: base[key] for key in _ROW_FIELDS})
    return MaterialRow(
        doc_id=int(pexels_id),
        title=str(base["title"]),
        description=str(base["description"]),
        tags=tuple(base["tags"]),
        duration=int(duration) if duration is not None else None,
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        orientation=str(orientation) if orientation is not None else None,
        license=str(license) if license is not None else None,
        embed_text=embed_text,
        doc_hash=doc_hash,
    )


def _validate_sample(rows: Sequence[MaterialRow], selected: Sequence[int], quotas: dict) -> None:
    if len(rows) != len(selected):
        raise SystemExit(
            f"sampling invariant broken: {len(selected)} ids selected, {len(rows)} rows fetched"
        )
    if sum(quotas.values()) != len(selected):
        raise SystemExit("quota invariant broken: quotas do not sum to the sample size")


def _tags_stats(rows: Sequence[MaterialRow]) -> dict:
    counts = sorted(len(row.tags) for row in rows)
    n = len(counts)
    if not n:
        return {"median": 0, "p90": 0, "empty_rate": 0.0}
    return {
        "median": counts[n // 2],
        "p90": util.percentile([float(value) for value in counts], 0.9),
        "empty_rate": sum(1 for value in counts if value == 0) / n,
    }


class Materials:
    """In-memory view of a frozen materials snapshot (what strategies see)."""

    def __init__(self, version: str, root: Path) -> None:
        self.version = version
        out_dir = root / "materials" / version
        self.manifest = util.load_json(out_dir / "manifest.json")
        self.rows: dict[int, MaterialRow] = {
            row.doc_id: row
            for row in (
                MaterialRow.from_json(data) for data in util.read_jsonl(out_dir / "rows.jsonl")
            )
        }
        self.ids = sorted(self.rows)
        self.content_hash = str(self.manifest["content_hash"])

    def passes_filters(self, row: MaterialRow, hard: Mapping[str, object]) -> bool:
        """Evaluate the query-side hard constraints against one row."""
        orientation = hard.get("orientation")
        if orientation and (row.orientation or "unknown") != orientation:
            return False
        duration_min = hard.get("duration_min")
        if duration_min is not None and (row.duration is None or row.duration < int(duration_min)):
            return False
        duration_max = hard.get("duration_max")
        if duration_max is not None and (row.duration is None or row.duration > int(duration_max)):
            return False
        min_width = hard.get("min_width")
        if min_width is not None and (row.width is None or row.width < int(min_width)):
            return False
        return True

    def filter_ids(self, hard: Mapping[str, object]) -> list[int]:
        """doc_ids passing the hard constraints (ascending)."""
        return [doc_id for doc_id in self.ids if self.passes_filters(self.rows[doc_id], hard)]
