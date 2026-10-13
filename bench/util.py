"""Shared helpers: hashing, canonical JSON, JSONL I/O, code version."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any


def hash_key(*parts: object, digest_size: int = 8) -> bytes:
    """Deterministic blake2b digest over the '|' -joined string parts."""
    payload = "|".join(str(part) for part in parts)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).digest()


def hash_hex(*parts: object, digest_size: int = 8) -> str:
    """Hex-encoded `hash_key` (sort keys, cache keys, version stamps)."""
    return hash_key(*parts, digest_size=digest_size).hex()


def canonical_json(obj: Any) -> str:
    """Stable JSON text: sorted keys, tight separators, real unicode."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    """blake2b over the canonical JSON form of `obj`."""
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=16).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSON Lines; returns count written (atomic-ish single write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json(row) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate JSON Lines rows (dicts)."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> Any:
    """Load one JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    """Pretty-write one JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def code_version() -> dict[str, Any]:
    """Current git commit + dirty flag (for run manifests; never a hard gate)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


REPO_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (fraction in [0,1]); 0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered))))
    return float(ordered[rank - 1])


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    """Median; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0
