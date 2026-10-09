"""Runner: execute one strategy over the fixed query set, persist the run.

A run directory is the atomic unit of comparison: `strategy.json` pins exactly
what ran (strategy spec + config hash + code version + materials/queries
versions and hashes) and `per_query.jsonl` holds ranked hits per query with
timing. Judging and scoring both consume these artifacts; nothing here calls
an LLM.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from bench import util
from bench.embedding_client import EmbeddingClient
from bench.materials import Materials
from bench.queries import QuerySet
from bench.strategies import Strategy, config_hash


def run_strategy(
    strategy: Strategy,
    spec: dict,
    materials: Materials,
    queries: QuerySet,
    root: Path,
    depth: int = 50,
    limit: int | None = None,
    embed_client: EmbeddingClient | None = None,
) -> dict:
    """Run `strategy` over every query; returns the run manifest."""
    run_id = "{}-{}-{}".format(
        dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        spec["strategy"],
        config_hash(spec)[:8],
    )
    run_dir = root / "runs" / run_id
    per_query_path = run_dir / "per_query.jsonl"
    if run_dir.exists():
        raise SystemExit(f"run id collision: {run_dir}")

    qids = queries.ordered[:limit] if limit else queries.ordered
    rows = []
    total_embed_ms = 0.0
    for qid in qids:
        record = queries.records[qid]
        result = strategy.search(record.text, depth, dict(record.hard))
        total_embed_ms += result.embed_ms
        rows.append(
            {
                "qid": qid,
                "hits": [[doc_id, score] for doc_id, score in result.hits],
                "search_ms": round(result.search_ms, 3),
                "embed_ms": round(result.embed_ms, 3),
            }
        )
    util.write_jsonl(per_query_path, rows)

    manifest = {
        "run_id": run_id,
        "spec": spec,
        "config_hash": config_hash(spec),
        "materials": {"version": materials.version, "hash": materials.content_hash},
        "queries": {"version": queries.version, "hash": queries.content_hash},
        "depth": depth,
        "n_queries": len(rows),
        "total_embed_ms": round(total_embed_ms, 1),
        "embed": (
            {
                "model": embed_client.config.model,
                "dim": embed_client.config.dim,
                "materials_tokens": _materials_cache_tokens(root, materials, embed_client),
            }
            if embed_client is not None
            else None
        ),
        "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "code": util.code_version(),
    }
    util.write_json(run_dir / "strategy.json", manifest)
    return manifest


def _materials_cache_tokens(root: Path, materials: Materials, client: EmbeddingClient) -> int:
    """Tokens spent on the materials side (one-time, from the cache metadata)."""
    from bench.strategies import _embeddings_cache_dir

    meta_path = (
        _embeddings_cache_dir(root, client.config.model, client.config.dim, materials) / "meta.json"
    )
    if not meta_path.is_file():
        return 0
    return int(util.load_json(meta_path).get("tokens", 0))


def load_run(run_dir: Path) -> dict:
    """Load one run's manifest + per-query rows (hits, timings)."""
    manifest = util.load_json(run_dir / "strategy.json")
    rows = {row["qid"]: row for row in util.read_jsonl(run_dir / "per_query.jsonl")}
    return {"manifest": manifest, "rows": rows, "run_id": str(manifest["run_id"])}


def find_runs(root: Path, materials_version: str, queries_version: str) -> list[Path]:
    """Run dirs pinned to the given materials/queries versions (any strategy)."""
    matches = []
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return matches
    for run_dir in sorted(runs_dir.iterdir()):
        manifest_path = run_dir / "strategy.json"
        if not manifest_path.is_file():
            continue
        manifest = util.load_json(manifest_path)
        if (
            manifest.get("materials", {}).get("version") == materials_version
            and manifest.get("queries", {}).get("version") == queries_version
        ):
            matches.append(run_dir)
    return matches
