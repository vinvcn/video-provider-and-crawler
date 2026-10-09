"""Shared fixtures: fabricate materials/queries/run artifacts without a DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import util
from bench.materials import Materials
from bench.queries import QuerySet


def material_row(
    doc_id: int,
    title: str,
    *,
    tags: tuple[str, ...] = (),
    description: str = "",
    duration: int | None = 20,
    width: int = 1920,
    orientation: str = "landscape",
) -> dict:
    embed_text = " — ".join(part for part in (title, description, ", ".join(tags)) if part)
    return {
        "doc_id": doc_id,
        "title": title,
        "description": description,
        "tags": list(tags),
        "duration": duration,
        "width": width,
        "height": 1080,
        "orientation": orientation,
        "license": "Pexels",
        "embed_text": embed_text,
        "doc_hash": util.hash_hex(doc_id, embed_text)[:16],
    }


def write_materials(root: Path, version: str, rows: list[dict]) -> Materials:
    out = root / "materials" / version
    out.mkdir(parents=True, exist_ok=True)
    util.write_jsonl(out / "rows.jsonl", rows)
    util.write_json(
        out / "manifest.json",
        {"version": version, "content_hash": util.content_hash(rows)},
    )
    return Materials(version, root)


def query_record(
    text: str,
    *,
    qid: str | None = None,
    source: str = "designed",
    category: str = "concept",
    lang: str = "en",
    facets: tuple[str, ...] = (),
    hard: dict | None = None,
    difficulty: str = "medium",
    split: str = "train",
) -> dict:
    from bench.queries import make_qid

    return {
        "qid": qid or make_qid(text),
        "text": text,
        "source": source,
        "category": category,
        "lang": lang,
        "facets": list(facets),
        "hard": hard or {},
        "difficulty": difficulty,
        "match_docs": 0,
        "split": split,
    }


def write_queries(root: Path, version: str, records: list[dict]) -> QuerySet:
    out = root / "queries" / version
    out.mkdir(parents=True, exist_ok=True)
    util.write_jsonl(out / "queries.jsonl", records)
    util.write_json(
        out / "manifest.json",
        {"version": version, "content_hash": util.content_hash(records)},
    )
    return QuerySet(version, root)


def write_run(
    root: Path,
    run_id: str,
    materials: Materials,
    queries: QuerySet,
    hits_by_qid: dict[str, list[list]],
    spec_strategy: str = "bm25-query",
) -> Path:
    from bench.strategies import config_hash, strategy_config

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = strategy_config(spec_strategy, None)
    util.write_jsonl(
        run_dir / "per_query.jsonl",
        [
            {"qid": qid, "hits": hits, "search_ms": 1.0, "embed_ms": 0.0}
            for qid, hits in hits_by_qid.items()
        ],
    )
    util.write_json(
        run_dir / "strategy.json",
        {
            "run_id": run_id,
            "spec": spec,
            "config_hash": config_hash(spec),
            "materials": {"version": materials.version, "hash": materials.content_hash},
            "queries": {"version": queries.version, "hash": queries.content_hash},
        },
    )
    return run_dir


@pytest.fixture()
def bench_root(tmp_path: Path) -> Path:
    return tmp_path / "bench"
