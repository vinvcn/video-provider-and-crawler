"""Export the bench's raw data as flat files (queries, judgments, run hits).

Everything the leaderboard is built from, as inspectable CSVs under
storage/bench/reports/: the query data points, every judgment (with grade and
reason), and each run's ranked hits. The markdown leaderboard summarizes
these; the exports are the audit trail behind it.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from bench import util
from bench.judge import label_records
from bench.queries import QuerySet
from bench.run import find_runs, load_run

MAX_HITS_EXPORT = 50  # every stored hit rank (matches the default run depth)


def _safe(name: str) -> str:
    """Filesystem-safe version stamp."""
    return name.replace("@", "_at_").replace("/", "_")


def export_queries(queries: QuerySet, out_dir: Path) -> Path:
    """One row per query data point, with its metadata and split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"export-queries-{_safe(queries.version)}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "qid",
                "text",
                "source",
                "category",
                "lang",
                "facets",
                "hard",
                "difficulty",
                "match_docs",
                "split",
            ]
        )
        for qid in queries.ordered:
            record = queries.records[qid]
            writer.writerow(
                [
                    record.qid,
                    record.text,
                    record.source,
                    record.category,
                    record.lang,
                    ";".join(record.facets),
                    util.canonical_json(dict(record.hard)) if record.hard else "",
                    record.difficulty,
                    record.match_docs,
                    record.split,
                ]
            )
    return path


def export_judgments(root: Path, judge_version: str, out_dir: Path) -> Path:
    """One row per (query, doc) judgment: grade, reason, status."""
    records = label_records(root / "judgments" / "labels.jsonl", judge_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"export-judgments-{_safe(judge_version)}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "qid",
                "query_text",
                "doc_id",
                "doc_hash",
                "doc_text",
                "grade",
                "reason",
                "status",
                "ts",
            ]
        )
        for record in sorted(records, key=lambda item: (item["qid"], item["doc_id"])):
            writer.writerow(
                [
                    record["qid"],
                    record["query_text"],
                    record["doc_id"],
                    record["doc_hash"],
                    record["doc_text"],
                    record.get("grade") if record.get("grade") is not None else "",
                    record.get("reason", ""),
                    record.get("status", ""),
                    record.get("ts", ""),
                ]
            )
    return path


def export_run_hits(root: Path, materials_version: str, queries: QuerySet, out_dir: Path) -> Path:
    """Every run's ranked hits: run, strategy, qid, rank, doc_id, score."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"export-hits-{_safe(materials_version)}-{_safe(queries.version)}.csv"
    run_dirs = find_runs(root, materials_version, queries.version)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "strategy", "qid", "rank", "doc_id", "score"])
        for run_dir in sorted(run_dirs):
            run = load_run(run_dir)
            strategy = run["manifest"]["spec"]["strategy"]
            for qid in queries.ordered:
                row = run["rows"].get(qid)
                if row is None:
                    continue
                for rank, (doc_id, score) in enumerate(row["hits"][:MAX_HITS_EXPORT], start=1):
                    writer.writerow([run["run_id"], strategy, qid, rank, doc_id, f"{score:.6f}"])
    return path


def export_all(
    root: Path, materials_version: str, queries_version: str, judge_version: str
) -> dict:
    """Write all three exports; returns their paths + record counts."""
    queries = QuerySet(queries_version, root)
    out_dir = root / "reports"
    queries_path = export_queries(queries, out_dir)
    judgments_path = export_judgments(root, judge_version, out_dir)
    hits_path = export_run_hits(root, materials_version, queries, out_dir)
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "queries_csv": str(queries_path),
        "judgments_csv": str(judgments_path),
        "hits_csv": str(hits_path),
        "n_queries": len(queries.ordered),
        "n_judgments": sum(1 for _ in open(judgments_path, encoding="utf-8")) - 1,
    }
