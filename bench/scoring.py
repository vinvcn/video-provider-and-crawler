"""Score one run against a judge's labels: per-query metrics, splits, cuts.

Shared by `vpc bench score` (single run) and `vpc bench leaderboard` (all runs
of one materials/queries/judge combination). Pure computation over loaded
artifacts — no I/O beyond what callers pass in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bench import metrics, util
from bench.queries import QuerySet

METRIC_NAMES = ("ndcg10", "recall1", "recall5", "recall10", "mrr10")


def query_scores(
    hits: list[tuple[int, float]], grades_for_query: Mapping[int, int | None]
) -> dict[str, Any]:
    """Metrics for one query: ranked grades from the run, pool from labels."""
    ranked_grades = [grades_for_query.get(doc_id) or 0 for doc_id, _ in hits[: metrics.METRIC_K]]
    nulls = sum(1 for doc_id, _ in hits[: metrics.METRIC_K] if grades_for_query.get(doc_id) is None)
    pool_grades = [grade if grade is not None else 0 for grade in grades_for_query.values()]
    relevant = metrics.n_relevant(
        [grade for grade in grades_for_query.values() if grade is not None]
    )
    return {
        "ndcg10": metrics.ndcg_at_k(ranked_grades, pool_grades),
        "recall1": metrics.recall_at_k(ranked_grades, relevant, 1),
        "recall5": metrics.recall_at_k(ranked_grades, relevant, 5),
        "recall10": metrics.recall_at_k(ranked_grades, relevant, 10),
        "mrr10": metrics.mrr_at_k(ranked_grades),
        "n_relevant": relevant,
        "n_judged": len(grades_for_query),
        "null_grade_top10": nulls,
    }


def score_run(run: dict, queries: QuerySet, labels: Mapping[str, Mapping[int, int | None]]) -> dict:
    """Full metric bundle for one loaded run under one judge version."""
    manifest = run["manifest"]
    per_query: dict[str, dict[str, Any]] = {}
    for qid, row in run["rows"].items():
        hits = [(int(doc_id), float(score)) for doc_id, score in row["hits"]]
        per_query[qid] = query_scores(hits, labels.get(qid, {}))

    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "holdout", "all"):
        qids = _qids_for_split(queries, split)
        qids = [qid for qid in qids if qid in per_query]
        bundle: dict[str, Any] = {"n": len(qids)}
        for name in METRIC_NAMES:
            bundle[name] = metrics.aggregate([per_query[qid][name] for qid in qids])
        splits[split] = bundle

    search_ms = [float(row["search_ms"]) for row in run["rows"].values()]
    embed_ms = [float(row["embed_ms"]) for row in run["rows"].values()]
    nulls = [per_query[qid]["null_grade_top10"] for qid in run["rows"]]
    return {
        "run_id": run["run_id"],
        "strategy": manifest["spec"]["strategy"],
        "config_hash": manifest["config_hash"],
        "splits": splits,
        "per_query": per_query,
        "latency": {
            "search_ms_p50": metrics.percentile(search_ms, 0.5),
            "search_ms_p95": metrics.percentile(search_ms, 0.95),
            "embed_ms_p50": metrics.percentile(embed_ms, 0.5),
            "embed_ms_p95": metrics.percentile(embed_ms, 0.95),
        },
        "null_grade_top10_total": sum(nulls),
        "judge_coverage": len(per_query),
    }


def cut_report(
    run: dict, scored: dict, queries: QuerySet, split: str
) -> dict[str, dict[str, float]]:
    """Mean nDCG@10 per cut (category/lang/source/difficulty/hard/facet) on a split."""
    qids = [qid for qid in _qids_for_split(queries, split) if qid in scored["per_query"]]
    buckets: dict[str, list[float]] = {}
    for qid in qids:
        record = queries.records[qid]
        for cut_key, value in metrics.cut_values(
            {
                "category": record.category,
                "lang": record.lang,
                "source": record.source,
                "difficulty": record.difficulty,
                "hard": dict(record.hard),
                "facets": list(record.facets),
            }
        ).items():
            buckets.setdefault(f"{cut_key}={value}", []).append(scored["per_query"][qid]["ndcg10"])
    return {
        key: {"mean_ndcg10": metrics.mean(values), "n": len(values)}
        for key, values in sorted(buckets.items())
    }


def _qids_for_split(queries: QuerySet, split: str) -> list[str]:
    if split == "all":
        return queries.ordered
    return [qid for qid in queries.ordered if queries.records[qid].split == split]


def write_metrics(run_dir, scored: dict, judge_version: str) -> None:
    """Persist metrics.json inside the run dir (annotated with the judge)."""
    payload = {**scored, "judge_version": judge_version}
    util.write_json(run_dir / "metrics.json", payload)
