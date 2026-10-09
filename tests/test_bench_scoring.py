"""Scoring + leaderboard: run metrics against labels, render the report."""

import json
from pathlib import Path

from conftest import material_row, query_record, write_materials, write_queries, write_run

from bench import leaderboard as lb
from bench import scoring
from bench.judge import judge_runs
from bench.run import load_run

ROWS = [
    material_row(1, "Ocean Waves at Sunset", tags=("ocean", "waves")),
    material_row(2, "City Night Skyline", tags=("city", "night")),
    material_row(3, "Forest Morning Fog", tags=("forest", "fog")),
    material_row(4, "Desert Dunes Aerial", tags=("desert", "aerial")),
]

QUERIES = [
    query_record("ocean waves", qid="q-en", split="holdout", facets=("subject",)),
    query_record("慢动作海浪", qid="q-zh", lang="zh", split="holdout"),
    query_record("city night", qid="q-train", split="train"),
]


def _setup(bench_root: Path):
    materials = write_materials(bench_root, "v1", ROWS)
    queries = write_queries(bench_root, "v1", QUERIES)
    good = write_run(
        bench_root,
        "run-good",
        materials,
        queries,
        {
            "q-en": [[1, 3.0], [2, 2.0]],
            "q-zh": [[1, 1.0]],
            "q-train": [[2, 2.0]],
        },
    )
    bad = write_run(
        bench_root,
        "run-bad",
        materials,
        queries,
        {
            "q-en": [[4, 3.0], [3, 2.0]],
            "q-zh": [[4, 1.0]],
            "q-train": [[3, 2.0]],
        },
    )
    judge_runs(
        [good, bad],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=2,
        max_pairs=None,
    )
    return materials, queries, good, bad


def test_score_run_math(bench_root: Path):
    materials, queries, good, _ = _setup(bench_root)
    from bench.judge import load_labels

    labels = load_labels(bench_root / "judgments" / "labels.jsonl", "mock@prompt-v1")
    # relabel deterministically so the math is hand-checkable
    labels["q-en"] = {1: 3, 2: 0}
    labels["q-zh"] = {1: 2}
    labels["q-train"] = {2: 2}
    scored = scoring.score_run(load_run(good), queries, labels)

    # q-en: ranked [3, 0] vs pool ideal [3, 0] -> ndcg 1.0
    assert scored["per_query"]["q-en"]["ndcg10"] == 1.0
    assert scored["per_query"]["q-en"]["recall10"] == 1.0
    assert scored["per_query"]["q-en"]["mrr10"] == 1.0
    # q-zh: ranked [2] vs pool [2] -> perfect as well
    assert scored["per_query"]["q-zh"]["ndcg10"] == 1.0
    # splits: holdout covers q-en + q-zh
    assert scored["splits"]["holdout"]["n"] == 2
    assert scored["splits"]["train"]["n"] == 1
    assert scored["splits"]["all"]["n"] == 3
    # null-grade accounting: doc 4 unjudged for q-en -> grade 0, counted
    scored_gap = scoring.score_run(load_run(bench_root / "runs" / "run-bad"), queries, labels)
    assert scored_gap["per_query"]["q-en"]["null_grade_top10"] >= 1
    assert scored_gap["per_query"]["q-en"]["ndcg10"] == 0.0


def test_cut_report_separates_language(bench_root: Path):
    materials, queries, good, _ = _setup(bench_root)
    from bench.judge import load_labels

    labels = load_labels(bench_root / "judgments" / "labels.jsonl", "mock@prompt-v1")
    labels["q-en"] = {1: 3, 2: 0}
    labels["q-zh"] = {1: 2}
    labels["q-train"] = {2: 2}
    run = load_run(good)
    scored = scoring.score_run(run, queries, labels)
    cuts = scoring.cut_report(run, scored, queries, "holdout")
    assert cuts["lang=en"]["n"] == 1
    assert cuts["lang=zh"]["n"] == 1
    assert cuts["facet:subject=yes"]["n"] == 1


def test_leaderboard_ranks_and_renders(bench_root: Path):
    materials, queries, good, bad = _setup(bench_root)
    # relabel so run-good is strictly better on the holdout
    from bench.judge import load_labels

    labels = load_labels(bench_root / "judgments" / "labels.jsonl", "mock@prompt-v1")
    labels["q-en"] = {1: 3, 4: 0}
    labels["q-zh"] = {1: 2, 4: 0}
    labels["q-train"] = {2: 2, 3: 0}
    (bench_root / "judgments" / "labels.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "key": f"manual-{qid}-{doc}",
                    "qid": qid,
                    "doc_id": doc,
                    "doc_hash": "x",
                    "judge_model": "mock",
                    "prompt_version": "v1",
                    "grade": grade,
                    "status": "ok",
                    "reason": "",
                    "ts": "2026-10-09T00:00:00+00:00",
                }
            )
            for qid, grades in labels.items()
            for doc, grade in grades.items()
        )
        + "\n",
        encoding="utf-8",
    )

    report = lb.build_leaderboard(bench_root, materials, queries, "mock@prompt-v1", split="holdout")
    assert report["runs"][0]["scored"]["strategy"] == "bm25-query"
    assert len(report["runs"]) == 2
    assert len(report["pairwise"]) == 1
    assert report["pairwise"][0]["win"] == "a"  # run-good wins

    paths = lb.write_reports(bench_root, report)
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "检索基准排行榜" in markdown
    assert "run-good" in markdown
    assert "配对 bootstrap" in markdown
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert csv_text.startswith("rank,run_id,strategy")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["judge_version"] == "mock@prompt-v1"
    assert len(summary["runs"]) == 2


def test_latest_judge_version(bench_root: Path):
    materials, queries, good, _ = _setup(bench_root)
    version = lb.latest_judge_version(bench_root / "judgments" / "labels.jsonl")
    assert version == "mock@prompt-v1"
