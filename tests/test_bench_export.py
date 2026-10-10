"""Exports: queries, judgments and run hits as flat CSVs."""

import csv
from pathlib import Path

from conftest import material_row, query_record, write_materials, write_queries, write_run

from bench import export as export_mod
from bench.judge import judge_runs, judge_version_for, label_key

ROWS = [
    material_row(1, "Ocean Waves at Sunset", tags=("ocean", "waves")),
    material_row(2, "City Night Skyline", tags=("city", "night")),
    material_row(3, "Forest Morning Fog", tags=("forest", "fog")),
]
QUERIES = [
    query_record("ocean waves", qid="q-a", split="holdout", facets=("subject",), difficulty="easy"),
    query_record("city night", qid="q-b", split="train", hard={"orientation": "landscape"}),
]


def _setup(bench_root: Path):
    materials = write_materials(bench_root, "v1", ROWS)
    queries = write_queries(bench_root, "v1", QUERIES)
    run_dir = write_run(
        bench_root,
        "run-x",
        materials,
        queries,
        {"q-a": [[1, 2.0], [2, 1.0]], "q-b": [[2, 5.0]]},
    )
    judge_runs(
        [run_dir],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=1,
        max_pairs=None,
    )
    return materials, queries, run_dir


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_all_writes_three_csvs(bench_root: Path):
    materials, queries, _ = _setup(bench_root)
    judge_version = judge_version_for("mock")
    summary = export_mod.export_all(bench_root, "v1", "v1", judge_version)

    q_rows = _rows(Path(summary["queries_csv"]))
    assert len(q_rows) == 2
    assert q_rows[0]["qid"] == "q-a"
    assert q_rows[0]["facets"] == "subject"
    assert q_rows[1]["hard"] == '{"orientation":"landscape"}'
    assert summary["n_queries"] == 2

    j_rows = _rows(Path(summary["judgments_csv"]))
    assert summary["n_judgments"] == len(j_rows)
    assert {row["status"] for row in j_rows} == {"mock"}
    assert all(row["grade"] in {"0", "1", "2", "3"} for row in j_rows)
    assert all(row["query_text"] and row["doc_text"] for row in j_rows)

    h_rows = _rows(Path(summary["hits_csv"]))
    assert {(row["qid"], row["rank"], row["doc_id"]) for row in h_rows} == {
        ("q-a", "1", "1"),
        ("q-a", "2", "2"),
        ("q-b", "1", "2"),
    }
    assert {row["strategy"] for row in h_rows} == {"bm25-query"}


def test_export_judgments_isolates_judge_versions(bench_root: Path):
    materials, queries, run_dir = _setup(bench_root)
    # a second, different judge version labels one pair
    from bench.judge import LabelStore

    store = LabelStore(bench_root / "judgments" / "labels.jsonl")
    record = queries.records["q-a"]
    key = label_key(record.text, 1, materials.rows[1].doc_hash, "other-judge")
    store.append(
        {
            "key": key,
            "qid": "q-a",
            "query_text": record.text,
            "doc_id": 1,
            "doc_hash": materials.rows[1].doc_hash,
            "doc_text": materials.rows[1].embed_text,
            "judge_model": "other-judge",
            "prompt_version": "v1",
            "grade": 3,
            "reason": "other",
            "status": "ok",
            "ts": "2026-10-09T00:00:00+00:00",
        }
    )
    mock_rows = _rows(
        export_mod.export_judgments(bench_root, judge_version_for("mock"), bench_root / "reports")
    )
    assert all(
        "mock" in row["status"] or row["reason"] == "mock grade (pipeline verification)"
        for row in mock_rows
    )
    other_rows = _rows(
        export_mod.export_judgments(
            bench_root, judge_version_for("other-judge"), bench_root / "reports"
        )
    )
    assert len(other_rows) == 1
    assert other_rows[0]["grade"] == "3"
