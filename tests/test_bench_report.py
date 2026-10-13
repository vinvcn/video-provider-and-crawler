"""Report page: self-contained HTML with leaderboard + interactive data browser."""

import json
from pathlib import Path

import pytest
from conftest import material_row, query_record, write_materials, write_queries, write_run

from bench import report_page
from bench.judge import judge_runs, judge_version_for

ROWS = [
    material_row(1, "Ocean Waves at Sunset", tags=("ocean", "waves")),
    material_row(2, "City Night Skyline", tags=("city", "night")),
    material_row(3, "Forest Morning Fog", tags=("forest", "fog")),
]
QUERIES = [
    query_record("ocean waves", qid="q-a", split="holdout", facets=("subject",)),
    query_record("city night", qid="q-b", split="train"),
]


def _setup(bench_root: Path):
    materials = write_materials(bench_root, "v1", ROWS)
    queries = write_queries(bench_root, "v1", QUERIES)
    for run_id, hits in (
        ("run-good", {"q-a": [[1, 3.0], [2, 1.0]], "q-b": [[2, 2.0]]}),
        ("run-bad", {"q-a": [[3, 1.0]], "q-b": [[3, 0.5]]}),
    ):
        write_run(bench_root, run_id, materials, queries, hits)
    judge_runs(
        [bench_root / "runs" / "run-good", bench_root / "runs" / "run-bad"],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=1,
        max_pairs=None,
    )
    return materials, queries


def test_report_page_renders_with_embedded_data(bench_root: Path):
    materials, queries = _setup(bench_root)
    path = report_page.build_report(bench_root, materials, queries, judge_version_for("mock"))
    assert path.is_file()
    html = path.read_text(encoding="utf-8")
    assert "bench 基线报告" in html
    assert "数据浏览" in html
    assert "ocean waves" in html  # query text embedded
    assert "Ocean Waves at Sunset" in html  # doc title embedded

    # the embedded JSON parses back and respects judge-version isolation
    data = html.split("const DATA = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</")
    payload = json.loads(data)
    assert {row["strategy"] for row in payload["rows"]} == {"bm25-query"}
    assert payload["judgments"] and all(j["s"] == "mock" for j in payload["judgments"])
    assert set(payload["docs"]) == {"1", "2", "3"}
    assert payload["runs"][0]["hits"]["q-a"][0][0] in (1, 3)


def test_report_page_requires_labels_for_the_judge_version(bench_root: Path):
    materials, queries = _setup(bench_root)
    with pytest.raises(SystemExit):
        report_page.build_report(bench_root, materials, queries, judge_version_for("no-such-judge"))


def test_report_page_survives_script_closing_text(bench_root: Path):
    materials = write_materials(
        bench_root,
        "v2",
        [material_row(1, "Sneaky </script> title", tags=("x",))],
    )
    queries = write_queries(bench_root, "v2", [query_record("sneaky", qid="q-x", split="holdout")])
    write_run(bench_root, "run-x", materials, queries, {"q-x": [[1, 1.0]]})
    judge_runs(
        [bench_root / "runs" / "run-x"],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=0,
        client=None,
        concurrency=1,
        max_pairs=None,
    )
    path = report_page.build_report(bench_root, materials, queries, judge_version_for("mock"))
    html = path.read_text(encoding="utf-8")
    assert "</script> title" not in html.split("const DATA = ")[1].split(";</script>")[0]
