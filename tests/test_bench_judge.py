"""Judge: grade parsing, pooling, label cache, mock judging end-to-end."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import material_row, query_record, write_materials, write_queries, write_run

from bench.judge import (
    LabelStore,
    judge_runs,
    judge_version_for,
    label_key,
    load_labels,
    parse_grade,
    pool_for_query,
)

ROWS = [
    material_row(1, "Ocean Waves at Sunset", tags=("ocean", "waves")),
    material_row(2, "City Night Skyline", tags=("city", "night")),
    material_row(3, "Forest Morning Fog", tags=("forest", "fog")),
    material_row(4, "Desert Dunes Aerial", tags=("desert", "aerial")),
    material_row(5, "Coffee Shop Interior", tags=("coffee", "shop")),
    material_row(6, "Snowy Mountain Peaks", tags=("snow", "mountain")),
]


def test_parse_grade_plain_json():
    assert parse_grade('{"grade": 2, "reason": "relevant"}') == (2, "relevant")


def test_parse_grade_string_grade_and_prose():
    grade, reason = parse_grade('The video matches. {"grade": "3", "reason": "exact"} great')
    assert grade == 3
    assert reason == "exact"


def test_parse_grade_out_of_range_and_garbage():
    assert parse_grade('{"grade": 4}')[0] is None
    assert parse_grade("no json at all")[0] is None
    assert parse_grade('{"grade": true}')[0] is None


def test_pacer_spaces_request_starts():
    from bench.judge import Pacer

    pacer = Pacer(per_minute=600)  # 0.1s between starts
    threads: list = []

    def hit() -> list[float]:
        starts: list[float] = []
        for _ in range(3):
            pacer.wait()
            starts.append(time.monotonic())
        return starts

    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            threads.append(pool.submit(hit))
    starts = sorted(stamp for future in threads for stamp in future.result())
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    # 6 paced starts: 5 gaps, each >= ~0.1s (allow scheduler slack downward)
    assert len(gaps) == 5
    assert all(gap > 0.08 for gap in gaps)


def test_pacer_disabled_when_zero():
    from bench.judge import Pacer

    pacer = Pacer(per_minute=0.0)
    started = time.monotonic()
    for _ in range(10):
        pacer.wait()
    assert time.monotonic() - started < 0.05


def test_label_key_changes_with_every_component():
    base = label_key("q", 1, "h", "model-a")
    assert base == label_key("q", 1, "h", "model-a")
    assert base != label_key("q2", 1, "h", "model-a")
    assert base != label_key("q", 2, "h", "model-a")
    assert base != label_key("q", 1, "h2", "model-a")
    assert base != label_key("q", 1, "h", "model-b")


def test_judge_version_format():
    assert judge_version_for("qwen-flash") == "qwen-flash@prompt-v1"


def test_pool_unions_runs_and_adds_negatives(bench_root: Path):
    materials = write_materials(bench_root, "v1", ROWS)
    queries = write_queries(
        bench_root,
        "v1",
        [query_record("ocean waves", split="train")],
    )
    run_a = write_run(bench_root, "run-a", materials, queries, {"q": [[1, 1.0], [2, 0.5]]})
    run_b = write_run(bench_root, "run-b", materials, queries, {"q": [[2, 2.0], [3, 0.1]]})
    from bench.run import load_run

    rows = [load_run(run_a)["rows"]["q"], load_run(run_b)["rows"]["q"]]
    pooled = pool_for_query(rows, "q", materials.ids, "v1", depth=20, negatives=2)
    assert set(pooled[:3]) == {1, 2, 3}  # union first
    assert len(pooled) == 5
    assert len(set(pooled)) == 5
    # deterministic across calls
    again = pool_for_query(rows, "q", materials.ids, "v1", depth=20, negatives=2)
    assert pooled == again
    # negative seed varies with materials version (union part is stable)
    other = pool_for_query(rows, "q", materials.ids, "v2", depth=20, negatives=2)
    assert other[:3] == pooled[:3]


def test_label_store_deduplicates(tmp_path: Path):
    store = LabelStore(tmp_path / "labels.jsonl")
    record = {"key": "k1", "grade": 2}
    store.append(record)
    store.append({**record, "grade": 3})  # duplicate key refused
    assert store.has("k1")
    lines = (tmp_path / "labels.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    reloaded = LabelStore(tmp_path / "labels.jsonl")
    assert reloaded.has("k1")
    assert not reloaded.has("k2")


def _mock_setup(bench_root: Path):
    materials = write_materials(bench_root, "v1", ROWS)
    queries = write_queries(
        bench_root,
        "v1",
        [
            query_record("ocean waves", qid="q-ocean"),
            query_record("city night", qid="q-city"),
        ],
    )
    hits = {"q-ocean": [[1, 1.0], [2, 0.5]], "q-city": [[2, 1.0], [3, 0.2]]}
    run_dir = write_run(bench_root, "run-mock", materials, queries, hits)
    return materials, queries, run_dir


def test_mock_judging_writes_labels_and_resumes(bench_root: Path):
    materials, queries, run_dir = _mock_setup(bench_root)
    outcome = judge_runs(
        [run_dir],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=2,
        max_pairs=None,
    )
    assert outcome.judge_version == "mock@prompt-v1"
    assert outcome.judged > 0
    assert outcome.cached == 0

    labels = load_labels(bench_root / "judgments" / "labels.jsonl", "mock@prompt-v1")
    assert set(labels) == {"q-ocean", "q-city"}
    assert all(isinstance(g, int) for grades in labels.values() for g in grades.values())

    # rerun: fully cached, nothing new judged
    second = judge_runs(
        [run_dir],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=2,
        max_pairs=None,
    )
    assert second.judged == 0
    assert second.cached == outcome.pool_size


def test_mock_judging_budget_guard(bench_root: Path):
    materials, queries, run_dir = _mock_setup(bench_root)
    outcome = judge_runs(
        [run_dir],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=2,
        max_pairs=2,
    )
    assert outcome.judged == 2
    # resume: the rest is still pending, then completes
    rest = judge_runs(
        [run_dir],
        materials,
        queries,
        bench_root,
        depth=20,
        negatives=1,
        client=None,
        concurrency=2,
        max_pairs=None,
    )
    assert rest.judged == outcome.pool_size - 2


def test_judge_rejects_version_mismatch(bench_root: Path):
    materials, queries, run_dir = _mock_setup(bench_root)
    other_queries = write_queries(bench_root, "v2", [query_record("ocean waves", qid="q-ocean")])
    with pytest.raises(SystemExit):
        judge_runs(
            [run_dir],
            materials,
            other_queries,
            bench_root,
            depth=20,
            negatives=1,
            client=None,
            concurrency=1,
            max_pairs=None,
        )
