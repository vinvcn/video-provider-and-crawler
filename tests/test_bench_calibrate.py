"""Calibration loop: worksheet export (stratified) + kappa scoring."""

import json
from pathlib import Path

import pytest
from conftest import material_row

from bench import calibrate

ROWS = [material_row(i, f"Clip {i} ocean city forest", tags=(f"tag{i}",)) for i in range(1, 30)]


def _write_labels(path: Path, grades: list[int]) -> None:
    records = []
    for index, grade in enumerate(grades):
        records.append(
            {
                "key": f"pair-{index:03d}",
                "qid": "q1",
                "query_text": "ocean waves",
                "doc_id": index + 1,
                "doc_hash": f"h{index}",
                "doc_text": f"clip {index}",
                "judge_model": "mock",
                "prompt_version": "v1",
                "grade": grade,
                "reason": "test",
                "status": "ok",
                "ts": "2026-10-09T00:00:00+00:00",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_export_is_stratified_and_deterministic(bench_root: Path):
    # 20 zeros, 20 ones, 5 twos, 5 threes available
    _write_labels(
        bench_root / "judgments" / "labels.jsonl",
        [0] * 20 + [1] * 20 + [2] * 5 + [3] * 5,
    )
    first = calibrate.export_worksheet(bench_root, "mock@prompt-v1")
    # quotas: 0->10, 1->15, 2->10 (only 5 exist), 3->5 => 35 picked
    assert first["picked"] == 35
    assert first["available"] == {0: 20, 1: 20, 2: 5, 3: 5}

    second = calibrate.export_worksheet(bench_root, "mock@prompt-v1")
    first_rows = Path(first["path"]).read_text(encoding="utf-8")
    second_rows = Path(second["path"]).read_text(encoding="utf-8")
    assert first_rows == second_rows

    # quota per grade respected where availability allows
    counts: dict[str, int] = {}
    for line in first_rows.splitlines()[1:]:
        counts[line.split(",")[4]] = counts.get(line.split(",")[4], 0) + 1
    assert counts == {"0": 10, "1": 15, "2": 5, "3": 5}


def test_blind_export_hides_judge_columns(bench_root: Path):
    _write_labels(bench_root / "judgments" / "labels.jsonl", [0] * 5 + [3] * 5)
    outcome = calibrate.export_worksheet(bench_root, "mock@prompt-v1", blind=True)
    assert outcome["blind"] is True
    text = Path(outcome["path"]).read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header == "pair_id,query_text,doc_id,doc_text,human_grade,notes"
    assert "judge" not in header


def test_blind_worksheet_scores_by_pair_id(bench_root: Path):
    _write_labels(bench_root / "judgments" / "labels.jsonl", [0] * 5 + [3] * 5)
    outcome = calibrate.export_worksheet(bench_root, "mock@prompt-v1", blind=True)
    worksheet = Path(outcome["path"])
    lines = worksheet.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    human_col = header.index("human_grade")
    filled = []
    for line in lines[1:]:
        cells = line.split(",")
        cells[human_col] = "1"  # uniform human grades; judge grades come from the store
        filled.append(",".join(cells))
    worksheet.write_text("\n".join([lines[0], *filled]) + "\n", encoding="utf-8")

    report = calibrate.score_worksheet(bench_root, "mock@prompt-v1", worksheet)
    assert report["n_graded"] == 10
    assert report["n_skipped"] == 0


def test_export_when_grades_are_scarce(bench_root: Path):
    _write_labels(bench_root / "judgments" / "labels.jsonl", [1, 1, 2])
    outcome = calibrate.export_worksheet(bench_root, "mock@prompt-v1")
    assert outcome["picked"] == 3  # takes what exists, never invents


def test_score_worksheet_computes_kappa(bench_root: Path):
    _write_labels(
        bench_root / "judgments" / "labels.jsonl",
        [0] * 20 + [1] * 20 + [2] * 5 + [3] * 5,
    )
    exported = calibrate.export_worksheet(bench_root, "mock@prompt-v1")
    worksheet = Path(exported["path"])
    lines = worksheet.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    judge_col = header.index("judge_grade")
    human_col = header.index("human_grade")
    filled = []
    for index, line in enumerate(lines[1:]):
        cells = line.split(",")
        human = cells[judge_col]  # human agrees by default
        if index < 4:  # ...except on the first four rows
            human = str((int(cells[judge_col]) + 1) % 4)
        cells[human_col] = human
        filled.append(",".join(cells))
    worksheet.write_text("\n".join([lines[0], *filled]) + "\n", encoding="utf-8")

    report = calibrate.score_worksheet(bench_root, "mock@prompt-v1", worksheet)
    assert report["n_graded"] == 35
    assert report["n_skipped"] == 0
    assert report["exact_agreement"] == pytest.approx(31 / 35)
    assert 0.7 < report["cohens_kappa"] < 1.0
    assert report["gate_passed"]
    assert report["path"].name == "calibration-report.json"


def test_score_requires_filled_grades(bench_root: Path):
    _write_labels(bench_root / "judgments" / "labels.jsonl", [2] * 10)
    exported = calibrate.export_worksheet(bench_root, "mock@prompt-v1")

    with pytest.raises(SystemExit):
        calibrate.score_worksheet(bench_root, "mock@prompt-v1", Path(exported["path"]))
