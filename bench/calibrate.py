"""Human calibration: export a stratified worksheet, score it for agreement.

The LLM judge is only trustworthy next to a measured human agreement rate
(docs/bench-harness-design-2026-10-09.md §5): export ~40 pairs weighted toward
the boundary grades (1/2), have a human grade them blind, then compute exact
agreement + Cohen's kappa + the confusion matrix. kappa < 0.4 means the rubric
or judge is broken — stop and fix before trusting any leaderboard.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections import Counter
from pathlib import Path

from bench import metrics, util
from bench.judge import label_records

# Boundary grades dominate: those are the pairs where judges disagree in
# practice, and where a broken rubric shows up first.
GRADE_QUOTA = {1: 15, 2: 10, 0: 10, 3: 5}
KAPPA_GATE = 0.4

WORKSHEET_COLUMNS = (
    "pair_id",
    "query_text",
    "doc_id",
    "doc_text",
    "judge_grade",
    "judge_reason",
    "human_grade",
    "notes",
)
BLIND_COLUMNS = ("pair_id", "query_text", "doc_id", "doc_text", "human_grade", "notes")


def export_worksheet(
    root: Path, judge_version: str, out_path: Path | None = None, blind: bool = False
) -> dict:
    """Write the calibration CSV; returns counts per grade + the path.

    `blind=True` drops the judge grade/reason columns so the human grades
    without anchoring (kappa stays meaningful); the scorer joins the judge
    grades back by pair_id either way.
    """
    records = [
        record
        for record in label_records(root / "judgments" / "labels.jsonl", judge_version)
        if record.get("status") == "ok"
    ]
    by_grade: dict[int, list[dict]] = {}
    for record in records:
        by_grade.setdefault(int(record["grade"]), []).append(record)

    picked: list[dict] = []
    for grade in sorted(GRADE_QUOTA):
        ordered = sorted(
            by_grade.get(grade, []),
            key=lambda record: (util.hash_key(record["key"]), record["key"]),
        )
        picked.extend(ordered[: GRADE_QUOTA[grade]])

    if out_path is None:
        name = "calibration-worksheet-blind.csv" if blind else "calibration-worksheet.csv"
        out_path = root / "reports" / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = BLIND_COLUMNS if blind else WORKSHEET_COLUMNS
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for record in sorted(picked, key=lambda record: record["key"]):
            writer.writerow(
                {
                    "pair_id": record["key"],
                    "query_text": record["query_text"],
                    "doc_id": record["doc_id"],
                    "doc_text": record["doc_text"],
                    "judge_grade": record["grade"],
                    "judge_reason": record["reason"],
                    "human_grade": "",
                    "notes": "",
                }
            )
    return {
        "path": out_path,
        "picked": len(picked),
        "blind": blind,
        "available": Counter({g: len(v) for g, v in by_grade.items()}),
    }


def score_worksheet(root: Path, judge_version: str, worksheet_path: Path) -> dict:
    """Compare filled-in human grades against the judge; writes the report."""
    judge_by_key = {
        record["key"]: int(record["grade"])
        for record in label_records(root / "judgments" / "labels.jsonl", judge_version)
        if record.get("status") == "ok"
    }
    human: list[int] = []
    judge: list[int] = []
    skipped = 0
    with worksheet_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("human_grade") or "").strip()
            if raw not in {"0", "1", "2", "3"}:
                skipped += 1
                continue
            pair_key = (row.get("pair_id") or "").strip()
            if pair_key not in judge_by_key:
                skipped += 1
                continue
            human.append(int(raw))
            judge.append(judge_by_key[pair_key])
    if not human:
        raise SystemExit("no usable human grades in the worksheet (fill the human_grade column)")

    exact = sum(1 for a, b in zip(judge, human, strict=True) if a == b) / len(human)
    kappa = metrics.cohen_kappa(judge, human)
    report = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "judge_version": judge_version,
        "n_graded": len(human),
        "n_skipped": skipped,
        "exact_agreement": exact,
        "cohens_kappa": kappa,
        "confusion_judge_x_human": metrics.confusion_matrix(judge, human),
        "gate": KAPPA_GATE,
        "gate_passed": kappa >= KAPPA_GATE,
    }
    out_path = root / "reports" / "calibration-report.json"
    util.write_json(out_path, report)
    report["path"] = out_path
    return report
