"""Leaderboard: score every run of one (materials, queries, judge) triple.

Ranks strategies on the holdout split's mean nDCG@10 by default, reports the
full-split score next to it, and adds a paired-bootstrap comparison matrix so
small deltas are never over-interpreted. Outputs markdown + csv + summary json
under storage/bench/reports/ (gitignored runtime data).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from bench import metrics, util
from bench.judge import judge_version_for, load_labels
from bench.materials import Materials
from bench.queries import QuerySet
from bench.run import find_runs, load_run
from bench.scoring import cut_report, score_run

PRIMARY = "ndcg10"


def latest_judge_version(labels_path: Path) -> str:
    """Most recently written judge model in the label store (for `--judge latest`)."""
    best_version, best_ts = None, ""
    if labels_path.is_file():
        for record in util.read_jsonl(labels_path):
            version = judge_version_for(str(record["judge_model"]))
            if str(record.get("ts", "")) >= best_ts:
                best_version, best_ts = version, str(record.get("ts", ""))
    if best_version is None:
        raise SystemExit("no labels yet: run `vpc bench judge` first (or pass --judge)")
    return best_version


def build_leaderboard(
    root: Path,
    materials: Materials,
    queries: QuerySet,
    judge_version: str,
    split: str = "holdout",
) -> dict:
    """Score all matching runs; returns the full report bundle."""
    run_dirs = find_runs(root, materials.version, queries.version)
    if not run_dirs:
        raise SystemExit(
            f"no runs found for materials {materials.version} / queries {queries.version}"
        )
    labels = load_labels(root / "judgments" / "labels.jsonl", judge_version)
    if not labels:
        raise SystemExit(f"no labels for judge {judge_version!r}; run `vpc bench judge` first")

    entries = []
    for run_dir in run_dirs:
        run = load_run(run_dir)
        scored = score_run(run, queries, labels)
        entries.append(
            {
                "run": run,
                "scored": scored,
                "cuts": {
                    split_name: cut_report(run, scored, queries, split_name)
                    for split_name in ("holdout", "all")
                },
            }
        )
    entries.sort(key=lambda entry: -entry["scored"]["splits"][split][PRIMARY]["mean"])

    pairwise = []
    for i, first in enumerate(entries):
        for second in entries[i + 1 :]:
            qids = [
                qid
                for qid in queries.ordered
                if queries.records[qid].split == split
                and qid in first["scored"]["per_query"]
                and qid in second["scored"]["per_query"]
            ]
            result = metrics.bootstrap_paired(
                [first["scored"]["per_query"][qid][PRIMARY] for qid in qids],
                [second["scored"]["per_query"][qid][PRIMARY] for qid in qids],
            )
            pairwise.append(
                {
                    "a": first["scored"]["run_id"],
                    "b": second["scored"]["run_id"],
                    **result,
                }
            )

    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "materials": {"version": materials.version, "hash": materials.content_hash},
        "queries": {"version": queries.version, "hash": queries.content_hash},
        "judge_version": judge_version,
        "split": split,
        "primary_metric": PRIMARY,
        "calibration": _calibration_for(root, judge_version),
        "runs": entries,
        "pairwise": pairwise,
    }


def _calibration_for(root: Path, judge_version: str) -> dict | None:
    """Attach the human-calibration report when one exists for this judge."""
    try:
        calibration = util.load_json(root / "reports" / "calibration-report.json")
    except FileNotFoundError:
        return None
    if calibration.get("judge_version") != judge_version:
        return None
    return calibration


def render_markdown(report: dict) -> str:
    """Human-readable leaderboard (the file the user actually opens)."""
    split = report["split"]
    lines = [
        "# 检索基准排行榜",
        "",
        f"- 材料:{report['materials']['version']}"
        f"(hash `{report['materials']['hash'][:12]}…`)"
        f"· 查询:{report['queries']['version']}"
        f"(hash `{report['queries']['hash'][:12]}…`)",
        f"- 判分:`{report['judge_version']}` · 主分:**{split} 平均 nDCG@10**"
        f"(full 分数并排展示,防小样本过读)",
        "",
        "| # | 策略 | 语料 | run | nDCG@10 ("
        + split
        + ") | nDCG@10 (full) | R@10 | MRR@10 | p50 ms | p95 ms | null@10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, entry in enumerate(report["runs"], start=1):
        scored = entry["scored"]
        holdout = scored["splits"].get(split, scored["splits"]["all"])
        full = scored["splits"]["all"]
        spec = entry["run"]["manifest"]["spec"]
        lines.append(
            "| {} | {} | {} | `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} |"
            " {:.1f} | {:.1f} | {} |".format(
                rank,
                scored["strategy"],
                spec.get("text_source", "raw"),
                scored["run_id"],
                holdout[report["primary_metric"]]["mean"],
                full[report["primary_metric"]]["mean"],
                holdout["recall10"]["mean"],
                holdout["mrr10"]["mean"],
                scored["latency"]["search_ms_p50"],
                scored["latency"]["search_ms_p95"],
                scored["null_grade_top10_total"],
            )
        )
    lines += ["", "## 配对 bootstrap(主分差,95% CI,B=1000)", ""]
    if report["pairwise"]:
        lines.append("| A | B | Δ(A−B) | CI | 结论 |")
        lines.append("|---|---|---|---|---|")
        for pair in report["pairwise"]:
            lo, hi = pair["ci"]
            verdict = {"a": "A 更优", "b": "B 更优", "tie": "无显著差异"}[pair["win"]]
            lines.append(
                f"| `{pair['a']}` | `{pair['b']}` | {pair['diff']:+.4f} "
                f"| [{lo:+.4f}, {hi:+.4f}] | {verdict} |"
            )
    else:
        lines.append("_(只有一条 run,无对比)_")
    if report.get("calibration"):
        cal = report["calibration"]
        verdict = "通过" if cal["gate_passed"] else "未通过(需修 rubric/判分器)"
        lines += [
            "",
            "## 人工校准(判分器 vs 人工抽检)",
            "",
            f"- n={cal['n_graded']} 对 · 一致率 **{cal['exact_agreement']:.1%}** · "
            f"Cohen's κ = **{cal['cohens_kappa']:.2f}**(闸门 ≥{cal['gate']}:{verdict})",
        ]
    lines += ["", "## 切分摘要(holdout 平均 nDCG@10,关键切面)", ""]
    lines.append("| 策略 | lang=zh | lang=en | hard=yes | catalog | designed |")
    lines.append("|---|---|---|---|---|---|")
    for entry in report["runs"]:
        cuts = entry["cuts"].get(split, {})
        cells = []
        for key in ("lang=zh", "lang=en", "hard=yes", "source=catalog", "source=designed"):
            cell = cuts.get(key)
            cells.append(f"{cell['mean_ndcg10']:.4f} (n={cell['n']})" if cell else "—")
        spec = entry["run"]["manifest"]["spec"]
        lines.append(
            f"| {entry['scored']['strategy']} ({spec.get('text_source', 'raw')}) | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_csv(report: dict) -> str:
    """Flat csv: one row per run (spreadsheet-friendly)."""
    split = report["split"]
    header = [
        "rank",
        "run_id",
        "strategy",
        "text_source",
        "config_hash",
        f"ndcg10_mean_{split}",
        "ndcg10_mean_full",
        f"recall10_mean_{split}",
        f"mrr10_mean_{split}",
        "search_ms_p50",
        "search_ms_p95",
        "embed_ms_p50",
        "null_grade_top10_total",
    ]
    rows = [",".join(header)]
    for rank, entry in enumerate(report["runs"], start=1):
        scored = entry["scored"]
        holdout = scored["splits"].get(split, scored["splits"]["all"])
        full = scored["splits"]["all"]
        rows.append(
            ",".join(
                str(value)
                for value in (
                    rank,
                    scored["run_id"],
                    scored["strategy"],
                    entry["run"]["manifest"]["spec"].get("text_source", "raw"),
                    scored["config_hash"][:12],
                    f"{holdout[report['primary_metric']]['mean']:.6f}",
                    f"{full[report['primary_metric']]['mean']:.6f}",
                    f"{holdout['recall10']['mean']:.6f}",
                    f"{holdout['mrr10']['mean']:.6f}",
                    f"{scored['latency']['search_ms_p50']:.3f}",
                    f"{scored['latency']['search_ms_p95']:.3f}",
                    f"{scored['latency']['embed_ms_p50']:.3f}",
                    scored["null_grade_top10_total"],
                )
            )
        )
    return "\n".join(rows) + "\n"


def write_reports(root: Path, report: dict) -> dict:
    """Render + persist md/csv/summary; returns the written paths."""
    stem = "leaderboard-{}-{}-{}".format(
        report["materials"]["version"],
        report["queries"]["version"],
        report["judge_version"].replace("@", "_at_").replace("/", "_"),
    )
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{stem}.md"
    csv_path = reports_dir / f"{stem}.csv"
    summary_path = reports_dir / f"summary-{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    csv_path.write_text(render_csv(report), encoding="utf-8")
    slim = {key: value for key, value in report.items() if key != "runs"}
    slim["runs"] = [
        {
            "scored": entry["scored"],
            "cuts": entry["cuts"],
            "spec": entry["run"]["manifest"]["spec"],
        }
        for entry in report["runs"]
    ]
    util.write_json(summary_path, slim)
    return {"markdown": md_path, "csv": csv_path, "summary": summary_path}
