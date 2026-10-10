"""`vpc bench` subcommand tree (materials / queries / run / judge / score / ...).

Registered from crawler.cli with a single call so the running crawler's CLI
surface only ever grows by addition. Every handler prints a JSON summary,
following the repo's CLI conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench import calibrate, env, report_page, scoring, strategies
from bench import captions as captions_mod
from bench import export as export_mod
from bench import judge as judge_mod
from bench import leaderboard as leaderboard_mod
from bench import materials as materials_mod
from bench import queries as queries_mod
from bench import run as run_mod
from bench.embedding_client import EmbeddingClient
from bench.materials import Materials
from bench.queries import QuerySet


def _prepare() -> Path:
    """Load .env overrides; returns the bench root directory."""
    env.load_env_file()
    return env.bench_dir()


def _resolve_run_dirs(root: Path, run_ids: str) -> list[Path]:
    dirs = []
    for name in [part.strip() for part in run_ids.split(",") if part.strip()]:
        run_dir = root / "runs" / name
        if not (run_dir / "strategy.json").is_file():
            raise SystemExit(f"run not found: {run_dir}")
        dirs.append(run_dir)
    return dirs


def _load_version(root: Path, kind: str, version: str, loader):
    try:
        return loader(version, root)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{kind} version {version!r} not built yet ({exc}); run build first"
        ) from exc


def cmd_materials_build(args: argparse.Namespace) -> int:
    root = _prepare()
    manifest = materials_mod.build_materials(args.dsn, args.n, args.version, root)
    manifest["out_dir"] = str(root / "materials" / args.version)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_queries_build(args: argparse.Namespace) -> int:
    root = _prepare()
    materials = _load_version(root, "materials", args.materials, Materials)
    manifest = queries_mod.build_queries(args.dsn, materials, args.version, root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = _prepare()
    materials = _load_version(root, "materials", args.materials, Materials)
    queries = _load_version(root, "queries", args.queries, QuerySet)
    captions = None
    if args.text_source == "caption":
        captions = captions_mod.Captions.load(args.captions, root)
        if captions.manifest.get("materials_version") != materials.version:
            raise SystemExit("captions were built for a different materials version")
    embed_client = None
    if args.strategy in ("dense", "rrf"):
        embed_client = EmbeddingClient(env.embed_config())
    spec = strategies.strategy_config(
        args.strategy, embed_client, text_source=args.text_source, captions_version=args.captions
    )
    strategy = strategies.build_strategy(
        args.strategy,
        materials,
        root,
        embed_client,
        text_source=args.text_source,
        captions=captions,
    )
    text_key = "raw" if args.text_source == "raw" else f"caption-{captions.content_hash[:16]}"
    manifest = run_mod.run_strategy(
        strategy,
        spec,
        materials,
        queries,
        root,
        depth=args.depth,
        limit=args.limit,
        embed_client=embed_client,
        text_key=text_key,
    )
    if embed_client is not None:
        embed_client.close()
    manifest["out_dir"] = str(root / "runs" / manifest["run_id"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_captions_build(args: argparse.Namespace) -> int:
    root = _prepare()
    materials = _load_version(root, "materials", args.materials, Materials)
    config = env.vlm_config()
    client = captions_mod.GemmaCaptionClient(config.keys, config.model, config.rpm)
    manifest = captions_mod.build_captions(
        args.dsn,
        materials,
        args.version,
        root,
        client,
        concurrency=args.concurrency,
        limit=args.limit,
    )
    client.close()
    manifest["out_dir"] = str(root / "captions" / args.version)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    root = _prepare()
    materials = _load_version(root, "materials", args.materials, Materials)
    queries = _load_version(root, "queries", args.queries, QuerySet)
    run_dirs = _resolve_run_dirs(root, args.runs)
    client = None
    if not args.mock:
        config = env.judge_config()
        client = judge_mod.JudgeClient(
            config.base_url, config.api_key, config.model, rpm=args.rpm or config.rpm
        )
    outcome = judge_mod.judge_runs(
        run_dirs,
        materials,
        queries,
        root,
        depth=args.depth,
        negatives=args.negatives,
        client=client,
        concurrency=args.concurrency,
        max_pairs=args.max_pairs,
    )
    if client is not None:
        client.close()
    print(
        json.dumps(
            {
                "judged": outcome.judged,
                "cached": outcome.cached,
                "unparsed": outcome.unparsed,
                "pool_size": outcome.pool_size,
                "judge_version": outcome.judge_version,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _judge_version_arg(root: Path, value: str) -> str:
    if value and value != "latest":
        return value
    return leaderboard_mod.latest_judge_version(root / "judgments" / "labels.jsonl")


def cmd_score(args: argparse.Namespace) -> int:
    root = _prepare()
    run_dir = _resolve_run_dirs(root, args.run)[0]
    run = run_mod.load_run(run_dir)
    queries = _load_version(root, "queries", run["manifest"]["queries"]["version"], QuerySet)
    judge_version = _judge_version_arg(root, args.judge)
    labels = judge_mod.load_labels(root / "judgments" / "labels.jsonl", judge_version)
    scored = scoring.score_run(run, queries, labels)
    scoring.write_metrics(run_dir, scored, judge_version)
    print(
        json.dumps(
            {
                "run_id": scored["run_id"],
                "strategy": scored["strategy"],
                "judge_version": judge_version,
                "holdout": scored["splits"]["holdout"],
                "full": scored["splits"]["all"],
                "latency": scored["latency"],
                "null_grade_top10_total": scored["null_grade_top10_total"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    root = _prepare()
    materials = _load_version(root, "materials", args.materials, Materials)
    queries = _load_version(root, "queries", args.queries, QuerySet)
    judge_version = _judge_version_arg(root, args.judge)
    report = leaderboard_mod.build_leaderboard(
        root, materials, queries, judge_version, split=args.split
    )
    paths = leaderboard_mod.write_reports(root, report)
    print(
        json.dumps(
            {
                "judge_version": judge_version,
                "split": args.split,
                **{k: str(v) for k, v in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_calibrate_export(args: argparse.Namespace) -> int:
    root = _prepare()
    judge_version = _judge_version_arg(root, args.judge)
    outcome = calibrate.export_worksheet(root, judge_version, blind=args.blind)
    print(
        json.dumps(
            {
                "judge_version": judge_version,
                "path": str(outcome["path"]),
                "picked": outcome["picked"],
                "blind": outcome["blind"],
                "available": {str(k): v for k, v in outcome["available"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_calibrate_score(args: argparse.Namespace) -> int:
    root = _prepare()
    judge_version = _judge_version_arg(root, args.judge)
    report = calibrate.score_worksheet(root, judge_version, Path(args.worksheet))
    print(
        json.dumps({k: v for k, v in report.items() if k != "path"}, ensure_ascii=False, indent=2)
    )
    if not report["gate_passed"]:
        print(
            f"⚠️ kappa {report['cohens_kappa']:.2f} < {calibrate.KAPPA_GATE}: "
            "fix the rubric/judge before trusting this leaderboard",
            file=sys.stderr,
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    root = _prepare()
    materials = _load_version(root, "materials", args.materials, Materials)
    queries = _load_version(root, "queries", args.queries, QuerySet)
    judge_version = _judge_version_arg(root, args.judge)
    path = report_page.build_report(root, materials, queries, judge_version, split=args.split)
    size_mb = path.stat().st_size / 1e6
    print(
        json.dumps(
            {
                "judge_version": judge_version,
                "split": args.split,
                "path": str(path),
                "size_mb": round(size_mb, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    root = _prepare()
    judge_version = _judge_version_arg(root, args.judge)
    summary = export_mod.export_all(root, args.materials, args.queries, judge_version)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    _prepare()
    result: dict[str, object] = {}

    embed_conf = env.embed_config()
    client = EmbeddingClient(embed_conf)
    try:
        vectors = client.embed(["ocean waves at sunset", "city night timelapse", "雨夜城市霓虹"])
        result["embed"] = {
            "model": embed_conf.model,
            "dim": len(vectors[0]),
            "tokens": client.total_tokens,
            "ok": True,
        }
    finally:
        client.close()

    judge_conf = env.judge_config()
    judge_client = judge_mod.JudgeClient(judge_conf.base_url, judge_conf.api_key, judge_conf.model)
    try:
        raw = judge_client.grade(
            "slow motion ocean waves",
            "Ocean Waves at Sunset — tags: ocean, waves, sunset, slow motion, sea, coast",
        )
        grade, reason = judge_mod.parse_grade(raw)
        result["judge"] = {
            "model": judge_conf.model,
            "grade": grade,
            "reason": reason[:200],
            "judge_version": judge_mod.judge_version_for(judge_conf.model),
            "ok": grade is not None,
        }
    finally:
        judge_client.close()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("judge", {}).get("ok") else 1


def register(sub: argparse._SubParsersAction) -> None:  # noqa: SLF001
    """Attach the whole `bench` command tree to the vpc parser."""
    parser = sub.add_parser(
        "bench", help="retrieval benchmark harness (see docs/bench-harness-design)"
    )
    bench_sub = parser.add_subparsers(dest="bench_command", required=True)

    materials_parser = bench_sub.add_parser("materials", help="materials set operations")
    materials_sub = materials_parser.add_subparsers(dest="materials_command", required=True)
    build = materials_sub.add_parser("build", help="sample + freeze a materials snapshot")
    build.add_argument("--n", type=int, default=10_000)
    build.add_argument("--version", default="v1")
    build.add_argument("--dsn")
    build.set_defaults(func=cmd_materials_build)

    queries_parser = bench_sub.add_parser("queries", help="query set operations")
    queries_sub = queries_parser.add_subparsers(dest="queries_command", required=True)
    qbuild = queries_sub.add_parser("build", help="sample real + designed queries, split 70/30")
    qbuild.add_argument("--materials", default="v1")
    qbuild.add_argument("--version", default="v1")
    qbuild.add_argument("--dsn")
    qbuild.set_defaults(func=cmd_queries_build)

    run = bench_sub.add_parser("run", help="run one strategy over the query set")
    run.add_argument("--strategy", required=True, choices=list(strategies.STRATEGY_IDS))
    run.add_argument("--materials", default="v1")
    run.add_argument("--queries", default="v1")
    run.add_argument(
        "--text-source",
        choices=list(strategies.TEXT_SOURCES),
        default="raw",
        help="retrieval text: raw (title/description/tags) or caption (VLM captions)",
    )
    run.add_argument(
        "--captions", default="v1", help="captions version (with --text-source caption)"
    )
    run.add_argument("--depth", type=int, default=50, help="ranked hits stored per query")
    run.add_argument("--limit", type=int, help="only run the first N queries (smoke)")
    run.set_defaults(func=cmd_run)

    captions_parser = bench_sub.add_parser("captions", help="thumbnail caption corpus operations")
    captions_sub = captions_parser.add_subparsers(dest="captions_command", required=True)
    cbuild = captions_sub.add_parser(
        "build", help="fetch thumbnails + caption them with the VLM (resumable)"
    )
    cbuild.add_argument("--materials", default="v1")
    cbuild.add_argument("--version", default="v1")
    cbuild.add_argument("--concurrency", type=int, default=90, help="in-flight caption requests")
    cbuild.add_argument("--limit", type=int, help="only caption the first N docs (smoke)")
    cbuild.add_argument("--dsn")
    cbuild.set_defaults(func=cmd_captions_build)

    judge = bench_sub.add_parser("judge", help="pool + grade (LLM) all runs' candidates")
    judge.add_argument("--runs", required=True, help="comma-separated run ids")
    judge.add_argument("--materials", default="v1")
    judge.add_argument("--queries", default="v1")
    judge.add_argument("--depth", type=int, default=20, help="pool depth per run")
    judge.add_argument("--negatives", type=int, default=5)
    judge.add_argument("--concurrency", type=int, default=8)
    judge.add_argument("--max-pairs", type=int, help="budget guard: stop after N new pairs")
    judge.add_argument(
        "--rpm",
        type=float,
        default=0.0,
        help="requests/min cap (0 = VPC_JUDGE_RPM from .env, else unlimited)",
    )
    judge.add_argument(
        "--mock", action="store_true", help="deterministic mock grading (pipeline test)"
    )
    judge.set_defaults(func=cmd_judge)

    score = bench_sub.add_parser("score", help="metrics for one run against labels")
    score.add_argument("--run", required=True)
    score.add_argument("--judge", default="latest")
    score.set_defaults(func=cmd_score)

    board = bench_sub.add_parser("leaderboard", help="aggregate + rank all runs")
    board.add_argument("--materials", default="v1")
    board.add_argument("--queries", default="v1")
    board.add_argument("--judge", default="latest")
    board.add_argument("--split", choices=["holdout", "train", "all"], default="holdout")
    board.set_defaults(func=cmd_leaderboard)

    export = bench_sub.add_parser("export", help="flat CSV exports: queries, judgments, run hits")
    export.add_argument("--materials", default="v1")
    export.add_argument("--queries", default="v1")
    export.add_argument("--judge", default="latest")
    export.set_defaults(func=cmd_export)

    report = bench_sub.add_parser(
        "report", help="self-contained HTML report: leaderboard + data browser"
    )
    report.add_argument("--materials", default="v1")
    report.add_argument("--queries", default="v1")
    report.add_argument("--judge", default="latest")
    report.add_argument("--split", choices=["holdout", "train", "all"], default="holdout")
    report.set_defaults(func=cmd_report)

    calibrate_parser = bench_sub.add_parser("calibrate", help="human calibration loop")
    calibrate_sub = calibrate_parser.add_subparsers(dest="calibrate_command", required=True)
    cexport = calibrate_sub.add_parser("export", help="write the human worksheet")
    cexport.add_argument("--judge", default="latest")
    cexport.add_argument(
        "--blind",
        action="store_true",
        help="hide judge grade/reason columns so the human grades blind",
    )
    cexport.set_defaults(func=cmd_calibrate_export)
    cscore = calibrate_sub.add_parser("score", help="score a filled worksheet (kappa report)")
    cscore.add_argument("--worksheet", required=True)
    cscore.add_argument("--judge", default="latest")
    cscore.set_defaults(func=cmd_calibrate_score)

    smoke = bench_sub.add_parser("smoke", help="connectivity check for judge + embeddings")
    smoke.set_defaults(func=cmd_smoke)
