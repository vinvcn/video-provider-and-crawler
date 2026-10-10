"""Self-contained HTML report: leaderboard + interactive data browser.

One file, no external dependencies: the leaderboard, pairwise significance,
per-cut scores, and the full data behind them (queries, judgments, run hits)
embedded as JSON with a small vanilla-JS browser. Point, click, filter —
the audit trail of a benchmark run without touching the CSVs.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from bench import util
from bench.judge import label_records
from bench.leaderboard import build_leaderboard
from bench.materials import Materials
from bench.queries import QuerySet
from bench.run import find_runs, load_run

MAX_HITS = 50
MAX_REASON_CHARS = 300
MAX_TAG_CHARS = 150

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1419; --panel:#161d26; --line:#28323f; --fg:#d8e1ea; --dim:#7a8a9c;
          --accent:#4cc2ff; --g3:#22c55e; --g2:#14b8a6; --g1:#f59e0b; --g0:#64748b; --gu:#ef4444; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f5f7fa; --panel:#ffffff; --line:#dde4ec; --fg:#1c2733; --dim:#5c6b7a; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line); }
  header h1 { margin:0 0 4px; font-size:19px; }
  header .meta { color:var(--dim); font-size:12.5px; }
  nav { display:flex; gap:4px; padding:10px 22px 0; border-bottom:1px solid var(--line); }
  nav button { background:none; border:1px solid var(--line); border-bottom:none; color:var(--dim);
               padding:8px 16px; cursor:pointer; border-radius:6px 6px 0 0; font-size:13.5px; }
  nav button.on { color:var(--fg); background:var(--panel); border-color:var(--line); font-weight:600; }
  main { padding:18px 22px; }
  section { display:none; }
  section.on { display:block; }
  table { border-collapse:collapse; width:100%; background:var(--panel); font-size:13px; }
  th, td { border:1px solid var(--line); padding:7px 10px; text-align:right; }
  th { color:var(--dim); cursor:pointer; user-select:none; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  td:first-child, th:first-child { white-space:nowrap; }
  .strat { font-weight:600; color:var(--accent); }
  .num-best { font-weight:700; }
  .filters { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
  .filters input, .filters select { background:var(--panel); color:var(--fg);
    border:1px solid var(--line); border-radius:6px; padding:6px 10px; font-size:13px; }
  .filters input { width:220px; }
  #browser { display:grid; grid-template-columns: 340px 1fr; gap:16px; }
  #qlist { max-height:72vh; overflow:auto; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; }
  #qlist .q { padding:8px 12px; border-bottom:1px solid var(--line); cursor:pointer; font-size:12.5px; }
  #qlist .q:hover { background:rgba(76,194,255,.08); }
  #qlist .q.sel { border-left:3px solid var(--accent); background:rgba(76,194,255,.12); }
  #qlist .q .cat { color:var(--dim); font-size:11px; margin-right:6px; }
  #qdetail { min-width:0; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:14px 16px; margin-bottom:14px; }
  .card h3 { margin:0 0 8px; font-size:14px; color:var(--accent); }
  .kv { color:var(--dim); font-size:12.5px; margin-bottom:10px; }
  .badge { display:inline-block; min-width:20px; text-align:center; border-radius:5px;
           padding:0 5px; font-size:12px; font-weight:700; color:#fff; margin-right:6px; }
  .b3 { background:var(--g3); } .b2 { background:var(--g2); }
  .b1 { background:var(--g1); } .b0 { background:var(--g0); } .bu { background:var(--gu); }
  .bq { background:var(--dim); }
  .doc { font-size:12.5px; margin:4px 0; }
  .doc .meta { color:var(--dim); }
  .doc .reason { color:var(--dim); font-size:12px; }
  .arm { margin-bottom:12px; }
  .arm h4 { margin:0 0 6px; font-size:13px; color:var(--fg); }
  .pair-table td, .pair-table th { text-align:left; }
  footer { padding:14px 22px; color:var(--dim); font-size:12px; border-top:1px solid var(--line); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="meta" id="meta"></div>
</header>
<nav>
  <button data-tab="board" class="on">排行榜</button>
  <button data-tab="pairs">显著性对比</button>
  <button data-tab="cuts">切分报告</button>
  <button data-tab="browser">数据浏览(查询 · 判分)</button>
</nav>
<main>
  <section id="board" class="on"></section>
  <section id="pairs"></section>
  <section id="cuts"></section>
  <section id="browser">
    <div>
      <div class="filters">
        <input id="qfilter" placeholder="搜索查询文本 / qid">
        <select id="fcat"><option value="">全部类目</option></select>
        <select id="fsplit"><option value="">全部切分</option></select>
      </div>
      <div id="qlist"></div>
    </div>
    <div id="qdetail"><div class="card"><div class="kv">从左侧选择一条查询,查看各臂 top-10 与全部判分结果。</div></div></div>
  </section>
</main>
<footer id="footer"></footer>
<script>
const DATA = __DATA__;
const $ = (sel) => document.querySelector(sel);

function gradeBadge(g, status) {
  if (status === "unparsed") return '<span class="badge bu">U</span>';
  if (g === null || g === undefined) return '<span class="badge bq">?</span>';
  return '<span class="badge b' + g + '">' + g + '</span>';
}
function docTitle(id) {
  const d = DATA.docs[id];
  return d ? d.t : ("doc " + id);
}
function fmt(n, digits) { return Number(n).toFixed(digits === undefined ? 4 : digits); }

/* ── leaderboard ─────────────────────────────────────────── */
function renderBoard() {
  const rows = DATA.rows.slice();
  let html = '<table id="boardTable"><thead><tr><th>#</th><th data-k="strategy">策略</th>' +
    '<th data-k="ts">语料</th>' +
    '<th data-k="run_id">run</th><th data-k="holdout">nDCG@10 holdout ▽</th>' +
    '<th data-k="full">nDCG@10 full</th><th data-k="recall10">R@10</th>' +
    '<th data-k="mrr10">MRR@10</th><th data-k="p50">p50 ms</th>' +
    '<th data-k="p95">p95 ms</th><th data-k="nulls">null@10</th></tr></thead><tbody>';
  const best = Math.max(...rows.map(r => r.holdout));
  rows.forEach((r, i) => {
    html += '<tr><td>' + (i + 1) + '</td><td class="strat">' + r.strategy + '</td>' +
      '<td style="color:var(--dim)">' + r.ts + '</td>' +
      '<td style="color:var(--dim)">' + r.run_id + '</td>' +
      '<td class="' + (r.holdout === best ? 'num-best' : '') + '">' + fmt(r.holdout) + '</td>' +
      '<td>' + fmt(r.full) + '</td><td>' + fmt(r.recall10) + '</td><td>' + fmt(r.mrr10) + '</td>' +
      '<td>' + fmt(r.p50, 1) + '</td><td>' + fmt(r.p95, 1) + '</td><td>' + r.nulls + '</td></tr>';
  });
  html += '</tbody></table>';
  html += '<p style="color:var(--dim);font-size:12.5px;margin-top:10px">' + DATA.meta.board_note + '</p>';
  $("#board").innerHTML = html;
  let sortKey = "holdout", asc = false;
  document.querySelectorAll("#boardTable th[data-k]").forEach(th => {
    th.onclick = () => {
      sortKey = th.dataset.k; asc = !asc;
      DATA.rows.sort((a, b) => (a[sortKey] - b[sortKey]) * (asc ? 1 : -1));
      renderBoard();
    };
  });
}

/* ── pairwise significance ───────────────────────────────── */
function renderPairs() {
  let html = '<table class="pair-table"><thead><tr><th>A</th><th>B</th><th>Δ(A−B)</th>' +
    '<th>95% CI</th><th>结论</th></tr></thead><tbody>';
  DATA.pairwise.forEach(p => {
    const verdict = p.win === "a" ? "A 显著更优" : p.win === "b" ? "B 显著更优" : "无显著差异";
    html += '<tr><td class="strat">' + p.a + '</td><td class="strat">' + p.b + '</td>' +
      '<td>' + (p.diff >= 0 ? "+" : "") + fmt(p.diff) + '</td>' +
      '<td>[' + fmt(p.lo) + ", " + fmt(p.hi) + ']</td><td>' + verdict + '</td></tr>';
  });
  html += '</tbody></table>';
  $("#pairs").innerHTML = html;
}

/* ── cuts ─────────────────────────────────────────────────── */
function renderCuts() {
  const strategies = Object.keys(DATA.cuts);
  let html = '<div class="filters"><select id="cutStrat">' +
    strategies.map(s => '<option value="' + s + '">' + s + '</option>').join('') +
    '</select></div><div id="cutTable"></div>';
  $("#cuts").innerHTML = html;
  $("#cutStrat").onchange = () => {
    const s = $("#cutStrat").value;
    const cuts = DATA.cuts[s];
    let t = '<table><thead><tr><th>切分</th><th>holdout nDCG@10</th><th>n</th></tr></thead><tbody>';
    Object.keys(cuts).sort().forEach(k => {
      t += '<tr><td>' + k + '</td><td>' + fmt(cuts[k].mean) + '</td><td>' + cuts[k].n + '</td></tr>';
    });
    t += '</tbody></table>';
    $("#cutTable").innerHTML = t;
  };
  $("#cutStrat").onchange();
}

/* ── data browser ────────────────────────────────────────── */
let selectedQid = null;
function judgmentsByQid() {
  const map = {};
  DATA.judgments.forEach(j => {
    (map[j.q] = map[j.q] || []).push(j);
  });
  Object.values(map).forEach(list => list.sort((a, b) => (b.g ?? -1) - (a.g ?? -1)));
  return map;
}
const JMAP = judgmentsByQid();

function renderQueryList() {
  const text = $("#qfilter").value.toLowerCase();
  const cat = $("#fcat").value, split = $("#fsplit").value;
  const list = DATA.queries.filter(q =>
    (!text || q.text.toLowerCase().includes(text) || q.qid.includes(text)) &&
    (!cat || q.category === cat) && (!split || q.split === split));
  let html = list.map(q =>
    '<div class="q' + (q.qid === selectedQid ? ' sel' : '') + '" data-qid="' + q.qid + '">' +
    '<span class="cat">' + q.category + (q.lang === "zh" ? "·zh" : "") + '</span>' +
    q.text + '</div>').join('');
  if (!list.length) html = '<div class="q" style="color:var(--dim)">无匹配查询</div>';
  $("#qlist").innerHTML = html;
  document.querySelectorAll("#qlist .q[data-qid]").forEach(el => {
    el.onclick = () => { selectedQid = el.dataset.qid; renderQueryList(); renderDetail(); };
  });
}

function renderDetail() {
  const q = DATA.queries.find(x => x.qid === selectedQid);
  if (!q) return;
  const judged = JMAP[q.qid] || [];
  const relevant = judged.filter(j => j.g >= 2).length;
  let html = '<div class="card"><h3>' + q.text + '</h3><div class="kv">' +
    'qid ' + q.qid + ' · ' + q.source + '/' + q.category + ' · ' + q.lang +
    ' · 难度 ' + q.difficulty + ' · 切分 ' + q.split +
    (q.facets.length ? ' · 分面 ' + q.facets.join("/") : '') +
    (q.hard ? ' · 硬约束 ' + q.hard : '') +
    ' · 已判 ' + judged.length + ' 对,相关(≥2)' + relevant + ' 条</div>';

  DATA.runs.forEach(run => {
    const hits = (run.hits[q.qid] || []).slice(0, 10);
    if (!hits.length) return;
    html += '<div class="arm"><h4>' + run.strategy + ' · ' + run.ts + ' — top 10</h4>';
    hits.forEach(([docId, score], i) => {
      const j = (judged.find(x => x.d === docId) || {});
      const d = DATA.docs[docId] || {};
      html += '<div class="doc">' + gradeBadge(j.g, j.s) + '<b>' + (i + 1) + '</b>. ' +
        docTitle(docId) +
        ' <span class="meta">(' + fmt(score, 3) + (d.w >= 3840 ? " · 4K" : "") +
        (d.d ? " · " + d.d + "s" : "") + ')</span></div>';
    });
    html += '</div>';
  });
  html += '</div>';

  html += '<div class="card"><h3>判分结果(' + judged.length + ' 对,按分值排序)</h3>';
  judged.forEach(j => {
    const d = DATA.docs[j.d] || {};
    html += '<div class="doc">' + gradeBadge(j.g, j.s) + docTitle(j.d) +
      ' <span class="meta">doc ' + j.d + (d.w >= 3840 ? " · 4K" : "") + '</span>' +
      (j.r ? '<div class="reason">' + j.r + '</div>' : '') + '</div>';
  });
  html += '</div>';
  $("#qdetail").innerHTML = html;
}

/* ── wiring ──────────────────────────────────────────────── */
document.querySelectorAll("nav button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("nav button").forEach(b => b.classList.toggle("on", b === btn));
    document.querySelectorAll("main section").forEach(s => s.classList.toggle("on", s.id === btn.dataset.tab));
  };
});
$("#meta").textContent = DATA.meta.line;
$("#footer").textContent = DATA.meta.footer;
[...new Set(DATA.queries.map(q => q.category))].sort().forEach(c => {
  $("#fcat").insertAdjacentHTML("beforeend", '<option value="' + c + '">' + c + '</option>');
});
["train", "holdout"].forEach(s => {
  $("#fsplit").insertAdjacentHTML("beforeend", '<option value="' + s + '">' + s + '</option>');
});
$("#qfilter").oninput = renderQueryList;
$("#fcat").onchange = renderQueryList;
$("#fsplit").onchange = renderQueryList;
renderBoard(); renderPairs(); renderCuts(); renderQueryList(); renderDetail();
</script>
</body>
</html>
"""


def _calibration_note(calibration: dict | None) -> str:
    """Human-calibration line for the report header, when one exists."""
    if not calibration:
        return ""
    return (
        f"人工校准:{calibration['n_graded']} 对,一致率 {calibration['exact_agreement']:.0%},"
        f"Cohen's κ {calibration['cohens_kappa']:.2f}"
        f"({'通过' if calibration['gate_passed'] else '未达闸门'})。"
    )


def build_report(
    root: Path,
    materials: Materials,
    queries: QuerySet,
    judge_version: str,
    split: str = "holdout",
) -> Path:
    """Render the self-contained report page; returns its path."""
    report = build_leaderboard(root, materials, queries, judge_version, split=split)

    rows = []
    for entry in report["runs"]:
        scored = entry["scored"]
        split_metrics = scored["splits"].get(split, scored["splits"]["all"])
        rows.append(
            {
                "strategy": scored["strategy"],
                "ts": entry["run"]["manifest"]["spec"].get("text_source", "raw"),
                "run_id": scored["run_id"],
                "holdout": split_metrics["ndcg10"]["mean"],
                "full": scored["splits"]["all"]["ndcg10"]["mean"],
                "recall10": split_metrics["recall10"]["mean"],
                "mrr10": split_metrics["mrr10"]["mean"],
                "p50": scored["latency"]["search_ms_p50"],
                "p95": scored["latency"]["search_ms_p95"],
                "nulls": scored["null_grade_top10_total"],
            }
        )
    rows.sort(key=lambda row: -row["holdout"])

    cuts = {
        f"{entry['scored']['strategy']}·{entry['run']['manifest']['spec'].get('text_source', 'raw')}": (
            entry["cuts"].get(split, {})
        )
        for entry in report["runs"]
    }

    docs = {
        row.doc_id: {
            "t": row.title,
            "o": row.orientation,
            "d": row.duration,
            "w": row.width,
            "g": " ".join(row.tags)[:MAX_TAG_CHARS],
        }
        for row in materials.rows.values()
    }

    judgments = [
        {
            "q": record["qid"],
            "d": int(record["doc_id"]),
            "g": record.get("grade"),
            "s": record.get("status", ""),
            "r": str(record.get("reason", ""))[:MAX_REASON_CHARS],
        }
        for record in label_records(root / "judgments" / "labels.jsonl", judge_version)
    ]

    runs = []
    for run_dir in find_runs(root, materials.version, queries.version):
        run = load_run(run_dir)
        runs.append(
            {
                "strategy": run["manifest"]["spec"]["strategy"],
                "ts": run["manifest"]["spec"].get("text_source", "raw"),
                "run_id": run["run_id"],
                "hits": {qid: row["hits"] for qid, row in run["rows"].items()},
            }
        )

    payload = {
        "meta": {
            "title": f"bench 基线报告 · materials {materials.version} × queries {queries.version}",
            "line": (
                f"材料 {materials.version}({materials.content_hash[:10]}…) · "
                f"查询 {queries.version}({queries.content_hash[:10]}…) · "
                f"判分 {judge_version} · 主分 {split} nDCG@10 · "
                f"{len(rows)} 臂 / {len(queries.ordered)} 查询 / {len(judgments)} 判分对 · "
                f"生成于 {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M UTC')}"
            ),
            "board_note": (
                "主分 = holdout 平均 nDCG@10(点击表头排序);full 列并排展示防小样本过读。"
                + _calibration_note(report.get("calibration"))
                + "「数据浏览」页可逐条查询查看各臂 top-10 与判分理由。"
            ),
            "footer": (
                "原始数据:storage/bench/reports/ 下的 export-queries / export-judgments / "
                "export-hits CSV 与 leaderboard md/csv/summary。"
            ),
        },
        "calibration": report.get("calibration"),
        "rows": rows,
        "pairwise": [
            {
                "a": next(f"{r['strategy']}·{r['ts']}" for r in rows if r["run_id"] == pair["a"]),
                "b": next(f"{r['strategy']}·{r['ts']}" for r in rows if r["run_id"] == pair["b"]),
                "diff": pair["diff"],
                "lo": pair["ci"][0],
                "hi": pair["ci"][1],
                "win": pair["win"],
            }
            for pair in report["pairwise"]
        ],
        "cuts": cuts,
        "queries": [
            {
                "qid": queries.records[qid].qid,
                "text": queries.records[qid].text,
                "source": queries.records[qid].source,
                "category": queries.records[qid].category,
                "lang": queries.records[qid].lang,
                "facets": list(queries.records[qid].facets),
                "hard": util.canonical_json(dict(queries.records[qid].hard))
                if queries.records[qid].hard
                else "",
                "difficulty": queries.records[qid].difficulty,
                "split": queries.records[qid].split,
            }
            for qid in queries.ordered
        ],
        "docs": docs,
        "judgments": judgments,
        "runs": runs,
    }

    stem = f"bench-report-{materials.version}-{queries.version}-" + _safe(judge_version)
    out_path = root / "reports" / f"{stem}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.replace("__TITLE__", payload["meta"]["title"]).replace(
        "__DATA__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _safe(name: str) -> str:
    return name.replace("@", "_at_").replace("/", "_")
