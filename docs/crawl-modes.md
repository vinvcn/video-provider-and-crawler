# 爬取模式与运行手册

本文件是爬虫操作的速查:各模式用途、状态文件、spool 布局、失败语义与限速约定。
背景与决策见 `docs/handoff-2026-10-07.md`;实测结果见
`docs/mvp-acceptance-2026-10-07.md`、`docs/phase1-full-feed-and-catalog-2026-10-07.md`。

## 1. 命令总览

| 命令 | 用途 | 状态文件 | spool 目录 |
|---|---|---|---|
| `vpc fetch-seed --pages N` | 热门 feed 链式翻页(游标向下) | `state/seed.json` | `spool/seed/` |
| `vpc fetch-seed --head [--until ISO]` | **每日增量**:从实时 head 往下走,走到上次 head 标记停 | `state/seed-head.json` | `spool/seed-head/` |
| `vpc fetch-search --terms "a,b" --pages-per-term N` | 关键词搜索(每词 ≤20 页 / 480 条) | `state/search.json` | `spool/search/` |
| `vpc fetch-sitemaps --kind all` | 落 video/query sitemap(目录 oracle + 查询宇宙) | (DB 表) | `spool/sitemap/` |
| `vpc ingest [--kind K] [--skip-thumbnails]` | spool → 规范化 → upsert →(可选)缩略图 | — | — |
| `vpc status` | 库存计数 | — | — |
| `vpc migrate` | 前向迁移 | — | — |

所有浏览器抓取都由 `browser-use` 驱动(匿名 browser context 内页内 `fetch()`);
spec 通过临时文件传递;不下载视频、不使用官方 API key。

## 2. 模式语义
### 2.1 feed 链(`fetch-seed`)
- 游标 `seed=<ISO>` 是链式键:每页返回 `pagination.cursor`,下一页用它。
- 游标单调向下;feed 全量实测 **524 页 / 9,416 条**(已在 2026-10-07 走到 `SEED-EXHAUSTED`)。
- `--until <ISO>`:游标降到该时间点即停。
- `--head`:从实时 head 开始(忽略深层游标),走到 `state/seed-head.json` 里的
  `head_cursor` 标记即停;**只有整轮完成才推进标记**(中断不推进,保证不丢增量窗口)。
  首次运行(无标记)= 只抓 head 一页并写下标记(`HEAD-BOOTSTRAP`)。
  每日增量即:`vpc fetch-seed --head`(无需传参)。
  ⚠️ `--head` 路径**尚未实跑验证**(实现后一直有别的爬取占用浏览器),首次使用时先冒烟。
  完整每日例程见 `scripts/daily_incremental.sh`(head 差量 → ingest → 刷新 sitemap → 覆盖率;
  脚本头部有 cron 示例)。

### 2.2 搜索扇出(`fetch-search`)
- 每词最多 20 页(`page=21..25` 站点已在 robots.txt 明确 disallow)。
- 某页返回 0 条即认为该词已尽,**自动跳过该词剩余页**(`TERM-END`),省请求。
- `state["done_keys"]` 记录已完成页;重跑同词表自动续跑。

### 2.3 sitemap(`fetch-sitemaps`)
- `/sitemaps/` 在 robots.txt 白名单内,不受 Cloudflare 挑战;索引 → 27+14 个 `.gz` shard。
- 原始 shard 落 `spool/sitemap/`,解析后 upsert 进 `catalog_videos` / `catalog_queries`。
- 用途:覆盖率 oracle(665k ID)与增量对账(lastmod / 新 ID)。

## 3. 失败语义与重试

| 情况 | 行为 |
|---|---|
| 401 / 403 / 429 | **立即停批**(`ABORT status=…`),不重试轰炸 —— 401 疑似 secret-key 轮换,需人工处理 |
| 连接类失败(status 0)连续 8 次 | `ABORT fail-streak=8`,视为浏览器/守护进程掉线 |
| `--retries N` | 非零退出即自动重试(退避 5s→30s):每次都启动新的 harness 进程,可自动重连掉线的 CDP;状态文件保证续跑 |
| seed 单页失败 | 写 `PAGE-FAIL` 记录并停批,游标停在最后成功页,重跑续接 |
| Chrome 崩溃 | 需外部拉起(`DISPLAY=:0 setsid nohup google-chrome &`);harness 会提示
"Allow remote debugging?" 一次。状态文件保证 0 丢失 |
| 抓取 tab 被别的任务导航走 | `TAB-FOREIGN … (abandoning context)`:自动弃用被污染的 context、新建匿名 context + pexels tab,不碰别人的 tab |

> **共享 daemon 注意**:本机 `browser-use` daemon 是多项目共用的,别人(或其他 agent)
> 可能导航我们的 tab,导致页内 fetch 全部 CORS 失败(status 0)。爬虫已能自动恢复
> (§3 最后一行);如需进一步隔离可用命名 daemon(`BU_NAME=vpc browser-use …`)。

## 4. 限速约定(AGENTS.md)

- seed 链页间 ≥1.5s(`--pace-ms`,默认 1500)。
- 搜索池 ≤4 并发(`--pool`)。
- 见到 429/403 立即退避停批。

## 5. 覆盖率对账(目录 oracle)

```sql
-- 当前库占目录的比例
SELECT (SELECT count(*) FROM stock_videos) * 100.0 /
       (SELECT count(*) FROM catalog_videos) AS pct;

-- 仍未获得的目录 ID(按 ID 大小抽样)
SELECT c.pexels_id FROM catalog_videos c
WHERE NOT EXISTS (SELECT 1 FROM stock_videos s WHERE s.pexels_id = c.pexels_id)
ORDER BY c.pexels_id DESC LIMIT 20;
```

## 6. spool / 状态清单

```
storage/
  spool/seed/        seed-000001.json …(524 个)
  spool/seed-head/   head-YYYYMMDD-NNNNNN.json(每日增量)
  spool/search/      search-<slug>-pN.json
  spool/sitemap/     videos-NN.xml.gz / queries-NN.xml.gz
  state/seed.json          深层游标 + seed_pages
  state/seed-head.json     head_cursor 标记 + run_date
  state/search.json        done_keys
  thumbnails/<pexels_id>.jpg
```

原始响应先落 spool 再入库,全部可重放(`vpc ingest` 幂等,`pexels_id` 为键)。
