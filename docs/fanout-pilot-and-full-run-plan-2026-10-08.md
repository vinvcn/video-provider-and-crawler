# 搜索扇出:试点结果与全量运行方案(2026-10-08)

上位文档:`docs/phase1-full-feed-and-catalog-2026-10-07.md`(feed 全量 + 目录 oracle)、
`docs/crawl-modes.md`(运行手册)、`docs/mvp-acceptance-2026-10-07.md`。

## 1. 试点设计与结果

- 样本:100 个词(50 个从 query 宇宙**随机**抽 / 50 个来自 `^[a-z][a-z ]{2,30}$` 干净池),每词 20 页(上限)
- 请求:2,000 页;全部成功(0 空词、0 提前结束)

| 指标 | 数值 |
|---|---|
| 原始条目 | 47,892 |
| 去重后 ID | 44,078(跨词重复仅 8%) |
| 对既有库新增 | **42,062** |
| 每词产出 | 479 条 / 479 ID(上限 480) |
| 试点 ID 在目录中 | 44,024 / 44,078 = **99.88%** |
| 库总量变化 | 16,805 → **58,913**(目录覆盖率 **2.53% → 8.86%**) |

两组(随机 vs 干净)产出几乎一致 → **词表不需要重过滤**;616,542 个 `^[a-z]+( [a-z]+){0,3}$`
候选词都在可用范围内(含看起来像噪声的词,如 `todays`、`pandem`,同样填满 480 条)。

## 2. 关键规律:边际新颖度随覆盖率线性衰减

试点结束时覆盖率 2.53%,实测 420 新/词;模型 `new/term ≈ 480 × (1 − coverage)` 预测 468
(略高,因词间存在相关性)。据此用指数趋近模型:

```
N(t) = C × (1 − e^(−480·t / C)),  C = 665,233, t = 已用词数
```

## 3. 全量投影

| 目标覆盖率 | 需新增 ID | 词数 | 请求数 | 时间(实测 ~6k req/h) |
|---|---|---|---|---|
| 25% | ~149k | ~410 | ~8.2k | ~1.5 h |
| 50% | ~316k | ~960 | ~19k | ~3.5 h |
| 80% | ~515k | ~2,230 | ~45k | ~8 h |
| 90% | ~540k | ~3,190 | ~64k | ~11 h |

> 90% 以上受"搜索可达集合"限制:目录里可能有视频任何关键词都搜不到,真实上限由
> 目录 oracle 实测决定,不要强推 100%。

资源估计:spool ~7.7GB(64k 页 × ~120KB);入库后 DB ~4GB;缩略图按约定不下载。

## 4. 运行方案:批式扇出 + 停点判据

- 每批 **250 词**(≈5k 请求 ≈1h)→ `vpc ingest --skip-thumbnails` → 记录覆盖率与该批**边际新颖度**
- 停点:覆盖率 ≥ 90% **或** 边际新颖度 < 50 新/词(说明可用词已被吃干)
- 词表:每批从 `catalog_queries` 随机抽 `^[a-z][a-z ]{2,40}$`,按 slug 排除已爬词
- 失败自恢复已就位:`--retries`(掉线重连)、tab 劫持自恢复、401/403/429 立即停批;
  批与批之间可任意中断续跑(状态在 `state/search.json`)

循环脚本:`scripts/fanout_loop.sh`(环境变量 `TARGET_PCT` / `BATCH_TERMS` /
`MAX_BATCHES` / `NOVELTY_FLOOR` 可调;进度追加在 `/tmp/opencode/fanout-progress.log`,
每批原始输出在 `/tmp/opencode/fanout-batch-N.log`)。

## 5. 已知风险与后续

| 项 | 说明 |
|---|---|
| 搜索可达上限未知 | 用 oracle 覆盖率实测;缺口再评估(逐页导航重、不建议) |
| 缩略图 49,441 张待下载 | 按约定暂缓;随时 `uv run vpc ingest --thumb-limit N` |
| 浏览器 daemon 被共享 | 已有 tab 劫持自恢复;Chrome 进程死亡仍需手动拉起(`DISPLAY=:0 setsid nohup google-chrome &`) |
| 词表噪声 | 随机抽样已验证产出与干净池一致;无需过滤,继续随机 |

## 6. 复现

```bash
# 单批
uv run vpc fetch-search --terms "$(python3 /tmp/opencode/pick_terms.py 250)" --pages-per-term 20 --pool 4 --retries 60
uv run vpc ingest --skip-thumbnails
# 覆盖率
docker exec dev-middleware-stock-db psql -U postgres -d stock -c \
  "SELECT round((SELECT count(*) FROM stock_videos)*100.0/(SELECT count(*) FROM catalog_videos),2)"
```
