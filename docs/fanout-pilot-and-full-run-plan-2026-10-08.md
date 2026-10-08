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

## 7. 实测更新(2026-10-08 09:30)

批循环实跑三个批次后的边际新颖度序列(每次 250 词,每词 20 页):

| 词序号 | 新/词 | 覆盖率(批后) |
|---|---|---|
| 1–100(试点) | 420 | 2.53% → 8.86% |
| 101–350 | 346 | → 21.88% |
| 351–600 | 262 | → 31.75% |

拟合:**`new/term ≈ 480 × (1 − coverage)²`**(p≈2,三个点分别 2.27 / 1.96 / 1.94)。
比 §2 的 `线性` 模型衰减更快 —— 因为搜索结果是**相关性排序**的,词与词的重叠随覆盖率上升。

推论(用 `C(t) = 1 − 1/(1 + 0.000727·t)`):

| 目标 | 需要的词数(累计) | 请求数 | 时间 |
|---|---|---|---|
| 50% | ~1,375 | ~27.5k | ~5.5 h |
| 68%(新颖度地板 50/词) | ~2,890 | ~58k | ~12 h |
| 80% | ~5,500 | ~110k | ~22 h |
| 90% | ~12,400 | ~248k | ~50 h |

**关键结论:search 通道在 ~68% 附近饱和**(边际新颖度跌破 50 新/词),循环会自动停在那里。
越过 ~70% 需要别的通道(逐视频页导航被 Cloudflare 挡;feed 已耗尽),或接受该上限。

批耗时实测 ≈ 64 分钟(5,000 请求 + ingest),约 4,900 请求/小时。

## 9. 2026-10-08 新发现:按 ID 的 `_next/data` 通道(补全覆盖率)

之前判定"逐视频页被 Cloudflare 挡、无按 ID 通道",**实测推翻了后半句**:

- 真实导航打开视频页本身**能过** Cloudflare 的 JS 挑战(页面正常加载);
- 更关键:**Next.js 数据路由不受挑战**
  ```
  https://www.pexels.com/_next/data/<buildId>/en-us/video/<slug>-<id>.json
  ```
  返回 `200`,~85KB JSON,`pageProps.medium.attributes` 就是与 v3 列表**同构的完整属性**
  (id/slug/description/width/height/status/video_files/…)。
  已验证:最新 ID、最老 ID(852038)、以及**不在库里的** ID 都返回 200。
- 注意事项:
  - `buildId` 从任意已加载页面的 `window.__NEXT_DATA__.buildId` 读取;站点发版会变,
    取到 404 即说明 buildId 过期(重新读取即可)。
  - 无 locale 前缀的变体(`/_next/data/<build>/video/…`)返回 404;必须带 `en-us`。
  - 按 ID 的普通 v3 路由不存在:`/en-us/api/v3/videos/<id>` 404;`media/<id>`、
    `getty-media/video/<id>` 均 401(Bad API credentials)——**不要用**。
  - 搜索的 Next 数据路由**不能**绕过 20 页上限(p21+ → 404),该上限是服务端的。

### 成本对比(补全剩余 ID)

| 通道 | 每个视频的请求数 | 覆盖 | 说明 |
|---|---|---|---|
| 搜索扇出 | 20 请求 / 480 条 ≈ **0.042** | 到 ~68% 饱和 | 便宜 24 倍,但够不着长尾 |
| 按 ID `_next/data` | **1** | **100%**(目录内任意 ID) | ~85KB/条;gap 213k 条 ≈ 18GB spool |

按实测 ~5k 请求/小时:补 ~213k 条(68% → ~100%)约 **40+ 小时**;补当前全部缺口
(454k)约 90 小时。若 pool 提到 4-6 并发且单请求 ~1s,有望压到 15-20 小时量级。

### 由此得到的策略

1. **继续搜索扇出到饱和**(最便宜,正在跑);
2. **再按 ID 补缺口**,优先按 `catalog_videos.lastmod`(新/近期变更的先补);
3. 这条通道同时解决**未来增量**:新 ID 或 lastmod 变化的条目可以按 ID 精准刷新。

**已实现(2026-10-08):** `vpc fetch-ids --limit N [--order lastmod|id|random]` ——
读 buildId → 拉数据路由 → 落 `spool/ids/` → `vpc ingest --kind ids`;冒烟 5 条全绿。

**质量注意:** 数据路由返回的 `tags` 数明显少于列表/搜索(3–10 vs 40–50;列表用 `seo_tags=true`)。
补全后建议挑选 tags 偏少的行做一次定向刷新(先验证数据路由是否接受 `seo_tags`)。

## 10. 复现

```bash
# 单批
uv run vpc fetch-search --terms "$(python3 /tmp/opencode/pick_terms.py 250)" --pages-per-term 20 --pool 4 --retries 60
uv run vpc ingest --skip-thumbnails
# 覆盖率
docker exec dev-middleware-stock-db psql -U postgres -d stock -c \
  "SELECT round((SELECT count(*) FROM stock_videos)*100.0/(SELECT count(*) FROM catalog_videos),2)"
```
