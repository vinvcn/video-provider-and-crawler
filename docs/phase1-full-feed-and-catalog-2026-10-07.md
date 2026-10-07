# Phase 1:feed 全量走完 + 目录 oracle(2026-10-07)

范围:A 步(热门 feed 走到耗尽·只存元数据)+ C 步(sitemap 目录/查询宇宙落库)。
上位文档:`docs/handoff-2026-10-07.md`(起点)、`docs/mvp-acceptance-2026-10-07.md`(MVP 指标)。本文档修正起点文档中"全量 ≈36.9k 请求"的估算。

## 1. 结论(先读这个)

- **热门 feed 的"全量"只有 524 页 / 9,416 条**(SEED-EXHAUSTED,游标下限 2017-03-08)。起点文档"≈36,900 请求"高估约 70 倍:feed 是该库约 **1.4%** 的精选子集,不是全集。
- **目录 oracle 已就位**:`catalog_videos` = **665,233** 个视频 ID;`catalog_queries` = **665,589** 条搜索查询。
- 当前库 **16,805 条** = feed 9,416 + search 7,608 − 交集 219;占目录 **2.53%**。
- 覆盖率核对正常:两通道 99.67% 的条目命中目录(缺 55 条 = 极新视频 + 少量未收录)。
- 推论:**"全量"不能由 feed 通道独立达成**(它已耗尽);量级通道 = search 扇出(词表来自查询宇宙)+ 目录 oracle 量化。

## 2. Feed 全量实测

| 项 | 数值 |
|---|---|
| 页数 / 唯一条数 | 524 页 / 9,416 条(链内重复仅 2) |
| 游标下限 | 2017-03-08(末页 n=4) |
| 游标步长 | 中位 2.76 天/页(最小 0.07,最大 906——尾部跳变) |
| 爬行速率 | ≈1,500 页/小时(约 2.4 秒/页) |
| 本次分段 | 2 / 105 / 109 / 308 页,实际爬行合计约 21 分钟 |
| 条目 created_at 分布 | 2016:6 · 2017:19 · 2018:18 · 2019:167 · 2020:1046 · 2021:1604 · 2022:339 · 2023:755 · 2024:1249 · 2025:2489 · 2026:1726 |

中断史:run2(105 页)后 Chrome 进程崩溃停摆约 2 小时;重启 Chrome 后由 seed-000217 接续,run3 一口气走完 308 页。**状态逐页持久化,3 次中断 0 数据丢失。**

## 3. 目录与查询宇宙(C 步)

| 表 | 行数 | 来源 |
|---|---|---|
| catalog_videos | 665,233 唯一 ID(sitemap 条目 665,406,173 条重复) | `en-US/video-sitemap*.xml.gz`(27 shard,~65MB) |
| catalog_queries | 665,589 条(14 shard) | `en-US/video-search-queries-sitemap*.xml.gz` |

- ID 范围 852,038 - 40,083,338;条目带 slug + lastmod(lastmod 集中于 2024+:批量刷新痕迹,作"变更信号"用,不是创建时间)。
- 实现:`vpc fetch-sitemaps`(+迁移 002 + 单测,commit `3c440fa`);原始 shard 落 `storage/spool/sitemap/`,可重放。
- oracle 缺口:约 0.1% 的站点视频(非英文标题等)不在任何语区 sitemap(de-DE 抽检同为缺失)→ 目录按 ~99.9% 完整性解读。

## 4. 当前库存与质量

- 16,805 条;4K(最长边 ≥3840)覆盖率 **59.8%**;缺 tags 1 条(18084535,标题为空,站点侧脏数据)。
- 缩略图:9,472 张在盘,**7,333 张待下载**(本轮按约定未下载,`vpc ingest` 随时补齐)。
- 目录对账:16,750 / 16,805 命中;缺 55 条(50 feed + 5 search)。
- 体积:DB 327MB · spool 162MB · 缩略图 1.9GB。

## 5. 策略更新(全量与增量)

1. **全量 = search 扇出**:665k 查询词(含噪声,需筛选)× ≤20 页/词;先做"词表命中率"试点(几百词样本 → 覆盖率/token 成本),再定全量批次。目录 oracle 让每步覆盖率可测。
2. **增量 = 双信号**:feed head 游标差量(快)+ sitemap new-ID/lastmod 差量(对账)。需要小功能 `fetch-seed --until <游标>`(待排期)。
3. **硬化**:401 停批告警;Chrome 崩溃自动恢复包装(本轮证明状态机可用,缺自动化)。

## 6. 复现

```bash
make fetch-seed PAGES=5000        # 从断点续走,直到 SEED-EXHAUSTED
uv run vpc ingest --skip-thumbnails
make fetch-sitemaps KIND=all
```

对账 SQL(例):

```sql
SELECT count(*) FROM stock_videos s
JOIN catalog_videos c USING (pexels_id);      -- 命中目录
SELECT count(*) FROM catalog_videos c
WHERE NOT EXISTS (SELECT 1 FROM stock_videos s WHERE s.pexels_id = c.pexels_id);  -- 缺口
```
