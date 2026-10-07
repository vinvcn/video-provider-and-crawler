# Phase 1 MVP 验收 — 2026-10-07

状态:**通过**。爬取通道(seed 链 + 搜索扇出)→ spool → 入库 → 缩略图全链路端到端验证完成,指标见下。

## 1. 结果摘要

| 指标 | 结果 |
|---|---|
| 入库总量(pexels_id 唯一) | **9,472** 条 |
| 来源构成 | seed 链 1,926(107 页 ×18)+ 搜索 7,608(480 请求,320 页有数据) |
| 交叉重叠 / 批内重复 | seed∩search = 62;搜索批内重复 54;总去重 116(1.2%) |
| 缩略图落盘 | **9,472 / 9,472(100%),0 失败**;磁盘 1.9GB(平均 ~205KB/张,w=1440) |
| tags 缺失率 | **0%**(0 / 9,472) |
| description 缺失 | 1 条(9,471 / 9,472);title / duration / fps / video_files / user / created_at / orientation 均 100% |
| 4K 覆盖率 | **60.3%**(最长边 ≥3840 宽高任一);≥2560 = 63.9%,≥1920 = 99.3% |
| 断点续爬 | ✅ 从 `seed-000003` 成功接续(`seed_pages=2, cursor=2026-10-01T13:45:01Z` → `seed_pages=107, cursor=2026-03-24T13:55:01Z`) |
| 幂等入库 | ✅ 旧 spool 重放 upsert 无重复(第一轮 36→36) |
| DB 体积 | 76MB(9472 行,含 raw jsonb + video_files jsonb) |

## 2. 用时明细

| 阶段 | 用量 | 备注 |
|---|---|---|
| 断点续爬复测(seed 5 页)+ ingest | ~1–2 min | seed-000003..000007 |
| seed 链 100 页 | 282s(≈2.7s/页,串行) | pace 1.5s + 请求 ~1.2s |
| 搜索 480 页(pool 4) | 336s(≈0.7s/请求) | 含 160 个空页(见 §3.1) |
| ingest + 9,346 张缩略图(8 并发走代理) | **14m20s** | 0 失败 |
| **本阶段合计** | **≈26 min** | 不含前期会话 |

## 3. 发现(对既有文档有修正)

### 3.1 内部 v3 搜索 API 单查询上限 = 20 页 / 480 条(修正:"10,000 条")

16 个词全部在 p1–p20 返回 24 条/页, **p21 起全部 0 条**(160 空页,白耗 1/3 请求量)。放大搜索覆盖只能**加词**,不能加页。后续 `fetch-search` 建议 `--pages-per-term 20`。

### 3.2 4K 覆盖率应以分辨率计,不以 `uhd` 标签计(低估 2.5 倍)

`video_files.quality == "uhd"` 只有 2,323 条(24.5%),但实测存在 3840×2160 文件被标注 `hd` 的情况。以最长边 ≥3840 计:**5,714 / 9,472 = 60.3%**。`vpc status` 的 `with_uhd`(标签口径)应解读为下界。

### 3.3 seed 链串行节奏 → 全量外推 ≈27h+,与 handoff §2 "4–7h(4–6 并发)"不符

seed 链是链式游标,天然串行,无法并发。实测 2.7s/页;36.9k 页 → **≈28h**。全量前需拍板:分片多晚跑 / 降低 pace(实测 15 连发无 429,1.5s pacing 偏保守)/ 或接受多日。另:游标约 1.8 天/页,与"每页游标约前进 1 天"描述接近但不精确。

> **后续实测作废本节外推**(见 `docs/phase1-full-feed-and-catalog-2026-10-07.md`):feed 全量只有 524 页,走完约 21 分钟;它是库的 ~1.4% 子集。量级通道改为 search 扇出 + 目录 oracle。

### 3.4 缩略图体积:全量外推 ≈136GB

平均 205KB(w=1440 压缩图)。664k 条时 ≈136GB(现 NVMe 剩余 ~333GB,可行;若要省可换 `w=480`,约降 4 倍)。属于存储决策输入,不阻塞。

### 3.5 全量前建议补的硬化项(未实施)

- **401 停批 + 告警**(AGENTS.md §6 已约定;`fetch_batch.py` 目前仅单次重试)
- 429/403 退避策略同样缺失
- 缩略图失败为静默计数(本次 0 失败,不构成问题)

## 4. 复现命令

```bash
make up && make migrate
uv run vpc fetch-seed --pages 5          # 断点续爬:seed-000003 起
uv run vpc ingest && uv run vpc status
make fetch-seed PAGES=100
uv run vpc fetch-search --terms "ocean waves,sunrise,forest,waterfall,mountains,city night,street traffic,aerial city,skyscraper,business woman,fitness workout,couple walking,cat,dog,birds,fire flames" --pages-per-term 30 --pool 4
uv run vpc ingest --thumb-limit 20000
```

验证 SQL(节选):

```sql
-- 4K 覆盖(最长边口径)
WITH t AS (SELECT (SELECT max(greatest((f->>'width')::int,(f->>'height')::int))
                   FROM jsonb_array_elements(video_files) f) md FROM stock_videos)
SELECT count(*) FILTER (WHERE md>=3840) ge4k, count(*) total FROM t;
```

## 5. 下一步

1. 全量策略拍板(§3.3 三条路线)+ 硬化项 §3.5(`feat/` 分支 + 测试)
2. 全量词表扇出(修正后:每次查询 480 条,覆盖长尾需数百词;来源:trending 端点 + tags 高频词)
3. Phase 2 可并行:embedding seam + 选型(仓库 handoff §5)
