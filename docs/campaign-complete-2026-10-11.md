# Pexels 全量爬取战役完成(2026-10-11)

## 0. TL;DR

**目录 oracle 全覆盖收官**:video sitemap 的 665,233 个条目,665,501 行入库(**100.04%**,含少量目录外 ID)、
647 个确认下架(`missing_since`)、可行动缺口 **0**。从 2026-10-08 14:25 的 365,103(54.88%)起,
历时约 **66 小时**。Phase 1 目标达成。

## 1. 最终状态

| 项 | 值 |
|---|---|
| stock_videos | **665,501 行**(100.04% / catalog 665,233) |
| catalog 缺口 | **0**(全部已入库或标记下架) |
| 确认下架(missing_since) | 647 |
| tags | 570,480 distinct(行均 ~46) |
| 4K(width≥3840) | 302,934 |
| DB / spool / 磁盘 | 5.7GB / ~8.4GB / 余 733GB |
| 缩略图 | **656,029 待下载**(按决策暂缓) |
| 向量 / 索引 | 0(Phase 2 未开始) |

## 2. 通道贡献

| 通道 | 规模 | 覆盖角色 |
|---|---|---|
| seed 链 | 524 页 / 9,416 条 | 热门 baseline(10-07 已走完) |
| search 扇出 | 66,798 done_keys(250 词/批 × ~40 批) | 语义长尾,至新词率 <50/词饱和 |
| **ids 数据路由** | **113,328 done_keys**(57 批 × 2,000) | **补缺口主力**:72% → 100% |

## 3. 时间线

| 时间(10-08 起) | 事件 |
|---|---|
| 14:25 | 交接:365,103(54.88%),search 扇出进行中 |
| 23:54 | 扇出触底(新词率 47<50)→ 72.19% |
| 23:55–10-09 01:49 | 首轮 gapfill(限 10 万)卡死:队头 8 个已下架 ID 触发 fail-streak 整批中止,仅完成 8k/100k |
| 10-09 06:54 | 修复后全缺口重跑(MAX_IDS=0) |
| 10-09 白天 | ~98% 命中率,速度 5,000–7,000 行/h |
| 10-10 凌晨 | **build 轮换事故**:旧 buildId 下所有请求 404 被误判下架,7,262 行误标(后清除,真下架仅 640) |
| 10-10 07:27 | **/tmp/opencode 被另一会话清空**:循环/监控/Chrome 全灭,僵尸 worker 残留 |
| 10-10 09:25 | 恢复:Chrome 拉起、日志迁 `storage/logs/`、双循环重启 |
| 10-10 全天→10-11 晨 | 稳定爬取,~100% 命中,尾段零下架 |
| **10-11 08:48** | **`catalog gap is empty; stopping` → LOOP-DONE** |

## 4. 战役中的事故与修复(全部已提交)

| # | 事故 | 修复(提交) |
|---|---|---|
| 1 | ingest 每轮全量重放 spool(单批 148 万次 upsert,`n_tup_upd` 累计 823 万) | 增量水位线 + 未变行跳过(`311839c`) |
| 2 | 队头已下架 ID 堵死 gapfill(误报"浏览器掉线") | `missing_medium` 精确分类 + `catalog_videos.missing_since` 标记(`f322fa6`,迁移 004) |
| 3 | build 轮换 → 全量 404 被当"下架",7,262 误标 | 只信 `missing_medium`;裸 404 算失败;整块 404 时**重载页面再读 buildId**(`0ba6448`);误标已按 spool 证据清除 |
| 4 | /tmp 共享区被清空(另一会话)+ Chrome/harness 死亡 | 日志迁入 `storage/logs/`(`chore`);僵尸 worker 清除;`done_keys` 以 gap 查询为唯一事实源重置 |

## 5. 仓库与分支状态

- 全部爬取修复在 `feat/bench-harness` 分支(其上另一会话已建好 **评测 harness**:`bench/` 模块 + 136 测试,已推 origin);栈:`311839c → f322fa6 → 0ba6448 → …bench`
- **未合并 main**(待用户决定;历史上走 PR)
- 监控:`storage/logs/checkin.sh` 循环仍在跑(每 30 分钟一条,写入 `storage/logs/monitor.log`);战役已结束,可随时停
- BM25 seam(idf²)已由 bench 分支的 `0c16fc2` 修复(idf_side 参数 + 教科书等价测试)——原 D4 闭环

## 6. 下一步(待拍板)

1. **缩略图 656,029**:Phase 2 图像/caption 的前提;一条命令幂等开跑(带宽 ~几十 GB),或继续暂缓
2. **每日增量**:`vpc fetch-seed --head` 通道已就绪,可定时保新鲜度
3. **Phase 2**:评测 harness 已就绪 → dense 模型 A/B(`text-embedding-v4` vs vision-flash)→ 索引迁移(HNSW+GIN)→ 多维分面(设计见 `docs/phase2-*.md`)
4. 合并 `feat/bench-harness` → main
