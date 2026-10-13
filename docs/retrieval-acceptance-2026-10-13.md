# 检索服务验收记录（2026-10-13/14）

对应 PR：`feat/service-docker`（容器化 + 打磨 + Airflow DAG）。
本文件把当日会话里测得、散落在日志与 PR 评论中的验收数据落到仓库里。

## 1. 延迟（live 665k 库，ef_search=100）

| 场景 | 结果 |
| --- | --- |
| 冷首查（HNSW 页未进 page cache） | ~1.3–3.9 s |
| **启动预热后首查**（`_warm_indexes`） | dense 17 ms / sparse 14 ms |
| 同词条热查（嵌入缓存命中） | 14–43 ms 端到端 |
| 硬过滤热查（portrait + 时长 ≥10s） | 43 ms 端到端 |
| `exclude_urls` 命中 | 剔除 + 候补补位，k 仍填满 |

结论：冷启动成本由启动预热覆盖；查询嵌入 API（~250ms 未缓存）是用户侧主要成本，
热态端到端稳定在几十毫秒级。

## 2. 端到端（vbl → MPT → vpc-retrieval）

- 2026-10-14 01:35，MPT 任务 `d2b9586e…`（`video_source="library"`）产物
  `final-1.mp4`：h264，1080×1920，12.07s；任务 state=1。
- 素材溯源：`script.json → material_sources` 5 条全部 `provider: library`
  （asset_id 16244927 / 35170756 / 39924139 / 9921064 / 39611010）。
- 检索服务查询日志（vbl 栈挂载卷）记录到真实调用：dense 213ms / sparse 286ms。
- 零 Pexels 官方 API 用量；素材阶段无需任何官方 key。
- 详见 vbl PR #356 的 “E2E acceptance” 评论。

## 3. 缩略图死链重试（507 行）

- `vpc ingest` 重试：`{"downloaded": 0, "failed": 507}` —— 507 行
  `thumbnail_path IS NULL` 确认为永久 CDN 死链，无自愈；行内保留
  `thumbnail_url` 供下游引用，不再重试。

## 4. 容器与网络

- `vpc-retrieval`（本仓库 compose 与 vbl 栈两种形态）健康检查均通过；
  MPT 容器内 `http://vpc-retrieval:8787/healthz` 200（共享网络 DNS）。
- 宿主/容器 `vpc search` 与 HTTP `/search` 结果一致（共用 `store/search.py`）。
