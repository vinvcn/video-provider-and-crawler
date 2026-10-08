# video-provider-and-crawler

爬取 pexels.com 视频元数据并构建 dense + sparse 混合检索索引。

## 两阶段路线

| 阶段 | 目标 | 状态 |
|---|---|---|
| **Phase 1 · 爬取** | 摄取全量视频元数据(664,546 条 / 3,533 小时)+ 缩略图 | 🚧 进行中 |
| **Phase 2 · 索引** | dense + sparse 向量化 + 混合检索服务(模型留 seam) | 📋 规划 |

完整背景、爬取通道规格、全部决策记录与研究结论见 **[docs/handoff-2026-10-07.md](docs/handoff-2026-10-07.md)**(起点文档,勿删)。

## Quickstart

```bash
make sync            # uv sync(安装依赖 + vpc CLI)
make up              # 起共享基建 dev_middleware 中的 stock-db(pgvector/pgvector:pg16, 127.0.0.1:15433)
make migrate         # 应用 migrations/*.sql

# Phase 1 爬取(需要本机 Chrome + browser-use)
make fetch-seed PAGES=5          # popular seed 链,先小量验证
make ingest                      # spool → 规范化 → upsert → 缩略图下载

make status                      # DB 计数 + spool 统计
make test                        # pytest
```

## 结构

```
crawler/               # Phase 1:浏览器爬取 + 规范化
  browser_scripts/     # 经 browser-use stdin 执行的页内 fetch 脚本
store/                 # 持久化:DB 访问 / 迁移 / 缩略图
migrations/            # 前向编号 SQL 迁移
storage/               # 运行时数据(gitignored):spool/ thumbnails/ state/
docs/                  # handoff 与设计文档
tests/                 # 单元测试(纯逻辑)
```

## 爬取架构(一句话)

`vpc fetch-*` 编排器 → 把页内 fetch 脚本喂给 `browser-use`(匿名专用 browser context)→ 页内 `fetch()` 打 Pexels 内部 v3 API → 原始 JSON 落 spool → `vpc ingest` 规范化后 upsert 进 `stock_videos`。断点续爬靠 `storage/state/*.json`,原始 spool 可重放。
