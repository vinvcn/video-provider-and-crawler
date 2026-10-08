# AGENTS.md — video-provider-and-crawler

## 1. 项目目标

两阶段:(1) 爬取 pexels.com 全量视频元数据;(2) 建立 dense + sparse 混合索引与检索服务。完整决策记录见 `docs/handoff-2026-10-07.md`(权威起点文档,除非用户明说否则不要推翻其中已锁定决策)。

## 2. 仓库结构

```
crawler/                # Phase 1:爬取编排 + 规范化 + CLI(vpc 入口)
  browser_scripts/      # browser-use stdin 脚本(页内 fetch,在 browser-use 环境执行)
store/                  # 持久化:db.py / migrate.py / thumbnails.py
migrations/NNN_*.sql    # 前向编号迁移(单文件,无需 down;应用记录在 schema_migrations)
storage/                # 运行时数据(gitignored):spool/ thumbnails/ state/
docs/                   # handoff 与设计文档
tests/                  # 单元测试
```

## 3. 环境事实

- 本机:Python 3.11 + uv;Docker;无 GPU;`storage/` 在 NVMe 上
- stock-db:`pgvector/pgvector:pg16` 容器,`127.0.0.1:15433`,库名/用户 `postgres`;由共享中间件基建 `$HOME/infrastructure/middleware/dev` 提供,环境相关信息见本仓库 `docker-compose.yml` 顶部说明与 `ENV.md`
- 爬取依赖 `browser-use` CLI(本机已装);爬虫用**匿名专用 browser context**,不得使用个人 Chrome profile 的登录态
- **Python 网络必须走环境代理**(localhost:7897):本机 Python 直连 pexels CDN 会 TLS 超时(curl 直连可用,Python 不行)。httpx 默认 `trust_env=True`;`VPC_USE_PROXY=0` 可强制直连
- `secret-key` 请求头是 Pexels 前端公开常量(非个人凭证),默认值在 `crawler/pexels_v3.py`,可用 `VPC_PEXELS_SECRET` 覆盖;不要提交任何**个人** API key

## 4. 规范命令

```
make sync / test / lint / fmt
make up / down / migrate
make fetch-seed PAGES=N / fetch-search TERMS="..." / ingest / status
```

## 5. 编码规则

- ruff 格式化与检查(`make fmt && make lint`);行宽 100
- 纯逻辑必须配单元测试(`tests/`);I/O 与外部服务走集成路径
- 导出函数/类型要有 docstring;错误必须处理,不用 `_` 丢弃
- 提交前:`make test && make lint`
- 提交信息使用 Conventional Commits(`feat:`, `fix:`, `docs:` …);分支 `feat/<short-desc>`
- 不提交:storage/ 数据、密钥、spool 内容

## 6. 爬取安全约定(重要)

- 礼貌限速:seed 链页间 ≥1.5s,搜索池 ≤4 并发;见到 429/403 立即退避并停批
- 401 = secret-key 疑似轮换 → 停批并报告,不要重试轰炸
- 原始响应一律先落 spool 再入库(可重放、可审计)
