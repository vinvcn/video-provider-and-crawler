# Phase 2 检索索引设计(合并提案,2026-10-08)

Phase 1 爬取仍在跑(365,103 / 665,233 = 54.88%),本文**只做 Phase 2 索引设计**,不改任何运行中的代码。
结论综合四份一手来源调研 + 本机实测,来源见 §9。**没有锁定的决策标了 ⬜**,等用户拍板;其余为按现有
锁定决策推导出的默认值。

## 0. 目标与约束(来自 handoff,不再重开)

- 用途:视频创作者按**自然语言概念**检索 b-roll,叠加结构化筛选(画幅/时长/分辨率/许可)。
- 关键指标:**快**(交互级,目标 sub-second p95)、**准**、**便宜**。
- 约束:单机、无 GPU、元数据 + 缩略图(视频文件不回源存储)、已在跑的 Postgres 16 + pgvector 0.8.7、
  dense + sparse 双路、模型留 seam。

## 1. TL;DR:推荐架构

**继续留在 Postgres + pgvector;两段式(召回→重排),dense + sparse 各一路,RRF 融合,SQL 预过滤。**

```
query(自然语言 + 可选筛选)
  │
  ├─ 查询理解:抽 facet(subject/motion/camera/setting/lighting/time/mood)
  │            → 结构化 facet 映射成 SQL WHERE;文本轻量同义词扩展(可关)
  │
  ├─ SQL 预过滤(两路都带):orientation / duration / resolution / license
  │
  ├─ Dense 臂  : HNSW cosine  on embedding  vector(768)         top-100
  └─ Sparse 臂 : HNSW inner-product on sparse_embedding sparsevec top-100   (本地 BM25)
                        │
                   RRF k=60 融合 → ~200 候选                      [阶段 1 召回]
                        │
                   cross-encoder 重排 → top-20~50                 [阶段 2 精排]
                        │
                   return top-30(检索服务契约)
```

分三层落地:**① 内容/向量 → ② 索引结构 → ③ 查询理解 + 融合 + 重排**,最后用 §7 的评测门收口。

## 2. 需要用户拍板的决策 ⬜

| # | 决策点 | 调研结论 | 我的建议 |
|---|---|---|---|
| D1 | Dense 文本模型 | `text-embedding-v4`@768 是 Qwen3-Embedding 服务版:原生 MRL(无需改列)、100+ 语言、**$0.07/1M → 全库约 $6**;handoff 原定 `tongyi-embedding-vision-flash` 是视觉模型,文本更强但**文本价更高($0.09)且固定 768** | **换用 `text-embedding-v4`@768**(保留 handoff 候选作备选)。只有当"与 MPT 查重门共用向量空间"是硬需求时才留 vision-flash |
| D2 | 缩略图跨模态(dense_image) | 行业惯例确实用**单帧预览图**做视频视觉代理(Shutterstock 2019);但单帧**无法表达运动**("slow motion/timelapse/aerial"),且图像 pass ≈ **$150 一次性**,约为文本 pass 的 20–30× | **v1 先不做**;下载缩略图后,仅对"视觉指定型"查询做 A/B,赢了再上 |
| D3 | 是否上阶段 2 重排 | CPU-only 下 0.6B cross-encoder 是交互延迟上限;33M 的 `answerai-colbert-small-v1`(Apache-2.0,BEIR 53.79)或 SiliconFlow 免费 `bge-reranker-v2-m3` 是低保真选择 | **v1 先只做阶段 1(RRF),留重排接口**;评测显示 top-10 精度不足再加 |
| D4 | BM25 seam 修正 | ⚠️ `store/bm25.py` 的 doc 侧与 query 侧**都带 idf**,点积 = idf²·tf_sat ≠ 教科书 BM25(见 §5.3) | **先修 + 补单测**,再 fit 语料 |
| D5 | sparse 升级路线 | SPLADE 质量高但**非商用许可(cc-by-nc-sa)** → 排除;BGE-M3(MIT)dense+sparse+ColBERT 一体,SiliconFlow 免费,但 card 自称"BM25 仍具竞争力" | v1 用本地 BM25;**BGE-M3 作评测门后的升级候选** |
| D6 | 评测集 | 学术 TVR benchmark(MSR-VTT 等)是 captioning 数据复用,假阴性严重,作者建议退役,且不像"moody b-roll" | **自建 100–300 条分面查询 + LLM 判分(0–3,UMBRELA)**,查询从 `catalog_queries`(665k 真实搜索词)采样 |

## 3. ① 内容与向量层

### 3.1 嵌入文本(两侧同一函数)

沿用 `build_embed_text`(title — description — tags),但**采用带字段标签的序列化**并**保留全部 tag**:

```
<title>. <description>. tags: t1, t2, …
```

- 每行约 **100–150 tokens** → 上下文长度不是选型轴,**不要为长上下文付费**。
- 查询侧用 instruction(Qwen3 系 +1~5%),**英文写**;文档侧不写 instruction。
- 现有 `HashEmbeddingProvider` 仅作离线冒烟,不代表检索质量。

### 3.2 Dense 模型选型(建议 D1 = `text-embedding-v4`@768)

| 维度 | 结论 |
|---|---|
| 质量 | = Qwen3-Embedding 系列服务版;CMTEB 71.99 / retrieval 75.01 @2048 |
| 维度 | 2048→64 原生 MRL,**请求 `dimension=768`** → 直接落现有 `vector(768)`,零迁移 |
| 语言 | 100+,中文查询可用(语料英文照常) |
| 成本 | $0.07/1M ≈ **$6 / 全库** |
| 升级 | 若 768 不够 → 重嵌 2048 + 迁移列(约 5.4GB) |

> 未决:文档名可能有 `qwen3.7-text-embedding` 新名,落地当天核对控制台模型名与价格。
> `text-embedding-v4`@768 的 MTEB 未单独公布(只给了 512/1024/2048),评测门应含一次 2048 对照。

### 3.3 缩略图跨模态(建议 D2 = 暂缓)

- 若做:`tongyi-embedding-vision-flash`(768 固定,text/image/video **同空间**)或
  `qwen3-vl-embedding`(`enable_fusion=true`,2560→256)可把文本查询与缩略图放进一个索引。
- **陷阱**:跨模态要与文本臂共用向量空间,就必须**用同一多模态模型同时嵌文本**(即文本臂也要换),否则
  两套空间不可比。这会把 D1 一起改掉 → 所以 v1 分离处理。
- **运动盲区**:单帧无法编码 slow-motion / timelapse / aerial → 这些继续靠 **tag + duration**(见 §6)。

## 4. ② 索引结构层(核心)

### 4.1 列 → 索引矩阵

| 列 | 索引 | 备注 |
|---|---|---|
| `embedding vector(768)` | **HNSW `vector_cosine_ops`** `(m=16, ef_construction=64)` | 现在全精度就放得下,**不上量化** |
| `sparse_embedding sparsevec` | **HNSW `sparsevec_ip_ops`**(内积) | **仅 HNSW 支持 sparsevec**;IVFFlat 不支持 |
| `tsv tsvector` | **GIN(tsv)** | 只作布尔预过滤;**不用 `ts_rank` 排序**(无 IDF/饱和/长度归一) |
| `tags text[]` | GIN(已存在,001) | 标签包含/重叠 |
| `orientation`/`duration` | B-tree(已存在,001) | 过滤预筛;**广/窄条件策略不同**(见 4.3) |
| 1,000 nnz 上限 | **建索引前断言 nnz ≤ 1000** | sparsevec HNSW 硬上限,超了会拒绝/报错 |

### 4.2 量化:为什么 v1 不做

- 768 维全精度:`vector` 3080 B/条 → 665k ≈ **1.91 GiB** 列 + HNSW 图 ≈ **2.3–2.5 GiB** 构建内存。
- `halfvec`(1544 B)≈ 1.37 GiB、`bit`(104 B)≈ 66 MiB —— 都是**兜底**,不是默认:
  容器 **6 GiB / shm 64 MiB**(实测)在 2.5 GiB 构建时"刚好够",若吃紧再降 halfvec。
- 二进制量化(BQ + 重排)只缩索引不缩表(重排仍读全精度堆),665k 无必要。

### 4.3 过滤 + ANN(HNSW 是**后过滤**)

- 默认 `WHERE` 在 ANN 扫描**之后**应用:筛选命中 ~10% 时 `ef_search=40` 只会平均剩 ~4 行 → **静默少返**。
- 本机实测画幅分布:**landscape 236k / portrait 129k / square 0.1k**,4K 149k → 都是**广条件**(非选择性),
  所以主力是 `hnsw.iterative_scan = relaxed_order`(`max_scan_tuples=20000` 默认),而不是 partial index。
- 文本标签类**窄条件**(如某具体 tag 命中很少)才走 GIN/B-tree 精确预筛;某个筛选值独大时才建 partial HNSW。
- **665k 不需要分区**(分区是"很多 distinct 过滤值"或千万级才考虑)。

### 4.4 建索引 DDL(爬取仍在写 → CONCURRENTLY)

```sql
-- 0) 核对:vector 0.8.7(实测)
SELECT extversion FROM pg_extension WHERE extname='vector';

-- 1) dense(HNSW 默认参数;665k 下提高 m/ef_construction 只增构建成本)
CREATE INDEX CONCURRENTLY idx_stock_videos_embedding_hnsw
    ON stock_videos USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);

-- 1b) 兜底(仅当 6GiB 顶不住):
-- CREATE INDEX CONCURRENTLY ... USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);

-- 2) sparse(BM25 内积;ORDER BY 必须用同一算子 <#>)
CREATE INDEX CONCURRENTLY idx_stock_videos_sparse_hnsw
    ON stock_videos USING hnsw (sparse_embedding sparsevec_ip_ops);

-- 3) lexical
CREATE INDEX CONCURRENTLY idx_stock_videos_tsv_gin
    ON stock_videos USING GIN (tsv);

ANALYZE stock_videos;
```

建索引会话(注意 shm 64 MiB 会挡并行 HNSW 构建):

```sql
SET maintenance_work_mem = '2560MB';              -- ≥ 预估图大小
SET max_parallel_maintenance_workers = 0;          -- 或把容器 --shm-size 提到 ≥ maintenance_work_mem
-- CONCURRENTLY 不能包在事务块里,用 session 级 SET 包在独立语句外
```

查询会话:

```sql
SET hnsw.ef_search = 100;                 -- 召回/延迟旋钮(默认 40)
SET hnsw.iterative_scan = relaxed_order;  -- 过滤查询仍能填满 LIMIT k
```

### 4.5 增长(何时离开 Postgres)

665k **不是**离开的理由。触发点是:(a) 向量到**千万级**、内存放不下 → 首选 **pgvectorscale**
(同 Postgres、StreamingDiskANN + SBQ,改动最小);(b) 多租户/多节点横向扩展 → 才考虑 Qdrant/Milvus 等。
turbopuffer/Qdrant 的基准都在更大规模/不同硬件,**不能外推到本机**。

## 5. ③ 查询理解 + 融合 + 重排

### 5.1 查询理解(高价值、低风险)

- **分面抽取 → SQL 过滤路由**:把 NL 查询拆成 subject / motion / camera / setting / lighting /
  time-of-day / mood,把 orientation/duration/resolution/license 映射成 `WHERE`。这一层是"词袋无法表达"
  的部分,收益最确定。
- **同义词扩展只做轻量**:注意一手结论——**LLM 扩展对弱检索器有用,对强检索器可能有害**
  (EACL 2024 发现"检索器越强、扩展收益越小甚至为负")。富标签本身已是人工同义层 → **扩展做成可开关的 A/B**,
  不作默认。
- 数据底座:`catalog_queries` 有 **665,589 个真实搜索词**(均 2.26 词、88% 多词)。这是**真查询分布**,
  既可用于查询理解的测试,也可直接采样构建评测集(见 §7)。

### 5.2 融合:RRF

- **RRF `k=60`** 起步(无需标注数据,Elastic/Qdrant 默认;Elastic 网格搜索曾得 k=20/N=1000)。
- 陷阱:**不要对原始分数线性加权**——cosine 有界、BM25 内积无界且随查询变化,固定 α 会被量纲支配。
  权重融合需要归一化 + ~40 条标注,且权重是模型特定的 → 有评测集后再试(convex / weighted RRF / DBSF)。
- 结构化过滤作为**两路共同的 `WHERE` 预筛**,**不要**当成第三个"排序器"。

### 5.3 sparse 臂与 BM25 seam 修正(D4)

- v1 保留**本地 BM25 → sparsevec**(零成本、确定性、已实现);`tsv` GIN 只作布尔预过滤。
- ⚠️ **seam bug**:`store/bm25.py` 的 `doc_weights` 已含 idf,`query_weights` **也含 idf** →
  `sparsevec <#>` 得到的是 `idf²·tf_sat`,不是标准 BM25 的 `idf·tf_sat`。rank 会被稀有词双重放大。
  **修法**:一侧带 idf(通常 doc=`idf·tf_sat`、query=1;或 doc=`tf_sat`、query=`idf`),并补一条
  "点积 == 教科书 BM25 公式"的单测。**先修再 fit 语料**。
- 硬约束:每行 nnz ≤ 1000(否则 sparsevec HNSW 建不了)→ 建前断言,必要时截断到 top-1000 权重。
- 升级候选 **BGE-M3 sparse**(MIT;SiliconFlow 免费;一体出 dense+sparse+ColBERT),过评测门再说。

### 5.4 重排(阶段 2,建议 v1 留接口)

| 选项 | 适用 | 许可 |
|---|---|---|
| `bge-reranker-v2-m3`(0.6B) | SiliconFlow **免费**,省运维 | Apache-2.0 |
| `answerai-colbert-small-v1`(33M) | **纯 CPU 交互延迟最优** | Apache-2.0 |
| `Qwen3-Reranker-0.6B` | 小模型 MTEB-R 最好(65.80) | Apache-2.0 |
| BGE-M3 ColBERT | 已在用 BGE-M3 时复用 | MIT |
| ~~jina-reranker/colbert-v2~~ | 质量好但**非商用** | cc-by-nc-4.0 ❌ |

预算:**召回每路 100 → RRF ~200 → 重排 20–50 → 返 30**(重排器官方都以 top-100 候选评测)。

### 5.5 运动盲区(显式记录)

单帧缩略图无法表达运动;无视频文件也做不了真 video-clip 嵌入(DashScope 视频上传上限 10–50MB,
Pexels 1080p ~15MB / 4K ~72MB)。→ **motion 继续靠 tag + duration**,不要指望图像向量。

## 6. 结构化过滤的取值域(本机实测)

| 过滤项 | 取值/分布 | 策略 |
|---|---|---|
| orientation | landscape 236k / portrait 129k / square 142 | 广条件 → HNSW + iterative_scan |
| 4K(width≥3840) | 149k / 365k(41%) | 广条件 → 同上 |
| duration | <5s:7k · 5–15s:196k · 16–60s:152k · >60s:9.6k | 区间条件 → B-tree 预筛 |
| tags | 中位 47 / p90 55,497k distinct(53% 只出现一次) | 长尾极大 → sparse 词表需考虑剪枝/上限 |

## 7. 评测门(什么算"准")

1. **查询集 100–300 条**,按 facet 分面;从 `catalog_queries` 采样真实词 + 人工/LLM 生成概念型查询。
2. **汇总池**:BM25、dense-text、(未来)dense-image、现有漏斗各自 top-k 取并集,避免只评一个系统。
3. **判分**:LLM 判相关性 0–3(UMBRELA 式),**判分模型 ≠ 改写/重排模型**(避免循环);
   留一小份人工标注做校准,报告一致率。注意已知偏差:打分膨胀、位置偏差。
4. **指标**:nDCG@10 / MRR / Recall@k,**按 facet 与筛选条件分别报告**(单一聚合会掩盖坏掉的那一面)。
5. **基线对比**:BM25-only → dense-only → RRF 混合 →(+缩略图 / +facet 抽取)。
6. 准入线沿用 handoff §5:召回@k / VLM 采纳率 对比现行漏斗,达标才扩默认。

## 8. 落地顺序(建议)

1. **修 BM25 seam**(D4)+ 单测;对全量 `embed_text` fit 一次 BM25(fit 可增量、可重放)。
2. **嵌 dense**:`text-embedding-v4`@768,增量可续(`embedding_model IS DISTINCT FROM`);先小样本 smoke。
3. **建索引**:dense HNSW + sparse HNSW + tsv GIN(CONCURRENTLY;先在小表或事务外演练)。
4. **实现检索服务**:查询理解(facet→WHERE)+ 双路召回 + RRF,先不带重排。
5. **建评测集 + 跑门**;再决定:D2 缩略图、D3 重排、D5 BGE-M3。
6. seam 保留:`embedding_model` / `sparse_model` 每行记录,支持双模型过渡与整体换模。

## 9. 来源

- 四份调研笔记(本次,含逐条一手引用):
  `docs/research/dense-embedding-models-2026-10-08.md` ·
  `docs/research/sparse-hybrid-rerank-2026-10-08.md` ·
  `docs/research/pgvector-index-engineering-2026-10-08.md` ·
  `docs/research/footage-retrieval-domain-2026-10-08.md`
- 本机实测(2026-10-08,`dev-middleware-stock-db`):PG **16.15**、vector **0.8.7**;
  容器内存 6 GiB / shm 64 MiB;`shared_buffers` 128MB、`maintenance_work_mem` 64MB、
  `max_parallel_maintenance_workers` 2;stock 365,103 行;画幅/4K/时长/tag 分布见 §6。
- 起点与锁定决策:`docs/handoff-2026-10-07.md`;接口层:`docs/phase2-embedding-seam.md`。

## 10. 遗留不确定(需实测/复核)

- `text-embedding-v4`@768 的实际检索质量未公布 → 评测门含 2048 对照。
- 本机 recall-vs-`ef_search` 未跑;HNSW 索引实际大小待建后 `pg_relation_size` 核对。
- 6 GiB 容器 + 实时爬写在**全精度并行构建**下是否稳住 → 建前演练;halfvec 兜底。
- 跨模态(facet 融合)分数校准 / 单帧代表性 / 运动覆盖 → 均需数据验证。
- 模型与价格快变(`qwen3.7-text-embedding`、pgvector 0.9.x?)→ 落地当天复核。
