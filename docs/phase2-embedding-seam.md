# Phase 2 起点:embedding seam(2026-10-08)

Phase 1 仍在跑(搜索扇出),Phase 2 的**接口层**先落地,避免后面被模型选型卡住。
模型候选与默认值见 `docs/handoff-2026-10-07.md` §5(dense 起步 `tongyi-embedding-vision-flash`
768 维;sparse 起步本地 BM25;升级候选 qwen3-vl-embedding / BGE-M3)。

## 1. 已落地

- **迁移 003**:`stock_videos` 增加 `embed_text` / `embedding vector(768)` /
  `embedding_model` / `embedded_at` / `sparse_embedding sparsevec` / `sparse_model` / `tsv`;
  安装 `vector` 扩展(镜像自带 0.8.7)。
  ⚠️ **索引(HNSW / GIN)刻意留到"有向量之后"的迁移**:现在建会在爬取写入路径上加开销。
- **`store/embedding.py`**:seam 契约
  - `EmbeddingSpec{model_id, dim, distance, max_batch, supports}`:声明模态与维度;
    sparse-only 模型 dim 可以为 0。
  - `EmbeddingProvider` Protocol:`dense_text / dense_image / sparse_text`(corpus 与 query 同走)。
  - `build_embed_text(row)`:规范文本 = `title — description — tag, tag…`(空白归一);
    入库与查询两侧必须用同一函数。
  - `HashEmbeddingProvider`:确定性本地 bag-of-words(无网络),用于离线把流水线跑通,
    **不代表检索质量**。
- **测试**:`tests/test_embedding.py`(spec 校验、文本拼装、确定性/归一化、稀疏计数、批上限)。

## 2. 刻意未做(下一步工单)

1. 真实 provider:DashScope `tongyi-embedding-vision-flash`(需 MPT config 里的 Key,
   与 MPT 查重门同向量空间);sparse 的 BM25 实现。
2. DB 写入路径:需要 `pgvector` python 包(注册 vector/sparsevec 适配器)+ `vpc embed`
   命令(`--model`,按 `embedding_model IS NULL` 增量回填、`max_batch` 分批、失败重试)。
3. 索引迁移:回填到一定比例后再建 HNSW(cosine)+ `tsv` GIN。
4. 评测门:召回@k / VLM 采纳率对比现行漏斗(见 handoff §5)。

## 3. 与 Phase 1 的衔接

- 爬取与 ingest 不受影响:新列全部可空,写入路径零改动;本迁移只加列(元数据级,秒级)。
- 回填顺序建议:先把覆盖率推到目标(搜索扇出循环),再统一嵌入——嵌入 API 按量计费,
  且避免对后续可能被更正的元数据重复嵌入。
