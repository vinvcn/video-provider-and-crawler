# 检索索引评测 harness 设计(2026-10-09 定稿)

为 Phase 2 索引策略提供**可复现的评测与排行榜**:固定材料集上,各策略按同一查询集检索,
LLM 判分,按指标排名。第一轮就把 D1(dense 模型价值)与 D4(BM25 idf² seam 实害)变成
可量化的比较,并顺带测出 BM25 的 CJK 盲区。调研依据见 `docs/research/*-2026-10-08.md`
(评测方法学:multidimensional-retrieval §Evaluation、footage-retrieval-domain §Evaluation plan)。

本文档是 T1–T4 工单的权威来源;与 `docs/phase2-index-design-2026-10-08.md` 的关系:
那篇定义生产索引架构,本篇只定义**离线评测 harness**,不建任何生产索引。

## 1. 已锁定决策(2026-10-09 与用户逐条确认)

| # | 决策 | 结论 |
|---|---|---|
| 1 | 材料集规模/时机 | **10k,冻结时点即 v1**(当时池子行数记入 manifest);不阻塞于爬取进度 |
| 2 | 采样口径 | 24 格(orientation × 4K × 时长桶)比例配额 + 稀有格保底 100;格内按 `blake2b(pexels_id)` 排序取前 N;排除空文本行;thin-tags 行照常纳入并记录分布 |
| 3 | 超集扩展 | 每格记录哈希截断点;crawl 全量后同阈值重跑 → v2 为 v1 **超集**,已判分 (query, doc) 对直接复用 |
| 4 | 判分 LLM | 用户提供 endpoint+key+model(OpenAI 兼容),存 `.env`(gitignored) |
| 5 | 嵌入模型 | 用户另给 OpenAI 兼容 embeddings endpoint;模型名/维度入 strategy 配置 |
| 6 | 查询集 | **150 = 90 真实(catalog_queries)+ 60 设计**;设计含 10 条中文探针;70/30 分层切分;**主榜以 holdout 排名** |
| 7 | 判分协议 | top-20/策略池 + 5 随机负例/查询;逐对独立调用(temp 0,JSON 输出);rubric 0–3;纯文本判分;按 (query_text, doc_id, doc_hash, judge_model, prompt_version) 缓存可续 |
| 8 | 指标 | 主分 = holdout per-query nDCG@10 平均(增益 2^g−1);Recall@{1,5,10}+MRR@10(二值阈值 grade≥2);配对 bootstrap 95% CI;延迟/成本两轴;按分面/过滤/难度/来源/语言切分 |
| 9 | 存储 | **全文件**(storage/bench/,gitignored),无 DB 表;排行榜出 md/csv/summary.json |

## 2. 材料集(T1)

- **总体**:stock_videos 中 title/description/tags 至少一项非空的行(全库空文本仅 1 行)。
- **格子**:orientation(landscape/portrait/square/unknown)× 4K(width≥3840 与否)×
  时长桶(<5s、5–15s、15<d≤60、>60s、unknown)。空值归 unknown 桶,有行才占格。
- **配额算法**(纯函数,单测):每格先保底 `min(100, 该格行数)`,剩余预算按各行数比例
  分配(largest remainder 取整);任何格配额不得超过该格行数,超出的预算按同法在
  未满格间再分配,直到收敛。总配额恰为 N=10,000。
- **取样**:格内按 `blake2b(str(pexels_id))` 升序(平手按 pexels_id)取前 N。
- **截断点**:每格记录被选中的最大哈希;v2 语义 = 同格子下哈希 ≤ 截断点的全部行(超集)。
- **行内容**:pexels_id、title、description、tags、duration、width、height、orientation、
  license、embed_text(`store.embedding.build_embed_text`)、doc_hash(规范化内容 blake2b)。
- **manifest**:版本、冻结时间、总体行数、各格 (行数/配额/实取/截断点)、内容总哈希、
  tags 分布(中位/p90/空率)、规则版本。
- **确定性**:同总体重跑 → rows.jsonl 逐字节一致。

## 3. 查询集(T1)

- **90 条真实词**(catalog_queries,665k,确定性采样):
  - 垃圾过滤:token(共享 `tokenize`)1–6 个、至少一个长度 ≥2 的字母 token、不全在停用词
    黑名单、不含 URL、非纯数字;
  - 命中量代理 = 查询各 token 在材料集 title+tags 倒排中的**最大文檔数**(命中下界);
    要求 ≥2,否则弃样;
  - 分层:head(≥1000)/ torso(100–999)/ tail(2–99)各取 30,层内按 `blake2b(term)` 序。
- **60 条设计词**(版本化数据文件 `bench/data/designed_queries.v1.json`):
  - 20 概念型(自然语言短语)、15 分面条件型(单一 soft facet 意图)、
    15 硬过滤型(概念 + orientation/时长/4K 硬约束)、10 组合型(≥2 分面意图);
  - 含 **10 条中文探针**(跨类目分布),量化 BM25 分词器丢 CJK vs dense 多语;
  - 每条:意图分面(subject/motion/camera/setting/lighting/time/mood/colour)、
    硬约束、难度(easy/medium/hard)。
- **切分**:按 (类目, 语言) 分层,层内按 `blake2b(text)` 序前 70% train、其余 holdout。
- **记录**:qid(内容哈希派生)、文本、来源、类目、语言、分面、硬约束、难度、命中量、切分。
- **manifest**:版本、各类目/语言/切分/命中层计数、内容总哈希。

## 4. 策略层(T2)

- **协议**:`search(query_text, k, filters) -> hits/doc_id+score + search_ms + embed_ms`;
  硬过滤必须生效(材料行内存掩码);分数平手按 doc_id 升序(重跑确定性)。
- **五臂**(命名即 strategy_id,配置哈希入 manifest):
  1. `bm25-both` —— 现状 doc/query 双侧 idf(点积 = idf²·tf_sat);
  2. `bm25-query` —— 修正:idf 移到 query 侧(doc = tf_sat,query = idf → 点积 = 教科书 BM25);
  3. `hash-dense` —— HashEmbeddingProvider 768 维 + 精确 cosine(离线冒烟基线);
  4. `dense` —— 用户 embeddings API(OpenAI 兼容),材料侧批量嵌入落缓存产物
     (`storage/bench/embeddings/<model|dim|材料哈希>/`,跨 run 复用),查询侧实时;
  5. `rrf` —— bm25-query + dense 各 top-100,RRF k=60 等权融合。
- **BM25 seam 修正(D4,expand 式)**:`store/bm25.py` 增加 `idf_side: both|query|doc`
  参数,默认 `both` 保持现有行为不变;单测断言 query/doc 模式点积 == 教科书公式。
- **BM25 fit 语料 = 材料集本身**(检索发生在哪,统计就在哪;生产用全库 fit 是已知差异)。
  检索用倒排索引(term_id → [(doc, weight)]),纯 Python。
- **numpy**:新增主依赖(10k×768 精确检索);BM25/hash 逻辑不依赖。
- **run 产物**:`storage/bench/runs/<run_id>/`,run_id = `<UTC时间戳>-<strategy>-<配置哈希8>`;
  `strategy.json`(spec、配置哈希、代码 git 版本、材料/查询版本与哈希)+
  `per_query.jsonl`(每查询 top-50 命中、search_ms、embed_ms)。

## 5. 判分与校准(T3)

- **池化**:给定若干同版本 run,每查询取各 run top-20 并集 + 5 个随机负例
  (rng 种子 = `blake2b(qid|materials_version)`,排除池内已有),去重,顺序确定。
- **调用**:每 (query, doc) 独立单次;system = rubric(0–3 锚点 + 跨语言说明),
  user = 查询文本 + 文档 embed_text;temperature 0;输出 JSON `{grade, reason}`;
  解析失败重试 ≤2,仍失败记 `status=unparsed`(打分时按 0 计但单独报率,不静默)。
- **rubric**:3 = 精确满足意图(直接可用);2 = 相关可用;1 = 边缘沾边;0 = 不相关。
- **判分器只看**:查询文本 + 文档文本(title—description—tags)。不看策略、不看其他候选、
  不看缩略图(v1 已知局限:文本相关性代理)。
- **缓存**:`storage/bench/judgments/labels.jsonl` append-only;键含 judge_model 与
  prompt 版本 → judge_version = `<model>@prompt-v1`;幂等续跑。
- **预算护栏**:`--max-pairs` 停机点;`--concurrency` 并发;`--mock` 用确定性哈希打分
  (仅管道验证,判分版本记 `mock@prompt-v1`,绝不进真实榜)。
- **人工校准**:`calibrate export` 按分层抽 ~40 对(侧重边界分 1/2)出 CSV 工作表 →
  人工填分 → `calibrate score` 算 exact agreement、Cohen's κ(无加权,4 类)、混淆矩阵;
  **κ < 0.4 → rubric/判分器需修,停下再跑**;0.4–0.6 属文献正常区间。

## 6. 指标与排行榜(T4)

- **per-query**:nDCG@10(增益 2^g−1,IDCG 取该查询池内判分档位理想排序)、
  Recall@{1,5,10} 与 MRR@10(二值相关 = grade ≥ 2;池化 recall,分母 = 池内判 ≥2 的
  文档数,标注为下界)、search_ms、embed_ms、top-10 内未解析数。
- **聚合**:mean + median;**配对 bootstrap**(B=1000,固定种子)对每对 run 报
  holdout 主分差的 95% CI 与 win/tie/loss;查询间天然配对(同集同序),比不配对灵敏。
- **切分报告**:train/holdout、来源(catalog/designed)、类目、语言、难度、
  是否带硬约束、每个意图分面。
- **排行榜**:按 (materials_version, queries_version, judge_version) 分组,不匹配不上同榜;
  排序键 = **holdout 平均 nDCG@10**;表列含 full 集分数、R@10、MRR@10、延迟 p50/p95、
  查询侧嵌入 token、未解析率;附成对 CI 矩阵与切分摘要。
- **产物**:`reports/leaderboard-<materials>-<queries>-<judge>.{md,csv}` + `summary-*.json`。
- **防泄漏**:holdout 为主榜;任何看到结果后改配置的再跑自动落入 holdout 保护;
  train 分数只作开发参考。

## 7. CLI 与工程

```
vpc bench materials build [--n 10000] [--version v1]
vpc bench queries build [--materials v1] [--version v1]
vpc bench run --strategy {bm25-both|bm25-query|hash-dense|dense|rrf} [--k 10] [--depth 50]
vpc bench judge --runs R1,R2 [--depth 20] [--negatives 5] [--concurrency 8] [--max-pairs N] [--mock]
vpc bench score --run R [--judge <version>]
vpc bench leaderboard [--materials v1] [--queries v1] [--judge latest]
vpc bench calibrate export [--n 40] / score --worksheet PATH
vpc bench smoke                      # 判分/嵌入连通性冒烟(读 .env)
```

- 环境变量(`.env` 或环境,`.env` 不覆盖已有环境变量):
  `VPC_JUDGE_BASE_URL/API_KEY/MODEL`、`VPC_EMBED_BASE_URL/API_KEY/MODEL/DIM/BATCH/QUERY_PREFIX`、
  `VPC_BENCH_DIR`(默认 storage/bench)、`VPC_DB_DSN`(复用)。
- 网络一律走环境代理(httpx trust_env);外部 API 调用有重试与退避。
- 纯逻辑全部单测(配额、采样确定性、BM25 三态、倒排检索、RRF、指标数学、bootstrap、
  池化、缓存键、JSON 解析、κ);DB/网络路径薄封装。
- 约束:`make fmt && make lint && make test`;行宽 100;Conventional Commits;
  storage/ 不入库;**不影响在跑的爬虫**(bench 只读 stock_videos;CLI 只增子命令)。

## 8. 验收线(第一版基线)

1. materials v1(10k)与 queries v1(150)构建可重放,manifest 齐全;
2. 五臂 run 可复现(配置哈希 + 重跑一致);
3. 判分可续跑、缓存幂等;校准工作表可导出/导入并出 κ 报告;
4. 排行榜 md/csv/summary 落盘,主榜 holdout nDCG@10,附成对 CI;
5. 产出结论:bm25-both vs bm25-query(D4)、dense vs bm25(D1)、中文探针(CJK)、
   硬过滤切分表现;人工校准一致率入报告。

## 9. 实跑纪要(2026-10-09,首版基线执行时)

外部端点由用户提供(`.scratch/model_provider.md`,钥匙只入 gitignored `.env`):

- **判分**:`deepseek-ai/deepseek-v4.1-flash` @ NVIDIA integrate(OpenAI 兼容)。
  实测**单次调用 ~45–50s**(免费端点排队型;`chat_template_kwargs.thinking=false`
  可关推理但延迟不变 → 慢在服务侧)。免费 tier **RPM 40** → 判分器加了
  线程共享 Pacer(`VPC_JUDGE_RPM` / `--rpm`),并发只重叠延迟、起点间隔钉死;
  实测并发 24 路 58.8s 全 200,线性扩展。
- **嵌入**:`BAAI/bge-m3` @ SiliconFlow(L0:RPM 2000 / TPM 500k),**1024 维**
  (生产列的 768 维 D1 候选 `text-embedding-v4` 未提供 → 本次 dense 臂实测的是
  **D5:BGE-M3**,同样过评测门,维度由 harness 记录,不影响比较方法)。
  材料侧 10k 行 ≈ **1.94M tokens**,批量 16,约 2 分钟;缓存命中后 rrf 臂零重嵌。
- **五臂池**:8,615 对(~57/查询,臂间重叠吸收了上限),判分全量 ≈ 4 小时 @ 36 RPM。
- bge-m3 材料向量缓存产物:`storage/bench/embeddings/BAAI_bge-m3-na-<材料哈希>/`。
