# Phase 2 多维索引设计提案(2026-10-08)

本文把"单索引(dense+sparse)"扩展为**跨模态的多维索引**:每个视频不再只有一个向量,而是
**帧图像向量 + 多个分面(facet)字幕向量**,查询时可对每个维度单独匹配、再加权融合。

- 与 `docs/phase2-index-design-2026-10-08.md` 的关系:那篇的 L0/L1(元数据过滤 + 整条文档向量)、
  HNSW 参数、RRF、BM25 seam 修正**仍然有效**;本文在其上叠加 L2+(分面字幕 / 帧 / 多帧)。
- 约束更新(用户 2026-10-08 明确):为做帧理解,**允许拉取缩略图与低清 SD 预览**(`preview_src`,
  0.6–3.3MB 典型)用于取帧;仍不落盘保存视频本体(预览下载后可即取即弃)。
- 来源:本仓库 `docs/research/` 下三篇新笔记(分面 taxonomy / 帧 captioning 管线 / 多维检索架构),
  及此前的四篇;本文只写合并结论。

## 1. TL;DR

**一个侧表 `video_embeddings(video_id, facet, model, caption, embedding, sparse)`,facet 是"数据"而不是 DDL;
每个 (facet, model) 建一个 partial HNSW;查询按分面解析 → 每维召回 → RRF 融合。**

```
视频帧(1 缩略图 或 4–8 帧 @640×360,从 SD 预览取)
   └─ VLM 一次 → 一个 JSON,约 10 个 creator 关心的分面字段
        ├─ 每分面短句 → 各自嵌入(256–384 halfvec)    ┐
        ├─ 一条 digest(拼接短句)→ 文档向量(768)      ├─► video_embeddings 侧表
        └─ 分面文本 → BM25(sparsevec,低权重)          ┘     每个 facet 一个 partial HNSW
                                                             + 帧图像向量(facet='frame')

查询:"航拍 黄金时刻 冷色调" → {camera:aerial, lighting:golden_hour, colour:cool, hard:{orientation:landscape}}
   ├─ 硬分面 → SQL 预过滤(两个 arm 都带)
   ├─ soft 分面 → 各自的 partial HNSW → top-k
   └─ RRF k=60 融合 →(L5)跨分面重排 → top-30
```

要点:**每一层都能单独开关;关掉一层只是查询少一个 arm,没有数据迁移。** 这就是"留尽可能多选项"的落法。

## 2. 哪些维度值得建(证据驱动的取舍)

三类来源,成本从低到高:

| 来源 | 得到的分面 | 成本 | 证据要点 |
|---|---|---|---|
| **已有列 + 解析** | orientation/aspect/duration/fps/分辨率/license;**motion/technique(aerial/drone/slow_motion/timelapse/handheld…)** 从 title+description 解析 | 0 | 运动词 17.4% 在正文、只有 3.1% 在 tag;单帧推断运动不可靠(MovieShots: scale 84–88%,movement 70–71%) |
| **从缩略图直接计算** | colour.temperature/saturation/monochrome/palette;DoF 代理 | CPU,无 API | 像素可测、零幻觉:暖色 tag 覆盖 9.3%,冷色 0.7%,B&W 2.2% |
| **VLM 结构化 caption** | camera.scale/angle、lighting(time_of_day/key/direction/quality/source)、setting(space/environment/place)、weather/season、people(count/group/activity)、mood(emotions+valence/arousal)、theme(topics/story)、composition(negative_space/symmetry/subject_position) | 1 次调用/视频 | 最大空白带:golden hour 1.9%、backlit 0.5%、low-key 0.15%、negative space 0.4% |

**不该由 VLM 做的:** genre/use-case(是"用途判断"不是画面属性 → 查询时路由,不落库);
people 的年龄/性别/族裔(**不自动推断**,伦理+错误,平台靠申报)。

**分面的嵌入面(3–6 条,facet 短句):** `subject+activity`, `lighting`, `mood`, `colour+composition`,
`camera+setting`(可按 A/B 调整)。另存一条 digest 供 L1 使用。

## 3. Caption JSON(facet schema,节选)

每行一个 JSONB `facets`(或并入 `video_embeddings.caption` 的来源),字段带 `null` + confidence,
来源标记 `provenance`(vlm / derived / prose / column);prompt/model/帧规格版本化,可重跑。

```json
{
  "camera":    {"scale":"medium_close","angle":"high","technique":["aerial","drone"],"movement":null},
  "lighting":  {"time_of_day":"golden_hour","key_style":"low_key","direction":"back","quality":"soft","confidence":0.6},
  "colour":    {"temperature":"warm","saturation":"vivid","monochrome":false,"palette":["teal","orange"]},
  "composition":{"dof":"shallow","negative_space":true,"subject_position":"thirds"},
  "setting":   {"space":"outdoor","environment":"nature","place":"mountainside","weather":["clear"],"season":"autumn"},
  "people":    {"count_bucket":"2","group_type":"couple","activity":["walking"]},
  "mood":      {"emotions":["relaxed"],"valence":0.6,"arousal":-0.2},
  "theme":     {"topics":["nature","travel"],"story_tags":["escape"]}
}
```

## 4. 索引布局(核心)

```sql
CREATE TABLE video_embeddings (
    video_id        BIGINT NOT NULL REFERENCES stock_videos(pexels_id) ON DELETE CASCADE,
    facet           TEXT   NOT NULL,   -- 'doc'|'subject'|'lighting'|'mood'|'colour'|'camera'|'frame'|...
    model           TEXT   NOT NULL,   -- 'text-embedding-v4@768' / 'qwen3-vl-embedding@1024' ...
    caption         TEXT,              -- 被嵌入的确切文本(审计/重嵌)
    embedding       vector,            -- 混合维度:索引端 cast
    sparse_embedding sparsevec,        -- 可选:该分面的词法 arm
    embedded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (video_id, facet, model)
);

-- 每 (facet, model) 一个 partial HNSW(维度一致才可建;pgvector FAQ 的混合维度官方模式)
CREATE INDEX CONCURRENTLY idx_ve_lighting_hnsw ON video_embeddings
    USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops) WHERE (facet = 'lighting');
```

- **为什么侧表而不是主表加列:** `vector` 列是 `external` TOAST(不压缩、行外);每加一个分面列 =
  `ALTER TABLE` + 新索引 + 回填,还没有地方放 model/caption 元数据。侧表让**新分面 = INSERT**。
- **为什么 partial HNSW 而不是一个大索引:** 大索引按 `WHERE facet=…` 后过滤,选择度 1/F,`ef_search=40`
  平均只剩 5–10 行;partial 索引图是分面纯净的,且可增量建、单独 drop。
- **四个已知坑(必须处理):**
  1. **参数化查询用不了 partial index** → facet 值走**校验过的白名单 + 字面量内联**,不用绑定参数;
  2. planner 可能因不感知 TOAST 而选错计划(实测 15× 慢)→ 每分面 `EXPLAIN (ANALYZE)` 验计划、常 `ANALYZE`;
  3. pgvector 还没有 per-index `ef_search` → 会话级 GUC 设成"最小分面"也能用,或按查询临时设;
  4. **F 大了就换 `PARTITION BY LIST (facet)`**(F≤8 用 partial;分面很多/互斥时分区对 planner 更友好)。
- **主表保留 1 个 `embedding vector(768)`(L1 digest)不变**;帧图像向量放同一侧表(`facet='frame'`,
  多帧则 `facet='frame'` + 另加 `frame_idx` 或独立表)。

## 5. 查询:每一维单独匹配,再融合

- **硬分面 = SQL 预过滤两路都有**(orientation/duration/分辨率/license;以及高精度新枚举如
  `camera.scale`/`lighting.time_of_day`/`people.count_bucket`/`colour.monochrome`)。
- **软分面 = 分数**(genre/lighting/mood/colour/camera);**别用布尔 AND 表示"氛围"**——
  产品搜索的相关性本身就是分级的(ESCI:Exact/Substitute/Complement/Irrelevant)。
- **融合:RRT k=60 起步**(无需标注);有 ~40 条标注后再试归一化加权 / convex(证据:Elastic +6–24%,
  但权重是模型特定的;原始分数跨模型不可比,必须先归一化)。
- **路由**:F 小就先"全分面都查";分面变多再加路由,并保留 **abstain → 查全部** 兜底,同时测
  "路由没漏掉命中分面"的 facet-set recall。
- **缺失值 fail-open**:HNSW 不索引 NULL;没有某分面向量的行不参与该 arm,融合不能奖励覆盖缺口。
- **L5 重排**:多分面组合查询("黄金时刻的航拍冷色调")用一个 cross-encoder / 小 VLM 在 top-50 上精排;
  线性分面加权不是组合算子(Fashion-IQ:~70% 属性查询是组合短语)。

## 6. 成本与容量算术(估算,不是报价)

| 项 | 数量级 | 说明 |
|---|---|---|
| Caption(cheap tier) | **≈$29(1 帧)/ $39(4 帧)/ $52(8 帧)** | `qwen3-vl-flash` 北京,$0.022/$0.215 每 1M;640×360;8 帧 @720p 升到 ~$131 |
| Caption(rich tier) | 5–20× | `qwen3-vl-plus` 8 帧 ≈ $346(batch ≈$173);只建议 5–10k 行 A/B 或补 caption |
| 字幕嵌入 | **≈$10** | text-embedding-v4 $0.07/1M,per-facet + digest ≈0.15B token |
| 帧图像嵌入 | **≈$40–150 一次性**(估) | 取决于分辨率/模型 token 计费(640×360 ≈227 img-token;1440px 缩略图 ≈900) |
| 帧获取带宽 | 单帧 ≈数十 GB;4–8 帧走 SD 预览 ≈1–2 TB 下载 | 只走带宽,不落盘;`preview_src` 典型 1–3MB |
| 存储:分面向量 | halfvec **256d**:F=8 ≈ 5.9 GiB 构建内存;F=4 ≈ 3.0 GiB | 全精度 768d **每分面 ≈2.3 GiB,4 分面 ≈9.3 GiB → 装不下 6 GiB 容器** |

**结论:分面向量必须用 halfvec 256–384 维**(MRL 截断,256 vs 768 约损失 2 个 MTEB 点,但 F 倍存储省下);
用 `subvector()` 表达式索引做"短向量召回 → 全向量重排"是最省内存的形态。
**先 500–1000 行试点校准 caption 质量与成本,再全量**;容器 6 GiB 不能并行建多个全精度 768d 索引。

## 7. 加性阶梯 L0–L5(每一层可独立开关)

| 层 | 内容 | schema 影响 | 解锁 | 怎么评 |
|---|---|---|---|---|
| **L0** | 元数据硬过滤(已存在) | 无 | 约束 | `EXPLAIN` + 手写用例 |
| **L1** | 整条文档向量(已存在 `embedding(768)`) | 无 | 整体语义 | Recall@k / nDCG@10 vs BM25 |
| **L2** | **分面字幕向量**(侧表 + 每分面 partial HNSW) | 新表 | genre/lighting/colour/mood/subject/camera 细粒度匹配 | per-facet nDCG@10 + 分面条件 Recall@k |
| **L3** | 帧图像向量(`facet='frame'`) | 同表 | 视觉指定型查询(构图、"从上方"、色彩) | 仅视觉查询子集 A/B、跨模态 Recall |
| **L4** | 多帧(k=3–8) | 同表多行 | 时间覆盖、镜头运动线索(仍非完整运动) | 运动查询子集,vs L3 |
| **L5** | 跨分面重排 | 无 | 组合查询("航拍+黄金时刻+冷色调") | 组合查询 nDCG 增量、去一分面 ablation |

**关层 = 查询少一个 arm**;`model` 列让两代模型共存做 A/B;partial 索引可随时 `CONCURRENTLY` 建/删。

## 8. 生成管线(帧 → caption → 嵌入)

1. **帧**:先缩略图(1 帧,立即可用);升级项 = 下载 `preview_src`(SD,1–3MB)→ ffmpeg 取 4–8 帧 @640×360。
   证据:1→6 帧增益显著(CLIP4Clip),5–60s 短片 **4–8 帧**是成本收益拐点。
2. **Caption**:一次请求带上全部帧 + 一个"只描述可见内容,不确定填 null"的 JSON prompt(约 10 个短字段);
   DashScope 多模态只支持 `json_object`(schema 会降级)→ 本地校验 + 失败重试;`enable_thinking=false`。
3. **运动字段**:单帧时 `motion=null`;**motion/technique 以正文解析为准**,不采信 VLM 的运动推断
   (视频 LLM 的"顺序性幻觉"是已知最严重失效模式)。
4. **嵌入**:每分面短句单独嵌(256–384 halfvec)+ digest(768);分面文本进 BM25 作低权重第二字段。
5. **QC**:raw JSON + prompt/model/帧规版本入 spool(仓库惯例"先落原始再入库");幂等键
   `(video_id, prompt_version, model_id)`;拒答/解析失败分类处理;抽几百行用**另一个模型**+人工抽检。

## 9. 评测(先建再调)

- **judgments 按分面存**:`(query_id, video_id, facet, grade 0–3, judge)`;TREC 式 pooled,未判 ≠ 不相关。
- **指标**:per-facet nDCG@10(不可只报平均,好的 lighting 会掩盖坏的 genre;分面条件 Recall@k;
  首段 Recall@100 分离召回与重排;每次"去一分面"的 nDCG 增量 = 该层的价值;分面覆盖率作为一等指标。
- **组合查询套件**:≥2 个分面意图的查询,固定其余分面、只动一个分面分别评测。
- **反循环**:判分模型 ≠ caption 模型 ≠ 查询改写模型;留人工小集校准。

## 10. 落地顺序与开放决策

1. **解析运动/technique**(零依赖)→ 2. **500–1k 行 caption 试点**(人工打分,校准成本/质量)→
3. **建侧表 + 2–4 个分面 partial HNSW**(halfvec 256–384)→ 4. **查询分面解析 + RRF** →
5. **评测门** → 再上 L3 帧 embedding / L4 多帧 / L5 重排。

开放决策(需拍板):
- **⬜ F1 帧数**:1(缩略图)/ 4 / 8(走 SD 预览,带宽 1–2TB)。
- **⬜ F2 caption 模型档**:flash 廉价全量 ≈$39–52,vs plus 先跑 5–10k A/B。
- **⬜ F3 分面数 F 与维度**:4 分面 × 384d vs 8 分面 × 256d(6 GiB 容器约束)。
- **⬜ F4 帧图像向量是否并入 L2 侧表**(还是独立表)。
- **⬜ F5 是否启用 `PARTITION BY LIST (facet)`**(现在 F≤8 用 partial 即可)。

## 11. 来源与不确定

- 三篇新笔记:`docs/research/footage-facet-taxonomy-2026-10-08.md`(平台/电影语法/实测 tag 覆盖/ facet schema)、
  `docs/research/frame-captioning-pipeline-2026-10-08.md`(VLM 价格/帧证据/prompt/成本)、
  `docs/research/multidimensional-retrieval-2026-10-08.md`(布局/分面融合/加性阶梯/评测);此前四篇见
  `docs/phase2-index-design-2026-10-08.md` §9。
- 最大不确定:**caption 质量**(upstream,无 Pexels 相关一手证据)、**本机多分面扫描延迟**(未实测)、
  **最优 F 与维度**(无公开答案,需自测)、**caption 是否真的超过 47-tags 基线**(有反证:生成字幕并非
  稳定增益 → 必须 A/B)。
