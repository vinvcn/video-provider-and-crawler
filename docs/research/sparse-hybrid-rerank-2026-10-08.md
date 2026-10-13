# Sparse / lexical retrieval, hybrid fusion & reranking (2026-10-08)

Scope: retrieval-quality decisions for the Phase-2 stock-video search backend
(~665k Pexels videos, `title` + 1-sentence `description` + median 47 / p90 55 `tags`,
metadata only, no GPU, no video/thumbnail bytes stored). Covers (1) BM25 in/next to
Postgres, (2) learned sparse (SPLADE/BGE-M3), (3) `pgvector sparsevec` capabilities,
(4) dense+sparse+filter fusion, (5) stage-2 reranking.

Every non-obvious number below carries a URL to a primary source (official docs, repo,
paper, pricing page). Items that are version-sensitive or that I could not confirm from
a primary page are explicitly marked **[uncertain]** / **[time-sensitive]**.

This note is read-only research; it does not change `store/bm25.py`, migrations, or the
handoff. It is meant to inform the Phase-2 eval gate in `docs/handoff-2026-10-07.md` §5.

## TL;DR recommendation

1. **Keep local BM25 → `sparsevec` as the default sparse arm for v1.** It is zero-cost,
   deterministic, and already implemented. Rank with the `sparsevec` dot/inner-product,
   not with `ts_rank`/`ts_rank_cd` (those lack IDF, term-frequency saturation and length
   normalization — see Findings §1). Use the `tsv` GIN index only as a *boolean prefilter*,
   never as the ranking signal.
2. **Do not ship SPLADE.** SPLADE++/SPLADE-v3 are the strongest zero-shot learned-sparse
   baselines (+~7.5 nDCG@10 over BM25 on BEIR-13), but the weights and the library are
   **non-commercial** (`cc-by-nc-sa-4.0`). This rules them out for a commercial stock-video
   product. (Findings §2)
3. **BGE-M3 is the commercial-safe learned-sparse upgrade candidate.** MIT license,
   one model emits dense + sparse (lexical weights) + ColBERT; it is **free via
   SiliconFlow** and available on OpenRouter. Gate it behind the recall@k eval vs. local
   BM25. Note there is *no* official sparse head in Qwen3-Embedding.
4. **`sparsevec` is fine for our scale, with one hard ceiling:** at 0.8.x only **HNSW**
   indexes sparse vectors, and only up to **1,000 non-zero elements** per row
   (storage allows 16,000; dimension up to 1e9). SPLADE docs can exceed 1,000 nnz
   (open pgvector issue #749), which is another reason BGE-M3 (short, sparse vectors) is
   the safer learned-sparse target. (Findings §3)
5. **Fuse with RRF (`k=60`) as the safe default.** It needs no labeled data. Once we have
   an eval set, a normalized weighted/convex combination can beat RRF (Bruch et al.;
   Elastic: +6–24% vs. +1.4–18%), but it is model-specific and needs calibration.
   Apply SQL metadata filters (`orientation`, `duration`, …) as a **prefilter on both
   arms**, not as a third pseudo-ranker. (Findings §4)
6. **Two-stage retrieval:** stage 1 RRF over dense top-100 + sparse top-100 + metadata
   prefilter → ~200–500 candidates; stage 2 cross-encoder rerank → 20–50 → return top-30.
   Best commercial-safe rerankers: **Qwen3-Reranker-0.6B** (Apache-2.0, best small-model
   MTEB-R 65.80) or **bge-reranker-v2-m3** (Apache-2.0, free on SiliconFlow). For CPU-only,
   **BGE-M3 ColBERT** (MIT) or **answerai-colbert-small-v1** (Apache-2.0, 33M) are the
   cheap late-interaction options. Avoid jina-reranker-v2 / jina-colbert-v2
   (both `cc-by-nc-4.0`). (Findings §5, Recommended pipeline)
7. **Do not adopt ParadeDB `pg_search` as the primary store yet.** It is a real,
   transactionally-consistent Postgres extension (not a fork), but the community edition
   is **AGPL-3.0**, it needs the ParadeDB image or `shared_preload_libraries='pg_search'`,
   and mixing it with our `sparsevec` column adds integration surface for little gain at
   665k short rows. Keep it as a documented escape hatch if `tsvector`/BM25 ops ever hurt.
   (Findings §1)

## Findings

### 1. BM25 in or beside Postgres

**Native `tsvector` + GIN is a Boolean index, not a ranking engine.**
`ts_rank`/`ts_rank_cd` score by term frequency (and proximity) but use **no global
corpus statistics**: no IDF, no term-frequency saturation, no length normalization, and
`@@` is a hard Boolean match (one missing term drops a document). Ranking requires reading
each matching `tsvector`, so top-k over a common term degrades with corpus size.
Primary: Tiger Data "From ts_rank to BM25" (describes IDF/saturation/length gaps, the
Boolean brittleness, and 25–30 s queries on ~800k rows after 1–2M-row degradation):
<https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres>.
Neon/ParadeDB benchmark on 10M log rows: ranked top-N query 81 ms with `pg_search` vs
38,797 ms with `ts_rank`, `COUNT` 770 ms vs 22,214 ms:
<https://neon.com/blog/pgsearch-on-neon>.
Conclusion: our `tsv` column should be a prefilter at most; score with BM25 (our
`sparsevec`), not with `ts_rank`.

**ParadeDB `pg_search` (BM25 via Tantivy) — quality and ops cost.**
It is a Rust/pgrx extension, "runs inside standard Postgres, with no fork or sidecar
process"; index updates happen in the same transaction as writes. On the same 10M-row
benchmark it indexes **faster** (BM25 covering index build 1.63 min vs GIN 2.60 min) but is
**larger** (647 MB vs 176 MB) because it indexes six columns at once:
<https://neon.com/blog/pgsearch-on-neon>. However, it is **AGPL-3.0** (community edition):
<https://www.paradedb.com/blog/agpl> and
<https://www.paradedb.com/docs/operate/deploy/enterprise>.
Install requires the `paradedb/paradedb` Docker image, or adding `pg_search` to
`shared_preload_libraries` and `CREATE EXTENSION pg_search CASCADE` (Postgres.app notes
`CASCADE` also enables pgvector):
<https://www.paradedb.com/docs/start/install>.
So it is *near* drop-in but is not a plain `CREATE EXTENSION` on a stock pgvector image,
and AGPL is a licensing decision the user should make deliberately.

**`bm25s` (pure Python, Numpy/Numba).** In-process, MIT, no JVM/PyTorch, eager sparse
scoring; supports the Lucene BM25 method (`method="lucene"` is the default, exact Lucene
BM25), plus robertson/atire/bm25l/bm25+; memory-mapping keeps a 2M-doc NQ index at
~0.49 GB RAM post-index. Throughput table vs Elasticsearch is in the README.
Primary: <https://github.com/xhluca/bm25s> (technical report arXiv:2407.03618,
<https://arxiv.org/abs/2407.03618>). Relevant because it is the same conceptual family as
`store/bm25.py` and a reference for parameter choices — but our current code already fills
the zero-cost role; `bm25s` would only help if our Python scorer becomes a latency
bottleneck (it won't at one query per user request).

**Other 2025–2026 BM25-in-Postgres options (context, not recommended now).**
Tiger Data `pg_textsearch` is **preview**: in-memory only, default 64 MB per index,
disk segments on the roadmap — not viable at 665k rows yet
(<https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres>).
TensorChord `VectorChord-bm25` uses Block-WeakAnd over a `bm25vector` sparse type; dual
**AGPLv3 / Elastic License v2**, English-only, and the index returns at most
`bm25_catalog.bm25_limit` rows (default 100, max 65,535):
<https://github.com/tensorchord/VectorChord-bm25>. Both are worth re-checking in a year.

**Pure-Python vs. engine tradeoff.** The handoff's plan (local BM25 → `sparsevec`,
optionally BGE-M3 sparse) is the right shape: it keeps ranking inside Postgres next to the
metadata filters, avoids a second system, and costs nothing per query. The failure mode to
watch is *quality*, not speed — hence the eval gate.

> ⚠️ **Seam check (I did not edit the code).** For a dot-product (`sparsevec <#>`) to equal
> a BM25 score, only one side should carry IDF: typically doc weight =
> `idf·tf_sat` and query weight = `1`, or doc weight = `tf_sat` and query weight = `idf`.
> In `store/bm25.py` *both* `doc_weights` (includes `idf`) and `query_weights` (also
> `idf`) carry IDF, so the dot product is `idf²·tf_sat`, which over-weights rare terms
> relative to standard BM25. Worth a unit test asserting dot product == the textbook BM25
> formula before trusting tuning. Flagged as uncertainty below.

### 2. Learned sparse: SPLADE / BGE-M3 / Qwen3

**Quality (primary).** SPLADE-v3 report: "statistically significantly more effective than
both BM25 and SPLADE++", **40.2 MRR@10** on MS MARCO dev and **51.7 avg nDCG@10 on BEIR-13**
(card): <https://huggingface.co/naver/splade-v3>; paper <https://arxiv.org/abs/2403.06789>
(which also reports "improves by 2% the out-of-domain results on the BEIR benchmark" over
SPLADE++). SPLADE++ (distillation + hard negatives): <https://arxiv.org/abs/2205.04733>.
For scale, BM25 averages **0.440 nDCG@10 on the same 13 BEIR sets** in Jina's table:
<https://huggingface.co/jinaai/jina-colbert-v2> — i.e. learned sparse buys roughly
**+7–8 nDCG@10 points** over BM25 on BEIR-like corpora.

**Licensing is the blocker for SPLADE.** `naver/splade-v3` is
**`cc-by-nc-sa-4.0`**, is a gated repo (must accept conditions), 30,522-dim, 512-token max:
<https://huggingface.co/naver/splade-v3>. The SPLADE paper/lib is likewise NC
(<https://github.com/naver/splade>, CC-BY-NC-SA). **Non-commercial → do not ship.**

**BGE-M3 (recommended upgrade path).** MIT, 1024-dim dense, 8,192-token context,
simultaneously dense + **sparse lexical weights** + multi-vector (ColBERT). Sparse output is
per-token learned weights (the card prints `{'What':0.08356,...}`) and the card itself
recommends "hybrid retrieval + re-ranking". Primary: <https://huggingface.co/BAAI/bge-m3>;
paper <https://arxiv.org/abs/2402.03216>. Honest caveat: the same card says "BM25 remains a
competitive baseline, especially in long document retrieval" — so BGE-M3 sparse is not
guaranteed to beat our BM25 on *short* video rows; it must pass the eval gate.

**Qwen3 has no sparse head.** The Qwen3 Embedding series is dense embedding + a separate
reranker (0.6B/4B/8B); there is no lexical/sparse output. Primary model list:
<https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>, paper <https://arxiv.org/abs/2506.05176>.
Any "Qwen3 sparse" model would be a community derivative **[uncertain]** — not something to
depend on.

**Cost / availability.** SiliconFlow lists **`BAAI/bge-m3` free** and
**`BAAI/bge-reranker-v2-m3` free**, with paid "Pro" mirrors at ¥0.07/M tokens:
<https://www.siliconflow.cn/pricing>. (English page: <https://www.siliconflow.com/pricing>.)
BGE-M3 is also on OpenRouter at ~$0.01/M tokens: <https://openrouter.ai/baai/bge-m3>.
Local BGE-M3 on CPU is feasible for a single query forward pass (0.6B) and fine for
offline corpus sparse-weight extraction, but expect CPU latency in the hundreds of ms to
seconds **[uncertain — not benchmarked on this box]**.

### 3. `pgvector sparsevec` at 0.8.7 — what it actually supports

From the pgvector 0.8.7 source `src/sparsevec.h` (primary):
`#define SPARSEVEC_MAX_DIM 1000000000` and `#define SPARSEVEC_MAX_NNZ 16000`
(<https://raw.githubusercontent.com/pgvector/pgvector/master/src/sparsevec.h>).
README confirms the storage formula and limits:

- **Storage:** `8 * non-zero elements + 16` bytes per vector; up to **16,000 non-zero**
  elements; dimension up to 1,000,000,000.
- **Operators:** `<->` (L2), `<#>` (negative inner product), `<=>` (cosine), `<+>` (L1);
  functions `l2_distance`, `inner_product`, `cosine_distance`, `l1_distance`, `l2_norm`,
  `l2_normalize`. No bit-style Hamming/Jaccard for sparsevec.
- **Indexes:** **HNSW only** for sparse vectors, using `sparsevec_l2_ops` /
  `sparsevec_ip_ops` / `sparsevec_cosine_ops`, and **HNSW supports sparsevec only up to
  1,000 non-zero elements** (added in 0.7.0). **IVFFlat's supported-types list does not
  include `sparsevec`.** Without an index, queries are exact scans.
  Primary: <https://github.com/pgvector/pgvector> (Indexing/HNSW/IVFFlat + Sparse Vectors
  sections).
- **The 1,000-nnz HNSW cap bites SPLADE:** open issue #749 requests raising it because
  SPLADE rows reach ~1,100 nnz: <https://github.com/pgvector/pgvector/issues/749>.

**Implication for us:** build an **HNSW `sparsevec_ip_ops`** index; assert per-row nnz
≤ 1,000 before index build (fail-loud), since a single >1,000-nnz row cannot be HNSW-indexed.
With 665k rows at ~50–200 nnz each, storage is ~ tens of MB — negligible. HNSW also
**skips NULL and zero vectors**, so rows without sparse embeddings simply drop out of the
sparse arm (aligns with the fail-open design).

### 4. Hybrid fusion: RRF vs. weighted vs. learned

**RRF.** Cormack et al. (SIGIR 2009) introduced RRF and suggested **k = 60**; the simplified
formula is `Σ 1/(k + rank)`. Elasticsearch's RRF retriever documents `rank_constant`
default **60** and notes RRF "requires no tuning":
<https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion>
(paper: <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>).
Qdrant's default is **k = 2** and it exposes `k` (default 60 in their example), plus
**weighted RRF (v1.17)** and **DBSF (v1.11)**:
<https://qdrant.tech/documentation/search/hybrid-queries/>. Elastic's own grid search on a
BEIR subset found **k = 20, N = 1000** optimal for several bi-encoders, and reported
**RRF +1.4% over ELSER, +18% over BM25**; a *well-calibrated* weighted sum gave
**+6% / +24%**, but needs score normalization and ~**40 annotated queries** to beat RRF, and
"the optimal weight is model specific": <https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid>.
Counterpoint (primary, peer-reviewed): Bruch et al., "An Analysis of Fusion Functions for
Hybrid Retrieval", find RRF **sensitive to its parameters** and that a **convex
combination (CC)** generally beats RRF in-domain and out-of-domain while needing only a
small labeled set: <https://arxiv.org/abs/2210.11934> (TOIS 2024).

**Pitfall — raw-score linear combination.** Dense cosine is bounded; BM25 (`sparsevec`
inner product) is unbounded and query-dependent. A fixed `α` over raw scores is
"dominated by whichever retriever has larger raw magnitudes on a given query" — normalize
(or use ranks) first. Primary: Qdrant, same page above. Elastic confirms min–max
(best: per-data-set) normalization is required and still doesn't make the weight transfer:
same Elastic blog.

**Metadata filters.** `orientation`, `duration`, `width/height`, `license` are exact
predicates on `stock_videos`. Apply them as a `WHERE` **prefilter in both arms** so the
candidate set is filter-clean before fusion, rather than adding a third fused "ranker".
When using HNSW + filtering, remember pgvector applies filters *after* the index scan and
can under-return; 0.8.0+ supports `hnsw.iterative_scan = strict_order|relaxed_order`, or use
partial/multicolumn indexes: <https://github.com/pgvector/pgvector> (Filtering /
Iterative Index Scans). For heavily filtered stock queries (e.g. `orientation='portrait'`),
a B-tree index on the filter columns plus iterative scan is the cheap path.

**Recommendation:** implement **RRF k=60** now (no eval set), fetch ~100 per arm; add a
small relevance eval set (creator/VLM accept labels, already the handoff's gate) and only
then try normalized weighted fusion / DBSF. Remember to test `tune` weights on a
train/val split — Qdrant explicitly warns that measuring on the tuned queries inflates
results (same page).

### 5. Stage-2 reranking

**Cross-encoders.**
- **Qwen3-Reranker-0.6B** — Apache-2.0, 32k context, instruction-aware (custom instruct
  gives ~1–5%). On MTEB-R it scores **65.80** vs **57.03** for bge-reranker-v2-m3; MLDR
  67.28 vs 59.51. Eval reranks the **top-100** dense candidates. Primary:
  <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>. (4B: MTEB-R 69.76; 8B: 69.02 but
  wins CMTEB-R/MMTEB-R/MLDR — more accurate, more expensive.)
- **bge-reranker-v2-m3** — Apache-2.0, 0.6B, multilingual, "fast inference"; card's
  evaluations rerank the **top-100** from bge-en-v1.5 / bge-m3. Primary:
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>. **Free on SiliconFlow.**
- **jina-reranker-v2-base-multilingual** — strong, but **`cc-by-nc-4.0`** (research only;
  commercial requires Jina's API/marketplace): <https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual>.
  Its own card puts BEIR nDCG@10 at **53.17** (jina) vs **53.65** (bge-reranker-v2-m3).
- **Cohere rerank** — API only, no GPU needed; current docs show `rerank-v4.0-pro` /
  `rerank-v4.0-fast` and `rerank-v3.5`, 100+ languages, billed in **search units**:
  <https://docs.cohere.com/docs/rerank-overview>. Pricing page is JS-rendered; a
  secondary source quotes **~$2.00 per 1,000 searches** for Rerank 3.5
  (<https://cohere.com/pricing>; secondary: <https://www.metacto.com/blogs/cohere-pricing-explained-a-deep-dive-into-integration-development-costs>)
  OpenRouter lists Cohere Rerank v3.5 at **$0.001/search**:
  <https://openrouter.ai/cohere/rerank-v3.5>.

**Late-interaction / ColBERT.**
- **BGE-M3 ColBERT mode** — MIT (same weights as the embedding model), 8k context;
  a MaxSim reranker reusing vectors we may already compute:
  <https://huggingface.co/BAAI/bge-m3>.
- **answerai-colbert-small-v1** — **Apache-2.0**, only **33M params / 96-dim**, BEIR avg
  **53.79** (beats ColBERTv2.0's 50.02 and bge-base-en). Best CPU/small-GPU late-interaction
  option: <https://huggingface.co/answerdotai/answerai-colbert-small-v1>.
- **jina-colbert-v2** — 8k context, Matryoshka 128/96/64, BEIR avg **0.531** vs BM25
  **0.440**, MS MARCO MRR@10 **0.396** vs **0.187** — but **`cc-by-nc-4.0`**:
  <https://huggingface.co/jinaai/jina-colbert-v2>. Do not ship.

**VLM reranking.** Out of scope for the current metadata-only corpus: thumbnails are
"not downloaded", so any vision reranker would need a new fetch+store path (and likely a
GPU or paid API). **[uncertain / deferred]** — revisit only if the handoff's VLM-walkthrough
gate (`docs/handoff-2026-10-07.md` §5) is actually reinstated; it changes the cost model
materially.

**Latency/cost intuition (mark as estimates).** Reranking K short documents = K cross-encoder
forward passes. Our `embed_text` is title + description + ~47 tags → short inputs, so a
0.6B cross-encoder on CPU is the practical ceiling for interactive use; for volume, the free
SiliconFlow `bge-reranker-v2-m3` API or a 33M ColBERT are the cheap options. Concrete
numbers require a local benchmark — not done here. **[uncertain]**

## Comparison tables

### Sparse retrieval options (for 665k short metadata rows)

| Option | Type / how it runs | Quality evidence (primary) | Cost | License | Verdict |
|---|---|---|---|---|---|
| Local BM25 → `sparsevec` | pure Python, in-DB scoring | BM25 BEIR-13 ≈ **0.440 nDCG@10** ([jina](https://huggingface.co/jinaai/jina-colbert-v2)) | $0, deterministic | our code | **Default v1** (fix idf seam) |
| `tsvector` + GIN + `ts_rank` | native PG | no IDF/saturation/length-norm; 38,797 ms vs 81 ms top-N ([Tiger](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres), [Neon](https://neon.com/blog/pgsearch-on-neon)) | $0 | PostgreSQL | **Prefilter only** |
| ParadeDB `pg_search` | Tantivy extension in PG | BM25 (Elastic-grade); build 1.63 vs 2.60 min; 647 vs 176 MB @10M ([Neon](https://neon.com/blog/pgsearch-on-neon)) | ops + image/preload | **AGPL-3.0** ([link](https://www.paradedb.com/blog/agpl)) | Escape hatch |
| `bm25s` | Python lib (Numpy/Numba) | Lucene BM25; 2M-doc index 0.49 GB mmap ([repo](https://github.com/xhluca/bm25s)) | $0 in-process | MIT | only if scorer is a bottleneck |
| **BGE-M3 sparse** | API (SiliconFlow) or local CPU | MIT; dense+sparse+ColBERT ([card](https://huggingface.co/BAAI/bge-m3)); "BM25 remains competitive" | **free** on [SiliconFlow](https://www.siliconflow.cn/pricing); ~$0.01/M on [OpenRouter](https://openrouter.ai/baai/bge-m3) | **MIT** | **Upgrade candidate (gated)** |
| SPLADE++ / SPLADE-v3 | local (needs GPU ideally) | v3: **40.2** MRR@10 MS MARCO, **51.7** BEIR-13 ([card](https://huggingface.co/naver/splade-v3)) | GPU/CPU heavy; 30,522-dim | **cc-by-nc-sa-4.0** | **Reject (non-commercial)** |
| VectorChord-bm25 / pg_textsearch | PG extensions | BM25; preview/English-only ([VCh](https://github.com/tensorchord/VectorChord-bm25), [pg_ts](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres)) | ops | AGPLv3/ELv2 / preview | watch list |

### `pgvector sparsevec` (0.8.7) vs dense `vector`

| Property | `sparsevec` | `vector(768)` dense |
|---|---|---|
| Storage | `8·nnz + 16` bytes (nnz ≤ **16,000**) | `4·dim + 8` bytes |
| Max dimensions | **1,000,000,000** | 16,000 |
| HNSW index | yes, **nnz ≤ 1,000** (`sparsevec_l2/ip/cosine_ops`) | yes (≤2,000 dims) |
| IVFFlat index | **no** | yes (≤2,000 dims) |
| Operators | `<-> <#> <=> <+>` | `<-> <#> <=> <+>` |
| NULL/zero vectors | not indexed | zero vectors not indexed (cosine) |
| Source | [sparsevec.h](https://raw.githubusercontent.com/pgvector/pgvector/master/src/sparsevec.h), [README](https://github.com/pgvector/pgvector) | same |

### Fusion methods

| Method | Tuning needed | Quality evidence (primary) | Pitfall |
|---|---|---|---|
| RRF | none (k=60 default; Elastic found k=20/N=1000) | +1.4% over ELSER, +18% over BM25 ([Elastic](https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid)) | Bruch: sensitive to k/N ([paper](https://arxiv.org/abs/2210.11934)); Qdrant default k=2 ([docs](https://qdrant.tech/documentation/search/hybrid-queries/)) |
| Weighted / convex | normalization + labeled set (~40 queries) | +6% ELSER, +24% BM25 **when calibrated** (Elastic) | model-specific weights; raw α unreliable ([Qdrant](https://qdrant.tech/documentation/search/hybrid-queries/)) |
| DBSF | none; needs score priors | Qdrant v1.11 | stats from small prefetch top-k; outlier-sensitive ([Qdrant](https://qdrant.tech/documentation/search/hybrid-queries/)) |
| Learned fusion | training pipeline | — | not justified at this scale |

### Rerankers (stage 2)

| Model | Params | Context | Quality (primary) | License | Notes |
|---|---|---|---|---|---|
| Qwen3-Reranker-0.6B | 0.6B | 32k | MTEB-R **65.80**; MLDR 67.28 ([card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)) | Apache-2.0 | best small cross-encoder; instruct +1–5% |
| Qwen3-Reranker-4B / 8B | 4B / 8B | 32k | MTEB-R 69.76 / 69.02 ([card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)) | Apache-2.0 | heavier; API or GPU |
| bge-reranker-v2-m3 | 0.6B | (multi) | BEIR 53.65 ([jina](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)) | Apache-2.0 | **free on SiliconFlow** |
| jina-reranker-v2-multilingual | 0.3B | 1024 | BEIR 53.17; MKQA 54.83 ([card](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)) | **cc-by-nc-4.0** | research only |
| Cohere rerank-v4.0 / v3.5 | API | — | 100+ langs; billed in search units ([docs](https://docs.cohere.com/docs/rerank-overview)) | commercial | no GPU; ~$0.001/search [OpenRouter](https://openrouter.ai/cohere/rerank-v3.5) |
| BGE-M3 ColBERT | 0.6B | 8k | MIT; MaxSim reuse ([card](https://huggingface.co/BAAI/bge-m3)) | MIT | cheap if vectors already exist |
| answerai-colbert-small-v1 | 33M | — | BEIR **53.79** ([card](https://huggingface.co/answerdotai/answerai-colbert-small-v1)) | Apache-2.0 | **best CPU late-interaction** |
| jina-colbert-v2 | 0.6B | 8k | BEIR 0.531 vs BM25 0.440 ([card](https://huggingface.co/jinaai/jina-colbert-v2)) | **cc-by-nc-4.0** | research only |

## Recommended 2-stage pipeline

```
query
  │  (dense embed)                (sparse: local BM25 → sparsevec; later BGE-M3 sparse)
  ├── SQL metadata prefilter  ──┬──► WHERE orientation=… AND duration BETWEEN … AND …
  │                            │
  ├─ dense arm (HNSW cosine)  ─┘   top-100
  └─ sparse arm (HNSW sparsevec_ip) top-100
                │
        RRF k=60  (no eval set)  ──►  ~200 candidates   [stage 1]
                │
        cross-encoder rerank  ──► top 20–50             [stage 2]
                │
        return top-30 (contract: POST /search → top-30)
```

- **Stage 1 budget:** retrieve **100–500 per arm** (Elastic's grid search favored N=1000 for
  max quality; ~100 keeps latency and the 1,000-nnz HNSW scan cheap). Prefilter with SQL on
  both arms.
- **Stage 2 budget:** rerank **20–50** with the cross-encoder and return top-30. Both
  Qwen3-Reranker and bge-reranker cards evaluate on **top-100** retrieved candidates, so
  100→rerank is the validated shape; 200–500 just buys recall for hard/rare concepts.
- **Fusion:** RRF **k=60** to start. Once the eval set exists, try normalized weighted RRF /
  DBSF / convex on a train/val split.
- **Reranker choice:** if a GPU/API budget exists → Qwen3-Reranker-0.6B (best small-model
  MTEB-R) or Cohere; if CPU-only and interactive latency matters → answerai-colbert-small-v1
  (33M) or BGE-M3 ColBERT; if minimizing ops → free SiliconFlow bge-reranker-v2-m3.
- **Keep the seam:** `embed_text` is the single canonical text for dense+sparse+filter+RAG;
  record `sparse_model` / `embedding_model` per row (already in migration 003) so BM25 and
  BGE-M3 sparse can coexist during eval.

## Open questions & uncertainty

1. **BM25 seam correctness (highest priority).** `store/bm25.py` puts IDF on *both* doc and
   query sides, so `sparsevec <#>` yields `idf²`-weighted scores, not textbook BM25. Verify
   with a unit test before tuning. (This note did not modify the file.)
2. **Does BGE-M3 sparse beat our local BM25 on *short* Pexels rows?** The card says BM25
   stays competitive, especially long docs. Only the Phase-2 recall@k / VLM-acceptance eval
   can answer this. **[uncertain]**
3. **CPU latency of any 0.6B model on this box (no GPU)** — BGE-M3 query forward pass, BGE-M3
   ColBERT, and Qwen3-Reranker-0.6B reranking of ~50 short docs. Not benchmarked here. **[uncertain]**
4. **Cohere rerank price** — the pricing page is JS-rendered; `~$2/1k searches` is from a
   secondary source, while OpenRouter quotes `$0.001/search`. Re-verify before budgeting.
   **[time-sensitive / uncertain]**
5. **`sparsevec` nnz > 1,000** silently cannot be HNSW-indexed. Need a pre-build assertion
   (fail-loud) and a fallback (truncate to top-1000 weights, or exact scan for the outliers).
6. **ParadeDB integration** — if `pg_search` is ever adopted as the primary store, confirm
   how its BM25 index coexists with the pgvector `sparsevec` column and whether the AGPL
   obligation fits the product's distribution model. **[decision for user]**
7. **Qwen3 "sparse"** — no official sparse/lexical head exists in the Qwen3-Embedding series
   ([model list](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)); any such model is a
   community derivative and should not be a plan dependency. **[uncertain]**
8. **VLM reranking** — deferred; requires downloading thumbnails/videos (currently not
   stored) and a GPU/API. Reconsider only if the handoff's VLM-walkthrough gate returns.
9. **Freshness of `jina-*`/Cohere model lineup** — reranker generations move fast (Cohere now
   advertises `rerank-v4.0`; Elastic/Jina are converging). Re-check before locking a vendor.
   **[time-sensitive]**

### Source index (primary)

- pgvector README + sparsevec source: <https://github.com/pgvector/pgvector> ·
  <https://raw.githubusercontent.com/pgvector/pgvector/master/src/sparsevec.h> ·
  <https://github.com/pgvector/pgvector/issues/749>
- ParadeDB: <https://www.paradedb.com/docs/start/install> · <https://www.paradedb.com/blog/agpl> ·
  <https://www.paradedb.com/docs/operate/deploy/enterprise> · <https://www.paradedb.com/docs/welcome/introduction>
- Neon/ParadeDB 10M-row benchmark: <https://neon.com/blog/pgsearch-on-neon>
- Tiger Data pg_textsearch: <https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres>
- VectorChord-bm25: <https://github.com/tensorchord/VectorChord-bm25>
- bm25s: <https://github.com/xhluca/bm25s> · <https://arxiv.org/abs/2407.03618>
- RRF: <https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion> ·
  <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
- Fusion analysis: <https://arxiv.org/abs/2210.11934>
- Elastic hybrid (RRF vs weighted): <https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid>
- Qdrant hybrid/fusion: <https://qdrant.tech/documentation/search/hybrid-queries/>
- BGE-M3: <https://huggingface.co/BAAI/bge-m3> · <https://arxiv.org/abs/2402.03216>
- bge-reranker-v2-m3: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- SPLADE: <https://huggingface.co/naver/splade-v3> · <https://arxiv.org/abs/2403.06789> ·
  <https://arxiv.org/abs/2205.04733>
- Qwen3 Embedding/Reranker: <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B> · <https://arxiv.org/abs/2506.05176>
- Jina: <https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual> ·
  <https://huggingface.co/jinaai/jina-colbert-v2>
- Answer.AI ColBERT: <https://huggingface.co/answerdotai/answerai-colbert-small-v1>
- Cohere: <https://docs.cohere.com/docs/rerank-overview> · <https://cohere.com/pricing> ·
  <https://openrouter.ai/cohere/rerank-v3.5>
- SiliconFlow: <https://www.siliconflow.cn/pricing> · <https://www.siliconflow.com/pricing> ·
  <https://docs.siliconflow.com/en/api-reference/rerank/create-rerank>
