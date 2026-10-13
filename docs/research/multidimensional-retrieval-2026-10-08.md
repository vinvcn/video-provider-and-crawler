# Multidimensional / multi-vector retrieval for stock-video search (2026-10-08)

Scope: how to store and query **many embeddings per video** — frame embeddings plus per-facet
caption embeddings (subject, genre, lighting, colour, mood, camera/shot, setting, …) — on the
fixed stack: single PostgreSQL 16 + pgvector 0.8.7, DB `stock_videos`, no GPU, ~665k target videos
(664,546 rows; 365,103 live as of 2026-10-08), container **6 GiB RAM / 64 MiB shm** (live facts
from [`pgvector-index-engineering-2026-10-08.md`](pgvector-index-engineering-2026-10-08.md) §1).

This note builds on, and does not repeat, the sibling research:

- [`pgvector-index-engineering-2026-10-08.md`](pgvector-index-engineering-2026-10-08.md) — HNSW
  sizing model (~670 B/element overhead), filtering/iterative scans, `sparsevec` limits, build GUCs.
- [`sparse-hybrid-rerank-2026-10-08.md`](sparse-hybrid-rerank-2026-10-08.md) — BM25→`sparsevec`,
  RRF (k=60), weighted-fusion calibration caveats, cross-encoder rerankers, ColBERT model options.
- [`dense-embedding-models-2026-10-08.md`](dense-embedding-models-2026-10-08.md) — MRL models,
  `text-embedding-v4`/Qwen3, multimodal image embedding cost (~US$150 one-time), motion caveat.

Every non-obvious claim/number below carries a URL to a primary source (pgvector repo/source/issues,
PostgreSQL docs, papers, vendor docs). Estimates are labelled **[estimate]**.

---

## TL;DR

1. **Layout: side table, facet as data, one PARTIAL HNSW index per (facet, model).** This is
   pgvector's own documented pattern for many mixed models/dimensions:
   `embeddings(model_id, item_id, embedding vector)` + partial expression index
   `... USING hnsw ((embedding::vector(3)) vector_l2_ops) WHERE (model_id = 123)`
   ([README FAQ](https://github.com/pgvector/pgvector/blob/master/README.md#can-i-store-vectors-with-different-dimensions-in-the-same-column)).
   Map `model_id → facet` (plus a `model` string for versioning). New facets = `INSERT`s, not DDL.
2. **Do not push past ~2 vector columns on `stock_videos`.** pgvector stores `vector` columns with
   `external` TOAST storage (out-of-line, uncompressed — [CHANGELOG 0.6.0](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md)),
   so every extra column is an 18-byte pointer + a separate out-of-line value; per-facet metadata
   (model, caption, dim) has nowhere to live; and adding facet #9 needs `ALTER TABLE` + a new HNSW
   build.
3. **Sizing: at full float32, more than ~2 facets × 768d does not fit this box.** For 664,546 videos
   the full-precision dense HNSW working set is ~2.3 GiB **per facet**; F=4×768d ≈ **9.3 GiB**,
   F=8×768d ≈ **18.6 GiB** build RAM. The same budget in **halfvec 256d** is **3.0 GiB (F=4)** /
   **5.9 GiB (F=8)**; one facet at a time (`F=1`) is 0.74 GiB. See §1(d) arithmetic.
4. **Dimensions: truncate per facet before you add more facets.** Matryoshka is a training objective
   ([MRL paper](https://arxiv.org/abs/2205.13147)); Google's own numbers show 3072→256 dims costs
   ~2 points MTEB (68.2→66.2 for `gemini-embedding-001`; 3072→768 costs ~0.2)
   ([Gemini embeddings docs](https://ai.google.dev/gemini-api/docs/embeddings)). Prefer 256–384d
   halfvec per facet over 768d float. `subvector()` expression indexing lets you truncate in-DB and
   re-rank with the full vector ([README → Indexing Subvectors](https://github.com/pgvector/pgvector/blob/master/README.md#indexing-subvectors)).
5. **Multiple HNSW indexes on one table are practical but have three sharp edges:** (a) partial
   indexes do not match **parameterized** query clauses ("Matching takes place at query planning
   time … parameterized query clauses do not work with a partial index" —
   [PostgreSQL 11.8](https://www.postgresql.org/docs/16/indexes-partial.html)); (b) the planner can
   refuse the partial index and under-estimate the competing filter index because it ignores TOAST —
   a real case measured **0.7 ms vs 11 ms (~15×)** ([pgvector issue #878](https://github.com/pgvector/pgvector/issues/878));
   (c) pgvector has **no per-index `ef_search` default** yet — session-wide GUC only
   ([#235](https://github.com/pgvector/pgvector/issues/235), [PR #933](https://github.com/pgvector/pgvector/pull/933), both open).
6. **Late interaction (ColBERT MaxSim) has no native index in pgvector 0.8.7** (nothing in the
   [CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md); issue
   [#640](https://github.com/pgvector/pgvector/issues/640) closed with "write a SQL function to
   compute MaxSim"). The **official pattern** is a token side table + per-token HNSW candidate
   generation + SQL `max_sim` rescore ([`pgvector-python/examples/colbert/approximate.py`](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/approximate.py),
   "based on section 3.6 of https://arxiv.org/abs/2004.12832"). At 665k, token vectors alone cost
   **~21–71 GiB** at 128d (64–220 tokens/doc) — treat it as a **stage-2 rerank**, not an index.
7. **Fuse facets with RRF k=60 first, weighted fusion later.** RRF needs no calibration
   ([Cormack 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf); sibling note).
   Cross-model/cross-facet raw scores are not comparable — normalize before weighting
   ([Qdrant hybrid docs](https://qdrant.tech/documentation/search/hybrid-queries/); Elastic found a
   calibrated weighted sum +6–24% vs RRF +1.4–18%, but only with ~40 labeled queries
   ([Elastic](https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid))).
8. **Hard filters for constraint facets, soft scores for preference facets.** `orientation`,
   `duration`, `resolution`, `license` are exact predicates (existing B-trees/GIN; pgvector: "Exact
   indexes work well for conditions that match a low percentage of rows" —
   [README Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering)).
   Grade genre/lighting/mood/colour as scores; product-search relevance is itself graded
   (Amazon ESCI: Exact/Substitute/Complement/Irrelevant, 2.6M judgments —
   [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)).
9. **A facet-conditioned single index is real but risky.** Instruction-conditioned embedders do work
   (INSTRUCTOR: 330 tasks, one model, +3.4% avg — [arXiv:2212.09741](https://arxiv.org/abs/2212.09741);
   TART/BERRI: ~40 retrieval tasks, one retriever — [arXiv:2211.09260](https://arxiv.org/abs/2211.09260);
   Google `task:` prefixes and Cohere `input_type` productionize it — [Gemini](https://ai.google.dev/gemini-api/docs/embeddings),
   [Cohere](https://docs.cohere.com/docs/embeddings)). But prompt phrasing alone moves benchmark
   scores and even leaderboard order ([arXiv:2605.22544](https://arxiv.org/abs/2605.22544)), and most
   retrieval models fail to follow instructions (FollowIR: only 3B+ models succeed —
   [arXiv:2403.15246](https://arxiv.org/abs/2403.15246)). Use it per facet as an *embedding style*,
   not as a substitute for per-facet indexes.
10. **Facet routing pays only when the query mix is single-facet dominated.** Resource-selection
    literature says routing is a mature idea ([Shokouhi & Si 2011](https://dl.acm.org/doi/10.1561/1500000010));
    zero-shot LLM routing works ([ReSLLM](https://arxiv.org/abs/2401.17645)), learned LTR routing
    works ([LTRR](https://arxiv.org/abs/2506.13743)), but naive single-retriever routing leaves
    gains on the table ([MoR, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.601.pdf)).
    Default to "query all facets" while F is small; add routing with an **abstain → search all**
    fallback and measure facet-level recall of the routed set.

---

## Layout options in pgvector

### (a) Side table `video_embeddings(video_id, facet, caption, embedding vector(D), sparse, model)` — one partial HNSW per facet

This is the recommended shape. pgvector's FAQ documents the exact pattern for mixed
models/dimensions: use the unconstrained `vector` type in one column, then create a **partial
expression index per model**:

```sql
CREATE TABLE embeddings (model_id bigint, item_id bigint, embedding vector, PRIMARY KEY (model_id, item_id));
CREATE INDEX ON embeddings USING hnsw ((embedding::vector(3)) vector_l2_ops) WHERE (model_id = 123);
SELECT * FROM embeddings WHERE model_id = 123 ORDER BY embedding::vector(3) <-> '[3,1,2]' LIMIT 5;
```

> "you can only create indexes on rows with the same number of dimensions (using expression and
> partial indexing)" — [pgvector README FAQ](https://github.com/pgvector/pgvector/blob/master/README.md#can-i-store-vectors-with-different-dimensions-in-the-same-column).

Sketch for this repo:

```sql
CREATE TABLE video_embeddings (
    video_id       BIGINT   NOT NULL REFERENCES stock_videos(pexels_id) ON DELETE CASCADE,
    facet          TEXT     NOT NULL,   -- 'subject' | 'genre' | 'lighting' | 'colour' | 'mood' | 'camera' | 'setting'
    model          TEXT     NOT NULL,   -- e.g. 'text-embedding-v4@768' / 'qwen3-vl-embedding@1024'
    caption        TEXT,                -- exact text embedded (audit, debug, re-embed)
    embedding      vector,              -- mixed dims live here; cast in the index/query
    sparse_embedding sparsevec,         -- optional lexical arm per facet
    embedded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (video_id, facet, model)
);
CREATE INDEX idx_ve_facet_video ON video_embeddings (facet, video_id);

CREATE INDEX CONCURRENTLY idx_ve_genre_hnsw ON video_embeddings
    USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops) WHERE (facet = 'genre');
-- …one per facet, each over ~665k rows (one embedding per video per facet)
```

**Why partial indexes per facet (vs one full index over the side table):**

- With a **single full HNSW index** over F facets (N×F rows), a query `WHERE facet='genre' ORDER BY
  embedding <=> q LIMIT k` post-filters: at facet selectivity 1/F and default `ef_search=40`, ~40×0.25
  = **10 rows (F=4)** or **5 rows (F=8)** survive on average — the documented 10%-filter example gives
  exactly 4 ([README Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering)).
  You must enable `hnsw.iterative_scan` or raise `ef_search` to ≈ k×F to fill `LIMIT k`
  ([README Iterative Index Scans](https://github.com/pgvector/pgvector/blob/master/README.md#iterative-index-scans);
  defaults: `max_scan_tuples=20000`, `scan_mem_multiplier=1`).
- **Partial HNSW indexes** contain only the facet's rows, so the graph search is facet-clean, the
  per-facet working set is smaller, and inserts of other facets don't touch the graph. HNSW needs no
  training data and can be built on an empty table, so facets can be indexed incrementally with
  `CREATE INDEX CONCURRENTLY` after each backfill
  ([README HNSW](https://github.com/pgvector/pgvector/blob/master/README.md#hnsw)).
- **Cost:** F graphs → F× vacuum/reindex/insert amplification (HNSW vacuuming is slow; "REINDEX …
  VACUUM" — [README Vacuuming](https://github.com/pgvector/pgvector/blob/master/README.md#vacuuming)).

**Sharp edges (primary-sourced):**

1. **Parameterized queries cannot use a partial index.** PostgreSQL: "Matching takes place at query
   planning time, not at run time. As a result, parameterized query clauses do not work with a
   partial index" ([PostgreSQL 11.8](https://www.postgresql.org/docs/16/indexes-partial.html)). The
   app must issue **facet constants** (validated allowlist, then inlined), not `$2`, wherever a
   partial index is expected.
2. **Planner selection is not automatic.** In issue [#878](https://github.com/pgvector/pgvector/issues/878)
   the ORM-fixed plan chose a boolean B-tree scan (11.1 ms) over the partial HNSW (0.74 ms, ~15×);
   the maintainer's diagnosis: "Postgres underestimates the cost of the `my_table_handy_bool_idx`
   index scan since it doesn't take **TOAST** into account", with `random_page_cost=1.1` it flips at
   ~50k rows; remedies: `ALTER COLUMN embedding SET STORAGE PLAIN` or `pg_hint_plan`. Verify each
   facet's plan with `EXPLAIN (ANALYZE, BUFFERS)` and keep stats fresh (`ANALYZE`).
3. **Many disjoint partial indexes are an anti-pattern for the *planner*.** PostgreSQL: "This is a
   bad idea! … the system does not understand the relationship among the partial indexes, and will
   laboriously test each one", recommending partitioning instead
   ([PostgreSQL Example 11.4](https://www.postgresql.org/docs/16/indexes-partial.html#INDEXES-PARTIAL-EX4)).
   pgvector says the same for high-cardinality filters: "If filtering by many different values,
   consider partitioning" ([README Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering)).
   **Route:** F ≤ ~8 → partial indexes; F grows large or facet values disjoint → `PARTITION BY LIST
   (facet)` with one HNSW per partition (partition pruning is planner-visible, and pgvector's
   multitenancy guidance endorses list partitioning for isolation —
   [README Multitenancy](https://github.com/pgvector/pgvector/blob/master/README.md#multitenancy)).
4. **No per-index `ef_search` yet.** `hnsw.ef_search` is a session GUC; the per-index default RFC is
   open ([#235](https://github.com/pgvector/pgvector/issues/235)) with an open PR
   ([#933](https://github.com/pgvector/pgvector/pull/933), `default_ef_search`/`default_probes`).
   With F facets of different sizes, one session setting applies to all scans in that query —
   acceptable, but tune for the smallest facet or set per query.

### (b) Extra vector columns on `stock_videos`

Fine for **1–2** first-class vectors (the repo already has `embedding vector(768)` from
`migrations/003_embedding_columns.sql`, plus `embedding_model`), not for a facet matrix:

- **TOAST:** rows wider than ~2 kB are compressed/moved out-of-line until ≤ ~2 kB
  ([PostgreSQL 73.2](https://www.postgresql.org/docs/16/storage-toast.html)). But pgvector already
  sets `vector` storage to `external` — out-of-line, **no compression** — since 0.6.0
  ([CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md)). So a 768d column
  (3,080 B) is always an out-of-line value; each extra facet column adds an 18-byte pointer to the
  main tuple plus its TOAST value/chunks (~2 kB chunks, 4/page —
  [TOAST §73.2.1](https://www.postgresql.org/docs/16/storage-toast.html)). Main rows stay narrow;
  the cost is per-facet fetch + no compression benefit.
- **DDL per facet:** every new facet = `ALTER TABLE` + `CREATE INDEX` + backfill; model/dim changes =
  another column (or `SET STORAGE`/cast games); per-facet model strings need per-column metadata.
- **Write amplification:** each HNSW index is independently updated on insert/update, and each index
  stores its **own copy of the vector** (index element tuple = vector + links —
  [issue #370](https://github.com/pgvector/pgvector/issues/370)), so table+index ≈ 2× vector bytes,
  ×F columns.
- **Planner upside:** part (a)'s partial-index pitfalls disappear — each facet column has its own
  plain HNSW index and no parameterized-predicate problem (PostgreSQL
  [partial-index matching](https://www.postgresql.org/docs/16/indexes-partial.html) doesn't apply).
- pgvector cannot build one HNSW across multiple columns; "add an index for each distance function"
  / each vector column ([README HNSW](https://github.com/pgvector/pgvector/blob/master/README.md#hnsw)).

**Verdict:** use main-table columns for L1 (whole-clip vector, already exists) and possibly one
visual vector; put every facet in the side table.

### (c) Lower dimensions and `halfvec`

- **MRL/truncation is first-class.** MRL trains embeddings whose prefixes are usable standalone —
  "up to 14× smaller embedding size at the same level of accuracy" ([arXiv:2205.13147](https://arxiv.org/abs/2205.13147)).
  Google's published table for `gemini-embedding-001` (3072d base): 2048→68.16, 1536→68.17,
  **768→67.99, 512→67.55, 256→66.19, 128→63.31** MTEB — i.e. 768 costs ~0.2 points, 256 ~2 points
  ([Gemini embeddings docs](https://ai.google.dev/gemini-api/docs/embeddings)). The repo's planned
  `text-embedding-v4` has native 768 (and 256/128/64) outputs, so no lossy cast for the L1 column
  (sibling note `dense-embedding-models`).
- **In-DB truncation:** `subvector(embedding,1,256)::vector(256)` can be **expression-indexed**;
  query with the same expression and re-rank candidates by the full vector
  ([README Indexing Subvectors](https://github.com/pgvector/pgvector/blob/master/README.md#indexing-subvectors)).
  Normalize after truncation if the model doesn't do it automatically (Google warns
  `gemini-embedding-001` requires manual renormalization for non-default dims; v2 auto-normalizes —
  [docs](https://ai.google.dev/gemini-api/docs/embeddings)).
- **`halfvec`:** storage `2·d+8` B vs `4·d+8` (README Reference; 768d: 1,544 vs 3,080 B), index up
  to 4,000 dims, index via expression cast `((embedding::halfvec(768)) halfvec_cosine_ops)`, and
  re-rank with the full vector for better recall ([README Half-Precision Indexing](https://github.com/pgvector/pgvector/blob/master/README.md#half-precision-indexing)).
  Community evidence that bfloat16-trained embeddings lose nothing: Discourse ran `halfvec` at half
  storage "losing us nothing" ([issue #521](https://github.com/pgvector/pgvector/issues/521));
  Qdrant's docs claim Float16 has "virtually no impact on the quality of vector search"
  ([Qdrant Datatypes](https://qdrant.tech/documentation/concepts/vectors/)) — different engine, same
  direction. For our box, `halfvec(256)` = 520 B/vector is the workhorse; `halfvec(384)` = 776 B.
- **Binary quantization + re-rank** (1 bit/dim) is the last-resort lever: index
  `binary_quantize(embedding)::bit(768)` with `bit_hamming_ops`, retrieve ~20, re-order by full
  vector ([README Binary Quantization](https://github.com/pgvector/pgvector/blob/master/README.md#binary-quantization)).
  Keeps the full vectors in the heap; shrinks index only. At 665k the plain halfvec index fits, so
  keep BQ for the image/multi-frame layer if it grows.

### (d) Multiple HNSW indexes: memory math for 665k × F facets × D dims

Per-element model from the sibling note (calibrated from [issue #769](https://github.com/pgvector/pgvector/issues/769)):
`bytes/vector(type,d) + ~670 B graph/struct overhead`; HNSW index tuples carry their own vector
(issue #370). Vector bytes: `vector(d)=4d+8`, `halfvec(d)=2d+8`
([README Reference](https://github.com/pgvector/pgvector/blob/master/README.md#reference)).

Rows: N = 664,546 videos → side table rows = N×F (F=4 → 2,658,184; F=8 → 5,316,368).

**Build/working-set RAM [estimate], GiB:**

| config | F=4 | F=8 | F=1 (one facet at a time) |
|---|---|---|---|
| 768d `vector` (3,080 B) | **9.3** | **18.6** | 2.3 |
| 768d `halfvec` (1,544 B) | 5.5 | 11.0 | 1.4 |
| 384d `vector` (1,544 B) | 5.5 | 11.0 | 1.4 |
| 384d `halfvec` (776 B) | 3.6 | 7.2 | 0.90 |
| 256d `vector` (1,032 B) | 4.2 | 8.4 | 1.05 |
| **256d `halfvec` (520 B)** | **3.0** | **5.9** | **0.74** |

Raw heap payload (vectors only, before indexes):

| config | per facet | F=4 total | F=8 total |
|---|---|---|---|
| 768d `vector` | 1.91 GiB | 7.63 GiB | 15.26 GiB |
| 768d `halfvec` | 0.96 GiB | 3.82 GiB | 7.65 GiB |
| 256d `vector` | 0.64 GiB | 2.55 GiB | 5.11 GiB |
| 256d `halfvec` | 0.32 GiB | 1.27 GiB | 2.55 GiB |

Total on-disk ≈ heap + index (index ≈ same per-element vector+overhead as build RAM):
F=4×768 `vector` ≈ **16.9 GiB**; F=8×768 `vector` ≈ **33.8 GiB**; F=8×256 `halfvec` ≈ **8.4 GiB**
[estimates]. NVMe disk is cheaper than RAM here (AGENTS.md), so disk is not the binding constraint —
**build memory and query working set are** (container 6 GiB,
[sibling note](pgvector-index-engineering-2026-10-08.md) §1/§5).

**Reading the table:**

- F=4×768 float does **not** fit a 6 GiB container working set with anything else; F=8×768 is 3× over.
- Per-facet partial indexes keep the **per-query** working set at one facet's graph (2.3 GiB at
  768d float; 0.74 GiB at 256d halfvec), even if total disk is larger. This is the main argument for
  partial indexes over one full index when RAM is capped.
- If a query touches several facet indexes, the working set is the sum of the facets touched; with
  4–8 facets and float 768 this is what breaks, not the single graph.
- Sparse: `sparsevec` HNSW is limited to **≤1,000 nnz** per vector (README HNSW supported types);
  keep facet lexical vectors small or use `tsv` + GIN ([sibling note](sparse-hybrid-rerank-2026-10-08.md) §3).

---

## Late interaction & multi-vector

**Status in pgvector 0.8.7: no native multi-vector type, no MaxSim index.** The
[CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md) has no multi-vector entry
through 0.8.7; issue [#640](https://github.com/pgvector/pgvector/issues/640) ("Late interaction
embedding support", closed 2024) answers: store `vector[]` and write a SQL `max_sim` function
(exact), and for approximate search "store each vector in a separate row and use the approach
described in Section 3.6 of the ColBERT paper".

**The official pgvector pattern** is in the maintainer-affiliated repo:

- Exact: `documents(embeddings vector(128)[])` + SQL `max_sim(document, query)`
  (unnest × cross join, max per query token, sum) —
  [`examples/colbert/exact.py`](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/exact.py).
- Approximate/rerank: token side table `document_embeddings(document_id, embedding vector(128))`
  with **one HNSW index over all token vectors**; stage 1 = for each **query token**, `ORDER BY
  embedding <=> $token LIMIT 5`, `UNION ALL` → candidate document_ids; stage 2 = exact `max_sim`
  over those documents only —
  [`examples/colbert/approximate.py`](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/approximate.py),
  headed "based on section 3.6 of https://arxiv.org/abs/2004.12832".

**Second Postgres implementation (same pattern):** VectorChord documents ColBERT rerank
(sentence-level ANN + SQL `max_sim` over token vectors) with pgvector
([docs.vectorchord.ai](https://docs.vectorchord.ai/vectorchord/use-case/colbert-rerank.html)).
So the community consensus inside Postgres is: **ANN over one representative/pooled vector or over
tokens → MaxSim rescore in SQL**; there is no index that scores MaxSim directly.

**Native multivector engines (for comparison, not adoption):**

- Qdrant v1.10+ stores matrices and scores `max_sim` natively:
  `score = Σ_i max_j Sim(A_i, B_j)`; hard limit `#subvectors × dim < 1,048,576` (4096d → ≤255
  subvectors), with advice to pool/split beyond that ([Qdrant Vectors](https://qdrant.tech/documentation/concepts/vectors/)).
- Weaviate 1.29+ supports multi-vectors **only with HNSW indexes**, and gives a storage intuition:
  1 single 1536d vector = 6 kB vs 64×96d = 25 kB ("over 4 times larger")
  ([tutorial](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings),
  [config](https://docs.weaviate.io/weaviate/manage-collections/vector-config)).
- Weaviate also ships **MUVERA encoding**, which compresses a multi-vector into one fixed-dimensional
  vector to "reduce memory usage compared to traditional multi-vector approaches"
  ([docs](https://docs.weaviate.io/weaviate/configuration/compression/multi-vectors)).

**Is MaxSim worth it here? Storage says no for index-resident token vectors.** At 665k videos and a
ColBERT-ish 128d per-token vector (520 B, or 264 B halfvec):

| tokens/doc | full float32 | halfvec |
|---|---|---|
| 64 | ~20.6 GiB | ~10.5 GiB |
| 100 | ~32.2 GiB | ~16.3 GiB |
| 220 (pgvector example `doc_maxlen`) | ~70.8 GiB | ~35.9 GiB |

[estimates; 664,546 × tokens × 520/264 B]. That is 10–40× the per-facet caption budget, for a rerank
stage. Quality-wise late interaction is strong — ColBERT/ColBERTv2 are the standard
([ColBERT, SIGIR 2020](https://arxiv.org/abs/2004.12832);
[ColBERTv2, NAACL 2022](https://aclanthology.org/2022.naacl-main.272/) reduced space 6–10×) — but it
buys most of its gain over single-vector embeddings by *reranking*: PLAID cuts latency up to
**7× GPU / 45× CPU** vs vanilla ColBERTv2 at 140M passages
([PLAID, CIKM 2022](https://arxiv.org/abs/2205.09707)); XTR shows the scoring stage can be
**2–3 orders of magnitude cheaper** and +2.8 nDCG@10 on BEIR by ranking with retrieved tokens only
([XTR, NeurIPS 2023](https://arxiv.org/abs/2304.01982)); MUVERA reduces multi-vector similarity to
single-vector MIPS via Fixed Dimensional Encodings, proving ε-approximation, "same recall … while
retrieving 2–5× fewer candidates", "10% improved recall with 90% lower latency" across BEIR
([MUVERA, NeurIPS 2024](https://arxiv.org/abs/2405.19504)).

**Verdict for this project:** do **not** build a token-vector index now. Add it only as **L5
rerank over shortlists** (top 50–200), using a cheap CPU model (BGE-M3 ColBERT or
`answerai-colbert-small-v1`, 33M/96d, Apache-2.0 — sibling note §5) — and even then, a facet-aware
cross-encoder is probably simpler to operate. MUVERA-style FDE is the fallback if multi-vector ever
must be index-resident, but pgvector has no implementation of it (Weaviate does).

---

## Query-time facet fusion

**Shape:** parse the query into facet intents → run one ANN scan per facet index (constant-facet SQL,
so partial indexes match) → fuse.

```text
query ──facet parse──► {genre: "aerial drone", lighting: "golden hour", hard: {orientation: landscape, duration:[5,30]}}
   ├─ WHERE orientation='landscape' AND duration BETWEEN … (exact prefilter, both arms)
   ├─ facet=genre    → partial HNSW → top-k₁
   ├─ facet=lighting → partial HNSW → top-k₂
   └─ facet=colour   → partial HNSW → top-k₃
        fuse (RRF → later weighted) → rerank → top-30
```

**Fusion methods and when they win:**

| method | tuning/eval needed | primary evidence | caveat |
|---|---|---|---|
| **RRF, k=60** | none | Cormack 2009 introduced RRF with k=60 ([paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)); Elastic default `rank_constant=60` ([docs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)); Qdrant defaults differ (k=2 in examples) ([docs](https://qdrant.tech/documentation/search/hybrid-queries/)) | Bruch et al.: RRF is sensitive to k/N; convex combination (CC) generally beats it in- and out-of-domain ([TOIS 2024](https://arxiv.org/abs/2210.11934)) |
| Weighted / convex over normalized scores | normalization + ~40 labeled queries | Elastic grid search: RRF +1.4% ELSER / +18% BM25; calibrated weighted +6% / +24%, with weights model-specific ([Elastic](https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid)) | raw scores not comparable across models; normalize first (min-max per query) |
| DBSF / relative-score | score-prior assumptions | Qdrant distribution-based score fusion ([docs](https://qdrant.tech/documentation/search/hybrid-queries/)); Weaviate join strategies incl. `relativeScore` weighted normalized distances ([docs](https://docs.weaviate.io/weaviate/search/multi-vector)) | outlier-sensitive at small top-k |
| Learned fusion / rerank | training pipeline + labels | LTRR learned retriever ranking ([arXiv:2506.13743](https://arxiv.org/abs/2506.13743)); MoR mixture-of-retrievers ([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.601.pdf)) | not justified at our scale until a per-facet eval set exists |

**Score calibration across facets/models** is the core trap: different models, dimensions, and facet
distributions produce different score scales; "a fixed α over raw scores is dominated by whichever
retriever has larger raw magnitudes" ([Qdrant](https://qdrant.tech/documentation/search/hybrid-queries/);
sibling note §4). Score normalization for fusion is an old metasearch result (Montague & Aslam,
[Relevance score normalization for metasearch, CIKM 2001](https://dl.acm.org/doi/10.1145/502585.502645)).
Practical rule: **normalize each facet's per-query top-k (min-max or z-score) before any weighted
sum**, or start with ranks/RRF and skip calibration until an eval set exists.

**Hard filter vs soft score — draw the line by facet semantics:**

- **Hard (SQL prefilter):** orientation, duration, resolution/fps, license — objective, binary,
  already indexed with B-trees/GIN in `stock_videos`; pgvector: exact indexes suit selective
  predicates, approximate indexes need iterative scan to fill `LIMIT` under filters
  ([README Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering)).
- **Soft (facet score):** genre/theme, lighting, colour palette, mood, camera/shot type — graded,
  subjective, composition-sensitive. Amazon's product search benchmark makes gradedness explicit
  with ESCI (Exact/Substitute/Complement/Irrelevant over 2.6M judgments) rather than binary
  ([arXiv:2206.06588](https://arxiv.org/abs/2206.06588)); Fashion-IQ shows ~70% of natural
  attribute queries are compositional phrases, not bags of attributes
  ([arXiv:1905.12794](https://arxiv.org/abs/1905.12794)). Both argue for scores + rerank, not
  Boolean AND.
- **Missing facet values:** HNSW does not index NULL vectors (README FAQ); facet arms must treat
  missing embeddings as neutral (fail-open) and the fusion must not reward coverage gaps. Track
  facet coverage as a metric (see Evaluation).

**Do not over-blend:** a single linear score over facet similarities is not a composition operator;
Fashion-IQ's evidence is that composition is more than a bag of attributes. Use fusion for
candidate generation, then a **composition-aware reranker (L5)** for the top of the list.

---

## Facet-conditioned single index

Idea: one embedding model called with a facet instruction/prompt ("genre: aerial drone", "lighting:
golden hour") so query and document sides share **one index** across facets; the facet is a string,
not a schema object.

**Primary evidence that this works at all:**

- **INSTRUCTOR** — one embedder conditioned on task instructions annotated for 330 tasks; beats
  prior best by 3.4% avg over 70 datasets and is "robust to changes in instructions"
  ([arXiv:2212.09741](https://arxiv.org/abs/2212.09741), ACL Findings 2023).
- **TART/BERRI** — multi-task instruction tuning over ~40 retrieval datasets produces one retriever
  that follows human instructions and generalizes zero-shot (BEIR/LOTTE), with **X²-Retrieval**: a
  pooled, diverse-domain setup where the system "needs to find documents aligning users' intents"
  ([arXiv:2211.09260](https://arxiv.org/abs/2211.09260)).
- **E5's prefixes** ("query:", "passage:") are the minimal two-instruction version of the same idea
  ([arXiv:2212.03533](https://arxiv.org/abs/2212.03533)).
- **Production APIs:** Google now recommends literal task prefixes
  (`task: search result | query: {content}`, `task: fact checking | …`) and warns "It is important
  that the task is used consistently" ([Gemini embeddings docs](https://ai.google.dev/gemini-api/docs/embeddings));
  Cohere requires `input_type` ∈ {search_document, search_query, classification, clustering, image}
  for v3+ ([Cohere](https://docs.cohere.com/docs/embeddings)). Qwen3-Embedding is instruction-aware
  with 1–5% gains (sibling note `dense-embedding-models` §3).

**Counter-evidence / risks:**

- **Prompt sensitivity is large and under-measured.** Across 6 embedding models × 11 datasets, the
  default prompt "can both systematically understate or overstate performance", leaderboard ranking
  is not robust to prompt choice, and "under adversarial prompt selection any model can be promoted
  to first place" ([arXiv:2605.22544](https://arxiv.org/abs/2605.22544), 2026).
- **Retrieval models often can't follow instructions.** FollowIR: existing retrieval models "fail to
  correctly use instructions, using them for basic keywords"; only 3B+ parameter models succeed
  ([arXiv:2403.15246](https://arxiv.org/abs/2403.15246), NAACL 2025). InstructIR is a concurrent
  instruction-following benchmark ([arXiv:2402.14334](https://arxiv.org/abs/2402.14334)).
- **Doc-side conditioning costs the same as per-facet embeddings.** If the document text must also
  carry the facet instruction (which TART-style quality generally needs), you compute/store one doc
  embedding per facet anyway — identical storage to the side table — while adding instruction
  sensitivity and losing per-facet index isolation. Only *query-side* conditioning truly gives you
  "one index, no extra rows".
- **Shared space, mixed graphs.** One index over all facets re-introduces the filtering/selectivity
  problem of §1(a) (post-filter under-return at `ef_search=40`) and forbids per-facet ANN tuning
  until [#235](https://github.com/pgvector/pgvector/issues/235)/[#933](https://github.com/pgvector/pgvector/pull/933) land.
- **Benchmarks are single-prompt and ad-hoc.** MTEB/BEIR contain no instructions; instruction
  benchmarks had to be built specially ([FollowIR](https://arxiv.org/abs/2403.15246)).

**Recommendation:** use facet-conditioned *embedding* (instruct both sides with the same facet
string) **inside** the side-table layout — it is a cheap way to make small dims (256/384) work and
lets one model serve all facets, while keeping one index per facet for ANN isolation. Do not replace
per-facet indexes with one shared index unless an A/B on our own query mix says otherwise; if you
do, evaluate prompt phrasings (≥5 paraphrases per facet) and pick the median, not the best
([arXiv:2605.22544](https://arxiv.org/abs/2605.22544)).

---

## The additive ladder (L0–L5)

Each layer is optional; L2+ live in the **same** `video_embeddings` side table (facet = data), so
layers can be enabled per facet without further schema churn. `model` + `embedded_at` columns let
two generations coexist for A/B; per-facet partial indexes can be created/dropped
`CONCURRENTLY` at any time (HNSW needs no training —
[README HNSW](https://github.com/pgvector/pgvector/blob/master/README.md#hnsw)).

| layer | schema impact | cost [estimate] | unlocks | how to evaluate |
|---|---|---|---|---|
| **L0 — metadata filters** | none (B-trees/GIN exist, `migrations/001`) | ~0 | exact constraints: orientation, duration, resolution, license, tags | `EXPLAIN` plan checks; filter correctness on hand-written cases |
| **L1 — doc embedding** | already present (`embedding vector(768)` + `embedding_model`, `migrations/003`); add 1 HNSW | **~2.3 GiB** build/working set full float; ~1.4 GiB halfvec (sibling note §2) | whole-clip semantics ("dramatic mountain sunrise, slow motion") | Recall@k / nDCG@10 vs BM25 funnel (sibling note; MTEB convention) |
| **L2 — per-facet caption embeddings** | new `video_embeddings(video_id, facet, model, caption, embedding, sparse)`; one partial HNSW per facet | F=4–6 × 256–384d halfvec: **1.3–2.6 GiB heap + 3.0–5.5 GiB indexes** (F=4) [estimate]; plus caption generation cost | facet-level matching: genre / lighting / colour / mood / subject / camera / setting; enables routing + per-facet scoring | **per-facet nDCG@10 + facet-conditional Recall@k** on judged captions; config ablation 256 vs 384 vs 768; halfvec vs float |
| **L3 — frame embeddings** | `video_frame_embeddings(video_id, frame_idx, embedding)` (or one column on `video_embeddings` with facet `'frame'`) | thumbnail download + image embedding ~US$150 one-time (sibling note `dense-embedding-models` §4); D=768 float ≈ 1.9 GiB heap + 2.5 GiB index | visually-specified queries (composition, "from above", colour grade) not captured by tags; reuses the MPT dedup space | A/B on **visually-specified query subset** only (sibling note); cross-modal Recall@k |
| **L4 — multi-frame** | `k` rows per video (k=3–8), same table; or array per frame | ~k× L3 (storage + embedding cost) [estimate] | temporal coverage, camera movement hints; motion still only sampled (a still frame cannot encode motion — sibling note) | frame-level judgments; motion-query subset; report gain vs L3 on the same queries |
| **L5 — cross-facet rerank** | none (stage-2 model or learned fusion weights) | K forward passes of a small cross-encoder / ColBERT / VLM per query (CPU-constrained — sibling note §5) | compositional multi-facet queries ("aerial at golden hour, cool tones"); replaces linear score blending at the top | compositional suite nDCG delta vs L2 fusion; per-facet attribution (remove-one-facet ablation); accept-rate online |

Layering notes:

- **L0–L1 are essentially built** (schema + sibling plan); L2 is the first new move and the one that
  changes the query model from "one score" to "one score per facet".
- **L2/L3/L4 are one table.** A frame is just another facet with multiple rows per video; if frames
  multiply per video, give them their own table to keep PK/upsert simple.
- **Backfill order mirrors index build order:** load rows → `ANALYZE` → `CREATE INDEX CONCURRENTLY` →
  verify plan → enable in router. Build one facet at a time; the 6 GiB container cannot host
  concurrent full-precision 768d builds (sibling note §5).
- **Turn-off story:** disabling a layer = removing its arm from the query; no data migration. This is
  the "leave options open" property the project asked for.

---

## Evaluation of multi-facet retrieval

**Judgments (build this before tuning anything):** pool candidates from all layers, grade
*per facet*, store `(query_id, video_id, facet, grade, judge, notes)`. Use TREC-style pooling so
unjudged ≠ irrelevant (BEIR's methodology, [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)).
Grade with creator labels and VLM/LLM assistance; keep graded labels (0–3) per facet — the
product-search precedent is graded ESCI classes rather than binary
([arXiv:2206.06588](https://arxiv.org/abs/2206.06588)).

**Metrics:**

1. **Per-facet nDCG@10** — standard, and what MTEB retrieval reports
   ([MTEB, arXiv:2210.07316](https://arxiv.org/abs/2210.07316); nDCG definition:
   Järvelin & Kekäläinen,
   [Cumulated gain-based evaluation of IR techniques, TOIS 2002](https://dl.acm.org/doi/10.1145/582415.582418)).
   Report per facet, not just averaged: a great lighting arm must not hide a broken genre arm.
2. **Facet-conditional Recall@k** — restrict the corpus to videos that have a value for that facet
   (coverage-aware), then measure recall of judged-relevant items; also report end-to-end
   Recall@100 to separate first-stage recall from rerank quality. This is the facet analogue of
   conditional evaluation and exposes failure when a facet is missing (HNSW skips NULLs — README
   FAQ).
3. **Arm attribution / counterfactual deltas** — run the system with each facet arm removed; the
   nDCG delta attributes value to the layer (cheap, no extra labels; the same logic as ablations in
   the fusion literature). Track "facet coverage" (% of results with captions per facet) as a
   first-class metric.
4. **Compositional query suites** — queries with ≥2 facet intents; measure per-facet nDCG *and*
   joint nDCG, changing one facet at a time while holding others fixed. Literature anchors:
   Fashion-IQ compositions (~70% of queries are compositional attribute phrases,
   [arXiv:1905.12794](https://arxiv.org/abs/1905.12794)) and TART's X²-Retrieval pooled-task setup
   ([arXiv:2211.09260](https://arxiv.org/abs/2211.09260)).
5. **Instruction-following / prompt robustness** — if using facet-conditioned prompts: measure
   instruction following with FollowIR's pairwise p-MRR idea
   ([arXiv:2403.15246](https://arxiv.org/abs/2403.15246)) and always report sensitivity across
   several paraphrases (mean ± spread), because single-prompt scores misrank models
   ([arXiv:2605.22544](https://arxiv.org/abs/2605.22544)). InstructIR is a second reference
   ([arXiv:2402.14334](https://arxiv.org/abs/2402.14334)).
6. **Routing metrics** — facet-set Recall (did the router include the facet that contains the
   winning results?) and per-facet exposure; the federated-search literature frames this as
   resource selection with its own evaluation ([Shokouhi & Si 2011](https://dl.acm.org/doi/10.1561/1500000010);
   [survey 2023](https://link.springer.com/article/10.1007/s10115-023-01922-6)), and modern routing
   work measures end-to-end utility ([LTRR](https://arxiv.org/abs/2506.13743),
   [MoR](https://aclanthology.org/2025.emnlp-main.601.pdf)).

**Protocol cautions:** tune fusion weights/routes on a train split and report on held-out queries —
measuring on tuned queries inflates results (Qdrant warns explicitly,
[docs](https://qdrant.tech/documentation/search/hybrid-queries/)); keep the BM25→`sparsevec` funnel
as the incumbent baseline that every new layer must beat (sibling note).

---

## Open questions & uncertainty

1. **Measured latency of F partial-index scans on this box is unmeasured.** Each facet arm ≈ one HNSW
   scan (0.7–2.3 GiB working set each); Postgres executes subqueries sequentially, so
   multi-facet queries scale ~O(F) unless the app parallelizes. **[unmeasured]**
2. **Caption quality per facet is the real upstream risk.** All layout math assumes accurate
   facet captions; if the captioner confuses lighting/mood, no index layout fixes it. No primary
   source addresses Pexels-style stock footage captions specifically. **[open]**
3. **Optimal F and dims** (how many facets, 256 vs 384 vs 768 per facet) has no published answer;
   the Google MTEB table is the closest proxy but for one model family
   ([docs](https://ai.google.dev/gemini-api/docs/embeddings)). Needs our own ablation on ~5k
   judgments. **[open]**
4. **halfvec quality on our embeddings** is unmeasured; the bfloat16 Discourse datapoint
   ([#521](https://github.com/pgvector/pgvector/issues/521)) and Qdrant's Float16 claim are
   adjacent, not ours. Include a float-vs-halfvec A/B in the L2 gate. **[open]**
5. **Partial-index planner behaviour at F=4–8 facets** is a documented hazard
   ([#878](https://github.com/pgvector/pgvector/issues/878); [PostgreSQL 11.8](https://www.postgresql.org/docs/16/indexes-partial.html));
   whether our query shapes always pick the facet index must be verified with `EXPLAIN` per facet,
   and may require literal-inlined facet predicates. **[verify at build]**
6. **Per-index `ef_search` defaults** (RFC [#235](https://github.com/pgvector/pgvector/issues/235),
   PR [#933](https://github.com/pgvector/pgvector/pull/933)) may land in a future pgvector; re-check
   before hard-coding session GUC workarounds. **[time-sensitive]**
7. **Facet routing accuracy for our query mix** is unknown (no query log yet); the literature
   supports routing in general but not our mix, and MoR shows naive routing leaves gains behind.
   Start with "search all facets", add routing only with measured facet-level recall. **[open]**
8. **Whether a shared facet-instruction space beats per-facet models** is empirically open for our
   corpus; both are supported by the same schema (`model` column), so run it as an A/B, not a
   migration. **[open]**
9. **Multi-frame sampling policy** (which k frames, whether motion queries can ever be served by
   stills) remains the largest quality unknown in the L3/L4 layers (sibling note flags motion as
   unserved by a single thumbnail). **[open]**

### Primary source index

- pgvector: [README](https://github.com/pgvector/pgvector/blob/master/README.md) ·
  [CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md) ·
  issues [#640](https://github.com/pgvector/pgvector/issues/640) ·
  [#878](https://github.com/pgvector/pgvector/issues/878) ·
  [#370](https://github.com/pgvector/pgvector/issues/370) ·
  [#769](https://github.com/pgvector/pgvector/issues/769) ·
  [#521](https://github.com/pgvector/pgvector/issues/521) ·
  [#235](https://github.com/pgvector/pgvector/issues/235) ·
  [PR #933](https://github.com/pgvector/pgvector/pull/933)
- pgvector ColBERT patterns:
  [exact.py](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/exact.py) ·
  [approximate.py](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/approximate.py)
- PostgreSQL 16 docs: [TOAST](https://www.postgresql.org/docs/16/storage-toast.html) ·
  [Partial Indexes](https://www.postgresql.org/docs/16/indexes-partial.html) ·
  [CREATE INDEX](https://www.postgresql.org/docs/16/sql-createindex.html) ·
  [Partitioning](https://www.postgresql.org/docs/16/ddl-partitioning.html)
- VectorChord ColBERT rerank: https://docs.vectorchord.ai/vectorchord/use-case/colbert-rerank.html
- Qdrant: [Vectors/multivectors](https://qdrant.tech/documentation/concepts/vectors/) ·
  [Hybrid fusion](https://qdrant.tech/documentation/search/hybrid-queries/)
- Weaviate: [multi-vector tutorial](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings) ·
  [vector config](https://docs.weaviate.io/weaviate/manage-collections/vector-config) ·
  [MUVERA compression](https://docs.weaviate.io/weaviate/configuration/compression/multi-vectors) ·
  [multi-target join strategies](https://docs.weaviate.io/weaviate/search/multi-vector)
- Late interaction: [ColBERT](https://arxiv.org/abs/2004.12832) ·
  [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) ·
  [PLAID](https://arxiv.org/abs/2205.09707) · [XTR](https://arxiv.org/abs/2304.01982) ·
  [MUVERA](https://arxiv.org/abs/2405.19504)
- Representations: [MRL](https://arxiv.org/abs/2205.13147) ·
  [INSTRUCTOR](https://arxiv.org/abs/2212.09741) · [TART](https://arxiv.org/abs/2211.09260) ·
  [E5](https://arxiv.org/abs/2212.03533) · [FollowIR](https://arxiv.org/abs/2403.15246) ·
  [InstructIR](https://arxiv.org/abs/2402.14334) · [One prompt is not enough](https://arxiv.org/abs/2605.22544)
- Task-conditioned production APIs:
  [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings) ·
  [Cohere embeddings](https://docs.cohere.com/docs/embeddings)
- Fusion/routing: [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) ·
  [Bruch et al.](https://arxiv.org/abs/2210.11934) ·
  [Elastic hybrid](https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-hybrid) ·
  [score normalization (CIKM 2001)](https://dl.acm.org/doi/10.1145/502585.502645) ·
  [Federated Search](https://dl.acm.org/doi/10.1561/1500000010) ·
  [federated survey 2023](https://link.springer.com/article/10.1007/s10115-023-01922-6) ·
  [ReSLLM](https://arxiv.org/abs/2401.17645) · [LTRR](https://arxiv.org/abs/2506.13743) ·
  [MoR](https://aclanthology.org/2025.emnlp-main.601.pdf)
- Evaluation: [nDCG (TOIS 2002)](https://dl.acm.org/doi/10.1145/582415.582418) ·
  [BEIR](https://arxiv.org/abs/2104.08663) · [MTEB](https://arxiv.org/abs/2210.07316) ·
  [MMTEB](https://arxiv.org/abs/2502.13595) · [Fashion-IQ](https://arxiv.org/abs/1905.12794) ·
  [ESCI Shopping Queries](https://arxiv.org/abs/2206.06588) ·
  [esci-data](https://github.com/amazon-science/esci-data)
