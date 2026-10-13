# pgvector index engineering for the stock-video corpus (2026-10-08)

Scope: how to index **dense (`embedding vector(768)`), sparse (`sparse_embedding sparsevec`) and
lexical (`tsv`/`tags`)** arms on a single PostgreSQL 16 + pgvector 0.8.7 instance, for a
~365k-now / ~665k-target Pexels video corpus, on a **no-GPU machine with ~30 GB RAM (~7 GB free)**,
interactive sub-second latency, and a "fast, effective, cheap" bias.

All non-obvious facts/numerics below carry a primary-source URL (pgvector README/source/issues,
PostgreSQL docs, or the relevant engine's own docs). Live facts about *this* instance were read
directly from `dev-middleware-stock-db` and are labelled as such.

## TL;DR recommendation

**Stay on PostgreSQL + pgvector. Build three indexes, dense first.** For ~665k × 768-dim vectors the
whole thing is small enough that a plain **HNSW cosine index at full `vector` precision (~2.0–2.5 GiB)
is affordable and gives the best speed/recall at this scale**; drop to `halfvec` (~1.4 GiB) only if
the 6 GiB container cap bites. Do **not** build IVFFlat (worse speed/recall tradeoff, needs training
data, and cannot index `sparsevec`). Do **not** reach for pgvectorscale/Qdrant/Milvus etc. yet —
their wins appear in the tens-of-millions-of-vectors / multi-node regime, not at 665k.

Concretely:

1. **Dense:** `hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)` — the defaults; at
   this scale raising `m`/`ef_construction` mostly buys build cost, not query quality.
2. **Sparse:** `hnsw (sparse_embedding sparsevec_ip_ops)` for the BM25 arm (dot-product / inner
   product). Keep the BM25 vectors **≤1000 non-zero terms**, the hard cap for indexing `sparsevec`.
3. **Lexical:** `GIN (tsv)` as the classic full-text arm and keep the existing `GIN (tags)`.
4. **Filters:** keep the B-tree indexes on `orientation`/`duration` that already exist; enable
   `hnsw.iterative_scan = relaxed_order` so a selective `WHERE` still returns `LIMIT k` rows; if one
   filter value dominates, add a **partial** HNSW index for it.
5. **Build procedure:** `CREATE INDEX CONCURRENTLY`, `maintenance_work_mem ≈ 3 GB`,
   `max_parallel_maintenance_workers = 0` **or** raise the container `--shm-size` to match
   `maintenance_work_mem` (this container is at a 64 MiB shm, which blocks parallel HNSW builds).
6. **Revisit** pgvectorscale only when the corpus heads toward tens of millions of vectors **and**
   the full-precision index stops fitting in RAM.

Why: pgvector's own guidance says "for a smaller working set use `halfvec`… [and] binary
quantization for indexes with re-ranking" only when scaling; at 665k the un-quantized index fits
([pgvector README → Scaling](https://github.com/pgvector/pgvector/blob/master/README.md#scaling)),
and independent engine docs corroborate that HNSW memory at 1M×768d is on the order of a few GB, i.e.
exactly our situation ([Weaviate → Resource requirements](https://weaviate.io/developers/weaviate/concepts/vector-index)).

## Findings

### 1. pgvector 0.8.x: HNSW vs IVFFlat, parameters, build behaviour

- **Two ANN index types:** HNSW and IVFFlat. pgvector's own wording: HNSW "has better query
  performance than IVFFlat (in terms of speed-recall tradeoff), but has slower build times and uses
  more memory"; IVFFlat "has faster build times and uses less memory than HNSW, but has lower query
  performance". HNSW needs no training data; IVFFlat needs rows present before the index is built.
  ([README → Indexing / HNSW / IVFFlat](https://github.com/pgvector/pgvector/blob/master/README.md#indexing))
- **HNSW parameters** (source defaults, `m=16`, `ef_construction=64`):
  - `m` — max connections per layer (default **16**), max 100.
  - `ef_construction` — size of dynamic candidate list during construction (default **64**), range 4–1000.
  - `ef_search` — dynamic candidate list at query time (default **40**), range 1–1000.
  ([README → Index Options / Query Options](https://github.com/pgvector/pgvector/blob/master/README.md#index-options);
  constants `HNSW_DEFAULT_M 16`, `HNSW_DEFAULT_EF_CONSTRUCTION 64`, `HNSW_DEFAULT_EF_SEARCH 40`,
  `HNSW_MAX_M 100`, `HNSW_MIN/MAX_EF_CONSTRUCTION` in
  [src/hnsw.h](https://github.com/pgvector/pgvector/blob/master/src/hnsw.h))
- **IVFFlat parameters:** `lists` (start at `rows/1000` up to 1M rows, `sqrt(rows)` above 1M),
  `probes` (start at `sqrt(lists)`, default 1). ([README → IVFFlat](https://github.com/pgvector/pgvector/blob/master/README.md#ivfflat))
- **Type/op-class support:** HNSW supports `vector` (≤2000 dims), `halfvec` (≤4000), `bit` (≤64000),
  `sparsevec` (≤1000 non-zero). **IVFFlat does NOT list `sparsevec`** — only `vector`, `halfvec`, `bit`.
  This is decisive for us: the sparse arm can only be HNSW. ([README → HNSW / IVFFlat supported types](https://github.com/pgvector/pgvector/blob/master/README.md#indexing))
- **Build memory:** "Indexes build significantly faster when the graph fits into `maintenance_work_mem`."
  When it stops fitting pgvector raises `NOTICE: hnsw graph no longer fits into maintenance_work_mem
  after N tuples`. ([README → Index Build Time](https://github.com/pgvector/pgvector/blob/master/README.md#index-build-time))
- **Parallel build:** HNSW and IVFFlat support parallel builds; raise
  `max_parallel_maintenance_workers` (default 2) plus `max_parallel_workers`. ([README → Index Build Time](https://github.com/pgvector/pgvector/blob/master/README.md#index-build-time))
  Note the core PostgreSQL doc text says parallel build is "currently, only B-tree", but pgvector
  implements its own parallel build path and documents `max_parallel_maintenance_workers` for HNSW/IVFFlat;
  trust the python-relevant pgvector README. ([PostgreSQL CREATE INDEX](https://www.postgresql.org/docs/16/sql-createindex.html))
- **`CREATE INDEX CONCURRENTLY`:** recommended in production to avoid blocking writes; it does **two
  table scans**, cannot run inside a transaction block, can leave an `INVALID` index on failure, and
  for extension/unusual op-classes it must wait for snapshots predating the second scan to drain.
  ([README → Indexing "create indexes concurrently"](https://github.com/pgvector/pgvector/blob/master/README.md#indexing);
  [PostgreSQL CREATE INDEX → Building Indexes Concurrently](https://www.postgresql.org/docs/16/sql-createindex.html))
- **Docker `--shm-size` caveat (matters here):** "If you increase `maintenance_work_mem`, make sure
  `--shm-size` is at least that size to avoid an error with parallel HNSW index builds." ([README → Docker](https://github.com/pgvector/pgvector/blob/master/README.md#docker))

**Live instance facts (read from `dev-middleware-stock-db`, 2026-10-08):** PostgreSQL **16.15**,
`vector` **0.8.7**, table `stock_videos` = **365,103 rows** (0 dense, 0 sparse, 0 tsv so far),
heap **909 MB** / indexes **173 MB** / total **3.27 GB**. GUCs: `shared_buffers=128 MB`,
`maintenance_work_mem=64 MB`, `work_mem=4 MB`, `max_parallel_maintenance_workers=2`,
`max_parallel_workers=8`, `max_worker_processes=8`. Container: **memory limit 6 GiB, shm 64 MiB**,
host 32 CPUs / ~9.5 GiB available. (These are the numbers that drive the sizing section.)

### 2. Memory & QPS math for 665k × 768-dim

pgvector storage formulas ([README → Reference](https://github.com/pgvector/pgvector/blob/master/README.md#reference)):

| type | bytes per vector | @768 dims |
|---|---|---|
| `vector` (float32) | `4·d + 8` | **3080 B** |
| `halfvec` (float16) | `2·d + 8` | **1544 B** |
| `bit` (binary) | `d/8 + 8` | **104 B** |
| `sparsevec` | `8·nnz + 16` | e.g. 100 nnz = 816 B |

Raw dense-column payload (just the vectors, no index):

- 365,103 × 3080 = **1,124,517,240 B ≈ 1.05 GiB**
- 664,546 × 3080 = **2,046,801,680 B ≈ 1.91 GiB**
- halfvec @664,546 = **1,026,059,024 B ≈ 0.96 GiB**
- bit (BQ) @664,546 = **69,112,784 B ≈ 66 MiB**

**HNSW index size / build memory.** pgvector does not publish a one-line formula in the README, but
the maintainer and a community issue pin it down:

- Each HNSW element's on-disk tuple holds its own vector plus the graph links; for high-dimensional
  vectors **each element is roughly one 8 KiB page regardless of `m`**, so `m=2` and `m=16` produced
  ~the same index size (7,036 MB vs 7,043 MB for 900k × 1280-dim).
  ([pgvector issue #370, maintainer reply](https://github.com/pgvector/pgvector/issues/370))
- A precise, source-derived per-element memory formula was posted (and the maintainer acknowledged
  the general dependency while declining to merge it): memory depends only on `m`, number of vectors,
  and vector byte size; empirically it matched a real 9.9M-vector build to **<0.01%** error. The
  per-element model (from the issue) is:

  ```
  MAXALIGN(sizeof(HnswElementData))
    + (avgLevel + 1) * sizeof(HnswNeighborArrayPtr)
    + MAXALIGN(HNSW_NEIGHBOR_ARRAY_SIZE(HnswGetLayerM(m, 0)))   -- ground layer: 2m candidates
    + avgLevel * MAXALIGN(HNSW_NEIGHBOR_ARRAY_SIZE(m))          -- upper layers: m candidates
  ```

  with `avgLevel = 1/(1-exp(-1/ml)) - 1`, `ml = 1/ln(m)`. For `m=16`: `avgLevel ≈ 0.067`.
  ([pgvector issue #769](https://github.com/pgvector/pgvector/issues/769);
  structure/link sizes in [src/hnsw.h](https://github.com/pgvector/pgvector/blob/master/src/hnsw.h):
  `HnswGetLayerM(m,0)=2m`, `HnswCandidate` = ptr + float + bool ≈ 16 B, ground neighbor array = `2m·16`
  ≈ 512 B + header.)
- The maintainer's practical guidance: **"set `maintenance_work_mem` to the max amount of memory
  you'd like the index build to consume rather than trying to be exact."**
  ([issue #769](https://github.com/pgvector/pgvector/issues/769))
- Corroboration from a different engine's docs: HNSW node cost ≈ `4 B × dims` and at 1M vectors is
  "2–12 GB" for 2–12 kB vectors — i.e. **768d ≈ 3 kB/node ≈ ~3 GB per million vectors**, edges add ~200 B/node.
  ([Weaviate → Resource requirements](https://weaviate.io/developers/weaviate/concepts/vector-index))

Applying that model (`~3080 B` vector + **~670 B** graph/struct overhead/element, the overhead
calibrated from issue #769's 96-dim measurement of 1,066 B/tuple − 392 B data = 674 B):

| corpus | full-precision HNSW build RAM (est.) | halfvec HNSW (est.) |
|---|---|---|
| 365,103 | ≈ **1.37 GB (1.28 GiB)** | ≈ 0.81 GB |
| 664,546 | ≈ **2.50 GB (2.32 GiB)** | ≈ 1.47 GB |

So: **a full-precision dense HNSW graph at the 665k target needs ~2.3 GiB to fit in
`maintenance_work_mem`, plus its on-disk index is the same order (~2–2.5 GiB).** On-disk *index*
size can be checked after building with `pg_relation_size` and should be monitored; pgvector says
indexes do **not** have to fit in memory, but "you'll likely see better performance if they do".
([README → FAQ "Do indexes need to fit into memory?"](https://github.com/pgvector/pgvector/blob/master/README.md#do-indexes-need-to-fit-into-memory))

**Quantization + rerank.** pgvector's own recipe for keeping the working set small is **binary
quantization + re-rank**: index `binary_quantize(embedding)::bit(768)` with `bit_hamming_ops`, retrieve
~20 candidates by Hamming distance, then re-order by the full-precision vector. This is "for smaller
indexes and faster build times at scale" and "with re-ranking for search". ([README → Binary
Quantization](https://github.com/pgvector/pgvector/blob/master/README.md#binary-quantization),
[README → Approximate Search](https://github.com/pgvector/pgvector/blob/master/README.md#approximate-search))
Tradeoff: BQ keeps the full vectors in the heap for the rerank pass, so it shrinks the *index* not the
*table*. At 665k the plain index already fits, so BQ is a fallback, not the default.

**Recall/latency levers:** `hnsw.ef_search` trades recall for speed (default 40); monitor recall by
comparing against exact search with `SET LOCAL enable_indexscan = off`. ([README → Query Options](https://github.com/pgvector/pgvector/blob/master/README.md#query-options),
[README → Monitoring](https://github.com/pgvector/pgvector/blob/master/README.md#monitoring))

**On "QPS math":** there is no primary, hardware-independent QPS number to quote for 665k; published
throughput figures are for much larger corpora and different hardware (see §6). The honest statement
is that sub-second latency at 665k is not the risk — **RAM residency and build memory are**. Do an
`EXPLAIN (ANALYZE, BUFFERS)` and recall-vs-`ef_search` sweep on our own box rather than trusting a blog.

### 3. Filtered vector search (orientation / duration / tags)

- **Default behaviour:** "With approximate indexes, filtering is applied *after* the index is
  scanned." With a filter matching 10% of rows and `ef_search=40`, only ~4 rows match on average, so
  you silently get fewer than `LIMIT k`. This is the single biggest correctness pitfall for our
  `orientation`/`duration`/`resolution` filters. ([README → Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering))
- **Fix — iterative index scans (0.8.0+):** automatically scan more of the index until enough rows
  are found (or `hnsw.max_scan_tuples` / `ivfflat.max_probes` is hit).
  `hnsw.iterative_scan = strict_order | relaxed_order`; relaxed gives better recall but results may be
  slightly out of distance order (use a materialized CTE to re-sort).
  `hnsw.max_scan_tuples` default **20,000**; `hnsw.scan_mem_multiplier` default 1; `ivfflat.max_probes`
  default 100. ([README → Iterative Index Scans / Iterative Scan Options](https://github.com/pgvector/pgvector/blob/master/README.md#iterative-index-scans))
- **Pre-filter vs post-filter:** for *selective* conditions (matching a small % of rows) an **exact**
  B-tree/GIN index on the filter column can give fast, exact NN by pre-filtering the candidate set;
  for *broad* conditions an approximate HNSW index with post-filtering + iterative scan works better.
  pgvector explicitly frames it this way: "Exact indexes work well for conditions that match a low
  percentage of rows. Otherwise, approximate indexes can work better." ([README → Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering))
- **Partial indexes:** "If filtering by only a few distinct values, consider partial indexing" e.g.
  `... WHERE (category_id = 123)`. ([README → Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering))
- **Partitioning:** if filtering by many distinct values, consider partitioning the table.
  ([README → Filtering](https://github.com/pgvector/pgvector/blob/master/README.md#filtering))
- We already hold `idx_stock_videos_orientation` and `idx_stock_videos_duration` B-trees and
  `idx_stock_videos_tags` GIN (from migration 001), so the exact-pre-filter path is available today.

### 4. Sparse (`sparsevec`) + lexical (`tsv`, `tags`) indexes

- **`sparsevec` storage:** `8·nnz + 16` bytes; up to 16,000 non-zero elements, but **an HNSW index
  supports only ≤1000 non-zero elements per vector**. Filters/index via `sparsevec_ip_ops` (inner
  product), `sparsevec_l2_ops`, `sparsevec_cosine_ops`. IVFFlat has **no** `sparsevec` support.
  ([README → Reference/Sparsevec, HNSW supported types](https://github.com/pgvector/pgvector/blob/master/README.md#sparse-vectors))
  → For BM25 this means **truncate each document vector to its top-1000 terms** (or use a smaller
  active vocabulary); otherwise the sparse HNSW index build will reject the vector.
- **Lexical `tsv`:** PostgreSQL's preferred full-text index is **GIN** on a `tsvector` column
  ("GIN indexes are the preferred text search index type"); GiST is the lossy alternative.
  ([PostgreSQL → Preferred Index Types for Text Search](https://www.postgresql.org/docs/16/textsearch-indexes.html))
- **`tags text[]`:** GIN is the right index for array containment/overlap (`&&`, `@>`). We already
  have `idx_stock_videos_tags` GIN. (Same PostgreSQL index-types reference; the repo's migration 001
  is the concrete source.)
- **Composition with the dense arm:** pgvector points to hybrid search by combining vector ranking
  with full-text search and fusing via **Reciprocal Rank Fusion** or a cross-encoder. In 0.8.x you can
  run each arm as a subquery and fuse in SQL/application code. ([README → Hybrid Search](https://github.com/pgvector/pgvector/blob/master/README.md#hybrid-search))
- We can keep BM25 in `sparsevec` **or** in `tsv`; the two are complementary: `tsv` gives the standard
  Postgres `ts_rank_cd` lexical arm, `sparsevec_ip` gives BM25 dot-product scoring comparable to the
  dense arm's score space.

### 5. Stay vs leave: pgvectorscale and external engines

**pgvectorscale (Timescale/Tiger Data):** adds a Rust **StreamingDiskANN** index (disk-resident, built
on Microsoft DiskANN), **Statistical Binary Quantization (SBQ)**, and **label-based filtered vector
search** (Filtered DiskANN). Index build params include `num_neighbors` (default 50),
`search_list_size` (100), `max_alpha` (1.2), `num_bits_per_dimension` (2 for <900 dims, else 1);
query params `diskann.query_search_list_size` (100) and `diskann.query_rescore` (50). It supports
**parallel build** only when the table has no labels, uses SBQ storage, and exceeds
`min_vectors_for_parallel_build` (default 65,536). Label filtering requires declaring a
`smallint[]` label column in the index and querying with `&&`. Arbitrary `WHERE` filters are
**post-filtered**. Its headline result is on **50M × 768d Cohere embeddings**: 28× lower p95 latency
and 16× higher QPS vs Pinecone s1 at 99% recall. ([pgvectorscale README](https://github.com/timescale/pgvectorscale/blob/main/README.md);
[Timescale "pgvector vs Pinecone"](https://www.timescale.com/blog/pgvector-vs-pinecone)) — note that
benchmark is an order of magnitude larger than our corpus and on a 128 GB EC2 box, so it does **not**
transfer to 665k on a 6 GiB container.

**External engines (when leaving becomes justified):**

- **Qdrant** — rich quantization: scalar (4×), binary (up to 32×), product (up to 64×), TurboQuant
  (up to 32×); binary/PQ best for high-dim/centered vectors; recommends rescoring. ([Qdrant → Quantization](https://qdrant.tech/documentation/manage-data/quantization/))
- **Milvus** — distributed, K8s-native; supports HNSW, IVF, FLAT, SCANN, DiskANN, quantization and
  `mmap`, plus native BM25/sparse and hybrid rerank. Aimed at "billions of vectors". ([Milvus README](https://github.com/milvus-io/milvus/blob/master/README.md))
- **Weaviate** — HNSW (default) + flat/dynamic + disk-backed **HFresh** (SPFresh-based, compressed
  centroid + 1-bit posting lists); documents HNSW memory growth and quantization. ([Weaviate → Vector Indexing](https://weaviate.io/developers/weaviate/concepts/vector-index))
- **Vespa** — tensor-based `nearestNeighbor` with exact or HNSW approximate search, phased
  re-ranking, six distance metrics, binary/hamming support; a full IR+serving stack, heavier to
  operate. ([Vespa → Nearest Neighbor Search](https://docs.vespa.ai/en/querying/nearest-neighbor-search.html))
- **LanceDB / Lance** — an *embedded* lakehouse format with IVF_PQ ANN, BM25 full-text and SQL on
  object storage; strong for training/data-engineering pipelines, not a drop-in OLTP search DB.
  ([Lance README](https://github.com/lance-format/lance/blob/main/README.md), [LanceDB docs](https://docs.lancedb.com/))
- **Turbopuffer** — object-storage-first, SPFresh centroid index, BM25, exact metadata filter
  indexes; warm p50 ≈ 14 ms / 1M docs, cold p50 ≈ 874 ms; ~10k+ vectors/s writes. ([turbopuffer → Architecture](https://turbopuffer.com/docs/architecture))

**Judgment:** none of these beats "already running Postgres 16 + pgvector, single node, 665k rows"
on cost or operational simplicity. The triggers to *leave* are (a) corpus into the tens of millions
where the in-memory HNSW stops fitting and disk-based DiskANN/SBQ wins, or (b) multi-tenant/multi-node
scale-out, or (c) a hard functional gap (e.g. GPU indexing). pgvectorscale is the *least disruptive*
upgrade path because it is the same Postgres and the same query syntax.

### 6. Concrete numbers caveat / uncertainty

Published recall/latency/QPS figures (pgvectorscale's 50M benchmark, Qdrant's model-specific
recall@k, turbopuffer's p50s) are all on different hardware, corpora and recall targets than ours.
They justify *relative* claims, not our latency budget. Marked uncertain: expected sub-second p95 at
665k (very likely fine, unmeasured), and the exact on-disk HNSW index size (estimate only).

## Sizing & tuning math

Assumptions: 768 dims; pgvector `vector` = 3080 B, `halfvec` = 1544 B, `bit` = 104 B per vector
([README Reference](https://github.com/pgvector/pgvector/blob/master/README.md#reference)).

1. **Dense column bytes**
   - `365,103 × 3080 = 1,124,517,240 B ≈ 1.05 GiB`
   - `664,546 × 3080 = 2,046,801,680 B ≈ 1.91 GiB`
2. **HNSW graph memory (build)** ≈ `vectors × (vector_bytes + ~670 B overhead)`
   - 664,546 × (3080 + 670) ≈ 664,546 × 3750 ≈ **2.49e9 B ≈ 2.32 GiB**
   - 365,103 × 3750 ≈ **1.37e9 B ≈ 1.28 GiB**
   - Overhead calibrated from [issue #769](https://github.com/pgvector/pgvector/issues/769)
     (1066 B/tuple at 96d, data = 4·96+8 = 392 B ⇒ ~674 B overhead) and the tuple layout in
     [src/hnsw.h](https://github.com/pgvector/pgvector/blob/master/src/hnsw.h).
3. **halfvec variant:** 664,546 × (1544 + 670) ≈ **1.47 GB ≈ 1.37 GiB**.
4. **binary-quantized index:** 664,546 × (104 + overhead) ≈ **~0.5 GB**, plus full vectors stay in
   the heap for rerank. ([README Binary Quantization](https://github.com/pgvector/pgvector/blob/master/README.md#binary-quantization))
5. **RAM budget check (live):** container cap **6 GiB**; plan ~2.5 GiB `maintenance_work_mem` +
   `shared_buffers` 128 MiB + bearer’s own overhead ⇒ **fits, barely**, which is why halfvec is the
   escape hatch. Raise `shared_buffers` toward 25% of available RAM per pgvector's tuning advice
   ([README → Tuning](https://github.com/pgvector/pgvector/blob/master/README.md#tuning)).
6. **Sparse arm:** BM25 with ~50–150 nnz/vector ⇒ `8·nnz+16` ≈ 0.4–1.2 kB/vector ⇒ ~0.3–0.8 GiB at 665k.
   Keep nnz ≤ 1000 for indexability ([README Sparsevec](https://github.com/pgvector/pgvector/blob/master/README.md#sparse-vectors)).

## Recommended DDL + build procedure

```sql
-- 0. Verify extension (already true here: 0.8.7)
SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- 0.8.7

-- 1. Dense arm: full-precision HNSW, cosine. Defaults are the right call at 665k.
CREATE INDEX CONCURRENTLY idx_stock_videos_embedding_hnsw
    ON stock_videos USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 1b. Optional: halfvec index if the 6 GiB cap is the binding constraint
-- CREATE INDEX CONCURRENTLY idx_stock_videos_embedding_hnsw_half
--     ON stock_videos USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);

-- 2. Sparse arm: BM25 via sparsevec, inner product (must match ORDER BY's operator)
CREATE INDEX CONCURRENTLY idx_stock_videos_sparse_hnsw
    ON stock_videos USING hnsw (sparse_embedding sparsevec_ip_ops);

-- 3. Lexical arm(s)
CREATE INDEX CONCURRENTLY idx_stock_videos_tsv_gin
    ON stock_videos USING GIN (tsv);
-- (idx_stock_videos_tags GIN already exists from migration 001)

-- 4. Optional partial dense index if one orientation dominates the query mix
-- CREATE INDEX CONCURRENTLY idx_stock_videos_landscape_hnsw
--     ON stock_videos USING hnsw (embedding vector_cosine_ops)
--     WHERE orientation = 'landscape';

ANALYZE stock_videos;
```

Build-session GUCs (this container has **shm = 64 MiB**, so parallel HNSW will error unless shm is
raised — see pgvector Docker note):

```sql
BEGIN;
SET LOCAL maintenance_work_mem = '2560MB';        -- >= estimated graph size (see §2)
SET LOCAL max_parallel_maintenance_workers = 0;    -- OR raise --shm-size >= maintenance_work_mem
-- SET LOCAL max_parallel_maintenance_workers = 4;  -- only after increasing shm-size
-- CREATE INDEX ...;  -- (must run OUTSIDE a txn for CONCURRENTLY)
COMMIT;
```

> Note: `CREATE INDEX CONCURRENTLY` cannot run inside a transaction block
> ([PostgreSQL CREATE INDEX](https://www.postgresql.org/docs/16/sql-createindex.html)), so issue the
> `SET`s as session-level (`SET`, not `SET LOCAL`) around the standalone statement.

Query-session GUCs:

```sql
SET hnsw.ef_search = 100;                  -- recall/latency dial (default 40)
SET hnsw.iterative_scan = relaxed_order;   -- so filtered queries still fill LIMIT k
SET hnsw.max_scan_tuples = 20000;          -- default; raise if selective filters starve results
```

Hybrid query shape (fuse the two arms with RRF in SQL/app):

```sql
-- dense arm (cosine) + lexical arm, filtered
WITH dense AS (
    SELECT pexels_id, embedding <=> $1 AS dist
    FROM stock_videos
    WHERE orientation = $2 AND duration BETWEEN $3 AND $4
    ORDER BY embedding <=> $1
    LIMIT 50
),
lex AS (
    SELECT pexels_id, ts_rank_cd(tsv, query) AS rank
    FROM stock_videos, plainto_tsquery($5) query
    WHERE tsv @@ query
    ORDER BY rank DESC
    LIMIT 50
)
SELECT pexels_id, 1.0/(60 + rank) AS score
FROM (SELECT pexels_id, row_number() OVER (ORDER BY dist) AS rank FROM dense
      UNION ALL
      SELECT pexels_id, 1000 + row_number() OVER (ORDER BY rank DESC) AS rank FROM lex) u;
```

**Build procedure on the live DB (crawl still writing):**

1. Wait until dense coverage is high enough that a rebuild is unlikely (embedding backfill is
   API-metered; the repo's own plan is "fill coverage first, then index" — `docs/phase2-embedding-seam.md`).
2. `CREATE INDEX CONCURRENTLY` so crawl inserts/updates don't block and aren't blocked. Expect two scans.
3. Watch progress: `SELECT phase, round(100.0*blocks_done/nullif(blocks_total,0),1) FROM pg_stat_progress_create_index;`
   (HNSW phases: `initializing`, `loading tuples`). ([README → Indexing Progress](https://github.com/pgvector/pgvector/blob/master/README.md#indexing-progress))
4. `ANALYZE stock_videos;` after the build.
5. Measure: `EXPLAIN (ANALYZE, BUFFERS)`, `pg_relation_size('idx_stock_videos_embedding_hnsw')`, and a
   recall sweep vs `SET LOCAL enable_indexscan = off`. ([README → Monitoring](https://github.com/pgvector/pgvector/blob/master/README.md#monitoring))
6. Vacuum note: HNSW vacuuming can be slow — `REINDEX INDEX CONCURRENTLY` first if needed.
   ([README → Vacuuming](https://github.com/pgvector/pgvector/blob/master/README.md#vacuuming))

**As the corpus grows past 665k:** 665k is *not* a partitioning scale. HNSW stays viable into the
low millions; a non-partitioned table supports up to 32 TB, so no time-partitioning is needed at
665k. Consider (a) partition or a partial index **only** if a small set of filter values dominates
queries (pgvector's own "many distinct values → partition" guidance), and (b) switch to halfvec /
binary-quantization+rerank when the full-precision index stops fitting RAM (tens of millions).
([README → Filtering / Scaling / FAQ](https://github.com/pgvector/pgvector/blob/master/README.md#filtering))

## Stay-vs-leave comparison

| Option | Index | Disk-resident? | Filtered search | Ops cost here | Verdict at 665k |
|---|---|---|---|---|---|
| **pgvector 0.8.7 (current)** | HNSW / IVFFlat | No (in-mem working set) | post-filter + iterative scan; partial/GIN pre-filter; partition | already running | **Use it. Fits in RAM, zero new services.** |
| **pgvectorscale** | StreamingDiskANN + SBQ | Yes | label-based (`&&`) + post-filter | same Postgres, add extension | Keep as the **first upgrade** if RAM becomes the limit or corpus → tens of millions |
| Qdrant | HNSW + scalar/binary/PQ/TurboQuant | optional | payload filters | new service + data sync | Not needed; quantization only matters at larger scale |
| Milvus | HNSW/IVF/DiskANN/SCANN + mmap | optional | scalar index | K8s/distributed, heavy | Overkill; designed for billions |
| Weaviate | HNSW / flat / HFresh | HFresh yes | inverted index + filters | new service | Not needed at 665k; HFresh interesting only if memory-bound at scale |
| Vespa | HNSW + exact | hybrid | rich filter language | full serving stack | Overkill for an internal search backend |
| LanceDB/Lance | IVF_PQ | yes (object storage) | SQL predicates + FTS | embedded, data-eng shaped | Wrong shape for OLTP interactive search |
| Turbopuffer | SPFresh | yes (S3) | exact metadata indexes | managed service (egress/latency) | Warm-cache 14 ms is great, but adds a remote hop and cost; not needed |

Primary-source basis for each row is linked in §5.

## Open questions & uncertainty

1. **Measured recall/latency at 665k is not in any primary source** — published benchmarks are larger
   corpora / different hardware. Need our own `ef_search` vs recall sweep. (unmeasured)
2. **Exact on-disk HNSW index size** for `vector(768)` — estimated ~2–2.5 GiB from the issue-#769
   model; verify post-build with `pg_relation_size`. (estimate)
3. **Will the 6 GiB container cap hold** during a full-precision parallel build concurrently with
   live crawl writes? Test before the real build; halfvec is the fallback. (time-sensitive: image/GUCs)
4. **BM25 vocabulary size / nnz per vector** — must stay ≤1000 for `sparsevec` HNSW; depends on the
   chosen BM25 tokenizer/vocab (repo: `store/bm25.py`). (open)
5. **IVFFlat rejected partly due to no `sparsevec` support** — confirmed by README's supported-types
   list, but worth a build-time sanity check on 0.8.7 since docs can lag. (verify at build)
6. **`tsv` vs `sparsevec` for the lexical arm**: both are viable; which gives better hybrid recall@k
   is an empirical A/B, not a documented fact. (open)
7. **Filter selectivity distribution** (orientation/duration) is unknown on our data; drives the
   partial-index/partition decision. (open)
8. **Version currency**: pgvector 0.8.7 released 2026-10-01 per CHANGELOG; re-check for 0.9.x before
   the real build. ([CHANGELOG](https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md)) (time-sensitive)

## Sources (primary)

- pgvector README (v0.8.7): https://github.com/pgvector/pgvector/blob/master/README.md
- pgvector CHANGELOG: https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md
- pgvector source `src/hnsw.h`: https://github.com/pgvector/pgvector/blob/master/src/hnsw.h
- pgvector issue #769 (HNSW memory formula): https://github.com/pgvector/pgvector/issues/769
- pgvector issue #370 (index size ≈ one page/element): https://github.com/pgvector/pgvector/issues/370
- PostgreSQL 16 CREATE INDEX (CONCURRENTLY, maintenance_work_mem, parallel): https://www.postgresql.org/docs/16/sql-createindex.html
- PostgreSQL 16 Preferred Index Types for Text Search (GIN for tsvector): https://www.postgresql.org/docs/16/textsearch-indexes.html
- pgvectorscale README (StreamingDiskANN, SBQ, label filtering, params): https://github.com/timescale/pgvectorscale/blob/main/README.md
- Timescale/Tiger Data "pgvector vs Pinecone" benchmark (50M×768d): https://www.timescale.com/blog/pgvector-vs-pinecone
- Qdrant quantization docs: https://qdrant.tech/documentation/manage-data/quantization/
- Weaviate vector indexing docs: https://weaviate.io/developers/weaviate/concepts/vector-index
- Vespa nearest-neighbor search docs: https://docs.vespa.ai/en/querying/nearest-neighbor-search.html
- Milvus README: https://github.com/milvus-io/milvus/blob/master/README.md
- Lance README: https://github.com/lance-format/lance/blob/main/README.md ; LanceDB docs: https://docs.lancedb.com/
- turbopuffer architecture docs: https://turbopuffer.com/docs/architecture

*Live-instance facts (§1) were read from `dev-middleware-stock-db` on 2026-10-08; the `postgres` MCP
tool in this environment pointed at a different database without the `vector` extension, so
`docker exec ... psql` was used as the primary source for the corpus/GUC/container state.*
