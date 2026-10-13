# Dense embedding model selection for stock-video search (text + multimodal), 2026-10

Scope: dense text-retrieval SOTA, concrete model comparison (Dim / modalities / max tokens /
price / China-accessibility), the text→image (thumbnail) question, and a ranked recommendation for
this corpus (~665k Pexels videos; `title` + 1-sentence `description` + `tags text[]` median 47 /
p90 55; no local video, thumbnails not yet downloaded; english corpus, queries may be Chinese;
no GPU; single Postgres 16 + pgvector 0.8.7 with an existing `embedding vector(768)` column).

> **All prices and model names are a 2026-10-08 snapshot and are time-sensitive.** Vendor pricing
> pages change; several entries below are flagged "verify before commit". Every non-obvious number
> carries a link to its primary source (official docs / model card / paper / vendor pricing page /
> leaderboard). Secondary sources are marked as such.

---

## TL;DR recommendation

1. **Start with DashScope `text-embedding-v4` at `dimension=768`.** It *is* the Qwen3-Embedding
   series served by Alibaba ([DashScope text-embedding API][ds-text]), it is natively Matryoshka
   (64–2048 dims, no schema migration needed for the existing `vector(768)`), it is multilingual
   (Chinese queries work), it is cheap (US$0.07 / 1M input tokens → roughly **US$6 for the whole
   665k-row corpus**), and the API key already exists in a sibling project.
   -> if a quality gate says 768 is not enough, re-embed at `dimension=2048` (best text MTEB of the
   family) and migrate the column to `vector(2048)`.
2. **Upgrade path once thumbnails are downloaded: `qwen3-vl-embedding` (DashScope) with
   `enable_fusion=true`.** It puts text + image (+ video) into **one** vector, supports 2560→256
   dims, 32k-token context, and is the strongest open multimodal embedder on MMEB-V2. Text
   US$0.10 / 1M, image/video US$0.258 / 1M.
3. **Do *not* build the whole pipeline around CLIP/SigLIP2 or a thumbnail-only cross-modal index
   yet.** For a corpus this tag-rich (median 47 tags), the visual signal is largely redundant with
   the tags; cross-modal only earns its cost for *visually-specified* queries ("from above",
   "slow motion", "aerial") and for the MPT dedup gate. Treat image embedding as an optional
   second pass, not the primary index.

Full ranking with reasoning and a decision criterion is in
[Ranked recommendation for this corpus](#ranked-recommendation-for-this-corpus).

---

## Findings

### 1. Dense text-retrieval SOTA, 2025–2026

The benchmark that matters is now **MMTEB / "MTEB (Multilingual, v2)"** (the successor to MTEB,
paper [arXiv:2502.13595](https://arxiv.org/abs/2502.13595); leaderboard
[huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard),
canonical host [mteb-leaderboard.hf.space](https://mteb-leaderboard.hf.space/benchmark/MTEB(Multilingual%2C%20v2))).

- **Open-weight leader (mid-2025 → 2026):** `Qwen3-Embedding-8B`, **70.58** Mean(Task) on MTEB
  Multilingual and 75.22 on MTEB English v2 (ranked #1 multilingual open model at release,
  June 2025) — [HF model card][qwen3-emb-8b]. The 4B (69.45 / 74.60) and 0.6B (64.33 / 70.70)
  are the same family; all are **MRL** and **instruction-aware**, Apache-2.0.
- **API leader (2025):** `gemini-embedding-001`, **68.37** MTEB Multilingual — same [HF table][qwen3-emb-8b].
- **Newer v2 entrants (2026, verify):** the leaderboard's current size-group tops include
  `harrier-oss-v1-27b`, `KaLM-Embedding-Gemma3-12B-2511` (reported MMTEB mean 72.32),
  `llama-embed-nemotron-8b`, `Nemotron-3-Embed-8B`, and Jina's **`jina-embeddings-v5-text-small`**
  (MMTEB 67.0, MTEB English 71.7, 677M params, reported released 2026-02-18) / `-nano` (768d)
  — [leaderboard][mteb-lb], [Jina v5-text announcement][jina-v5-text], [vLLM recipe (release
  date)][jina-v5-recipe]. These names move fast; re-check before locking.
- **Chinese (C-MTEB):** `text-embedding-v4` scores **CMTEB 71.99 / CMTEB-retrieval 75.01** at
  2048 dims (DM 70.14 @1024, 68.79 @512) — [DashScope embedding page][ds-embed]. C-MTEB itself:
  [arXiv:2309.07597][cpack], [BGE docs][cmteb].

**Take-away:** for this corpus the useful frontier is "small multilingual model, MRL, cheap" (0.6B–8B
Qwen3 class) rather than "largest MTEB score". We do not need long context (metadata is ~100–150
tokens); we *do* need multilingual (Chinese queries) and Matryoshka (768-dim column).

### 2. Model-by-model facts

| Model | Provider / access | Dim (MRL) | Modalities | Max tokens | Reference |
|---|---|---|---|---|---|
| `text-embedding-v4` (= Qwen3-Embedding served) | Alibaba DashScope, API | 2048/1536/**1024**/768/512/256/128/64 | text | 8,192 | [ds-text][ds-text], [ds-embed][ds-embed] |
| `tongyi-embedding-vision-flash` | Alibaba DashScope, API | **768 (fixed)** | text+image+video (**independent**, no fusion) | 1,024 (text) | [ds-mm][ds-mm], [ds-price][ds-price] |
| `tongyi-embedding-vision-plus` | Alibaba DashScope, API | **1152 (fixed)** | text+image+video (`multi_images`) | 1,024 (text) | [ds-mm][ds-mm] |
| `qwen3-vl-embedding` | Alibaba DashScope, API (China-Beijing only) | **2560**/2048/1536/1024/768/512/256 | text+image+video, **fused or independent** | 32,000 | [ds-mm][ds-mm] |
| `Qwen3-Embedding-0.6B/4B/8B` | Qwen, open weights (Apache-2.0) | 1024 / 2560 / 4096 (MRL to 64) | text | 32,768 | [HF 0.6B][qwen3-emb-06], [HF 8B][qwen3-emb-8b], [paper arXiv:2506.05176][qwen3-paper] |
| `Qwen3-VL-Embedding-2B/8B` | Qwen, open weights (Apache-2.0) | 2048 / 4096 (MRL to 64) | text+image+video+screenshot, fused | 32,000 | [HF 2B][qwen3vl-2b], [paper arXiv:2601.04720][qwen3vl-paper] |
| `BGE-M3` | BAAI, open (MIT); API via SiliconFlow | 1024 (no MRL) | text, **dense+sparse+ColBERT** | 8,192 | [HF BGE-M3][bgem3], [paper arXiv:2402.03216][bgem3-paper] |
| `multilingual-e5-large-instruct` | Microsoft/intfloat, open | 1024 (no MRL) | text | 512 | [HF][e5], [paper arXiv:2402.05672][e5-paper] |
| `jina-embeddings-v4` | Jina, API / open weights | 2048 (MRL to 128) | text+image (query/passage task adapters) | 32,768 | [HF][jina-v4], [paper arXiv:2506.18902][jina-v4-paper] |
| `jina-embeddings-v5-text-small/nano` | Jina, API / open | 1024 / 768 (MRL to 32) | text | 32,768 / 8,192 | [Jina v5-text][jina-v5-text] |
| `jina-embeddings-v5-omni-small` | Jina, API / open | 1024 (MRL to 32) | text+image+video+**audio** | — | [Jina model page][jina-v5-omni], [paper arXiv:2605.08384][jina-v5-omni-paper] |
| `embed-v4` | Cohere, API | 1536 (MRL: 256/512/1024/1536) | text+image (interleaved) | ~128,000 | [Cohere blog][cohere-v4], [AWS Bedrock][cohere-bedrock] |
| `voyage-3.5` | Voyage (MongoDB), API | 1024 (MRL 256/512/2048) | text | 32,000 | [Voyage pricing][voyage-price], [MongoDB docs][voyage-models] |
| `voyage-multimodal-3.5` | Voyage (MongoDB), API | 1024 (MRL 256/512/2048) | text+image+video | 32,000 | [Voyage pricing][voyage-price], [blog][voyage-mm35] |
| `text-embedding-3-large` / `-small` | OpenAI, API | 3072 (MRL 256/1024/3072) / 1536 | text | 8,191 | [OpenAI pricing][oai-price], [OpenAI launch][oai-launch] |
| `gemini-embedding-001` | Google, API | 3072 (MRL 128–3072; rec. 768/1536/3072) | text | 2,048 | [AI.google.dev][gemini-doc], [Google blog][gemini-blog] |
| `siglip2-*` | Google, open weights | 1536 (giant) | text+image (dual encoder) | — | [paper arXiv:2502.14786][siglip2] |
| `jina-clip-v2` | Jina | 1024 | text+image | — | [Milvus comparison][milvus-2026] |

### 3. Input formatting, dimension, context for *short, keyword-dense* metadata

- **Documents are tiny.** title (~5–15 tok) + description (~20–40 tok) + ~47 tags (~1–2 tok each)
  is roughly **100–150 tokens**. Every shortlist model except `multilingual-e5` (512) accommodates
  this with enormous headroom, so **max-token limits are not a selection axis here** — do not pay
  for 128k-context models (Cohere v4) for this.
- **Instruction-aware models want a query instruction, not a document one.** Qwen3-Embedding and
  `text-embedding-v4` take an `instruct` / `prompt_name="query"` on the query side and plain
  documents ([HF usage][qwen3-emb-06], [Milvus hands-on][milvus-qwen3]); E5 uses
  `Instruct: …\nQuery: …` for queries and `passage:` for docs ([HF e5][e5]). Instructions are
  reported to add **1–5 %** for Qwen3-family models ([Qwen3-VL card][qwen3vl-2b]); write them in
  English even for Chinese queries.
- **Recommended document serialization:** one string per row with light field labels, e.g.
  `"<title>. <description>. tags: t1, t2, …"`. Keep the tags — they are the dominant keyword
  signal and what sparse/BM25 will otherwise own. Do **not** truncate tags to hit a small token
  budget; there is none to hit.
- **Dimension / Matryoshka.** The repo already stores `vector(768)`. `text-embedding-v4` exposes a
  *native* 768 output (request `dimension=768`), so no lossy truncation and no migration.
  Qwen3 open models MRL to 64; Jina v5 to 32; OpenAI-3-large to 256; Cohere to 256; Gemini to 128.
  Storage math (float32): 768d ≈ 3 KB/row → **~2 GB** for 665k; 1024d ≈ 2.7 GB; 2048d ≈ 5.4 GB;
  2560d ≈ 6.8 GB. BGE-M3 has **no MRL** — requesting <1024 would be lossy truncation.
- **Multilingual gap is real but small.** Corpus is English, queries may be Chinese. `text-embedding-v4`
  lists 100+ languages incl. Chinese ([ds-text][ds-text]); BGE-M3 and multilingual-e5 are explicitly
  cross-lingual ([BGE-M3][bgem3], [E5][e5]); Qwen3 open weights cover 100+ languages
  ([Qwen3 GitHub][qwen3-gh]). English-only models (OpenAI-3, base BGE-en) would need query
  translation.

### 4. Multimodal text→image (thumbnails): is it worth it?

**Context.** We have *no* video locally and thumbnails are not yet downloaded. The natural question
is whether a cross-modal (text↔image) embedding buys recall over text-only on tag-rich metadata.

- **Cross-modal quality is model-dependent and improving fast.** On MMEB-V2 (2026), open
  `Qwen3-VL-Embedding-2B/8B` reach Image-Overall 75.0 / 80.1 and All 73.2 / **77.8** — competitive
  with closed multimodal embedders and far above 2024-era GME/VLM2Vec ([HF card][qwen3vl-2b]).
  A vendor-run 2026 comparison put `Qwen3-VL-2B` first on text→image R@1 (0.945) ahead of
  Gemini-Embedding-2 (0.928), Voyage-MM-3.5 (0.900), JinaCLIP-v2 (0.873) and CLIP ViT-L-14 (0.768)
  ([Milvus 2026][milvus-2026] — secondary, vendor-run).
- **Plain CLIP/SigLIP2 is a weak tool for this job.** Zero-shot dual encoders are known to fail on
  fine-grained/attribute queries and on text-to-person retrieval; a 2025 SigLIP2 re-ID evaluation
  reported text→image **Rank-1 of 0.4–2.7 %** on Market-1501/RSTPReid ([HF discussion][siglip2-reid]).
  SigLIP2 still beats SigLIP on standard retrieval ([paper][siglip2]), but it is a 2024/2025-era
  dual encoder, not a 2026 retrieval embedder.
- **MIEB shows there is no universal winner** across 130 image-embedding tasks / 38 languages
  ([arXiv:2504.10471][mieb]) — so the choice must be made against *our* query set, not a leaderboard.
- **Fusion into one vector is available and cheap to try.** `qwen3-vl-embedding` with
  `enable_fusion=true` fuses text+image (and video) into a single embedding in one space
  ([ds-mm][ds-mm]); `jina-embeddings-v5-omni` and `voyage-multimodal-3.5` likewise fuse
  ([jina-v5-omni][jina-v5-omni], [voyage-mm35][voyage-mm35]). This lets image-only "what is *in*
  the frame" queries share the index with text queries.

**Verdict.** For *this* corpus the tradeoff is:
- **Text-only on rich tags** already captures most recall; the median row has 47 tags, so a
  text query like "slow motion ocean waves at sunset" is well matched by `waves, ocean, sunset,
  slow motion` tags. Cost ~US$6, tiny embeddings, no thumbnail dependency.
- **Cross-modal fusion** adds value for genuinely visual queries (camera motion, composition,
  "aerial", "from above") and for reusing the MPT dedup gate in the same space — but it costs image
  tokens (~US$0.26 / 1M; e.g. the flash model bills ~896 image tokens/image per the response example
  in [ds-mm][ds-mm]) **and** requires downloading + maintaining thumbnails. Estimate: 665k images ×
  ~900 image-tokens ≈ 0.6B image-tokens → **~US$150 one-time** at US$0.258/1M, plus storage — i.e.
  the image pass is ~**20–30× the cost of the text pass**, for a recall gain that only shows up on a
  subset of queries.

**Recommendation:** ship text-only first; run a small A/B (text-only vs text+image-fused) on
*visually-specified* queries only, and only adopt the fused vector if it wins on those.

---

## Comparison table

Prices are US$ per **1M input tokens** at standard context, from vendor pages, 2026-10-08.
"China ✓" = reachable/operable from mainland China without a special egress path (Alibaba
DashScope, SiliconFlow, Jina's mainland-friendly APIs); international APIs (OpenAI, Google, Cohere,
Voyage) generally require the local proxy (the repo already routes Python via `localhost:7897`).

| Model | Provider | Access | Dim (default → MRL min) | Modalities | Max tok | Price /1M | China | Notes |
|---|---|---|---|---|---|---|---|---|
| `text-embedding-v4` | Alibaba DashScope | API | 2048 → 64 (req. `dimension=768`) | text | 8,192 | **$0.07** [ds-text] | ✓ | = Qwen3-Embedding; MTEB 71.58 / CMTEB 71.99 @2048 |
| `qwen3-vl-embedding` | Alibaba DashScope | API (Beijing only) | 2560 → 256 | text+image+video, **fused** | 32,000 | **Text $0.10; img/vid $0.258** [ds-mm] | ✓ | best open multimodal (MMEB All 77.8 @8B) |
| `tongyi-embedding-vision-flash` | Alibaba DashScope | API | **768 fixed** | text+image+video (independent) | 1,024 | Text $0.09; img/vid $0.03 (SG); CNY0.15 intl. [ds-price] | ✓ | shares space with MPT's dedup gate |
| `tongyi-embedding-vision-plus` | Alibaba DashScope | API | **1152 fixed** | text+image+video (independent) | 1,024 | $0.09 (all) [ds-mm] | ✓ | |
| `Qwen3-Embedding-0.6B` | Qwen open / SiliconFlow | weights or API | 1024 → 64 | text | 32,768 | ~$0.01–0.02 [sf-doc] | ✓ (SF) | MTEB-M 64.33 |
| `Qwen3-Embedding-4B` | Qwen open / SiliconFlow | weights or API | 2560 → 64 | text | 32,768 | ~$0.02–0.04 [sf-doc] | ✓ (SF) | MTEB-M 69.45 |
| `Qwen3-Embedding-8B` | Qwen open / SiliconFlow | weights or API | 4096 → 64 | text | 32,768 | ~$0.01–0.02 [sf-doc] | ✓ (SF) | MTEB-M **70.58** |
| `Qwen3-VL-Embedding-2B/8B` | Qwen open | weights | 2048 / 4096 → 64 | text+image+video fused | 32,000 | self-host (needs GPU) | ✓ (SF) | MMEB All 73.2 / 77.8 |
| `BGE-M3` | BAAI open / SiliconFlow | weights or API | 1024 (no MRL) | text (dense+sparse+colbert) | 8,192 | ~$0.01 [sf] | ✓ (SF) | gives sparse for free; MTEB-M 59.56 |
| `multilingual-e5-large-instruct` | intfloat open | weights | 1024 (no MRL) | text | 512 | self-host | ✓ | MTEB-M 63.22; short context |
| `jina-embeddings-v4` | Jina | API / weights | 2048 → 128 | text+image | 32,768 | ~$0.05 (10M pack) [jina-price] | ~ | task adapters |
| `jina-embeddings-v5-text-small/nano` | Jina | API / weights | 1024 / 768 → 32 | text | 32,768 / 8,192 | ~$0.05 [jina-price] | ~ | MMTEB 67.0; best <1B claim |
| `jina-embeddings-v5-omni-small` | Jina | API / weights | 1024 → 32 | text+image+video+audio | — | ~$0.05 [jina-price] | ~ | 1.6B, "one index all media" |
| `embed-v4` | Cohere | API | 1536 → 256 | text+image | 128,000 | **$0.12** text; $0.47 img [cohere-price] | ✗/proxy | 128k context overshoots our need |
| `voyage-3.5` | Voyage (MongoDB) | API | 1024 → 256 | text | 32,000 | **$0.06** [voyage-price] | ✗/proxy | |
| `voyage-multimodal-3.5` | Voyage | API | 1024 → 256 | text+image+video | 32,000 | $0.12 + $0.60/1B px [voyage-price] | ✗/proxy | |
| `text-embedding-3-large` | OpenAI | API | 3072 → 256 | text | 8,191 | **$0.13** [oai-price] | ✗/proxy | MTEB ~64.6 |
| `text-embedding-3-small` | OpenAI | API | 1536 | text | 8,191 | **$0.02** [oai-price] | ✗/proxy | strong value |
| `gemini-embedding-001` | Google | API | 3072 → 128 | text | 2,048 | **$0.15** [oai-price2] | ✗/proxy | MTEB-M 68.37 |
| `siglip2-*` | Google open | weights | 1536 | text+image | — | self-host (GPU) | ✓ | weak fine-grained text→image |
| `jina-clip-v2` | Jina | API / weights | 1024 | text+image | — | ~$0.05 [jina-price] | ~ | CLIP-style baseline |

Rough one-time corpus cost (est.: 665k rows × ~120 tokens ≈ **80M text tokens**; not a vendor quote):

| Model | Est. corpus cost | Comment |
|---|---|---|
| BGE-M3 @ SiliconFlow | **~$0.8** | cheapest; gives sparse too |
| Qwen3-Embedding-8B @ SiliconFlow | ~$1–2 | best open text quality |
| `text-embedding-v4` @ DashScope | **~$5.6** | recommended starter |
| OpenAI 3-small | ~$1.6 | |
| Voyage-3.5 | ~$4.8 | |
| Jina v4 | ~$4 | |
| tongyi-embedding-vision-flash (text) | ~$7.2 | |
| `qwen3-vl-embedding` (text) | ~$8.0 | |
| Cohere v4 (text) | ~$9.6 | |
| OpenAI 3-large | ~$10.4 | |
| Gemini 001 | ~$12 | |
| **+ image pass** (665k thumbnails × ~900 img-tok) | **~$150** | the reason to defer cross-modal |

---

## Ranked recommendation for this corpus

**Decision criterion (run this before committing):** on a held-out set of ~200 English + ~50 Chinese
natural-language concept queries against a 50k–100k sample of the corpus, measure **Recall@k /
nDCG@10** (judge with the existing VLM funnel or an LLM-as-judge), against (a) the text-only dense
model and (b) the current keyword/BM25 funnel. Adopt the dense model only if it beats the funnel on
the same query mix *and* stays within the per-corpus cost budget. Re-verify prices the day you run it.

1. **#1 — DashScope `text-embedding-v4`, `dimension=768`, instruction on query side.**
   Best text quality that natively fits the existing `vector(768)` column, multilingual (Chinese
   queries), ~US$6 for the whole corpus, key already available. [ds-text][ds-text], [ds-embed][ds-embed].
   *If* the gate says 768 truncates too much, re-embed at 2048 and migrate the column (storage still
   ~5.4 GB).
2. **#2 — `qwen3-vl-embedding` fused text+image @768–1024 dims (upgrade, after thumbnails).**
   One space for text and images, strongest open multimodal on MMEB-V2, 32k context, China-BJ
   hosted. Adopt only if visually-specified-query A/B beats #1. [ds-mm][ds-mm], [qwen3vl-2b][qwen3vl-2b].
3. **#3 — `BGE-M3` (self-host on a future GPU, or SiliconFlow API), 1024 dims.**
   Cheapest, and the only shortlist model that gives **dense + sparse + ColBERT in one forward pass**,
   which lets one model feed both the dense index *and* the sparse leg of the hybrid. Cost = a schema
   change 768→1024 (no MRL ⇒ no lossy 768 truncation) and loss of cross-modal. [bgem3][bgem3].
4. **#4 (quality ceiling / future GPU) — `Qwen3-Embedding-8B`** (or the newer 2026 open leaders such
   as `KaLM-Embedding-Gemma3-12B-2511`) at 768–1024 dims. Only worth it with local GPU; via API the
   quality-per-dollar over #1 is small for short metadata. [qwen3-emb-8b][qwen3-emb-8b], [mteb-lb][mteb-lb].

**Why not the handoff's original starter (`tongyi-embedding-vision-flash`)?** It is the right choice
*only if* sharing a vector space with MPT's image dedup gate is a hard requirement. As a *text*
retriever it is a vision model with a fixed 768 dims and 1,024-token text limit, and its text price
($0.09) is higher than `text-embedding-v4` ($0.07) — so it is dominated for the text-only index
unless the dedup-space reuse is decisive.

**Why not the long-context / big-API models?** Cohere v4 (128k), voyage, OpenAI-3-large and Gemini
are all more expensive and none of their differentiators (document length, huge context) apply to
100–150-token metadata rows.

---

## Open questions & uncertainty

- **Verify before committing (time-sensitive):** the exact current DashScope text model name. The
  official docs now show **`qwen3.7-text-embedding`** in examples ([ds-embed][ds-embed]) alongside the
  older `text-embedding-v4`; unclear whether it is a rename, a newer generation, or a preview. Check
  console model list + price page on implementation day.
- **`text-embedding-v4` @768 (our target) MTEB is not published** — the official table gives 512 /
  1024 / 2048 only ([ds-embed][ds-embed]). Quality at 768 is interpolatable but unmeasured; the
  evaluation gate should include a 2048-dim run.
- **Vision-flash price inconsistency.** The handoff note records ¥0.18 / 1M for
  `tongyi-embedding-vision-flash`; the vendor pages show CNY 0.15 / 1M (China) and US$0.09 text /
  US$0.03 image (Singapore) ([ds-price][ds-price]). One of these is stale — reconcile against the
  billing console.
- **`qwen3-vl-embedding` is China-Beijing-only** per the vendor doc ([ds-mm][ds-mm]); confirm the
  MPT key's region before relying on it.
- **SiliconFlow prices for BGE-M3 / Qwen3-Embedding** are cited from secondary aggregators
  (~$0.01–0.02 / 1M); the official API docs show the models and dims but not the per-token price
  ([sf-doc][sf-doc]). Confirm on the pricing page.
- **The 2026 open-weight leaders move weekly.** `harrier-oss-v1-*`, `KaLM-Embedding-Gemma3-12B`,
  `Nemotron-3-Embed-8B`, `jina-embeddings-v5-*` all appear on MTEB v2 but are not yet all in the
  stable published tables. Treat the leaderboard as a moving target.
- **`gemini-embedding-2`** is referenced (natively multimodal, preview) by multiple 2026 sources but
  is not fully documented on the official pricing/doc pages I could reach; exclude from shortlist
  until GA.
- **Cross-modal fusion quality for *motion* queries** ("slow motion", "timelapse", "aerial") is the
  real unknown: a single thumbnail is a still frame and cannot encode motion, so thumbnail-based
  fusion likely cannot serve the motion half of this corpus's query distribution. A video-frame or
  `qwen3-vl-embedding` video path is required if motion is important — measure before investing.
- **Interaction with sparse leg.** None of the dense prices here cover BM25/sparse cost; the hybrid
  design (dense + BM25, or dense + BGE-M3 sparse) is a separate decision and affects which dense
  model wins (BGE-M3 can serve both).

---

[ds-text]: https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api
[ds-embed]: https://help.aliyun.com/en/model-studio/embedding
[ds-mm]: https://www.alibabacloud.com/help/en/model-studio/multimodal-embedding-api-reference
[ds-price]: https://help.aliyun.com/en/model-studio/model-pricing
[mteb-lb]: https://huggingface.co/spaces/mteb/leaderboard
[mteb-v2]: https://mteb-leaderboard.hf.space/benchmark/MTEB(Multilingual%2C%20v2)
[qwen3-emb-8b]: https://huggingface.co/Qwen/Qwen3-Embedding-8B
[qwen3-emb-06]: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
[qwen3-emb-4b]: https://huggingface.co/Qwen/Qwen3-Embedding-4B
[qwen3-gh]: https://github.com/QwenLM/Qwen3-Embedding
[qwen3-paper]: https://arxiv.org/pdf/2506.05176
[qwen3vl-2b]: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
[qwen3vl-paper]: https://arxiv.org/abs/2601.04720
[bgem3]: https://huggingface.co/BAAI/bge-m3
[bgem3-paper]: https://arxiv.org/pdf/2402.03216.pdf
[e5]: https://huggingface.co/intfloat/multilingual-e5-large-instruct
[e5-paper]: https://arxiv.org/abs/2402.05672
[jina-v4]: https://huggingface.co/jinaai/jina-embeddings-v4
[jina-v4-paper]: https://arxiv.org/html/2506.18902v3
[jina-v5-text]: https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings/
[jina-v5-recipe]: https://recipes.vllm.ai/jinaai/jina-embeddings-v5-text-small
[jina-v5-omni]: https://jina.ai/models/jina-embeddings-v5-omni-small/
[jina-v5-omni-paper]: https://arxiv.org/html/2605.08384v2
[jina-price]: https://jina.ai/embeddings/
[cohere-v4]: https://cohere.com/blog/embed-4
[cohere-bedrock]: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
[cohere-price]: https://embeddingcost.com/cohere
[voyage-price]: https://docs.voyageai.com/docs/pricing
[voyage-models]: https://www.mongodb.com/docs/voyageai/models/
[voyage-mm35]: https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/
[oai-price]: https://platform.openai.com/docs/pricing
[oai-price2]: https://embeddingcost.com/google
[oai-launch]: https://openai.com/index/new-embedding-models-and-api-updates/
[gemini-doc]: https://ai.google.dev/gemini-api/docs/embeddings
[gemini-blog]: https://developers.googleblog.com/gemini-embedding-available-gemini-api/
[siglip2]: https://arxiv.org/html/2502.14786v1
[siglip2-reid]: https://huggingface.co/google/siglip2-base-patch16-224/discussions/17
[mieb]: https://arxiv.org/html/2504.10471v1
[milvus-2026]: https://milvus.io/blog/choose-embedding-model-rag-2026.md
[milvus-qwen3]: https://milvus.io/blog/hands-on-rag-with-qwen3-embedding-and-reranking-models-using-milvus.md
[cpack]: https://arxiv.org/html/2309.07597v3
[cmteb]: https://bge-model.com/tutorial/4_Evaluation/4.2.3.html
[sf-doc]: https://docs.siliconflow.com/cn/api-reference/embeddings/create-embeddings
