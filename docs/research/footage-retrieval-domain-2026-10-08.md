# Footage retrieval — domain research (SOTA stock-video search, query understanding, evaluation)

Date: 2026-10-08 · Scope: the **domain** (how footage search works, query understanding, retrieval
models, evaluation). Primary sources only; every non-obvious claim carries a URL. Items that are
time-sensitive or vendor-marketing are flagged **⚠️**.

Context assumed for this repo: ~665k Pexels video **metadata** rows (`title`, `description`, median
47 / p90 55 `tags`, `duration`, `width/height`, `orientation`, `license`, `video_files` JSONB,
`thumbnail_url`); **no video files stored**; **no GPU**; current plan = dense + sparse hybrid in
Postgres/pgvector, with an embedding seam already in place (`docs/phase2-embedding-seam.md`).

---

## TL;DR recommendation

**Ship the "text + filters" hybrid first; add thumbnail-image embedding as the single highest-ROI
enrichment; treat motion as a known gap; build your own small LLM-judged eval set instead of
trusting academic benchmarks.**

1. **Index what you already have, now.** Dense over `title — description — tags` (the existing
   `build_embed_text` contract) + local BM25, fused with RRF in pgvector. This is exactly the
   "keyword + semantic" shape every stock platform exposes via API parameters
   ([Shutterstock video search params](https://www.shutterstock.com/developers/documentation/searching),
   [Adobe](https://developer.adobe.com/stock/docs/getting-started/apps/05-search-for-assets)).
2. **Embed the thumbnail too — same API, same vector space.** The repo's chosen default
   `tongyi-embedding-vision-flash` **already accepts image and video inputs** (768-dim fixed,
   image up to 3 MB / 8 images) ([Alibaba Model Studio embedding docs](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/embedding)).
   Since thumbnails are already being fetched as URLs, this is a near-zero-cost addition that
   unlocks *text → image* retrieval — i.e. the "visually similar / search by look-and-feel" feature
   Shutterstock built on its **video preview image**
   ([Shutterstock 2019 reverse-search-for-video](https://www.shutterstock.com/blog/reverse-image-search-for-video)).
   Thumbnail = strong proxy for **subject, setting, lighting, colour, style**; weak for **motion**.
3. **Do NOT auto-caption every row from a single thumbnail as a first step.** It costs an extra VLM
   pass over 665k rows and a still can't describe motion — CLIP4Clip's headline finding is that
   "one single image is far from enough for video encoding"
   ([CLIP4Clip §1](https://arxiv.org/abs/2104.08860)). Add captions only for *style/mood/lighting*
   language if the eval shows a gap (auxiliary captions do help retrieval, but the payoff is
   small-to-moderate: +~2–7 R@1 on MSR-VTT in
   [Cap4Video](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Cap4Video_What_Can_Auxiliary_Captions_Do_for_Text-Video_Retrieval_CVPR_2023_paper.pdf)).
4. **Query understanding: facet-extract + lightly expand, then route facets to structured filters.**
   Decompose NL queries into subject / motion / camera / setting / lighting / time-of-day / mood;
   use synonym expansion for BM25; map orientation/duration/resolution/license to SQL filters.
   Caveat from the literature: LLM expansion helps **weak** retrievers but can **hurt strong** ones
   ([Weller et al., EACL 2024](https://aclanthology.org/2024.findings-eacl.134)).
5. **Evaluate with your own set.** Academic TVR benchmarks (MSR-VTT/MSVD…) are captioning datasets
   repurposed for retrieval; they suffer false negatives and are near saturation — the authors
   recommend retiring them ([Fighting FIRe with FIRE](https://aclanthology.org/2023.findings-eacl.3/)).
   They also don't resemble "moody b-roll". Build ~100–300 faceted queries, pool top-k, judge with an
   LLM (UMBRELA-style graded 0–3), report nDCG@10 / MRR / Recall@k per facet.

Feasibility summary (no GPU, no video files): text embedding ✅ API; thumbnail image embedding ✅ API;
true video-clip embedding ⚠️ requires the video (only via remote URL, or download+store) and is the
one thing you cannot do cheaply today.

---

## How stock platforms do search

**Pexels (your source).** Tagging is contributor-supplied plus **automatically generated** tags;
the tagging guide says a "well-tagged" item has "at least 10 tags, but no more than 25", organised
as *subject* tags + *descriptive* tags (action/emotion/setting/time-of-day/weather) + abstract
"story" tags ([Pexels tagging guidelines](https://www.pexels.com/blog/photography/photo-video-tagging-guidelines)).
For photos, Pexels extracts `Title`/`Tags` from **EXIF/IPTC** and pre-populates up to **30 tags**
([Pexels metadata help](https://help.pexels.com/hc/en-us/articles/37285267631769-How-does-Pexels-use-the-metadata-on-my-photos)).
The official API exposes `query`, `orientation`, `size`, `color`, `locale`, and for videos
`min_width/min_height/min_duration/max_duration` — but **not** the rich tag array
([pexels-api-py client, mirroring the official API](https://github.com/meetsohail/pexels-api-py)).
⚠️ Pexels' own API help states tags/titles/metadata are *not* returned by the API
([Pexels API media-details help](https://help.pexels.com/hc/en-us/articles/47677748516121-Are-media-details-and-statistics-available-through-the-API)) —
which is why this repo scrapes the internal v3 endpoint.

**Shutterstock.** Two search families:
- *Keyword search* with strong structured filters: video params include `orientation`, `aspect_ratio`,
  `resolution` (`4k|high_definition|standard_definition`), `duration_from/duration_to`, `fps_from/fps_to`,
  `license`, `category`, `sort` (`newest|popular|relevance|random`)
  ([Shutterstock searching docs](https://www.shutterstock.com/developers/documentation/searching),
  [VideosApi SDK](https://github.com/shutterstock/public-api-javascript-sdk/blob/master/docs/VideosApi.md)).
- *Visual search*: "reverse image search" + "visually similar" powered by a custom CNN that "sees the
  pixels" rather than keywords ([Shutterstock help: visually similar](https://www.shutterstock.com/help/en/articles/10617103-how-does-shutterstock-recommend-images-similar-to-the-one-i-m-viewing),
  [2016 launch PR](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-launches-reverse-image-search-and-visually-similar)).
  Crucially, the **video** version "analyzed every video clip … to identify the **video preview image**
  and capture the characteristics of the clip" ([Shutterstock 2019 blog](https://www.shutterstock.com/blog/reverse-image-search-for-video)) —
  i.e. a big player already uses a single representative frame as the video's visual proxy. There is
  also a `findSimilarVideos(id)` endpoint ([VideosApi SDK](https://github.com/shutterstock/public-api-javascript-sdk/blob/master/docs/VideosApi.md))
  and a "Composition Aware Search" that places keywords spatially on a canvas
  ([NVIDIA technical blog](https://developer.nvidia.com/blog/shutterstocks-new-ai-tool-lets-you-search-by-image-composition)).

**Adobe Stock.** Keyword + filters (`content_type`, colour, etc.) and **visual similarity search**
powered by Adobe Sensei: search by uploaded image, by image URL, or by an existing Stock ID, then
narrow with keyword/colour qualifiers
([Adobe Stock Search API](https://developer.adobe.com/stock/docs/getting-started/apps/05-search-for-assets),
[Find similar assets help](https://helpx.adobe.com/stock/web/search-for-assets/find-similar-assets.html)).
These are image-level "find similar"; the same assets include video.

**Getty Images.** The API exposes `orientation` (horizontal/vertical/square/panoramic), `quality_rank`
(0–3, where for editorial 0–1 = highly representative, 3 = "background, filler shots"),
`number_of_people`, `ethnicity`, `compositions`, and **facets** (artists/events/locations/specific
people) for refining a search
([Getty asset metadata](https://developers.gettyimages.com/asset-metadata),
[Getty release notes](https://developers.gettyimages.com/release-notes)).
Getty markets "enhanced search" that "uses machine-learning models to understand natural,
conversational language" ⚠️ (vendor claim, [API aggregator page](https://apis.io/apis/getty-images/getty-images-search-api)).
Note Getty faceting is *metadata* faceting, not visual.

**Artgrid.** Explicitly facet-driven: filter by **video theme** (Sport, Medical, Green Planet,
abstract, timelapse), **shot type** (close-up, handheld, interiors, exteriors, 4K), **people**,
**framing** (wide), and curated **Collections** (Wildlife, City Life, colour-scheme clips)
([Artgrid help: how to search](https://artgrid.zendesk.com/hc/en-us/articles/8069563020701-How-to-Find-Download-and-Use-Footage),
[Artgrid "stock video" guide](https://artgrid.io/insights/stock-video)).
Storyblocks exposes similar facets: 360°/VR, Aerial, Slow Motion, Time Lapse, Categories, Orientation,
Resolution (HD/4K/8K), Frame Rate, Duration, Model/Property Released
([Storyblocks footage filters](https://www.storyblocks.com/video/search/art-grid)).
Motion Elements offers a drag-and-drop **VisualSearch** for footage "reflecting a certain style or
mood [you] just can't describe … when required to type keywords"
([Motion Elements VisualSearch](https://www.motionelements.com/visualsearch)).

**Takeaways for us.**
- Every serious platform = **keyword search + structured facets + visual (image) search**. None of
  the public docs claim a text→*full-video* neural retriever over the catalogue for ad-hoc search.
- Motion/style facets (Slow Motion, Time Lapse, Aerial, handheld) are usually **attributes**
  (tags/enums), not something recovered by an image encoder.
- "Visual similar" is routinely built on **one representative frame**, which is precisely the proxy
  a no-video-files shop needs.

---

## Query understanding approaches

Creative-footage queries are **multi-facet** natural language ("slow motion ocean waves at sunset
from above" = motion + subject + time-of-day + camera). The literature supports decomposing them.

**Facet / sub-query decomposition.**
- Breaking a complex query into independent "clues/subqueries" routed to specialised retrievers and
  ensembled improved Recall@5 by up to **6 points absolute** on a 14,441-query set
  ([Decomposing Complex Queries for Tip-of-the-tongue, EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.367.pdf)).
- For RAG, an LLM decomposes a query → retrieve per sub-query → merge/dedup → rerank; the paper notes
  decomposition increases **coverage** but also noise
  ([Question Decomposition for RAG, 2025](https://arxiv.org/html/2507.00355v1)).
- "Facets" are an established IR concept: extract/generate facets to refine or diversify queries
  ([Revisiting Open Domain Query Facet Extraction, UMass CIIR](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1455)).
- An LLM can be used as a **facet decomposer** for documents (paper abstracts), then blended for
  faceted retrieval ([Multi-Facet Blending, ACL 2025](https://aclanthology.org/2025.acl-long.1388.pdf)).
- ⚠️ Hierarchical/adaptive budget allocation across sub-queries reported a **30% precision gain** over
  single-level decomposition in one RAG study ([Query Decomposition Exploration-Exploitation, 2025](https://arxiv.org/html/2510.18633)) —
  promising but not footage-specific.

**Query expansion / rewriting.**
- LLM-generated document-based expansions improved a zero-shot dense retriever by **>8%**
  ([LLM-QE, 2025](https://arxiv.org/abs/2502.17057)).
- **But** a large multi-dataset/multi-retriever study finds a **strong negative correlation between
  retriever strength and expansion benefit**: expansions help weak models and generally harm strong
  ones, adding noise that hurts top-rank precision
  ([When do Generative Query and Document Expansions Fail?, EACL 2024](https://aclanthology.org/2024.findings-eacl.134)).
  A reproduction study similarly finds the gain shrinks as the retriever moves from BM25 to neural
  encoders with implicit expansion (e.g. SPLADE), and that **BM25 + good LLM expansion approaches
  unexpanded dense retrievers** ([Reproducibility study of LLM query reformulation, 2026](https://arxiv.org/pdf/2604.27421v1)).
- Best-practice paper for LLM expansion across lexical + dense retrievers (the MUGI pipeline)
  ([Exploring Best Practices of Query Expansion with LLMs, EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.103.pdf)).
- Aligned query expansion cuts latency by ~70% vs generate-then-filter
  ([AQE, 2025](https://arxiv.org/html/2507.11042v1)).

**Implication for a footage backend.** Given rich tags already act as a hand-built synonym layer,
treat LLM expansion as an **A/B-tested add-on**, not a default. The safest, highest-value LLM step is
**structured facet extraction → filter routing** (orientation/duration/resolution/license + hard
constraints), which cannot be expressed as a bag of words at all.

---

## Video/text retrieval models & feasibility (no GPU, no video files)

### The models (2020 → 2026)

| Model | What it is | Retrieval signal | Key number / source |
|---|---|---|---|
| **CLIP** | image-text dual encoder, 400M pairs | image↔text | [Radford et al., ICML 2021](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) |
| **CLIP4Clip** | CLIP → video via frame sequence + similarity calc (mean/seq/tight) | video↔text | finding: "one single image is far from enough" ([arXiv:2104.08860](https://arxiv.org/abs/2104.08860)) |
| **LanguageBind** | align N modalities to a frozen language space; VIDAL-10M | video↔text + audio/depth/IR | **+3.7% R@1 over CLIP4Clip on MSR-VTT** zero-shot ([ICLR 2024, arXiv:2310.01852](https://arxiv.org/abs/2310.01852)) |
| **InternVideo2** | 6B progressive ViFM (masked recon + contrastive + NTP) | video-text, dialogue | SOTA on 60+ tasks; zero-shot T2V reported on MSR-VTT/LSMDC/MSVD/ActivityNet/VATEX ([ECCV 2024, arXiv:2403.15377](https://arxiv.org/abs/2403.15377)) |
| **VideoPrism** | frozen general video encoder | classification/dense/retrieval | pretrained on **36M video-caption + 582M clips**; zero-shot MSRVTT-1K t2v R@1 **50.1** (LvT-L public) ([ICML 2024, arXiv:2402.13217](https://arxiv.org/abs/2402.13217), [repo](https://github.com/google-deepmind/videoprism)) |
| **SigLIP 2** | multilingual **image**-text encoder, B/L/So400m/g | image↔text | better than SigLIP at all scales ([arXiv:2502.14786](https://arxiv.org/abs/2502.14786)) |
| **Qwen3-VL-Embedding** | multimodal embed (text/image/**video**), 2B/8B, MRL | unified | MMEB-v2 overall **77.8** (8B); video-retrieval subtask **53.6/55.8** (2B embed/rerank), **~67** for text-only 8B ([tech report arXiv:2601.04720](https://arxiv.org/abs/2601.04720), [Qwen blog](https://qwen.ai/blog?id=qwen3-vl-embedding)) ⚠️ Jan 2026, fast-moving |
| **Qwen2.5-VL** | VLM (caption/VQA, dynamic-FPS video) | captioner / judge | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) |
| **BLIP-2** | frozen encoders + Q-Former, captioner | captioning | SOTA captioning/VQA at release ([arXiv:2301.12597](https://arxiv.org/abs/2301.12597)) |
| **Cap4Video / NarVid** | auto-captions from video boost T-V retrieval | video↔text | MSR-VTT R@1 **51.4** (Cap4Video) vs 49.3 baseline ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Cap4Video_What_Can_Auxiliary_Captions_Do_for_Text-Video_Retrieval_CVPR_2023_paper.pdf)); frame-level captions further help ([NarVid, 2025](https://arxiv.org/html/2503.05186v1)) |

Also relevant: **keyframe/clip aggregation** is an active subfield. Query-relevant frame selection can
be training-free: AdaRD-Key scores frames with BLIP-2 and picks a relevance+diversity set
([arXiv:2510.02778](https://arxiv.org/html/2510.02778)); a coverage-driven selector adapts how many
frames to score ([CSES, 2026](https://arxiv.org/pdf/2608.00714)). This matters only if you ever touch
real video.

### Hard feasibility filter: **no GPU, no stored video files**

**✅ Fully feasible today (API, CPU-only client):**
- **Text embedding** over metadata → dense index. The repo default
  `tongyi-embedding-vision-flash` is 768-dim, accepts text, image, **and video**, with
  `res_level` / `max_video_frames` knobs ([Alibaba Model Studio](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/embedding)),
  text-embedding-v4 also returns dense+sparse ([DashScope text-embedding docs](https://docs.qwencloud.com/api-reference/text-embedding/dashscope-embedding)).
- **Thumbnail image embedding** → text→image retrieval. Same multimodal endpoint takes an image URL;
  Cohere `embed-v4.0` (256/512/1024/1536 dims, up to 96 images/request) and Voyage
  `voyage-multimodal-3.5` (1024 default, Matryoshka) and Jina `jina-clip-v2` (1024, 89 languages,
  Matryoshka) are alternatives ([Cohere embeddings](https://docs.cohere.com/docs/embeddings),
  [Voyage multimodal docs](https://docs.voyageai.com/docs/multimodal-embeddings),
  [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2)).
- **Auto-caption from thumbnail** (optional) via Qwen2.5-VL / Qwen3-VL API; BLIP-2/qwen captioning is
  well-documented ([BLIP-2](https://arxiv.org/abs/2301.12597)). Cheap to *try*, but see the motion caveat.

**⚠️ Not cheap / not direct:**
- **True video-clip embedding** requires the video. Some APIs accept a video **URL** (Voyage,
  DashScope), but DashScope caps video at **10 MB** for the flash model / **50 MB** for
  qwen3-vl-embedding ([Model Studio](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/embedding)),
  while Pexels 1080p clips are ~15 MB and 4K ~72 MB (repo handoff §2). So you'd have to download,
  truncate/transcode (CPU/ffmpeg) and pay per call over hundreds of thousands of rows — an order of
  magnitude more cost than thumbnail embedding, for a marginal ad-hoc-search gain. **Recommend deferring.**
- **Local open models** (SigLIP2, VideoPrism, InternVideo2, LanguageBind, CLIP4Clip): all need a GPU
  to be practical at 665k scale; the repo has none. Use managed APIs or skip.

**Is thumbnail-based/image retrieval a reasonable proxy for video content?** *Partially — and the
industry already does it.* Shutterstock's video reverse search is built on the video **preview image**
([2019 blog](https://www.shutterstock.com/blog/reverse-image-search-for-video)). A single frame
captures **subject, setting, colour, lighting, composition, style** well. It does **not** capture
**motion/temporal** semantics — the explicit motivation behind CLIP4Clip's frame-sequence design
([arXiv:2104.08860](https://arxiv.org/abs/2104.08860)) and why Voyage treats a video as an *ordered
sequence* of frames ([Voyage 3.5 blog](https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/)).
For "moody b-roll" the visual frame is most of the signal; for "slow motion" or "timelapse" it is
not — keep motion as a **tag/enum + duration** attribute.

⚠️ **Modality gap** caveat: CLIP-style dual encoders route images and text through separate towers,
so a text query can outrank a relevant image with an irrelevant text vector; single-backbone
embedders (Voyage) claim to avoid this ([Voyage multimodal docs](https://docs.voyageai.com/docs/multimodal-embeddings)).
Test this on your own data before trusting cross-modal scores.

---

## Evaluation plan

### Metrics
- Ranking: **Recall@k**, **MRR**, **nDCG@k** (graded). nDCG's gain-discount formulation is
  [Järvelin & Kekäläinen, TOIS 2002](https://dl.acm.org/doi/10.1145/582415.582418). TVR papers report
  R@1/R@5/R@10 + median/mean rank ([X-Pool, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Gorti_X-Pool_Cross-Modal_Language-Video_Attention_for_Text-Video_Retrieval_CVPR_2022_paper.pdf)).
- Report **per-facet** (subject / motion / camera / setting / lighting / time-of-day / mood) and
  **filter-constrained** (orientation/duration/resolution) cuts — a single aggregate hides the facet
  that's actually broken.

### Why *not* the standard TVR benchmarks
- **MSR-VTT**: 10K clips, ~20 captions each, 41.2 h ([MSR-VTT paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf)).
- **MSVD**: 1,970 videos (~40 captions); **VATEX**: 41,250 videos / 825,000 captions
  ([DiffusionRet supp.](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Jin_DiffusionRet_Generative_Text-Video_ICCV_2023_supplemental.pdf),
  [VATEX](https://huggingface.co/papers/1904.03493));
  **LSMDC**: 118,081 movie clips; **ActivityNet Captions**: 20K videos/42K clips.
- **They are captioning datasets repurposed for retrieval.** One positive video per caption → many
  *false negatives*; correcting them gives a recent SOTA model **+25 recall points**, and the authors
  recommend **retiring** MSR-VTT/MSVD ([Fighting FIRe with FIRE, EACL 2023](https://aclanthology.org/2023.findings-eacl.3/)).
- They are only **coarse-grained**: captions differ mostly by nouns/verbs, and models' fine-grained
  (adjective/adverb/preposition) sensitivity is much lower
  ([Beyond Coarse-Grained Matching, 2024](https://arxiv.org/html/2410.12407v2)).
- They do **not** target commercial stock footage, though the motivation is explicitly stock-adjacent:
  DiDeMo frames the task as "finding … desired **B-roll stock video footage** from a large video
  library (e.g., Adobe Stock, Getty, Shutterstock)"
  ([DiDeMo, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hendricks_Localizing_Moments_in_ICCV_2017_paper.pdf)).
- Related footage/aesthetic datasets exist but are for **generation/aesthetics**, not text→b-roll
  retrieval: LAION aesthetic predictor ([LAION](https://laion.ai/blog/laion-aesthetics/)),
  VADB video aesthetics ([arXiv:2510.25238](https://arxiv.org/html/2510.25238)),
  VideoAesBench ([arXiv:2601.21915](https://arxiv.org/pdf/2601.21915v1)).
  ⚠️ Also note the LAION aesthetic predictor has documented cultural bias and rates realistic
  landscapes/portraits highest ([audit, FAccT 2026](https://arxiv.org/abs/2601.09896)) — if used at all,
  use as a *rank* signal, never as a quality gate.

### Cheap, principled eval-set construction
1. **Query set (100–300).** Sample real creator-style queries per facet; add the structured
   constraints (orientation/duration/resolution) separately so you can score facet and filter effects
   independently.
2. **Pool.** Union top-k (k≈20–50) from: BM25, dense-text, dense-image, and the current
   keyword/funnel baseline. Pooling protects against judging only one system's results.
3. **Judge.** LLM relevance judging with a graded 0–3 rubric is standard and reproducible:
   UMBRELA is the open-source reproduction of the Bing relevance assessor and is used in TREC
   ([UMBRELA, SIGIR 2024, arXiv:2406.06519](https://arxiv.org/pdf/2406.06519),
   [repo](https://github.com/castorini/umbrela)). Typical agreement is modest — e.g. Cohen's κ≈0.31
   on the 4-point scale with GPT-4o ([LLM-judge generalizability study](https://arxiv.org/html/2507.09483v1)) —
   so **never over-interpret small deltas**.
   ⚠️ Known LLM-judge hazards: **overrating/inflation** of relevance labels
   ([When LLM Judges Inflate Scores, 2026](https://arxiv.org/html/2602.17170v3)), **position bias**
   ([Judging the Judges, IJCNLP 2025](https://aclanthology.org/2025.ijcnlp-long.18)),
   prompt sensitivity, and — critically for a system that *itself* uses an LLM — **circularity**: if the
   same LLM both ranks and judges, the evaluation is invalid
   ([Clarke & Dietz, LLM assessment can't replace humans](https://arxiv.org/html/2412.17156)).
   Mitigations: use a **different** model as judge than as rewriter/reranker; hold out a small
   **human-labelled** seed set to calibrate the LLM judge; report judge-vs-human agreement.
4. **Mine hard negatives** from the pool (relevant-looking-but-wrong) so the metric discriminates.
5. **Track cost** alongside quality; the acceptance gate in `docs/handoff-2026-10-07.md` §5
   (recall@k and VLM adoption vs the current funnel) is the right shape.

### Baselines to beat
- BM25 only; dense-text only; the current MPT funnel; then **RRF hybrid** of dense+sparse (pgvector
  documents hybrid via Postgres FTS + RRF, [pgvector README](https://github.com/pgvector/pgvector));
  finally **+ thumbnail-image** and **+ facet-extraction**.

---

## Open questions & uncertainty

- **Motion.** Nothing in scope recovers motion without video frames. Is "slow motion / timelapse /
  aerial" adequately covered by tags + `duration`? ⚠️ Needs a real query sample to confirm.
- **Tag quality & coverage.** Pexels auto-generates tags and admits they "aren't always that
  accurate" ([Pexels](https://www.pexels.com/blog/photography/photo-video-tagging-guidelines)); the
  repo's median 47 tags/row is *far above* Pexels' own 10–25 recommendation. Are the extra tags
  helpful or is the dense text getting diluted? Measure recall@k with/without tags.
- **Thumbnail representativeness.** Pexels thumbnails are usually a single frame (often the first
  frame). A first frame can be unrepresentative of a clip's content — how much does this cost versus
  a mid-clip keyframe? We cannot test a mid-clip frame without the video. ⚠️
- **Cross-modal score calibration.** Whether `tongyi-embedding-vision-flash` text→thumbnail scores
  are comparable to text→text scores (modality gap) is unverified for this model — test by mixing
  modalities in one ranking and checking for bias. ⚠️
- **Vendor numbers are self-reported.** Voyage claims +4.65% NDCG@10 over Google Multimodal 001 on
  3 video-retrieval datasets ([blog](https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/));
  Qwen claims MMEB-v2 leadership ([blog](https://qwen.ai/blog?id=qwen3-vl-embedding)). Treat as
  directional until reproduced on in-domain data.
- **Cost/latency of 665k embedding calls** (dense text + thumbnail + optional caption) is not yet
  measured; the embedding seam's incremental, resumable backfill design is the right hedge.
- **Model landscape is fast-moving** (Qwen3-VL-Embedding is Jan 2026; several 2025–26 preprints
  cited here may be superseded). Keep the seam so the default can be swapped without re-architecting.
- **Legal/licence facets.** `license` is a filter but commercial-use nuance (model/property release)
  exists in other platforms ([Getty metadata](https://developers.gettyimages.com/asset-metadata));
  confirm whether Pexels metadata gives enough to answer "can I actually use this in a paid ad?".
