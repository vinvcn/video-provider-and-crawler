# Frame captioning pipeline for stock-video search — VLMs, prompts, frame sampling, cost (frames → structured multi-facet captions)

Date: 2026-10-08 · Scope: the **captioning pipeline** (frames → structured facet captions → what to
embed / index). Companion to [footage-retrieval-domain](footage-retrieval-domain-2026-10-08.md)
(domain, thumbnail findings, Cap4Video, motion gap) and
[dense-embedding-models](dense-embedding-models-2026-10-08.md) (text/multimodal embedders).

Every non-obvious number carries a primary-source URL. **⚠️ = time-sensitive price/list, verify on
implementation day** (all vendor prices below are a 2026-10-08 snapshot). Estimates computed for this
repo's corpus are marked *(est.)* and are **not vendor quotes**.

Assumed inputs (from this repo / task): ~665k Pexels videos, metadata only; thumbnail URLs (up to
1440 px) not yet downloaded; **one low-res SD preview mp4 URL per video (0.6–18 MB, typical 1–3 MB,
repo handoff [`docs/handoff-2026-10-07.md`](../handoff-2026-10-07.md) §2 and task brief)**; no GPU;
Python egress via proxy `localhost:7897`; DashScope (Alibaba) key already exists in a sibling
project.

---

## TL;DR

1. **Pipeline shape:** download the SD preview → extract 4–8 frames with ffmpeg → one request per
   video containing all frames → **one JSON object with one short field per facet** (subject,
   action, setting, camera, lighting, time-of-day, colour, mood, people, style, text-in-frame,
   motion) → store raw JSON → embed per-facet text and a short concatenated digest → add facet text
   to the sparse arm → A/B against tags-only.
2. **Cheapest sane tier (recommended first):** `qwen3-vl-flash` on DashScope **Beijing** at
   **$0.022 / $0.215 per 1M in/out tokens** (≤32K-token requests) ⚠️
   ([qwen3-vl-flash model page](https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-flash)).
   Estimated one-time cost for all 665k videos: **≈ $29 (1 frame @640×360) → ≈ $39 (4 frames) →
   ≈ $52 (8 frames)** *(est.)*; Singapore region is ~2×; `qwen3-vl-plus` is ~7×.
3. **Frame sampling is worth it.** A still is strong for subject/setting/colour/lighting but blind
   to motion — CLIP4Clip: "one single image is far from enough", big gain 1→6 frames, 12 used
   ([CLIP4Clip §4.4](https://arxiv.org/abs/2104.08860)). For 5–60 s clips, **4–8 uniformly (or
   scene-change) sampled frames at 640×360** is a good token/cost point; the preview download is
   bandwidth (~1–2 TB total) + CPU, no storage.
4. **Prompt for per-facet short captions, not one long paragraph.** Retrieval evidence: short,
   clear captions containing single actions/colour attributes get *higher* recall while long
   fine-grained scene descriptions stay hard ([Performance Plateau, 2026](https://arxiv.org/abs/2605.00826));
   generic MLLM captions are "generic and indistinguishable across visually similar videos"
   ([CaRe-DPO, 2025](https://arxiv.org/abs/2509.16560)); captions help via **score fusion +
   augmentation**, not by replacing the visual match ([Cap4Video](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Cap4Video_What_Can_Auxiliary_Captions_Do_for_Text-Video_Retrieval_CVPR_2023_paper.pdf),
   +2.1 R@1 MSR-VTT vs 49.3 baseline per the domain note).
5. **Usage:** embed each facet separately (multi-vector) + one digest vector; fuse with RRF like the
   rest of the hybrid. Feed facet text into BM25 as a second, lower-weighted field — but keep the
   tags-only baseline: generative captions "do not consistently enhance retrieval accuracy"
   ([Performance Plateau](https://arxiv.org/abs/2605.00826)) and expansions can hurt strong
   retrievers ([EACL 2024](https://aclanthology.org/2024.findings-eacl.134)).
6. **QC:** JSON-validate + retry; "describe only what is visible" instruction; motion must be
   `null`/`unknown` on single frames — motion hallucination and sequential over-inference are the
   worst failure modes of video LLMs ([MoHallBench](https://arxiv.org/abs/2607.01117),
   [SEASON](https://arxiv.org/abs/2512.04643)); store raw JSON + prompt/model version for
   re-captioning.

---

## VLM options & pricing

All prices ⚠️ per **1M tokens**, standard/real-time tier, captured 2026-10-08. "Image pricing
model" = how pixels become tokens (that formula, not the headline text price, determines cost).

### Comparison table

| Model (access) | Input / output per 1M | Image pricing model | Max images / request | Structured output | Rate limits (region) | Notes |
|---|---|---|---|---|---|---|
| `qwen3-vl-flash` (DashScope Beijing) | **$0.022 / $0.215** ≤32K req; $0.043/$0.43 ≤128K ⚠️ ([page](https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-flash)) | 32×32 px per token: `tokens = h̄·w̄/1024 + 2`; default cap 2,621,440 px (~2,562 tok) ([vision doc](https://www.alibabacloud.com/help/en/model-studio/vision)) | 256 (URL/local), 250 base64; video-as-image-list 4–2,000 | `json_object` only, multimodal `json_schema` falls back ([structured-output doc](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)) | 3,000 RPM / 5M TPM | cache read $0.002/1M; batch listed at 50% on the price page but model page says "Batch Inference: Unsupported" — verify |
| `qwen3-vl-flash` (Singapore) | $0.050 / $0.400 ⚠️ (same page) | same | same | same | 1,200 RPM / 1M TPM | same model, 2×+ price |
| `qwen3-vl-plus` (DashScope Beijing) | **$0.144 / $1.434** ≤32K ⚠️ ([page](https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-plus)) | same | same | same | 3,000 RPM / 5M TPM | Batch **supported** in Beijing (50%: $0.072/$0.717); strongest Qwen VL API tier |
| `qwen3-vl-plus` (Singapore) | $0.200 / $1.600 ⚠️ (same page) | same | same | same | 1,200 RPM / 1M TPM | |
| Qwen3-VL-8B / 32B / 30B-A3B (SiliconFlow intl) | $0.18/$0.68 · $0.20/$0.60 · $0.29/$1.00 ⚠️ ([8B](https://www.siliconflow.com/models/qwen3-vl-8b-instruct), [32B](https://www.siliconflow.com/models/qwen3-vl-32b-instruct), [30B-A3B](https://www.siliconflow.com/models/qwen3-vl-30b-a3b-instruct)) | not documented on the page — assume Qwen 32-px tokenizer | n/a on page | not documented | not documented | China-accessible alternative; verify RPM/TPM + image-token rules with SiliconFlow |
| Gemini 3.5 Flash-Lite (Google) | **$0.30 / $2.50** (batch $0.15/$1.25) ⚠️ ([pricing](https://ai.google.dev/gemini-api/docs/pricing)) | 258 tok per ≤384 px image or per 768×768 tile ([image doc](https://ai.google.dev/gemini-api/docs/image-understanding)) | **3,600 image files** | JSON schema via `response_format` (documented example) | spend caps $10/$50/$200 per 10 min by tier; batch 100 concurrent jobs, tier caps ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)) | `media_resolution` knob (Gemini 3) trades tokens/quality |
| Gemini 3.1 Flash-Lite (Google) | $0.25 / $1.50 (batch $0.125/$0.75) ⚠️ | same | 3,600 | same | same | older/cheaper Flash-Lite |
| Gemini 3.8 Flash (Google) | $0.75 / $3.75 (batch $0.375/$1.875) ⚠️ | same | 3,600 | same | same | quality tier |
| `gpt-5.6-luna` (OpenAI) | **$0.20 / $1.20** (batch 50%) ⚠️ ([pricing](https://platform.openai.com/docs/pricing)) | 32×32 patches ×1.2 multiplier, 2,500-patch high cap ([vision doc](https://platform.openai.com/docs/guides/images-vision)) | **1,500 images**, 512 MB payload | `json_schema` strict ([structured outputs](https://platform.openai.com/docs/guides/structured-outputs)) | Build 5k RPM/2M TPM; Launch 10k/10M; Grow 30k/180M ([rate limits](https://platform.openai.com/docs/guides/rate-limits)) | vision listed in sizing table; cheapest vision-capable GPT-5.6 tier |
| `gpt-4.1-mini` (OpenAI) | $0.40 / $1.60 (batch $0.20/$0.80) ⚠️ | patch ×1.62 multiplier, 6,144-patch budget | 1,500 | `json_schema` strict | Astra/Sol/Terra pools | widely used baseline |
| `gpt-4o` (OpenAI) | $2.50 / $10.00 ⚠️ | tile-based: 85 base + 170/tile ("high") | 1,500 | `json_schema` strict | — | legacy, expensive per image at high detail |
| Claude Sonnet 5 / Haiku 4.5 (Anthropic) | $2/$10 · $1/$5 ⚠️ | per-image tokens; formula not verified here | not verified | tool-use/JSON | not verified | **docs unreachable from this box** (direct + proxy timeouts); prices captured from the docs search index ([platform.claude.com pricing](https://platform.claude.com/docs/en/about-claude/pricing)) — treat as unverified |
| `qwen3-omni-30b-a3b-captioner` (DashScope) | CNY 15.8/12.7 Beijing ⚠️ ([price page](https://help.aliyun.com/en/model-studio/model-pricing)) | — | — | unsupported ([model page](https://www.alibabacloud.com/help/en/model-studio/qwen3-omni-30b-a3b-captioner)) | 60 RPM/100K TPM | **audio-only** captioner (promptless); *not usable for frames* — listed to avoid confusion |

Practical reads:

- **Qwen3-VL on DashScope Beijing is ≈5–13× cheaper than the Western APIs for this workload**, and
  the only shortlisted option operated without special egress from mainland China. Region matters:
  Singapore is 2.3× the Beijing input price and 5× lower TPM.
- **Token accounting decides cost, and the expensive term differs by model.** At flash prices a
  1-frame pass is *output-dominated* (JSON ≈ 150 out-tokens cost $21/corpus vs $8–18 for input
  images); at 8 frames and on plus/OpenAI/Gemini tiers, **input/image tokens dominate**. So frame
  resolution is a first-order knob. Roughly, at these sizes (DashScope tokenizer): 640×360 ≈ 227
  tok · 960×540 ≈ 508 · 1280×720 ≈ 902 · 1440×810 ≈ 1,141. At OpenAI: ×1.2 patches
  (gpt-5.6-luna) ≈ 288 / 612 / 1,104 / 1,404. Gemini bills ≈ 1.5k tokens for all of these (tile
  rule is nearly flat); that makes Gemini's per-video cost much less sensitive to resolution — and
  much higher for multi-frame.
- DashScope bills image tokens inside the input count; a documented example shows a 486-input-token
  call with 418 of them image tokens ([structured-output doc](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)).
- **Keep `enable_thinking=false`** (default for Qwen3-VL) — thinking tokens are billed output and
  add latency; `thinking_budget` up to 81,920 exists if ever needed
  ([vision doc](https://www.alibabacloud.com/help/en/model-studio/vision)).
- **Context caching:** explicit-cache read $0.002/1M for flash — only useful if the fixed prompt
  prefix is long; our prompt is short, so low priority.
- **Batch vs real-time:** DashScope says batch calls are "not subject to real-time rate limiting";
  50% discount where supported (plus, Beijing). OpenAI and Gemini batch = 50% off. This is the
  right lever for a 665k backfill.

---

## Prompt design for multi-facet captions

### What the evidence says

- **Short, clear, single-attribute captions retrieve better than rich paragraphs.** An empirical
  study of 14 TVR systems across 3 datasets: "short, clear, and simple captions, such as those
  describing single actions or colour attributes, achieve higher recall, while complex events,
  multi-step activities, or fine-grained scene descriptions remain challenging for all existing
  models" ([arXiv:2605.00826](https://arxiv.org/abs/2605.00826)). → prefer **one short sentence per
  facet** over one long caption.
- **Generic captions destroy discriminative power for retrieval.** Zero-shot MLLM captions "tend to
  be generic and indistinguishable across visually similar videos"; captions should be optimized
  for retrieval relevance, not BLEU ([CaRe-DPO, arXiv:2509.16560](https://arxiv.org/abs/2509.16560)).
  → ask for concrete, visible specifics (colours, shot size, materials, weather cues), not
  "beautiful footage".
- **Captions help via multiple channels, and fusion is one of them.** Cap4Video uses generated
  captions as (i) extra training pairs, (ii) feature interaction, (iii) a **complementary
  query–caption matching score**; MSR-VTT R@1 51.4 vs 49.3 for the same visual backbone
  ([Cap4Video, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Cap4Video_What_Can_Auxiliary_Captions_Do_for_Text-Video_Retrieval_CVPR_2023_paper.pdf),
  per the [domain note](footage-retrieval-domain-2026-10-08.md)).
- **Frame-level captions + filtering beat one caption.** NarVid adds query-aware adaptive filtering
  "to suppress irrelevant or incorrect information" and a dual query–video / query–narration score
  ([NarVid, arXiv:2503.05186](https://arxiv.org/abs/2503.05186)). → keep per-frame notes internally
  or a per-video merged caption, but never let a wrong frame caption outvote the visual index.
- **Structured attribute indexing works as an architecture.** MAVIS parses videos into a
  "Structured Semantic Library" and retrieves via attribute-level indexing + planner decomposition,
  competitive without task-specific fine-tuning
  ([MAVIS, arXiv:2606.09641](https://arxiv.org/abs/2606.09641)). Facet/attribute text is a
  first-class retrieval surface — consistent with the domain note's faceted-query plan.
- **Attribute-level video search is a real user need**: Composed Video Retrieval (modify an existing
  visual with text, e.g. a lighting/camera change) is an active benchmark and detailed language
  descriptions + discriminative embeddings give up to ~7% R@1 gains
  ([CoVR, arXiv:2403.16997](https://arxiv.org/abs/2403.16997)).
- **Schema adherence is a solved API feature**, not a prompt trick: OpenAI `json_schema` strict mode
  starting with gpt-4o-mini ([structured outputs](https://platform.openai.com/docs/guides/structured-outputs)),
  Gemini schema in `response_format` ([image doc](https://ai.google.dev/gemini-api/docs/image-understanding)).
  **DashScope caveat:** with image/video input, `json_schema` "does not support … and will
  automatically fall back to `json_object`", and `json_object` requires the word "JSON" in the
  prompt ([structured-output doc](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)).
  So for Qwen3-VL: put the schema in the prompt + validate locally + retry.

### Concrete prompt skeleton (Qwen3-VL / DashScope-compatible)

Use **one user message** with frame(s) + text (DashScope recommends avoiding system messages for
Qwen3-VL general use — [vision doc](https://www.alibabacloud.com/help/en/model-studio/vision)),
`response_format={"type":"json_object"}`, `temperature` low/0.

```text
You are labeling stock footage for a video search engine. The input is [N frames sampled
uniformly from one short clip / one thumbnail frame]. Describe ONLY what is visibly
verifiable. Never guess sounds, off-screen context, or identity. If a field cannot be
determined from the frame(s), use null. Output one JSON object with exactly these fields;
keep each field short (one phrase, max ~12 words):

{
  "subject":      "main objects/entities + count, e.g. 'two hikers'",
  "action":       "visible activity, e.g. 'walking uphill'",
  "setting":      "indoor/outdoor + location type, e.g. 'coastal cliff at sunset'",
  "camera": {
    "shot_size":  "wide|medium|close-up|extreme close-up|macro",
    "angle":      "eye-level|low|high|top-down|aerial",
    "movement":   "static|pan|tilt|zoom|tracking|handheld|drone|null if single frame"
  },
  "lighting":     "e.g. 'golden hour backlight', 'soft overcast', 'neon artificial'",
  "time_of_day":  "dawn|morning|day|afternoon|dusk|night|null",
  "colors":       ["dominant colours, 2-4 items"],
  "mood":         "e.g. 'calm', 'dramatic', 'energetic'",
  "people":       {"present": true|false, "count": "approx or null",
                   "attire_or_activity": "e.g. 'business attire'|null"},
  "style":        "e.g. 'cinematic', 'documentary', 'archival', 'graphic'",
  "text_in_frame":"any readable on-screen text, or null",
  "weather":      "clear|rain|snow|fog|wind|null",
  "motion_note":  "for multi-frame input: what changed between frames; for one frame: null",
  "uncertain":    ["fields you had to guess; [] if none"]
}
Return JSON only, no markdown fences.
```

Notes and knobs:

- **Field set maps to the domain note's facet list** (subject/motion/camera/setting/lighting/
  time-of-day/mood) plus colour/style/people/text, i.e. exactly the facets a stock search UI filters
  on ([domain note](footage-retrieval-domain-2026-10-08.md); Pexels' own tagging guide uses
  subject + descriptive + abstract "story" tags —
  [Pexels](https://www.pexels.com/blog/photography/photo-video-tagging-guidelines)).
- **`motion_note` is the one temporal field.** With 4–8 frames a VLM can report *coarse* change
  (subject moves, camera pans); it should not be trusted for absolute speed. Keep "slow motion /
  timelapse / aerial" primarily as tag/enum + `duration`/`fps` metadata (domain-note motion gap),
  and use `motion_note` as a secondary signal.
- **Ask for ~120–200 output tokens.** Long JSON = output-token cost (which dominates the sentence
  budget at flash prices) and more chances to hallucinate.
- **Few-shot 1 example** helps key stability in `json_object` mode; DashScope explicitly recommends
  describing fields + examples and warns JSON-Object mode "does not guarantee stable key names"
  ([structured-output doc](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)).
  If key stability matters more than cost, use a strict-schema provider (OpenAI/Gemini) for a
  sample and normalize.
- **Per-facet wording must be discriminative** (CaRe-DPO); e.g. instead of "nature", ask "pine
  forest, snow on ground, blue-grey palette".
- If per-frame captions are wanted (NarVid-style), prompt for a `frames` array with one short
  caption + timestamp per frame and let a local merge produce the final facets; do not embed the
  merged JSON raw until the A/B says it helps.

---

## Frame sampling

### Evidence

| Question | Finding | Source |
|---|---|---|
| Is one image enough? | "It is required for video to actually model with a sequence of multiple frames instead of one single frame"; significant gain 1→6 frames; 12 frames used | [CLIP4Clip Fig. 3(d), §4.4](https://arxiv.org/abs/2104.08860) |
| First frame vs middle vs uniform? | MSR-VTT R@1: Head **42.3**, Tail **40.5**, Uniform **42.6**; MSVD/LSMDC differences ≤0.5 | [CLIP4Clip Table 6](https://arxiv.org/abs/2104.08860) |
| Does smarter frame selection beat uniform? | Token/frame selection transformer sets records on MSRVTT/VATEX/LSMDC/ActivityNet/DiDeMo | [TS2-Net, arXiv:2207.07852](https://arxiv.org/abs/2207.07852) |
| What makes a sampled set good? | Not precision alone: **scene coverage and sampling balance** also drive QA accuracy; new sampling-quality metric | [KFS-Bench, arXiv:2512.14017](https://arxiv.org/abs/2512.14017) |
| Can a few well-chosen frames replace many? | FOCUS selects query-relevant frames under a token budget using <2% of frames on long-video QA | [FOCUS, arXiv:2510.27280](https://arxiv.org/abs/2510.27280) |
| Beyond uniform: subset-aware selection | Greedy marginal-utility selection (relevance + coverage − redundancy) beats isolated scoring on 6 benchmarks | [MarKey, arXiv:2609.15408](https://arxiv.org/abs/2609.15408) |
| Shot-aware sampling | Partition into shots, take 2 complementary keyframes/shot; improves anomaly hit rate + downstream accuracy | [InfoShot, arXiv:2603.17374](https://arxiv.org/abs/2603.17374) |
| Video-LLM frame counts in the wild | Fixed 8-frame models suffer "excessively sparse information density" on longer videos; per-model counts 8–384 | [Video-MME, arXiv:2405.21075](https://arxiv.org/abs/2405.21075) |
| Thumbnail representativeness | Not tested here; Pexels thumbnails are usually a single frame (often the first) — domain note flags this as an unverified risk | [domain note](footage-retrieval-domain-2026-10-08.md) |

### Recommendation for 5–60 s Pexels clips

- **Tier 0 (thumbnail only):** 0 extra bandwidth, reuses URLs already in the plan; blind to motion,
  and first-frame bias is untested. Good as the absolute-cheapest floor, not as the target.
- **Tier 1 (recommended): 4–6 frames** at ~1 fps (or 6 evenly spaced timestamps) from the SD
  preview, downscaled to **640×360**. Uniform slightly beat Head/Tail in CLIP4Clip, but the delta
  was ~0.3 R@1 — the bigger win is "some sequence instead of one frame"; 4–6 is the elbow.
- **Tier 2: 8 frames** for durations ≥30 s (or if the eval shows short-clip recall benefits).
  InfoShot-style shot splitting is a cheap upgrade (`ffmpeg` scene detection) and fixes the
  "unrepresentative first frame" problem without an ML keyframe model.
- **Do not** go above ~8–12 frames at 640×360 for this corpus: token cost is linear in frames,
  motion understanding saturates (MoHallBench), and clip durations are short.
- **Throw the frames away after captioning** (task brief says bandwidth-only, nothing stored
  long-term). Keep the timestamps + frame hashes in the caption record for audit/replay.

### Frame extraction cost (no money, only scanning)

- Download: 665k × 1–3 MB typical preview ≈ **0.7–2.0 TB**; at 100 Mbps ≈ 15–44 h, at 500 Mbps
  ≈ 3–9 h *(est.)*. Proxy `localhost:7897` egress applies (repo AGENTS).
- Decode/sample: h264 640p software decode ≈ 20–40× realtime per core; 665k clips × ~10–15 s mean
  ≈ 1,850–2,770 video-hours → **~50–140 core-hours** → one overnight run on 8 cores *(est.)*.
- This is the cheapest part of the pipeline; the VLM bill is the real spend.

---

## Cost & throughput @665k

### Assumptions (unchanged across rows)

- 665k videos; **prompt ≈ 300 text tokens**, **JSON output ≈ 150 tokens** per video *(est.)*.
- Image token counts from each vendor's formula at 640×360 / 1280×720 (see table above).
- Prices as of 2026-10-08 ⚠️; no batch discount in the first column, batch in brackets.
- **These are estimates, not quotes**; measure 100 clips end-to-end before committing.

### One-time captioning cost (whole corpus)

| Provider / model | Frames | Resolution | Input tokens | Output tokens | Cost *(est.)* |
|---|---|---|---|---|---|
| qwen3-vl-flash · Beijing | 1 | 640×360 | 0.35B | 0.10B | **≈ $7.7 + $21.4 = $29** |
| qwen3-vl-flash · Beijing | 1 | 1280×720 | 0.80B | 0.10B | ≈ $17.6 + $21.4 = $39 |
| qwen3-vl-flash · Beijing | 4 | 640×360 | 0.80B | 0.10B | ≈ $17.7 + $21.4 = $39 |
| qwen3-vl-flash · Beijing | 8 | 640×360 | 1.41B | 0.10B | **≈ $31 + $21.4 = $52** |
| qwen3-vl-flash · Beijing | 8 | 1280×720 | 5.00B | 0.10B | ≈ $110 + $21.4 = $131 |
| qwen3-vl-flash · Singapore | 8 | 640×360 | 1.41B | 0.10B | ≈ $70 + $40 = $110 |
| qwen3-vl-plus · Beijing | 8 | 640×360 | 1.41B | 0.10B | ≈ $203 + $143 = **$346** (batch ≈ $173) |
| qwen3-vl-plus · Singapore | 8 | 1280×720 | 5.00B | 0.10B | ≈ $1,000 + $160 = $1,160 |
| Qwen3-VL-32B · SiliconFlow | 8 | 640×360 | 1.41B | 0.10B | ≈ $281 + $60 = $341 (tokenizer unpriced/undocumented) |
| Gemini 3.5 Flash-Lite | 1 | 1280×720 | 1.23B | 0.10B | ≈ $369 + $249 = $618 (batch ≈ $309) |
| Gemini 3.5 Flash-Lite | 8 | 1280×720 | 8.43B | 0.10B | ≈ $2,530 + $249 = $2,780 (batch ≈ $1,390) |
| gpt-5.6-luna | 1 | 1280×720 | 0.93B | 0.10B | ≈ $187 + $120 = $307 (batch ≈ $153) |
| gpt-5.6-luna | 8 | 640×360 | 1.73B | 0.10B | ≈ $346 + $120 = $466 (batch ≈ $233) |
| gpt-4.1-mini | 1 | 1280×720 | 1.19B | 0.10B | ≈ $476 + $160 = $636 (batch ≈ $318) |

Then **embed the resulting captions** (per-facet + digest ≈ 150–250 tokens/video → ~0.1–0.17B
tokens): ≈ **$7–12** one-time at `text-embedding-v4` $0.07/1M ⚠️
([dense-embedding note](dense-embedding-models-2026-10-08.md)). Storage for per-facet vectors
(≈10 facets × 768d × 4B ≈ 30 KB/video) ≈ **20 GB** for 665k — size the pgvector table accordingly.

### Throughput feasibility (real-time, no batch)

Theoretical minimum = max(total_tokens / TPM, requests / RPM), using published limits above:

| Config | Binding limit | Min wall time | Realistic |
|---|---|---|---|
| flash BJ, 1 frame @640 | RPM 3,000 (450M tok / 5M TPM = 90 min) | **3.7 h** | 1–2 days with pacing/retries |
| flash BJ, 4 frames @640 | TPM (0.90B/5M = 180 min) vs RPM 222 min | **3.7 h** | 1–2 days |
| flash BJ, 8 frames @640 | TPM (1.51B/5M ≈ 300 min) | **5 h** | 1–2 days |
| flash SGP, 8 frames @640 | TPM (1.51B/1M ≈ 25 h) | **25 h** | 2–4 days |
| qwen3-vl-plus BJ, 8 frames @720 (batch) | batch not rate-limited; queue unknown | — | days, but cheapest per-quality |

DashScope's account-level limiter also enforces RPS = RPM/60 smoothing and returns 429 on short
bursts ("Request rate increased too quickly"), and batches are exempt where supported
([rate-limit doc](https://www.alibabacloud.com/help/en/model-studio/rate-limit)). Gemini Tier-1
spend caps ($10 per 10 min) and batch enqueue caps (Flash-Lite 10M tokens at Tier 1 → 500M at
Tier 2) make a 665k Gemini backfill a multi-day, tier-dependent job unless run in Batch and
chunked ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).

**Cheap vs rich tiers (recommendation):** run the **cheap tier first** — `qwen3-vl-flash` Beijing,
**8 frames @640×360 ≈ $52** (or 4 frames ≈ $39 if the eval says frames beyond 4 don't help) +
embedding ~$10. Reserve the **rich tier** (`qwen3-vl-plus`, or an OpenAI/Gemini model) for (a) a
5–10k-row quality A/B, and (b) re-captioning only rows whose facet fields came back `uncertain` or
whose captions lose the eval. A full rich-tier pass is 5–20× the cheap tier for an unproven gain.

---

## Caption usage (embed / sparse)

**Recommendation: store the raw JSON; embed facets separately *and* one digest; feed facet text to
the sparse arm as a weighted extra field; keep the tags-only baseline in the eval.**

1. **Per-facet vectors (multi-vector).** One embedding row per `(video_id, facet)` with a `field`
   column; query matched per facet and fused (e.g. facet-level max or weighted sum, then RRF with
   tag/BM25 arms). Precedent: multi-vector/late-interaction retrieval is a well-established
   architecture (ColBERT, [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)), and Cap4Video's
   complementary query–caption score branch is exactly this idea inside one model
   ([Cap4Video](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Cap4Video_What_Can_Auxiliary_Captions_Do_for_Text-Video_Retrieval_CVPR_2023_paper.pdf)).
   It also lets you *weight* facets (e.g. subject/setting > mood) and inspect failures per facet.
2. **One digest vector** (concatenated short facets, `"subject: … | setting: … | lighting: …"`) as
   the fallback that reuses the existing single-`vector(768)` seam
   ([`docs/phase2-embedding-seam.md`](../phase2-embedding-seam.md)); keep it even if facets are
   indexed, because some queries span facets and the digest is the cheapest thing to A/B.
3. **Do not embed the pretty-printed JSON as-is** unless the A/B demands it: braces, key names and
   `null`s are noise tokens, and the embedding model's instruction format expects natural language
   (dense-embedding note §3).
4. **Sparse arm:** add caption facets as a second field (BM25F-style weighting) or a separate
   index row set, at lower weight than title+tags. Rationale: tags already carry the bulk of the
   keyword signal (median 47/row — domain note), but captions add vocabulary tags lack ("golden
   hour", "top-down", "backlit", "confetti"). Counter-evidence to respect: generative captions
   "do not consistently enhance retrieval accuracy"
   ([Plateau](https://arxiv.org/abs/2605.00826)), and LLM expansion helps weak-but-hurts-strong
   retrievers ([EACL 2024](https://aclanthology.org/2024.findings-eacl.134)) — so this is an A/B
   item, not a default.
5. **Evaluation before rollout** (reuse the domain note's plan): ~200 queries, per-facet, pool
   BM25 + dense-text + digest + facet vectors + thumbnail vectors; score nDCG@10 / R@1 with an
   LLM judge **different from the captioner** (circularity warning in the domain note,
   [Clarke & Dietz](https://arxiv.org/html/2412.17156)). Gate: adopt captions only where the
   facet-specific cuts improve; keep motion queries on the metadata/tag path.
6. **Cost recap:** caption embedding is ~$10 and ~20 GB; per-facet storage and one more table is
   the main engineering cost, not money. If storage is tight, index only 4–6 facets
   (subject/action/setting/lighting/colour/mood) — those are the retrieval-heavy ones.

---

## Risks & QC

- **Motion misattribution is the sharpest risk.** Video LLMs over-infer human motions absent from
  the video; "sequential inference hallucination" (inferring expected outcomes from partial cues)
  is the most severe mode, and action-recognition skill does **not** predict hallucination
  resistance ([MoHallBench, arXiv:2607.01117](https://arxiv.org/abs/2607.01117)). Mitigations:
  single-frame → `motion_note=null`; multi-frame → ask strictly for observed change; cross-check
  "slow motion/timelapse" against Pexels tags + `fps`/`duration`; never let `motion_note` feed a
  hard filter.
- **Object/attribute hallucination** is driven by co-occurrence priors and even by the instruction
  wording; standard caption-quality metrics do not measure it
  ([POPE, arXiv:2305.10355](https://arxiv.org/abs/2305.10355);
  [Rohrbach et al., arXiv:1809.02156](https://arxiv.org/abs/1809.02156)). Mitigations: "describe
  only what is visible", permit `null`, require an `uncertain` list, and spot-audit a few hundred
  rows with a *different* model + human reads.
- **Temporal consistency across frames** is still weak in VideoLLMs; expect occasional
  inconsistent/implausible event descriptions
  ([SEASON, arXiv:2512.04643](https://arxiv.org/abs/2512.04643)). Keep frame timestamps in the
  stored record so inconsistent captions can be re-derived.
- **JSON reliability:** DashScope multimodal is `json_object` only (schema falls back), and JSON
  mode "does not guarantee stable key names"
  ([DashScope](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)). Pipeline:
  `json.loads` → validate against a local schema (pydantic/jsonschema) → on failure, re-ask with the
  error; log raw completions in the spool (repo convention: raw first, then ingest).
- **Prompt/version provenance:** store per row `prompt_version`, `model_id` (+ region), `temperature`,
  `frames_spec` (timestamps/fps), `image_hash`, `caption_json`, `raw_response_ref`, `created_at`.
  Idempotent key `(video_id, prompt_version, model_id)` makes re-captioning a targeted backfill
  (the embedding seam's incremental design generalizes to captions).
- **Safety refusals:** some frames will trip vendor safety filters or return empty JSON; classify
  refusals separately from parse errors and re-route only those rows to a different provider.
- **Judge circularity:** do not evaluate with the captioning model (domain note's warning), and
  remember that LLM judges inflate/overtrust their own generations.
- **Price drift:** every price above is ⚠️; re-verify the console/pricing page the day the backfill
  runs, and set a DashScope consumption limit + fee alert before launching
  ([rate-limit doc](https://www.alibabacloud.com/help/en/model-studio/rate-limit)).
- **Language:** keep caption text in English (matches the corpus; Chinese queries are handled
  cross-lingually by `text-embedding-v4` — dense-embedding note). Do not caption in Chinese and
  English in one pass; A/B later if needed.

---

## Open questions & uncertainty

1. **Marginal value over rich tags.** With median 47 tags/row, facet captions may be largely
   redundant; the Plateau paper's "generative captions do not consistently enhance retrieval" is a
   real warning. The qualifying experiment is 5–10k rows × 200 queries, captions vs tags-only,
   before the full pass.
2. **Best frame count for 5–60 s clips** is extrapolated from CLIP4Clip (12 frames, MSR-VTT) and
   long-video samplers; no in-domain evidence. A/B 1 vs 4 vs 8 on 1–2k clips.
3. **Thumbnail representativeness** (first frame?) remains unknown; extracting from the SD preview
   sidesteps it, but at download cost.
4. **DashScope batch support for `qwen3-vl-flash`** is contradictory between the pricing page
   (50% batch) and the model page ("Batch Inference: Unsupported"); verify in console. plus is
   confirmed batch-capable in Beijing.
5. **Anthropic reachability** from this environment failed (docs and site timeouts, direct and via
   proxy), so Claude pricing/limits here are unverified; also confirm whether it is usable at all
   from the target network before considering it.
6. **SiliconFlow Qwen3-VL** image-token accounting, RPM/TPM, and batch options are not documented
   on the pages I could reach — verify directly if it is to be a fallback.
7. **Facet fusion weights** (per-facet vector vs digest vs tags vs BM25) and whether per-facet
   embeddings beat one digest at all for this tag-heavy corpus.
8. **Motion fields:** will `motion_note` from 8 frames beat (or add anything to) tags + duration +
   fps for "slow motion/timelapse/aerial" queries? MoHallBench suggests low expectations.
9. **`qwen3-vl-flash` snapshot instability:** the Jan-2026 snapshot shows "Structured Outputs:
   Unsupported" while the stable alias is "Supported" — pin the stable alias and re-verify at
   run time.
10. **Corpus scale drift:** 365k ingested today vs 665k target; cost estimates scale linearly, and
    incremental captioning should key on `video_id` so the second half is just more rows.

---

### Related notes

- [footage-retrieval-domain-2026-10-08.md](footage-retrieval-domain-2026-10-08.md) — domain,
  Cap4Video numbers, motion gap, eval plan.
- [dense-embedding-models-2026-10-08.md](dense-embedding-models-2026-10-08.md) — embedder choice
  (`text-embedding-v4` @768) and multimodal upgrade path (`qwen3-vl-embedding`).
- [`docs/phase2-embedding-seam.md`](../phase2-embedding-seam.md) — incremental, resumable backfill
  design the captioning pass should mirror.
