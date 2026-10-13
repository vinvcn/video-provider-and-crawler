# Footage facet taxonomy — what stock platforms filter on, what cinematography gives us as controlled vocabulary, and what a single frame can actually tell you

Date: 2026-10-08 · Scope: the **facet taxonomy** for the Phase-2 stock-video search backend
(~388k Pexels video metadata rows in `stock_videos`; `title`, 1-sentence `description`,
`tags text[]` median 47 / p90 55; one thumbnail per row; **no video files**; no GPU).

This note builds on, and does not repeat, three sibling notes:

- `docs/research/footage-retrieval-domain-2026-10-08.md` — platform search mechanics, the motion
  gap, the eval plan ("the domain note" below);
- `docs/research/dense-embedding-models-2026-10-08.md` — model choice ("the dense note");
- `docs/research/sparse-hybrid-rerank-2026-10-08.md` — BM25/SPLADE/RRF/rerank ("the sparse note").

Every non-obvious claim below links to a primary source (vendor docs, standards, university film
references, dataset papers). **Our own tag-coverage numbers are measured against this repo's DB**
(method and caveats in [§5](#5-measured-tag-coverage-in-our-corpus-method)).

---

## TL;DR

1. **Platforms converge on two facet families, and they are not the ones a VLM can invent.**
   (a) *hard technical/compliance facets* — orientation, aspect ratio, duration, fps, resolution,
   licence/releases — which every platform exposes as filters and we already store as columns;
   (b) *semantic facets* — subject, setting, activity, people — which every platform pushes into
   keywords/tags, **not** into filters. Cinematography facets (shot scale, lighting, movement,
   composition) are filters at only a handful of platforms (Getty `compositions`,
   Artgrid "shot type", Storyblocks "Aerial / Slow Motion / Time Lapse", Pond5 "Aerial / 360VR /
   Loopable"), which is evidence of demand, not of coverage.

2. **Only part of the cinematography vocabulary is inferable from one still frame.** The
   MovieShots study (46K shots, 7.8K trailers, professional annotators) reports **84–88 % accuracy
   for shot *scale* from a single image**, but only **70–71 % for shot *movement* from image-only**,
   rising to **80–84 % when optical flow / a temporal variance map over ≥8 clips is added**
   ([ECCV 2020, arXiv:2008.03548](https://arxiv.org/html/2008.03548v1) §3.1, Table 3). Our
   thumbnails are single frames, so: **scale yes, movement no**. This matches the domain note's
   `CLIP4Clip` argument and Shutterstock's choice to build video reverse-search on the *preview
   image* alone.

3. **Our `tags` already cover subject / setting / season / daytime well, and lighting /
   composition / mood only thinly.** On a 5 % sample (n = 19,341 videos): outdoor ≈ 43 %, nature ≈ 41 %,
   people ≈ 40 %, urban ≈ 20 %, aerial/drone ≈ 17.5 %, close-up ≈ 15.6 %, but **golden hour ≈ 1.9 %,
   backlit ≈ 0.5 %, low-key ≈ 0.15 %, negative space ≈ 0.4 %, leading lines ≈ 0.02 %, neon ≈ 1.7 %,
   studio lighting ≈ 0.7 %**. That thin band is exactly where one VLM caption pass earns its cost.

4. **Motion words live in prose, not tags.** Slow-motion appears in a *tag* on ≈ 3.1 % of rows and
   time-lapse ≈ 1.6 %, but slow-motion/time-lapse/aerial/drone/hyperlapse words appear in
   `title`+`description` on **17.4 %**. Rule-based parsing of prose + `duration` is a cheaper and
   more complete motion facet than captioning or embedding.

5. **Pexels' own guidance already encodes 7 facets in its 10–25 tags** — subject, action, emotion,
   setting, time-of-day, weather, and "story" — but says nothing about lighting, colour, composition
   or shot scale ([Pexels tagging guidelines](https://www.pexels.com/blog/photography/photo-video-tagging-guidelines)).
   Our median of 47 tags is ~2× their ceiling; the extras are mostly auto-generated subject/location
   expansions, which is consistent with the domain note's finding that Pexels admits auto-tags
   "aren't always that accurate".

6. **Proposed shape:** keep technical facets as SQL filters (already present), *derive* colour /
   monochrome / depth-of-field proxies from the thumbnail without a VLM, *parse* motion from
   title+description, and **caption ~10 VLM-justified fields** (shot scale, camera angle,
   lighting/time-of-day, setting/environment, weather/season, people bucket + group type + activity,
   mood/emotion, subject/theme, composition, colour grade) into one JSON facet record per row.
   Embed 3–5 *facet strings* separately rather than one blob, per the dense note's "facet-routed
   queries" argument.

---

## 1. Platform taxonomies (primary)

Source key used in the tables below:

| Key | Primary source |
|---|---|
| **[SH]** | Shutterstock public API SDK docs, `VideosApi` (mirrors the official API contract): <https://github.com/shutterstock/public-api-javascript-sdk/blob/master/docs/VideosApi.md> (raw: <https://raw.githubusercontent.com/shutterstock/public-api-javascript-sdk/master/docs/VideosApi.md>) · images variant: <https://raw.githubusercontent.com/shutterstock/public-api-javascript-sdk/master/docs/ImagesApi.md> |
| **[AD]** | Adobe Stock Search API reference: <https://developer.adobe.com/stock/docs/api/11-search-reference> · CategoryTree reference: <https://developer.adobe.com/stock/docs/api/17-categorytree> · video contributor specs (resolution/fps/duration): <https://helpx.adobe.com/stock/contributor/submit-your-content/submit-videos/technical-requirements-for-video-submissions.html> |
| **[GI]** | Getty Images API asset metadata: <https://developers.gettyimages.com/asset-metadata> · release notes (creative video filters, 2024-01-19 and 2023-10-25): <https://developers.gettyimages.com/release-notes> · iStock video search guide: <https://www.istockphoto.com/search-guide/search-for-video> |
| **[AG]** | Artgrid help centre: <https://artgrid.zendesk.com/hc/en-us/articles/8069563020701-How-to-Find-Download-and-Use-Footage> · Artgrid "stock video" guide: <https://artgrid.io/insights/stock-video> |
| **[SB]** | Storyblocks video search filters (live page as fetched 2026-10-08): <https://www.storyblocks.com/video/search> |
| **[P5]** | Pond5 blog, "Using Pond5 Filters": <https://blog.pond5.com/80631-how-to-find-the-perfect-stock-assets> · Pond5 search page filter panel: <https://www.pond5.com/search?kw=ai&media=footage> · colour/composition visual search: <https://blog.pond5.com/22617-new-explore-footage-by-color-or-composition-with-visual-search> |
| **[EN]** | Envato Elements stock-video filters (as fetched): <https://elements.envato.com/stock-video/farm+animals> · Envato search help: <https://help.elements.envato.com/hc/en-us/articles/360000629186-How-to-Search-Browse-Find-Items-on-Envato> · VideoHive category index: <https://videohive.net/category> |
| **[ME]** | MotionElements VisualSearch: <https://www.motionelements.com/visualsearch> · user guide: <https://www.motionelements.com/begin> · help: <https://help.motionelements.com/en/docs/use-visualsearch-for-video-image-templates> |
| **[PX]** | Pexels tagging guidelines: <https://www.pexels.com/blog/photography/photo-video-tagging-guidelines> (see caveat §6) · Pexels search-page filters, observed via a search-engine snapshot of <https://www.pexels.com/search/videos/elements> (Orientation / People / Age / Duration / Frame rate / Resolution / Date) · Pexels metadata help: <https://help.pexels.com/hc/en-us/articles/37285267631769-How-does-Pexels-use-the-metadata-on-my-photos> |

### 1.1 What each platform actually exposes

| Platform | Hard filters (exact enums observed) | Semantic vocabularies | Notes |
|---|---|---|---|
| **Shutterstock** **[SH]** | `orientation` vertical/horizontal; `aspect_ratio` 4_3/16_9/nonstandard; `resolution` 4k/standard_definition/high_definition; `duration_from`/`duration_to` (s); `fps_from`/`fps_to`; `license` commercial/editorial; `category` (Shutterstock-defined, name or ID); `people_age` infants→older (9 buckets); `people_ethnicity` (15 values incl. african_american, east_asian, hispanic, middle_eastern…); `people_gender` male/female/both; `people_number` (integer); `people_model_released`; `added_date*`; `sort` newest/oldest/popular/relevance/random; `library` shutterstock/footage_select; `exact_match` | Every asset carries contributor `keywords` and up to two Shutterstock-assigned **`categories`** (the getVideo example returns `[Nature, People]`); a category catalogue endpoint exists (`total_count` 13 in the doc example — treat the count as example data, not the live enum) | Reverse search lives at `GET /v2/videos/{id}/similar`; the domain note documents image-based video discovery built on the preview frame |
| **Adobe Stock** **[AD]** | `filters[content_type:photo|illustration|vector|video|template|3d]`; `filters[orientation]` horizontal/vertical/square/all; `filters[video_duration]` ≤10 / ≤20 / ≤30 / 30+ / all; `filters[colors]` comma-separated hex; `filters[copy_space]`; `filters[is_loop]`; `filters[transparent]`; `filters[has_releases]`; `filters[premium]`; `filters[offensive:2]`; `filters[isolated:on]`; `filters[panoramic:on]`; `filters[gentech]`; `search_parameters[category]` via CategoryTree; sort relevance/creation/featured/nb_downloads/undiscovered | CategoryTree top levels include **"States of Mind", "Food", "Graphic Resources"** (IDs 261/289/444); sub-trees are 3 deep (Food → Cheese and Dairy → Cheddar). Adobe explicitly warns categories are **contributor-set, not curated**, and that "**searching by keywords instead of categories is considered a best practice**" | Adobe's own docs call attention to the human-error factor of its semantic taxonomy — a direct argument for machine-derived facets with confidence scores |
| **Getty Images / iStock** **[GI]** | Video search: `min_clip_length`/`max_clip_length` (s); `aspect_ratio` (16:9, 9:16, 3:4, 4:3, 4:5, 2:1, 17:9, 9:17); `orientation` horizontal/vertical; `format_available` (e.g. hd_web); `exclude_nudity`; `safe_search`; `exclude_editorial_use_only`; `event_ids` (editorial); **creative video**: `number_of_people` **none/one/two/group**, `ethnicity` (12 values), `compositions` (added **medium_shot, wide_shot, part_of_a_series** in 2024; enum otherwise in Swagger); image-only `moods` **black_and_white/bold/cool/dramatic/natural/vivid/warm** | Per-asset metadata includes `shot_speed` (e.g. real time / time lapse), `color_type` color/black_and_white, `quality_rank` 0–4 (creative: 0 = highest quality; editorial: 0 = most representative, 4 = background/filler), `release_info`, `key_frames` (**six frame URLs per video**), `era` contemporary/archival, `mastered_to`, `originally_shot_on` | iStock's user-facing Refine panel exposes duration, resolution, frame rate, location, number of people, age, ethnicity, upload date, nudity toggle, exact-match toggle. Getty's `compositions` is the only *shot-scale-ish* filter found in any platform's API |
| **Artgrid** **[AG]** | Filter menu: **Video Themes**, **Shot Type**, **People**, **Collections**, framing ("wide"), sort Staff pick/new | Shot Type examples: *close-ups, handheld, interiors, exteriors, 4K clips*. Themes: Sport & Fitness, Medical & Health, Green Planet, abstract, timelapse. Collections: Wildlife, City Life, Technological Devices, Corporate, plus **colour-scheme collections (Blue/Yellow/Green/Red)** | Curated-collection model: the taxonomy is partly editorial, not exhaustive |
| **Storyblocks** **[SB]** | Media Type (Footage/Backgrounds/Templates); Orientation (vertical/horizontal); Resolution HD/4K/8K; Frame Rate **23.98/24, 25, 29.97/30, 50, 59.94/60**; Duration slider; Model Released; Property Released; Usage Rights commercial/editorial | **Categories mix genre and technique**: 360° and VR, Aerial, Animals, Business and Professions, Effects and Lower Thirds, Food and Beverage, Green Screen, Historical and Archival, Holidays and Celebrations, Lifestyle, Medical, Nature, People, Places and Landmarks, **Slow Motion**, Sports and Recreation, Technology, **Time Lapse**, Transportation | The clearest evidence that **movement/technique belongs in the browse taxonomy side-by-side with subject** |
| **Pond5** **[P5]** | Price band; Collections (News & Archival, Subscription, Marquee); **Visually Similar** (by *composition* or *colour*); Duration; FPS 23.97→60+; People (number 0/1/2/3+, gender female/male/both); Artist; Additional: **Aerial, Alpha Channel, Alpha Matte, Contains Audio, Green Screen, Loopable, RED Camera, 360/VR, Free, Public Domain** | Visual Search explicitly offers "search by composition" and "search by colour" as *retrieval modes*, not stored enums | Loop / alpha / audio-present are facets we could not infer from a still but could infer from the SD preview if we ever store it |
| **Envato (Elements)** **[EN]** | Stock Video filters: Categories (Stock Footage / Motion Graphics), Orientation horizontal/vertical, Resolution 720p/1080p/2K/4K, Frame Rate 23.98/24/25/29.97/30/50 fps | Landing-page categories: People, Lifestyle, Sport, Food, Nature, Business; VideoHive category index includes Buildings, Business/Corporate, City, Nature, Sky/Clouds, Space, Sports, Technology, Medical, Retro, Kids, Light, Water | No lighting/mood/shot-scale vocabulary at all |
| **MotionElements** **[ME]** | "Filters" alongside three search modes | Keyword search is explicitly documented as "object **or mood** (e.g. 'happy', 'romantic')"; Category browse; **AI VisualSearch** (drop an image or video → similar style/mood) | The vendor rationale for VisualSearch is exactly the facet gap: "you've been trying to find videos reflecting a certain style or mood but just can't describe it accurately … when required to type keywords" |
| **Pexels (our source)** **[PX]** | Site filters (observed): Orientation any/horizontal/vertical/square; **People 0 / 1 / 2 / 3+**; Age baby/child/teenager/adult/senior; Duration (min 5 s, max 2 m+); **Frame rate 24/25/30/50/60+**; Resolution 1080p/4K/8K; Date last 24 h/week/month/year. API (per the domain note): `query`, `orientation`, `size`, `color`, `locale`, `min_width/min_height`, `min_duration/max_duration` | Contributor + auto tags; guidance = **10–25 tags** in three buckets: *subject* + *descriptive* (action, emotion, setting, time-of-day, weather) + abstract *story* tags; photos get up to 30 tags pre-populated from EXIF/IPTC | Pexels exposes a **frame-rate filter** on site — a facet we can serve from our `fps` column today |

### 1.2 Synthesis: which facet families are actually filterable anywhere

| Facet family | Exposed as a filter by | Our column? |
|---|---|---|
| Orientation / aspect ratio | all eight **[SH][AD][GI][SB][P5][EN][ME][PX]** | `orientation`, `aspect_ratio` ✅ |
| Duration / fps / resolution | all eight | `duration`, `fps`, `width/height` ✅ |
| Licence / releases | [SH][AD][GI][SB][P5] (Pond5 via price/collection) | `license` ✅ (model/property release ✗) |
| People count / age / ethnicity / gender | [SH] (age/ethnicity/gender/number), [GI] (number/ethnicity), [P5] (number/gender), [PX] (0/1/2/3+, age) | ✗ — only implicit in tags |
| Movement / technique (aerial, slow-mo, timelapse, loop, 360/VR) | [SB] (categories), [AG] (shot type), [P5] (Aerial, Loopable, 360/VR), [GI] (`shot_speed`= time lapse), [AD] (`is_loop`) | ✗ (tag/prose only) |
| Shot scale / composition | **[GI]** (`compositions`: medium_shot, wide_shot, part_of_a_series); [P5] composition *as retrieval mode* | ✗ |
| Lighting / mood / colour grade | [GI] `moods` (colour palette of image searches only), [P5] colour *as retrieval mode*, [AD]/[SH] hex-colour filters, [PX] `color` (per the domain note) | ✗ (colour only via tag words) |
| Genre / theme / use-case | all eight (see §1.1) | ✗ |

**Takeaway.** No platform ships a lighting or mood *filter*, and exactly one ships a shot-scale
filter. Creators express those needs through keywords (MotionElements says so explicitly **[ME]**)
or through image search (Pond5/Shutterstock/Adobe/Getty all offer "visually similar" **[P5][SH][AD][GI]**).
For us that means: the caption pass should produce **keyword-like facet text** (retrievable) and a
**small number of new enums worth filtering on** (§4), rather than trying to reproduce 26 platform
categories.

---

## 2. Cinematography taxonomy (controlled-vocabulary candidates)

Primary references used: **Yale Film Analysis** (film-studies teaching site, close to Bordwell's
*Film Art* vocabulary) — <https://filmanalysis.yale.edu/cinematography/> and
<https://filmanalysis.yale.edu/mise-en-scene/>; **Columbia Film Language Glossary**
(<https://filmglossary.ccnmtl.columbia.edu/term/>, curated by Columbia Film faculty);
**MovieShots** (ECCV 2020, 46K annotated shots — <https://arxiv.org/html/2008.03548v1>,
project page <https://movienet.github.io/> via MovieNet); **Getty's controlled vocabularies** via the
Library of Congress **Thesaurus for Graphic Materials II** (genre/format, ~700 terms —
<https://guides.loc.gov/tgm-ii>, term file <https://guides.loc.gov/tgm-ii/download>);
**IPTC Media Topics** (1,085 news-subject terms, 17 top-level —
<https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html>);
**W3C EmotionML vocabularies** (<https://www.w3.org/TR/emotion-voc/>);
**WMO Code Table 4677** (present weather, 100 code figures —
<https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM>);
**Astral** for sun-angle definitions (<https://astral.readthedocs.io/en/latest/>).

### 2.1 Shot scale (size)

**Schemes disagree — pick one and be consistent.** That is itself a finding:

- Yale: *extreme long shot, long shot, medium long shot, medium close-up, close-up, extreme
  close-up* — **no plain "medium shot"** ([Yale, Scale](https://filmanalysis.yale.edu/cinematography/#scale)).
- Columbia: *long shot* (characters in their entirety), *medium shot* (waist up), *close-up* (face fills frame), plus *establishing shot* ([Columbia](https://filmglossary.ccnmtl.columbia.edu/term/)).
- MovieShots (machine-validated, 46K shots): **long shot (LS), full shot (FS), medium shot (MS),
  close-up (CS), extreme close-up (ECS)** — with the explicit note that scales are "established in
  relationship with other frames from the same film" rather than absolute ([arXiv:2008.03548](https://arxiv.org/html/2008.03548v1) §3.1).

→ Proposed controlled values: `extreme_wide | wide | full | medium | medium_close | close | extreme_close`
(and `n/a` when no human figure grounds the scale). Report confidence: MovieShots shows a 2D
image-only model reaching **84.1 % (TSN-ResNet50) / 87.2 % (SGNet, image only)** top-1 for scale.

### 2.2 Camera angle, level, and POV

- Yale: *low angle, high angle* ("angle of framing"), *low level, high level* ("level of framing"),
  *canted framing* (unlevel frame = chaos/instability), *point-of-view shot* (explicitly "relatively
  rare" and defined by editing context — cut before/after a shot of the looker)
  ([Yale, Framing](https://filmanalysis.yale.edu/cinematography/#framing)).
- Columbia: *high-angle shot*, *low-angle shot*, *canted angle (Dutch angle)*, *point of view*
  ([Columbia](https://filmglossary.ccnmtl.columbia.edu/term/)).
- Overhead vocabulary in a library-grade controlled list: TGM II **"Views, Aerial"** and
  **"Views, Bird's-eye"** ([TGM download](https://guides.loc.gov/tgm-ii/download) — searched in the
  downloaded term file).

→ Proposed values: `eye_level | low | high | overhead_top_down | aerial | dutch | pov`.
`pov` should be marked *low confidence* from a still (it is an editing relation).

### 2.3 Camera movement

- Yale: *crane, handheld/Steadicam, pan, tilt, tracking, whip pan, zoom* ([Yale, Movement](https://filmanalysis.yale.edu/cinematography/#movement)).
- Columbia: *aerial shot, crane shot, dolly, handheld shot, pan shot, circular pan, Steadicam shot,
  swish pan, tracking (trucking) shot, virtual camera movement, zoom shot* ([Columbia](https://filmglossary.ccnmtl.columbia.edu/term/)).
- MovieShots' reducible machine vocabulary: **static / motion / push / pull** ([arXiv:2008.03548](https://arxiv.org/html/2008.03548v1) §3.1).

→ Doctrine for our pipeline: store two layers — a coarse machine layer
`static | moving | zoom_in | zoom_out | unknown` (validated at ~80–84 % when temporal evidence is
available) and a **prose-parsed technique layer** (`aerial`, `drone`, `slow_motion`, `timelapse`,
`hyperlapse`, `handheld`, `tracking`) taken from title/description, which is where Pexels puts it
(§5). **The thumbnail alone must not be trusted for movement** (image-only AccM ≈ 70 %).

### 2.4 Lighting

- Yale: *three-point lighting* (the classical scheme: key + fill + backlight), *high-key lighting*
  (bright, few shadows — musicals/comedies), *low-key lighting* (strong contrast, shadows — noir/
  suspense); plus *contrast* and *exposure* as measurable image properties
  ([Yale, Lighting](https://filmanalysis.yale.edu/mise-en-scene/#lighting) ·
  [Yale, Quality](https://filmanalysis.yale.edu/cinematography/#quality)).
- Columbia: *lighting*, *chiaroscuro* ("strong contrasts between light and dark"), *day for night*
  ([Columbia](https://filmglossary.ccnmtl.columbia.edu/term/)).
- Industry machine vocabulary for the same axes appears in Getty's GenAI camera controls:
  `lens_type` ∈ {wide_angle, telephoto}, `depth_of_field` ∈ {shallow, deep}
  ([Getty release notes 2024-07-11](https://developers.gettyimages.com/release-notes)).

→ Proposed values: `key_style` ∈ `high_key | low_key | neutral`; `direction` ∈
`front | side | back | top | mixed`; `source` ∈ `natural | studio | practical | neon | fire | mixed`;
`quality` ∈ `hard | soft`. Note that "three-point studio look" vs. "available-light look" is a
recognizable but partly subjective distinction — store a confidence.

### 2.5 Colour

- Yale treats colour as an aesthetic system (Fellini's reds, Bergman's whites) and defines
  *contrast* as "the ratio of dark to light" ([Yale, Quality](https://filmanalysis.yale.edu/cinematography/#quality)).
- Library-grade machine vocabularies: Getty `color_type` = color / black_and_white; Getty `moods`
  = black_and_white / bold / cool / dramatic / natural / vivid / warm (image search)
  ([Getty asset metadata](https://developers.gettyimages.com/asset-metadata),
  [release notes 2023-10-25](https://developers.gettyimages.com/release-notes)).
- Shutterstock offers a hex-colour filter for images (`color=4F21EA` or `grayscale`), Adobe accepts
  comma-separated hex `filters[colors]` — both treat colour as a *similarity* facet, not a taxonomy
  ([SH](https://raw.githubusercontent.com/shutterstock/public-api-javascript-sdk/master/docs/ImagesApi.md), [AD](https://developer.adobe.com/stock/docs/api/11-search-reference)).

→ Proposed values: `temperature` ∈ `warm | neutral | cool`; `saturation` ∈ `muted | natural | vivid`;
`monochrome` boolean; `palette` free k-means labels (tree-green, teal-orange, pastel, neon);
`dominant_hex[]`. This is cheaply **derived** from the thumbnail (no VLM).

### 2.6 Composition / framing / lens

- Yale: *aspect ratio, deep focus, shallow focus, depth of field, deep space / shallow space,
  frontality, offscreen space* ([Yale](https://filmanalysis.yale.edu/cinematography/) ·
  <https://filmanalysis.yale.edu/mise-en-scene/#space>).
- Columbia: *deep focus, depth of field, fisheye lens, wide-angle lens/shot, framing = mise-en-scène*
  ([Columbia](https://filmglossary.ccnmtl.columbia.edu/term/)).
- Industry machine vocabulary: `lens_type`, `depth_of_field` (Getty, above); rule-of-thirds /
  negative-space language has no standards body but is exactly what Shutterstock productized in
  Composition-Aware Search (cited in the domain note: <https://developer.nvidia.com/blog/shutterstocks-new-ai-tool-lets-you-search-by-image-composition>).

→ Proposed values: `dof` ∈ `shallow | deep`; `negative_space` boolean; `symmetry` ∈
`symmetric | asymmetric`; `leading_lines` boolean; `subject_position` ∈ `centered | thirds | off_center`;
`perspective` ∈ `wide_angle | normal | telephoto`. Flag `leading_lines`/`rule_of_thirds` as
**aesthetic judgements** with the lowest reliability of the composition set.

### 2.7 Time of day

Astral documents the operational definitions that photographers actually mean (and derives them from
the NOAA solar calculators): **golden hour = sun between 4° below and 6° above the horizon; blue hour
= between 6° and 4° below** ([Astral docs](https://astral.readthedocs.io/en/latest/)).

→ Proposed values: `dawn | sunrise | morning | midday | afternoon | golden_hour | sunset | blue_hour | dusk | night`.
Caveat: the same frame can be dawn or dusk; for search that is usually acceptable, but do not claim
to distinguish them confidently.

### 2.8 Weather and season

- Weather: **WMO Code Table 4677** defines present weather in ~100 code figures across drizzle/rain/
  freezing/snow/showers/thunderstorm/fog/dust/squall classes
  ([NOAA mirror](https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM)).
  For search we collapse to: `clear | partly_cloudy | overcast | rain | storm | snow | fog | haze | wind | n/a`.
- Season: `spring | summer | autumn | winter | n/a` — physically ambiguous outside temperate
  regions; treat as *partly* inferable and never as a hard filter (a topical query, not a truth).

### 2.9 Mood / emotion

W3C's EmotionML vocabulary repository lists what the field considers defensible:
Ekman's **big six** (`anger, disgust, fear, happiness, sadness, surprise`); Cowie et al.'s
**everyday 17** (`affectionate, afraid, amused, angry, bored, confident, content, disappointed,
excited, happy, interested, loving, pleased, relaxed, sad, satisfied, worried`); OCC (22) and FSRE
(24) category sets; and dimensional models — Mehrabian's **PAD** (pleasure/arousal/dominance) and
FSRE's four dimensions (**valence, potency, arousal, unpredictability**)
([W3C emotion vocab](https://www.w3.org/TR/emotion-voc/)). The document states plainly there is a
"lack of agreement in the community" on a single emotion vocabulary — so don't invent one.

→ Proposed: reuse the **everyday 17** as the closed label set for footage mood (they read like
stock-search language), plus **valence / arousal** floats for ranking. Cite a confidence.

### 2.10 Subject, place, action

- **IPTC Media Topics**: 1,085 terms, 17 top-level topics (as of 2026-07-09), e.g. level‑1
  `arts, culture, entertainment and media`, `economy, business and finance`, `environment`,
  `human interest`, `lifestyle and leisure`, `sport`, `weather`; level‑2 examples
  `lifestyle and leisure → leisure | lifestyle | wellness`, `weather → weather phenomena`,
  `environment → nature` ([IPTC treeview](https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html);
  machine-readable XLSX: <https://iptc.org/std/NewsCodes/IPTC-MediaTopic-NewsCodes.xlsx>).
- **MovieNet** annotates 42K scenes with **90 place classes and 80 action classes**, plus 1.1M
  character boxes, scene segmentation, and *shot scale + shot movement* tags per shot
  ([MovieNet site](https://movienet.github.io/), [ECCV 2020 paper](https://arxiv.org/pdf/2007.10937.pdf)).
- Platform categories are the market-facing version of the same thing (§1.1) — use them for
  **topic** but not as the controlled vocabulary (Adobe's own caveat **[AD]**).

### 2.11 People

- Getty creative **video** filters: `number_of_people` ∈ **none / one / two / group**; `ethnicity`
  ∈ black, white, east asian, hispanic_latinx, japanese, middle_eastern, multiracial_person,
  multiethnic_group, native_american_first_nations, pacific_islander, south_asian, southeast_asian
  ([Getty release notes 2024-01-19](https://developers.gettyimages.com/release-notes)).
- Shutterstock: `people_number` (integer), `people_age` (9 buckets), `people_gender`,
  `people_ethnicity` (15 values) **[SH]**.
- Pexels UI: 0 / 1 / 2 / 3+ and an age filter **[PX]**.

→ Proposed: `count_bucket` ∈ `0 | 1 | 2 | 3+` and `group_type` ∈ `solo | couple | family | team | crowd | n/a`.
**Do not auto-infer ethnicity or age** — every platform that has those facets gets them from
contributor declarations or curated editorial metadata, and inference carries real harm plus
unmeasurable error. Keep `age`/`gender_presentation` as *optional descriptive text* only if a human
reviewed it (or omit). This is also why Getty's `quality_rank` and `release_info` exist as
*declared* fields ([Getty metadata](https://developers.gettyimages.com/asset-metadata)).

### 2.12 Genre / use-case

No film-school tree exists for "commercial / corporate / travel / documentary"; the working
vocabulary is platform categories (§1.1) plus Pexels' own **"story"** tag bucket **[PX]**. Use it as
an open-vocabulary *theme* field, not an enum.

---

## 3. Map: what a VLM caption of one thumbnail can and cannot deliver

Evidence anchors: MovieShots accuracy table (**AccS** = scale, **AccM** = movement) and its
ablation showing a variance map over 8 clips lifts AccM from 70.5 → 80.6 with image-only input
([arXiv:2008.03548](https://arxiv.org/html/2008.03548v1) Tables 3–4). Yale/Columbia define the
terms as *perceptible from a shot*, which for scale/lighting/colour/composition is a single frame;
movement/POV/editing relations are defined across shots.

| Facet cluster | Single still frame | Why |
|---|---|---|
| Shot scale | **High** (84–88 % machine-validated) | Scale is a spatial property of the frame |
| Camera angle (low/high/overhead/dutch) | **High** for low/high/overhead/dutch; **low** for POV | Angle is visible; POV requires the editing relation |
| Camera movement | **No** (70 % vs 80–84 % with temporal evidence) | Needs ≥8 clips / optical flow |
| Time-lapse / slow-motion / aerial | **Partly** as *style* (blur streaks, altitude cues), but unreliable; aerial ≈ high | Aerial is a viewpoint (inferable); motion is not |
| Lighting key/contrast/direction | **Partly–High** | Key style and contrast are frame properties; "studio vs available" is inferential |
| Golden/blue hour, night, daytime | **Partly–High** | Strong colour/sun-angle cues, but dawn vs dusk ambiguous |
| Colour palette / temperature / saturation / monochrome | **High** | Directly measurable from pixels |
| Composition (DoF, negative space, symmetry, leading lines) | **Partly–High** (geometry yes; "intentional composition" is a judgement) | Depth/blur and layout are measurable; rule-of-thirds is an interpretation |
| Setting indoor/outdoor, urban/nature, place | **High** | Core visual scene understanding |
| Weather / season | **Partly** | Snow/rain visible; "spring" is a botanical/hemisphere inference |
| Subject nouns (objects, species, activities) | **High** for nouns, **Partly** for activity | A still shows a pose, not the trajectory of an action (MovieNet annotates actions over *scenes*, not frames) |
| People count / group type | **High** (faces/bodies are countable) | But occlusion; store confidence |
| Mood / emotion | **Partly** | Scene affect is real but ambiguous; W3C notes community disagreement |
| Genre / use-case | **No** | A use-case judgement about the production, not the frame |
| Motion/tech (loop, alpha, contains audio, RED camera) | **No** | Requires the clip metadata or audio |

---

## 4. Facet-by-facet assessment (still-frame inferability, tag coverage, payoff)

"Tag coverage" = share of videos whose `tags` contain the facet, from a 5 % sample (n = 19,341);
strict token matching, so treat as a band not a decimal (§5). "Payoff" = expected value for creator
queries, ranked H/M/L on *frequency × current gap*.

| Facet (proposed field) | Typical controlled values | Still frame | Our tag coverage | Payoff | Verdict |
|---|---|---|---|---|---|
| `setting.space` indoor/outdoor | indoor, outdoor, mixed | High | **43 % outdoor / 23 % indoor** | H | Already strong; caption only to fill nulls |
| `setting.environment` | urban, nature, rural, industrial, water, studio, abstract | High | urban ≈ 20 %, nature ≈ 41 % | H | Strong; keep as filter + caption delta |
| `subject.theme` (topical) | platform-style topics (nature, business, food, travel, people, tech, health, sport, fashion, transport, animals) | High | **≈ 8–40 %** per topic (varies widely by topic) | H | Well covered in tags; densify with captions for long-tail topics |
| `people.count_bucket` | 0, 1, 2, 3+ | High | explicit count words ≈ **3.9 %** | H | **Gap** — Pexels UI + Getty + Shutterstock all filter on it; we can't |
| `people.group_type` | solo, couple, family, team, crowd | High | partial (`couple`/`group of people` in tags) | H | Caption |
| `people.activity` | open vocab (walking, cooking, typing …) | Partly | via tags/titles | M | Caption short verb phrase |
| `people.age/gender` | platform buckets | Partly | partial | M | Store caption text only; **no ethnicity** |
| `lighting.time_of_day` | dawn, sunrise, morning, midday, golden_hour, sunset, blue_hour, night | Partly–High | sunset ≈ **5.8 %**, sunrise ≈ 2.2 %, night ≈ 6.3 %, golden hour ≈ **1.9 %**, daytime ≈ 24 % | **H** | **Biggest gap**; VLM task with clear visual cues |
| `lighting.key_style` | high_key, low_key, neutral | Partly | low-key ≈ **0.15 %** | H | Gap; also derivable from image statistics |
| `lighting.direction/source` | front/side/back/top; natural/studio/practical/neon/fire | Partly | backlit ≈ **0.5 %**, studio ≈ 0.7 %, neon ≈ 1.7 % | H | Gap |
| `lighting.quality` hard/soft | hard, soft | Partly | ~0 | M | Caption |
| `colour.temperature` | warm, neutral, cool | High | warm ≈ 9.3 %, cool/teal ≈ **0.7 %** | H | Derive from pixels |
| `colour.saturation` | muted, natural, vivid | High | vibrant ≈ 14.9 % | M–H | Derive from pixels |
| `colour.monochrome` | true/false | High | B&W ≈ **2.2 %** | M | Derive (cheap, exact) |
| `colour.palette` | k-means labels / hex | High | ~0 as taxonomy | M | Derive; enables "teal & orange", "pastel" queries |
| `composition.dof` | shallow, deep | Partly–High | bokeh/DoF ≈ **2.2 %** | H | Derive bokeh proxy + caption |
| `composition.negative_space` | true/false | Partly | ≈ **0.4 %** | H | Gap; also useful as a layout constraint |
| `composition.symmetry` | symmetric, asymmetric | Partly | symmetry/pattern ≈ 1.6 % | M | Caption |
| `composition.leading_lines`, `rule_of_thirds` | true/false | Partly (judgement) | leading lines ≈ **0.02 %** | L–M | Caption with low weight only |
| `camera.scale` | ELS→ECU | High | close-up ≈ 15.6 %, wide/panorama ≈ 8.5 %, macro ≈ 1.5 % | H | MovieShots-validated; beware scheme mismatch |
| `camera.angle` | eye, low, high, overhead, aerial, dutch, pov | High (pov low) | top-down/overhead ≈ 6.3 %, aerial ≈ 17.5 % | M–H | Caption |
| `camera.technique` (motion-ish) | aerial, drone, handheld, tracking, timelapse, slow_motion, hyperlapse, loop | **No** (except aerial) | slow-motion tag ≈ 3.1 %, timelapse ≈ 1.6 %, handheld ≈ 1.5 %; **but 17.4 % of titles+descriptions contain a motion word** | **H** | **Parse prose first**; caption cannot help |
| `setting.weather` | clear, cloudy, overcast, rain, storm, snow, fog, haze, wind | Partly | rain ≈ 5.4 %, snow ≈ 2.3 %, fog ≈ 1.5 %, clouds/sky ≈ 19 % | M | Caption + parse |
| `setting.season` | spring, summer, autumn, winter | Partly | summer ≈ 16.9 %, spring ≈ 6.0 %, autumn ≈ 3.0 %, winter ≈ 4.0 % | M | Caption; not a hard filter |
| `mood.emotions` | everyday-17 (W3C) | Partly | calm ≈ 36.9 %, happy ≈ 13.8 %, love ≈ 6.3 %, dramatic/moody ≈ 4.2 % | **H** | Caption; current tag mood is dominated by "calm/happy" |
| `mood.valence/arousal` | floats | Partly | n/a | M–H | Derive from caption model; useful for rank, not filter |
| `style.vintage/retro`, `style.cinematic`, `style.minimal` | booleans / labels | Partly–High | vintage ≈ 2.6 %, cinematic ≈ **1.7 %**, minimal ≈ 5.4 % | M | Caption for "cinematic/moody" queries |
| `theme.use_case` genre | commercial, corporate, documentary, travel, food, fashion, lifestyle, social | **No** | via topics | M | Query-time routing, not a stored facet |
| Hard facets: orientation, aspect, duration, fps, resolution, licence | exact enums (§1.2) | n/a | n/a | H | Already columns; no work |

**Reading of the table.** The four highest-value *caption* targets are (1) time-of-day & lighting
key/source, (2) mood/emotion, (3) shot scale + angle, (4) people count/group + setting fill-rate.
The two highest-value *non-caption* targets are colour (derive) and motion/technique (parse prose).

---

## 5. Measured tag coverage in our corpus (method)

- Source: `stock_videos` in `dev-middleware-stock-db` (`127.0.0.1:15433`, DB `stock`) — **388,584 rows**
  at query time (2026-10-08). Full-table stats: **median 47 tags, p90 55, mean 46.5** — i.e. roughly
  **2× Pexels' own 10–25 tag guidance** **[PX]**.
- Coverage query: `TABLESAMPLE SYSTEM (5)` → **n = 19,341 distinct videos**; tags lower-cased,
  matched with anchored token lists + phrase regexes (e.g. `close.?up`, `slow.?mot`). 95 % CI is
  roughly ±0.6 pp at p≈20 % and ±0.15 pp at p≈1 %; matching is approximate by nature (multi-word
  tags, plurals, synonyms), so read the numbers as bands. Query text kept in the session's shell
  history (`/tmp/opencode/facets2.sql`).
- Title/description pass: same 5 % sample, `lower(title || ' ' || description)` matched with the
  same phrase regexes; motion 17.4 %, lighting 6.0 % (mostly "sunset"), close-up 11.1 %.
- Observed tag style: tags include multi-word descriptive phrases — real examples from two rows:
  `{"natural lighting","candid","group of people","daylight","urban exploration",…}` and
  `{"at home","cozy","festive","indoors","holiday spirit","warmth","seasonal",…}`. This confirms
  Pexels' three-bucket guidance is present in the data (subject + descriptive[action/emotion/
  setting/time-of-day/weather] + story) — but lighting terms are limited to a handful of stock
  phrases ("natural lighting", "daylight"), and composition/lighting-quality vocabulary is absent.

**Answer to "which facets do Pexels' own tags encode?"** — the seven the guidelines name:
subject, action, emotion (mood), setting, time-of-day, weather, and abstract story/theme.
**Not encoded:** shot scale/angle (beyond close-up/wide), lighting key/source, colour grade,
composition, camera technique (except when the contributor happened to write it), people count
(3.9 % explicit), demographics. Our 47-tag median is *more subject/location surface*, not more
facet dimensions.

---

## 6. Proposed facet schema

One JSONB `facets` column per row (or a `facets_v1` table), produced once per row from three
sources: **DB columns** (no computation), **thumbnail derivations** (CPU, no VLM), **one VLM
caption** (Qwen2.5-VL-class, per the dense note's feasibility section), plus a **prose parse** of
title+description. All enums carry `null` + confidence; nothing is guessed into a hard enum.

```json
{
  "schema_version": 1,
  "generated_at": "2026-10-08T12:00:00Z",
  "caption_model": "qwen2.5-vl-…",
  "provenance": { "scale": "vlm", "palette": "derived", "slow_motion": "prose", "orientation": "column" },

  "camera": {
    "scale": "medium_close",          // extreme_wide|wide|full|medium|medium_close|close|extreme_close|null
    "scale_confidence": 0.82,
    "angle": "high",                  // eye_level|low|high|overhead|aerial|dutch|pov|null
    "angle_confidence": 0.7,
    "movement": "static",             // static|moving|zoom_in|zoom_out|unknown   (temporal; low trust)
    "technique": ["aerial", "drone"], // parsed: aerial|drone|handheld|tracking|timelapse|slow_motion|hyperlapse|loop|360vr
    "lens_hint": "wide_angle"         // wide_angle|normal|telephoto|null
  },

  "lighting": {
    "time_of_day": "golden_hour",     // dawn|sunrise|morning|midday|afternoon|golden_hour|sunset|blue_hour|dusk|night|null
    "key_style": "low_key",           // high_key|low_key|neutral|null
    "direction": "back",              // front|side|back|top|mixed|null
    "quality": "soft",                // hard|soft|null
    "source": ["natural", "practical"],
    "confidence": 0.6
  },

  "colour": {
    "temperature": "warm",            // warm|neutral|cool
    "saturation": "vivid",            // muted|natural|vivid
    "monochrome": false,
    "palette": ["teal", "orange"],    // free labels
    "dominant_hex": ["#0e2a35", "#e8a23a"]
  },

  "composition": {
    "dof": "shallow",                 // shallow|deep|null
    "negative_space": true,
    "symmetry": "asymmetric",         // symmetric|asymmetric
    "subject_position": "thirds",     // centered|thirds|off_center
    "leading_lines": false
  },

  "setting": {
    "space": "outdoor",               // indoor|outdoor|mixed
    "environment": "nature",          // urban|nature|rural|industrial|water|studio|abstract|aerial
    "place": "mountainside",          // open vocab, 1–3 words
    "weather": ["clear"],             // WMO-collapsed set
    "season": "autumn"                // spring|summer|autumn|winter|null
  },

  "people": {
    "count_bucket": "2",              // 0|1|2|3+
    "group_type": "couple",           // solo|couple|family|team|crowd|null
    "activity": ["walking", "laughing"],
    "age_text": null,                 // descriptive text only; never an enum
    "gender_text": null
  },

  "mood": {
    "emotions": ["relaxed", "loving"], // W3C EmotionML "everyday 17" subset
    "valence": 0.6,
    "arousal": -0.2
  },

  "theme": {
    "topics": ["nature", "travel"],    // platform-style topics
    "story_tags": ["freedom", "escape"],
    "use_cases": []                    // commercial|corporate|documentary|social|... (query-time only)
  },

  "embeddings": {
    "facet_text": {
      "subject": "couple walking on a mountain trail at golden hour",
      "lighting": "warm low sunlight, backlit, golden hour",
      "mood": "relaxed, loving, hopeful",
      "composition": "shallow depth of field, negative space, rule of thirds"
    }
  }
}
```

### 6.1 Field-by-field: caption vs derive vs parse vs filter

| Field | Mechanism | Cost | Notes |
|---|---|---|---|
| `camera.scale`, `camera.angle`, `setting.*`, `people.count_bucket/group_type/activity`, `mood.emotions` (+V/A), `theme.topics/story_tags`, `lighting.time_of_day/key_style/direction/source/quality`, `composition.dof/negative_space/symmetry/subject_position` | **VLM caption** (one prompt producing one JSON per thumbnail) | 1 API call/row; **measure on 1k rows first** | Ask for confidence per field; allow `null` |
| `colour.*` | **Derive from thumbnail** (k-means palette, saturation/temperature statistics, mono detection) | CPU, ~ms/row | Exact, no hallucination, free |
| `composition.dof` (proxy) | **Derive** (Laplacian/blur + bokeh blob detection) then reconcile with caption | CPU | Cheap signal even before VLM |
| `camera.technique` motion words | **Parse** title+description (regex/lexicon) | free | 17.4 % of rows carry a motion word in prose vs 3.1 % in tags |
| `composition.leading_lines`, `rule_of_thirds` | Caption only, low weight; optionally skip | — | Weakest inferences; don't filter on them |
| orientation, aspect_ratio, duration, fps, resolution, license | **Columns** — filter | free | Already present (schema confirmed) |
| `people.age`, `people.gender`, `ethnicity` | **Do not auto-infer.** Optional human/contributor text only | — | Ethics + error; platforms get these from declarations **[GI][SH]** |

### 6.2 Retrieval wiring (per the sibling notes)

1. **Filters = SQL prefilters on both arms** (sparse note §4): the existing columns plus the new
   high-precision enums (`camera.scale`, `lighting.time_of_day`, `setting.space`,
   `people.count_bucket`, `colour.monochrome`, `camera.technique[]`). Keep them prefiltered in dense
   *and* sparse arms, not as a third ranker.
2. **Dense = 3–5 facet strings per row**, embedded separately, plus the existing
   `title — description — tags` blob as fallback (dense note: `text-embedding-v4` @768, ~US$6 for
   the whole corpus). Facet-routed queries ("moody night neon") then hit the `lighting`/`mood`
   vectors instead of diluting one long blob. A/B separate vectors vs one concatenated facet text on
   the domain note's eval set before adopting.
3. **Sparse = BM25 over the same facet strings + tags**; RRF k=60 as in the sparse note.
4. **Motion queries route to the parsed `technique` + `duration`**, never to the image vector
   (image-only movement inference is ~70 % accurate).

### 6.3 Sequencing (cheap → expensive)

1. Parse motion/technique from title+description (no new dependencies) and expose it as a facet.
2. Derive colour fields from thumbnails once thumbnails are fetched (already planned).
3. Caption 500–1,000 stratified rows with the target VLM; have a human score each facet for
   precision/recall; only then run the full corpus.
4. Embed facet strings; gate on the eval plan in the domain note (per-facet nDCG@10, filter-constrained cuts).

---

## 7. Open questions & uncertainty

1. **Pexels tagging guidelines page is not fetchable from this environment** (HTTP 403 to automated
   fetches; web.archive.org returned 429). The 10–25-tag guidance and the subject/descriptive/story
   three-bucket split are taken from the URL and were independently verified by the domain note on
   2026-10-08. Re-verify before quoting it in user-facing docs.
2. **Shutterstock's live category enum is not public without an API key.** The SDK doc example shows
   `total_count: 13` (clearly example data) and the site breadcrumb says "Video Categories"
   (<https://www.shutterstock.com/category/food-and-drink>); do not hard-code a count.
3. **Getty `compositions` full enum is only partially public** — the release notes name
   `medium_shot`, `wide_shot`, `part_of_a_series`; the rest lives in Swagger behind a key. Same for
   iStock's filter labels.
4. **VLM reliability on our data is untested.** Golden-hour vs sunset, "studio vs natural", mood
   labels, and rule-of-thirds are the likely failure modes. The single-frame movement limit is the
   one we *know* (MovieShots). Run the 1k-row calibration before committing to a corpus-scale pass.
5. **Cost of the caption pass is unmeasured.** Image embedding was estimated at ~US$150 for 665k
   rows (dense note); an output-token-billed VLM caption pass is a different (likely larger) number.
   Compute it on 1k rows first, then extrapolate for 388k.
6. **Dawn vs dusk is visually ambiguous** (same sun angles); treat `dawn`/`sunset` as interchangeable
   for ranking unless the user's query is explicit.
7. **Season and weather are geography-dependent** — "spring" footage shot in the southern hemisphere
   or the tropics will be mis-captioned; keep both as soft facets.
8. **People facets carry policy risk.** Getty/Shutterstock/Pexels all obtain age/ethnicity/count from
   contributors or curated metadata, not from a model. Recommend: never auto-label ethnicity or age;
   count/group only, with confidence.
9. **Do our 47 tags already dilute the dense text?** (open in the domain note). The facet strings in
   §6.2 are partly an answer — but the A/B (tags-blob vs facet-blob vs both) must be run.
10. **Series/duplicate detection** ("part_of_a_series" at Getty; Shutterstock similar/collections) has
    no Pexels equivalent in our columns; if creators ask "give me 5 different clips from the same
    shoot", that's a future facet built from `user_name` + thumbnail similarity, not from captions.
11. **Time-lapse/slow-motion could be recovered from the SD preview mp4s** (0.6–18 MB, already
    URL-known) by sampling ≥8 frames and running the MovieShots-style variance-map classifier — the
    one path that could upgrade movement from "no" to "coarse yes". Deferred because it means
    downloading/storing clips (dense note's tradeoff).
12. **Everything vendor-facing is time-sensitive** — platform filter lists change (Getty added video
    filters in 2024; Storyblocks' category set is clearly evolving). Re-check the source key table at
    implementation time.

---

### Source index (primary, all fetched 2026-10-08 unless noted)

- Shutterstock API contract: <https://raw.githubusercontent.com/shutterstock/public-api-javascript-sdk/master/docs/VideosApi.md> · <https://raw.githubusercontent.com/shutterstock/public-api-javascript-sdk/master/docs/ImagesApi.md>
- Adobe Stock: <https://developer.adobe.com/stock/docs/api/11-search-reference> · <https://developer.adobe.com/stock/docs/api/17-categorytree> · <https://helpx.adobe.com/stock/contributor/submit-your-content/submit-videos/technical-requirements-for-video-submissions.html>
- Getty Images: <https://developers.gettyimages.com/asset-metadata> · <https://developers.gettyimages.com/release-notes> · <https://www.istockphoto.com/search-guide/search-for-video>
- Artgrid: <https://artgrid.zendesk.com/hc/en-us/articles/8069563020701-How-to-Find-Download-and-Use-Footage> · <https://artgrid.io/insights/stock-video>
- Storyblocks: <https://www.storyblocks.com/video/search>
- Pond5: <https://blog.pond5.com/80631-how-to-find-the-perfect-stock-assets> · <https://www.pond5.com/search?kw=ai&media=footage> · <https://blog.pond5.com/22617-new-explore-footage-by-color-or-composition-with-visual-search>
- Envato: <https://elements.envato.com/stock-video/farm+animals> · <https://help.elements.envato.com/hc/en-us/articles/360000629186-How-to-Search-Browse-Find-Items-on-Envato> · <https://videohive.net/category>
- MotionElements: <https://www.motionelements.com/visualsearch> · <https://www.motionelements.com/begin> · <https://help.motionelements.com/en/docs/use-visualsearch-for-video-image-templates>
- Pexels: <https://www.pexels.com/blog/photography/photo-video-tagging-guidelines> (blocked; see §7.1) · <https://help.pexels.com/hc/en-us/articles/37285267631769-How-does-Pexels-use-the-metadata-on-my-photos>
- Film grammar: <https://filmanalysis.yale.edu/cinematography/> · <https://filmanalysis.yale.edu/mise-en-scene/> · <https://filmglossary.ccnmtl.columbia.edu/term/>
- Datasets: <https://movienet.github.io/> · <https://arxiv.org/abs/2008.03548> · <https://arxiv.org/pdf/2007.10937.pdf>
- Controlled vocabularies: <https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html> · <https://guides.loc.gov/tgm-ii> · <https://www.w3.org/TR/emotion-voc/> · <https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM> · <https://astral.readthedocs.io/en/latest/>
- Repo DB measurements: `dev-middleware-stock-db` @ `127.0.0.1:15433/stock`, `stock_videos`, 388,584 rows, `TABLESAMPLE SYSTEM (5)` (n = 19,341), 2026-10-08.
