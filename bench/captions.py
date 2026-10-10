"""Thumbnail captioning: fetch thumbnails for the materials, caption them with a VLM.

The caption corpus is the second retrieval text source next to raw text
(title/description/tags): `storage/bench/captions/<version>/captions.jsonl`
holds one caption per materials doc, versioned by (model, prompt version) and
content-hashed. Judging deliberately stays anchored to the raw text (option B
in the 2026-10-10 plan): captions change what the retrievers read, not what
"relevant" means.

The Google Generative Language API is keyed per request; the client rotates a
pool of keys (each with its own RPM pacer) and retries on 500/429 by moving to
the next key. Raw model responses are kept inside the caption records so a run
is replayable without re-billing the VLM.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from bench import util
from bench.judge import Pacer
from bench.materials import Materials

RULE_VERSION = "captions-v1"
PROMPT_VERSION = "caption-v2"
GEMMA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# Phrasing matters: the prescriptive "You are cataloging... requirements" style made
# gemma-4 echo the task instead of describing the image; this imperative form is the
# one verified against the endpoint (2026-10-10 probe).
CAPTION_PROMPT = (
    "Describe this stock video thumbnail for a search index in 2-4 sentences: "
    "main subject, setting, lighting, mood, colors, camera framing. Text only."
)
MAX_OUTPUT_TOKENS = 1024
CAPTION_TEMPERATURE = 0.2
MAX_ATTEMPTS_FACTOR = 2  # attempts = factor * key pool size

# OCR-family models on SiliconFlow return empty on the caption phrasing; their
# default chat mode answers short imperative descriptions instead (verified
# 2026-10-10: 1.6s, structured, high quality).
DEEPSEEK_OCR_PROMPT = "Describe the image in detail: subject, setting, lighting, colors, framing."
DEEPSEEK_OCR_PROMPT_VERSION = "caption-ocr-v1"
MODEL_PROMPTS: dict[str, tuple[str, str]] = {
    "deepseek-ai/deepseek-ocr": (DEEPSEEK_OCR_PROMPT, DEEPSEEK_OCR_PROMPT_VERSION),
}
DEFAULT_SF_PROMPT = (DEEPSEEK_OCR_PROMPT, DEEPSEEK_OCR_PROMPT_VERSION)

# Field markers the prompt asks for; used to cut task-echo preambles that gemma
# sometimes prepends ("Task: Describe ... Constraints: ...") before the answer.
_FIELD_MARKERS = ("main subject:", "subject:", "setting:")
_TASK_ECHO_PREFIXES = ("task:", "constraints:", "requirements:", "required elements:", "format:")


class CaptionError(RuntimeError):
    """Raised when every key in the pool failed for one image."""


def parse_caption(payload: Mapping[str, object]) -> str:
    """Extract and normalise the text answer from a generateContent response.

    Gemma occasionally echoes the task as a preamble ("Task: Describe …
    Constraints: …") before the real answer. That boilerplate would be copy-
    pasted into thousands of captions and pollute the corpus, so when an echo
    is detected the text is cut at the first field marker; otherwise known
    echo lines are dropped individually.
    """
    candidates = payload.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        for part in (candidate.get("content") or {}).get("parts") or []:  # type: ignore[union-attr]
            text = part.get("text")  # type: ignore[union-attr]
            if text:
                texts.append(str(text))
    raw = " ".join(texts).strip()
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("*-# ").strip()
        if line:
            lines.append(line)
    lowered = [line.lower() for line in lines]
    if any(line.startswith(_TASK_ECHO_PREFIXES) for line in lowered):
        for index, line in enumerate(lowered):
            if line.startswith(_FIELD_MARKERS):
                return " ".join(lines[index:])
        lines = [line for line in lines if not line.lower().startswith(_TASK_ECHO_PREFIXES)]
    return " ".join(lines)


class _KeyState:
    """Per-key health bookkeeping: consecutive failures + blocked-until stamp."""

    def __init__(self) -> None:
        self.fails = 0
        self.blocked_until = 0.0


COOLDOWN_SECONDS = 300.0  # after 3 consecutive failures a key rests for 5 minutes
FAST_FAIL_BACKOFF = 0.5  # seconds per attempt when the failure was instant
BREAKER_THRESHOLD = 40  # consecutive failed requests with zero successes


class _Breaker:
    """Circuit breaker: a wedged endpoint aborts the pass instead of grinding.

    Without it, an outage turns every pending doc into 12 attempts x 300s
    timeouts — hours of dead time before the loop can back off.
    """

    def __init__(self, threshold: int = BREAKER_THRESHOLD) -> None:
        self.threshold = threshold
        self._lock = threading.Lock()
        self._consecutive = 0

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._consecutive = 0
                return
            self._consecutive += 1
            if self._consecutive >= self.threshold:
                raise SystemExit(
                    f"caption endpoint unreachable: {self._consecutive} consecutive failures — aborting pass"
                )


class GemmaCaptionClient:
    """Google generateContent client with a rotating, health-aware key pool.

    Free-tier project keys fail individually (some return 500 for a model, some
    are 403-blocked), so the pool tracks per-key health: 401/403 blocks a key
    permanently, three consecutive 5xx/429 failures cool it down for five
    minutes, and request starts stay paced per key (RPM). Fast failures back
    off quickly instead of eating 30-second sleeps.
    """

    def __init__(
        self,
        keys: Sequence[str],
        model: str,
        rpm: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not keys:
            raise CaptionError("caption client needs at least one API key")
        self.model = model
        self._keys = list(keys)
        self._states = [_KeyState() for _ in self._keys]
        self._pacers = [Pacer(rpm) for _ in self._keys]
        self._lock = threading.Lock()
        self._rr = 0
        self.requests = 0
        self._breaker = _Breaker()
        self.prompt = CAPTION_PROMPT
        self.prompt_version = PROMPT_VERSION
        self._client = httpx.Client(
            base_url=GEMMA_BASE_URL,
            # 300s: evening queueing on the free tier can hold a request well
            # past 120s; timing out would miscount as key failure and trigger
            # cooldown spirals.
            timeout=httpx.Timeout(300.0),
            trust_env=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _pick_key(self) -> int:
        """Round-robin over the keys not in cooldown (eariest unblock if all cooling)."""
        now = time.monotonic()
        available = [i for i, s in enumerate(self._states) if s.blocked_until <= now]
        if not available:
            available = [min(range(len(self._keys)), key=lambda i: self._states[i].blocked_until)]
        with self._lock:
            index = available[self._rr % len(available)]
            self._rr += 1
            self.requests += 1
            return index

    def _mark_ok(self, index: int) -> None:
        self._states[index].fails = 0

    def _mark_fail(self, index: int, status_code: int) -> None:
        state = self._states[index]
        state.fails += 1
        if status_code in (401, 403):
            state.blocked_until = float("inf")
        elif state.fails >= 3:
            state.blocked_until = time.monotonic() + COOLDOWN_SECONDS

    def probe_keys(self, timeout: float = 45.0) -> dict[str, int]:
        """One tiny request per key (in parallel); unblock only healthy keys.

        401/403 keys are blocked permanently; failing keys rest for an hour.
        Returns counts: {"healthy": n, "cooldown": n, "blocked": n}.
        """
        from concurrent.futures import ThreadPoolExecutor

        def probe(index: int) -> int:
            try:
                response = self._client.post(
                    f"/models/{self.model}:generateContent",
                    json={
                        "contents": [{"parts": [{"text": "Say OK."}]}],
                        "generationConfig": {"maxOutputTokens": 8},
                    },
                    headers={"X-goog-api-key": self._keys[index]},
                )
                return response.status_code
            except httpx.HTTPError:
                return 599

        with ThreadPoolExecutor(max_workers=len(self._keys)) as pool:
            codes = list(pool.map(probe, range(len(self._keys))))
        summary = {"healthy": 0, "cooldown": 0, "blocked": 0}
        for index, code in enumerate(codes):
            if code == 200:
                self._mark_ok(index)
                summary["healthy"] += 1
            elif code in (401, 403):
                self._states[index].blocked_until = float("inf")
                summary["blocked"] += 1
            else:
                self._states[index].blocked_until = time.monotonic() + 3600.0
                summary["cooldown"] += 1
        return summary

    def caption_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Caption one image; rotates keys and retries transient failures."""
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": CAPTION_PROMPT},
                        {"inline_data": {"mime_type": mime_type, "data": _b64(image_bytes)}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": CAPTION_TEMPERATURE,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
        last_error = ""
        for attempt in range(MAX_ATTEMPTS_FACTOR * len(self._keys)):
            index = self._pick_key()
            self._pacers[index].wait()
            started = time.perf_counter()
            try:
                response = self._client.post(
                    f"/models/{self.model}:generateContent",
                    json=body,
                    headers={"X-goog-api-key": self._keys[index]},
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._mark_fail(index, 599)
                self._breaker.record(False)
                time.sleep(min(FAST_FAIL_BACKOFF * (attempt + 1), 5.0))
                continue
            elapsed = time.perf_counter() - started
            if response.status_code == 200:
                self._mark_ok(index)
                self._breaker.record(True)
                return parse_caption(response.json())
            last_error = f"HTTP {response.status_code}: {response.text[:120]}"
            if response.status_code in (429, 500, 502, 503, 504):
                self._mark_fail(index, response.status_code)
                self._breaker.record(False)
                backoff = (
                    min(FAST_FAIL_BACKOFF * (attempt + 1), 5.0)
                    if elapsed < 5.0
                    else min(2**attempt, 30)
                )
                time.sleep(backoff)
                continue
            if response.status_code in (401, 403):
                self._mark_fail(index, response.status_code)
                continue
            raise CaptionError(f"caption endpoint rejected request: {last_error}")
        raise CaptionError(
            f"caption failed after {len(self._keys) * MAX_ATTEMPTS_FACTOR} attempts: {last_error}"
        )


class SiliconFlowCaptionClient:
    """SiliconFlow vision-language captioner (OpenAI-compatible chat).

    OCR-family models (DeepSeek-OCR, PaddleOCR-VL) live here; prompts are
    selected per model — their chat modes reject the gemma caption phrasing.
    Single key with RPM pacing; retries transient failures with backoff.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        rpm: float,
        base_url: str = SILICONFLOW_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.prompt, self.prompt_version = MODEL_PROMPTS.get(model.lower(), DEFAULT_SF_PROMPT)
        self._pacer = Pacer(rpm)
        self._breaker = _Breaker()
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(300.0),
            trust_env=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def caption_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Caption one image via chat completions with an inline data URL."""
        body = {
            "model": self.model,
            "temperature": CAPTION_TEMPERATURE,
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{_b64(image_bytes)}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
        }
        last_error = ""
        for attempt in range(4):
            self._pacer.wait()
            try:
                response = self._client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._breaker.record(False)
                time.sleep(min(2**attempt, 15))
                continue
            if response.status_code == 200:
                message = response.json()["choices"][0]["message"]
                content = str(message.get("content") or "").strip()
                self._breaker.record(True)
                return " ".join(content.split())
            last_error = f"HTTP {response.status_code}: {response.text[:120]}"
            if response.status_code in (429, 500, 502, 503, 504):
                self._breaker.record(False)
                time.sleep(min(2**attempt, 15))
                continue
            raise CaptionError(f"caption endpoint rejected request: {last_error}")
        raise CaptionError(f"caption failed after 4 attempts: {last_error}")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


_THUMB_SQL = """
SELECT pexels_id, thumbnail_path, thumbnail_url
FROM stock_videos
WHERE pexels_id = ANY(%s)
"""


def ensure_thumbnails(
    dsn: str | None,
    materials: Materials,
    root: Path,
    concurrency: int = 8,
    doc_ids: Sequence[int] | None = None,
) -> dict[int, dict]:
    """Return {doc_id: {"path": Path, "source": str}} for the requested docs.

    Reuses thumbnails the crawler already downloaded (read-only), then fetches
    the rest into the bench's own directory. Docs with no thumbnail anywhere
    are simply absent from the result.
    """
    import psycopg

    from store.db import DEFAULT_DSN

    wanted = list(doc_ids) if doc_ids is not None else list(materials.ids)
    if not wanted:
        return {}
    crawler_dir = Path(
        os.environ.get("VPC_THUMBNAIL_DIR", util.REPO_ROOT / "storage" / "thumbnails")
    )
    bench_dir = root / "thumbnails"
    bench_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn or DEFAULT_DSN) as conn:
        rows = conn.execute(_THUMB_SQL, (wanted,)).fetchall()

    resolved: dict[int, dict] = {}
    pending: list[tuple[int, str]] = []
    for pexels_id, thumbnail_path, thumbnail_url in rows:
        doc_id = int(pexels_id)
        if thumbnail_path:
            local = Path(thumbnail_path)
            if not local.is_absolute():
                local = crawler_dir / local
            if local.is_file() and local.stat().st_size > 0:
                resolved[doc_id] = {"path": local, "source": "crawler-local"}
                continue
        bench_path = bench_dir / f"{doc_id}.jpg"
        if bench_path.is_file() and bench_path.stat().st_size > 0:
            resolved[doc_id] = {"path": bench_path, "source": "bench-download"}
            continue
        if thumbnail_url:
            pending.append((doc_id, str(thumbnail_url)))

    if pending:
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
        )
        with httpx.Client(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": ua}, trust_env=True
        ) as client:

            def fetch_one(doc_id: int, url: str) -> tuple[int, Path | None]:
                target = bench_dir / f"{doc_id}.jpg"
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    tmp = target.with_suffix(".jpg.tmp")
                    tmp.write_bytes(response.content)
                    tmp.replace(target)
                    return doc_id, target
                except Exception:
                    return doc_id, None

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for doc_id, path in pool.map(lambda item: fetch_one(*item), pending):
                    if path is not None:
                        resolved[doc_id] = {"path": path, "source": "bench-download"}
    return resolved


def build_captions(
    dsn: str | None,
    materials: Materials,
    version: str,
    root: Path,
    client: GemmaCaptionClient,
    concurrency: int = 90,
    limit: int | None = None,
    key_health: dict | None = None,
) -> dict:
    """Caption every materials doc that has a thumbnail; returns the manifest."""
    out_dir = root / "captions" / version
    captions_path = out_dir / "captions.jsonl"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        raise SystemExit(f"captions version {version!r} already frozen: {manifest_path}")

    from bench.judge import LabelStore

    store = LabelStore(captions_path)
    todo = [doc_id for doc_id in materials.ids if not store.has(f"cap-{doc_id}")]
    if limit is not None:
        todo = todo[:limit]
    thumbs = ensure_thumbnails(dsn, materials, root, doc_ids=todo)
    if not thumbs:
        raise SystemExit("no thumbnails resolved; check thumbnail_url in stock_videos")
    pending = [doc_id for doc_id in todo if doc_id in thumbs]

    failed: list[int] = []

    def caption_one(doc_id: int) -> None:
        thumb = thumbs[doc_id]
        image_bytes = thumb["path"].read_bytes()
        try:
            caption = client.caption_image(image_bytes)
        except CaptionError:
            failed.append(doc_id)
            return
        if not caption:
            failed.append(doc_id)
            return
        store.append(
            {
                "key": f"cap-{doc_id}",
                "doc_id": doc_id,
                "caption": caption,
                "model": client.model,
                "prompt_version": client.prompt_version,
                "thumb_source": thumb["source"],
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
        )

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(caption_one, pending))

    records = list(util.read_jsonl(captions_path)) if captions_path.is_file() else []
    manifest = {
        "version": version,
        "rule": RULE_VERSION,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_materials": len(materials.ids),
        "n_with_thumbnail": len(thumbs),
        "n_captions": len(records),
        "n_failed": len(failed),
        "model": client.model,
        "prompt_version": client.prompt_version,
        "prompt": client.prompt,
        "key_health": key_health,
        "materials_version": materials.version,
        "materials_hash": materials.content_hash,
        "content_hash": util.content_hash(records),
        "code": util.code_version(),
    }
    manifest["frozen"] = False
    complete = manifest["n_captions"] + manifest["n_failed"] >= manifest[
        "n_with_thumbnail"
    ] and manifest["n_failed"] <= max(1, int(0.05 * manifest["n_with_thumbnail"]))
    if limit is None and complete:
        manifest["frozen"] = True
        util.write_json(manifest_path, manifest)
    elif limit is not None:
        manifest["note"] = "smoke run (--limit): not frozen, resume freely"
    else:
        manifest["note"] = "incomplete: rerun the same command to resume missing docs"
    return manifest


@dataclass
class Captions:
    """In-memory view of a frozen caption corpus (what caption arms read)."""

    version: str
    texts: dict[int, str]
    content_hash: str
    manifest: dict

    @classmethod
    def load(cls, version: str, root: Path) -> Captions:
        manifest_path = root / "captions" / version / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"captions version {version!r} not built yet: {manifest_path}")
        manifest = util.load_json(manifest_path)
        texts = {
            int(record["doc_id"]): str(record["caption"])
            for record in util.read_jsonl(root / "captions" / version / "captions.jsonl")
        }
        return cls(
            version=version,
            texts=texts,
            content_hash=str(manifest["content_hash"]),
            manifest=manifest,
        )
