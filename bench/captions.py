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
PROMPT_VERSION = "caption-v1"
GEMMA_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

CAPTION_PROMPT = (
    "You are cataloging a stock video by its thumbnail image. Describe what is "
    "visible in 2-4 sentences: the main subject, the setting, the lighting, the "
    "mood, the dominant colors, and the camera framing. Describe only what you "
    "can see; do not speculate about motion, sound, or story. Plain text only, "
    "no markdown, no lists."
)
MAX_OUTPUT_TOKENS = 200
CAPTION_TEMPERATURE = 0.2
MAX_ATTEMPTS_FACTOR = 2  # attempts = factor * key pool size


class CaptionError(RuntimeError):
    """Raised when every key in the pool failed for one image."""


def parse_caption(payload: Mapping[str, object]) -> str:
    """Extract and normalise the text answer from a generateContent response."""
    candidates = payload.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        for part in (candidate.get("content") or {}).get("parts") or []:  # type: ignore[union-attr]
            text = part.get("text")  # type: ignore[union-attr]
            if text:
                texts.append(str(text))
    raw = " ".join(texts).strip()
    cleaned: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("*-# ").strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


class GemmaCaptionClient:
    """Google generateContent client with a rotating key pool (per-key pacing)."""

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
        self._pacers = [Pacer(rpm) for _ in self._keys]
        self._lock = threading.Lock()
        self._next_key = 0
        self.requests = 0
        self._client = httpx.Client(
            base_url=GEMMA_BASE_URL,
            timeout=httpx.Timeout(120.0),
            trust_env=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _pick_key(self) -> int:
        with self._lock:
            index = self._next_key
            self._next_key = (self._next_key + 1) % len(self._keys)
            self.requests += 1
            return index

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
            try:
                response = self._client.post(
                    f"/models/{self.model}:generateContent",
                    json=body,
                    headers={"X-goog-api-key": self._keys[index]},
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempt, 30))
                continue
            if response.status_code == 200:
                return parse_caption(response.json())
            last_error = f"HTTP {response.status_code}: {response.text[:120]}"
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2**attempt, 30))
                continue
            raise CaptionError(f"caption endpoint rejected request: {last_error}")
        raise CaptionError(
            f"caption failed after {len(self._keys) * MAX_ATTEMPTS_FACTOR} attempts: {last_error}"
        )


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
) -> dict[int, dict]:
    """Return {doc_id: {"path": Path, "source": str}} for every material doc.

    Reuses thumbnails the crawler already downloaded (read-only), then fetches
    the rest into the bench's own directory. Docs with no thumbnail anywhere
    are simply absent from the result.
    """
    import psycopg

    from store.db import DEFAULT_DSN

    crawler_dir = Path(
        os.environ.get("VPC_THUMBNAIL_DIR", util.REPO_ROOT / "storage" / "thumbnails")
    )
    bench_dir = root / "thumbnails"
    bench_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn or DEFAULT_DSN) as conn:
        rows = conn.execute(_THUMB_SQL, (materials.ids,)).fetchall()

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
) -> dict:
    """Caption every materials doc that has a thumbnail; returns the manifest."""
    out_dir = root / "captions" / version
    captions_path = out_dir / "captions.jsonl"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        raise SystemExit(f"captions version {version!r} already frozen: {manifest_path}")

    from bench.judge import LabelStore

    store = LabelStore(captions_path)
    thumbs = ensure_thumbnails(dsn, materials, root)
    pending = [doc_id for doc_id in materials.ids if not store.has(f"cap-{doc_id}")]
    if not thumbs:
        raise SystemExit("no thumbnails resolved; check thumbnail_url in stock_videos")
    pending = [doc_id for doc_id in pending if doc_id in thumbs]
    if limit is not None:
        pending = pending[:limit]

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
                "prompt_version": PROMPT_VERSION,
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
        "prompt_version": PROMPT_VERSION,
        "prompt": CAPTION_PROMPT,
        "materials_version": materials.version,
        "materials_hash": materials.content_hash,
        "content_hash": util.content_hash(records),
        "code": util.code_version(),
    }
    manifest["frozen"] = False
    complete = manifest["n_captions"] + manifest["n_failed"] >= manifest["n_with_thumbnail"]
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
