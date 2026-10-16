"""Lazy persisted image vectors for the MPT dedup gate.

``get_or_embed`` is the only entry point: answer from the ``image_embeddings``
table when the (key, model) row exists, otherwise read the local thumbnail,
embed it through the DashScope multimodal endpoint and upsert. An in-process
single-flight lock per key keeps concurrent funnels from double-embedding one
image; the unique row is the durable side of the same dedup.

Keys are ``pexels:<id>`` — the flat thumbnail dir shares the id, so a hit is a
pure DB lookup: no file read, no hashing. The CDN ``thumbnail_url`` from the
``stock_videos`` row stays as fallback when the local file is absent.

Every call logs one usage line (hit or miss) so the A/B cost axis can read
cache-hit rate, tokens and latency straight off the service log.

Environment (injected by compose ``env_file``, machine-local ``.env``):

- ``VPC_IMAGE_EMBED_MODEL``      default ``tongyi-embedding-vision-flash``
- ``VPC_IMAGE_EMBED_API_KEY``     DashScope key, required for a miss
- ``VPC_IMAGE_EMBED_BASE_URL``   DashScope multimodal embeddings endpoint
- ``VPC_IMAGE_EMBED_TIMEOUT``    seconds, default 30 (provider AND CDN fetch)
- ``VPC_THUMBNAIL_DIR``          local thumbnail dir (``load_settings`` owns it)
- ``VPC_USE_PROXY``              0 forces direct connections (AGENTS.md §3)
- ``VPC_LOG_LEVEL``              service-side stdlib log level, default INFO
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psycopg

from crawler.config import load_settings
from store import db
from store.embedding import parse_vector_literal, to_vector_literal

logger = logging.getLogger("vpc.image_embeddings")

DEFAULT_EMBED_MODEL = "tongyi-embedding-vision-flash"
DEFAULT_EMBED_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
_RATE_LIMIT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


@dataclass(frozen=True)
class ImageEmbedSettings:
    """Resolved provider settings for one embed call."""

    model: str
    api_key: str
    base_url: str
    timeout: float
    trust_env: bool


def embed_settings() -> ImageEmbedSettings:
    """Read the VPC_IMAGE_EMBED_* environment (settings loader ran first).

    ``trust_env`` threads AGENTS.md §3's ``VPC_USE_PROXY=0`` direct-connection
    override into every outbound httpx call this module makes.
    """
    return ImageEmbedSettings(
        model=os.environ.get("VPC_IMAGE_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        api_key=os.environ.get("VPC_IMAGE_EMBED_API_KEY", ""),
        base_url=os.environ.get("VPC_IMAGE_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL),
        timeout=float(os.environ.get("VPC_IMAGE_EMBED_TIMEOUT", "30") or 30),
        trust_env=load_settings().use_proxy,
    )


def pexels_key(pexels_id: int) -> str:
    """Identity of one library image."""
    return f"pexels:{int(pexels_id)}"


def thumbnail_path(pexels_id: int) -> Path:
    """Local file for one pexels id, flat dir ``{pexels_id}.jpg``."""
    return load_settings().thumbnail_dir / f"{int(pexels_id)}.jpg"


def lookup(conn: psycopg.Connection, key: str, model: str) -> list[float] | None:
    """Vector for (key, model) or None; a hit never touches the file."""
    row = conn.execute(
        "SELECT embedding FROM image_embeddings WHERE key = %s AND model = %s",
        (key, model),
    ).fetchone()
    return parse_vector_literal(row[0]) if row is not None else None


def upsert(conn: psycopg.Connection, key: str, model: str, vector: list[float]) -> None:
    """Persist one vector; (key, model) conflict refreshes the vector only."""
    conn.execute(
        """
        INSERT INTO image_embeddings (key, model, embedding)
        VALUES (%s, %s, %s::vector)
        ON CONFLICT (key, model) DO UPDATE SET
            embedding = EXCLUDED.embedding
        """,
        (key, model, to_vector_literal(vector)),
    )
    conn.commit()


def _row_thumbnail_url(conn: psycopg.Connection, pexels_id: int) -> str:
    """CDN fallback URL from the stock row, empty when unknown."""
    row = conn.execute(
        "SELECT thumbnail_url FROM stock_videos WHERE pexels_id = %s", (int(pexels_id),)
    ).fetchone()
    return str(row[0]) if row and row[0] else ""


def image_bytes(conn: psycopg.Connection, pexels_id: int) -> tuple[bytes, str]:
    """(jpeg bytes, source) local-first; CDN only when the file is absent."""
    path = thumbnail_path(pexels_id)
    if path.is_file():
        return path.read_bytes(), "local"
    url = _row_thumbnail_url(conn, pexels_id)
    if not url:
        raise FileNotFoundError(f"no local thumbnail and no thumbnail_url: pexels {pexels_id}")
    settings = embed_settings()
    response = httpx.get(
        url,
        timeout=settings.timeout,
        follow_redirects=True,
        trust_env=settings.trust_env,
    )
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"empty thumbnail response: {url}")
    return response.content, "cdn"


def embed_image(payload: bytes, settings: ImageEmbedSettings) -> tuple[list[float], dict[str, Any]]:
    """Embed one image through DashScope; returns (vector, usage).

    Retries 429 with the fixed backoff sequence; every other failure — HTTP
    error, malformed JSON, missing/empty embedding — raises ValueError so the
    endpoint maps it to a 5xx and the MPT gate fails open. No other exception
    type escapes this function.
    """
    data_uri = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
    body = {"model": settings.model, "input": {"contents": [{"image": data_uri}]}}
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    for attempt, backoff in enumerate((0.0, *_RATE_LIMIT_BACKOFF_SECONDS), start=1):
        if backoff:
            time.sleep(backoff)
        try:
            response = httpx.post(
                settings.base_url,
                headers=headers,
                json=body,
                timeout=settings.timeout,
                trust_env=settings.trust_env,
            )
        except httpx.HTTPError as exc:
            raise ValueError(f"provider request failed: {type(exc).__name__}") from exc
        if response.status_code == 429 and attempt <= len(_RATE_LIMIT_BACKOFF_SECONDS):
            logger.warning("image embed rate limited, attempt=%s", attempt)
            continue
        if response.status_code != 200:
            raise ValueError(f"provider returned status={response.status_code}")
        try:
            result = response.json()
            embedding = result["output"]["embeddings"][0]["embedding"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("malformed provider response") from exc
        if not embedding:
            raise ValueError("empty embedding from provider")
        return list(embedding), dict(result.get("usage") or {})
    raise ValueError("image embed rate limited after retries")


_FLIGHT_GUARD = threading.Lock()
_FLIGHT: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    """One lock per key so concurrent funnels embed one image once."""
    with _FLIGHT_GUARD:
        return _FLIGHT.setdefault(key, threading.Lock())


def _release_key_lock(key: str) -> None:
    """Drop the per-key lock once the flight is done so the registry stays bounded."""
    with _FLIGHT_GUARD:
        _FLIGHT.pop(key, None)


def _payload(
    key: str, model: str, vector: list[float], cached: bool, image: str
) -> dict[str, Any]:
    """Endpoint response body."""
    return {
        "key": key,
        "model": model,
        "dim": len(vector),
        "vector": vector,
        "cached": cached,
        "image": image,
    }


def get_or_embed(pexels_id: int) -> dict[str, Any]:
    """Lazy vector for one pexels id.

    Cache hit: one DB lookup, no file read, no provider call. Miss: single
    flight, local-first image read, one DashScope embed, upsert. Every path —
    including the race-loser that finds the row inside the lock — logs one
    usage line (cached, image source, tokens, latency) for the A/B cost axis.
    """
    key = pexels_key(pexels_id)
    settings = embed_settings()
    started = time.monotonic()
    conn = db.connect()
    try:
        vector = lookup(conn, key, settings.model)
        if vector is not None:
            logger.info(
                "image embed hit: key=%s model=%s image=none latency_ms=%.1f",
                key,
                settings.model,
                (time.monotonic() - started) * 1000,
            )
            return _payload(key, settings.model, vector, cached=True, image="none")
    finally:
        conn.close()

    with _key_lock(key):
        try:
            conn = db.connect()
            try:
                vector = lookup(conn, key, settings.model)
                if vector is not None:  # another funnel won the race — also a hit
                    logger.info(
                        "image embed hit: key=%s model=%s image=none latency_ms=%.1f",
                        key,
                        settings.model,
                        (time.monotonic() - started) * 1000,
                    )
                    return _payload(key, settings.model, vector, cached=True, image="none")
                payload, source = image_bytes(conn, pexels_id)
                vector, usage = embed_image(payload, settings)
                upsert(conn, key, settings.model, vector)
            finally:
                conn.close()
            tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            logger.info(
                "image embed miss: key=%s model=%s image=%s tokens=%s latency_ms=%.1f",
                key,
                settings.model,
                source,
                tokens,
                (time.monotonic() - started) * 1000,
            )
            return _payload(key, settings.model, vector, cached=False, image=source)
        finally:
            _release_key_lock(key)
