"""Pexels internal v3 API: URL builders and response normalization.

The `secret-key` header below is a *public frontend constant* shipped in every
pexels.com page's JavaScript (not a personal API key). If the site rotates it,
derive a fresh one from any pexels.com page's network panel, or override with
the VPC_PEXELS_SECRET environment variable — no code change needed.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

BASE = "https://www.pexels.com/en-us/api/v3"
SEED_BASE_URL = f"{BASE}/videos?sort=popular&per_page=24&seo_tags=true"
SEARCH_BASE_URL = f"{BASE}/search/videos"
PEXELS_PUBLIC_SECRET = "H2jk9uKnhRmL6WPwh89zBezWvr"


def request_headers(secret: str) -> dict[str, str]:
    """Headers the Pexels web frontend attaches to v3 API calls."""
    return {"secret-key": secret, "x-client-type": "react"}


def build_seed_url(cursor: str | None = None) -> str:
    """Popular-feed page; pagination is chained via &seed=<previous cursor>."""
    if cursor:
        return f"{SEED_BASE_URL}&seed={quote(cursor, safe='')}"
    return SEED_BASE_URL


def build_search_url(term: str, page: int = 1, per_page: int = 24) -> str:
    """Keyword search page (each query caps at 10,000 results)."""
    return f"{SEARCH_BASE_URL}?query={quote(term, safe='')}&per_page={per_page}&page={max(1, page)}"


def _orientation(width: Any, height: Any) -> str | None:
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if w == h:
        return "square"
    return "portrait" if h > w else "landscape"


def _user_name(user: Any) -> str | None:
    if not isinstance(user, dict):
        return None
    name = " ".join(
        str(user.get(key)).strip() for key in ("first_name", "last_name") if user.get(key)
    ).strip()
    return name or user.get("username") or None


def _thumbnail_url(video: Any) -> str | None:
    if not isinstance(video, dict):
        return None
    thumb = video.get("thumbnail")
    if isinstance(thumb, str) and thumb:
        return thumb
    if isinstance(thumb, dict):
        for key in ("large", "medium", "small"):
            value = thumb.get(key)
            if isinstance(value, str) and value:
                return value
        for value in thumb.values():
            if isinstance(value, str) and value:
                return value
    preview = video.get("preview_src")
    return preview if isinstance(preview, str) and preview else None


def normalize_video(attrs: dict[str, Any]) -> dict[str, Any]:
    """Map one v3 `attributes` object to a stock_videos row (dict)."""
    video = attrs.get("video") if isinstance(attrs.get("video"), dict) else {}
    video_files = video.get("video_files") if isinstance(video.get("video_files"), list) else []
    return {
        "pexels_id": int(attrs["id"]),
        "slug": attrs.get("slug"),
        "title": attrs.get("title"),
        "description": attrs.get("description"),
        "tags": [str(tag) for tag in (attrs.get("tags") or []) if tag],
        "duration": attrs.get("duration"),
        "width": attrs.get("width"),
        "height": attrs.get("height"),
        "fps": attrs.get("fps"),
        "aspect_ratio": attrs.get("aspect_ratio"),
        "orientation": _orientation(attrs.get("width"), attrs.get("height")),
        "license": attrs.get("license"),
        "user_name": _user_name(attrs.get("user")),
        "created_at": attrs.get("created_at"),
        "video_files": video_files,
        "thumbnail_url": _thumbnail_url(video),
        "download_link": video.get("download_link"),
        "source": "pexels",
        "raw": attrs,
    }
