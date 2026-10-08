"""Per-video metadata through Pexels' Next.js data route (gap-fill channel).

```
https://www.pexels.com/_next/data/<buildId>/en-us/video/<slug>-<id>.json
```

returns the page props as JSON; `pageProps.medium.attributes` is the same
attribute object the internal v3 list API returns, so `normalize_video` handles
it unchanged. The route sits outside Cloudflare's HTML challenge (unlike
`/video/<slug>-<id>/` itself), which makes full-coverage gap filling possible.

`buildId` comes from any loaded pexels page (`window.__NEXT_DATA__.buildId`); the
site rotates it on deploys, and a stale id yields 404s — the crawler re-reads it
when it sees a run of 404s.
"""

from __future__ import annotations

from typing import Any

DATA_BASE = "https://www.pexels.com/_next/data"
LOCALE = "en-us"

#: Template carried into the in-page script, which fills `{build}` at runtime.
DATA_URL_TEMPLATE = f"{DATA_BASE}/{{build}}/{LOCALE}/video/{{slug}}-{{id}}.json"


def build_data_url(build_id: str, slug: str, pexels_id: int | str) -> str:
    """Compose the Next data URL for one video."""
    return DATA_URL_TEMPLATE.format(build=build_id, slug=slug, id=pexels_id)


def extract_attributes(payload: Any) -> dict[str, Any] | None:
    """Pull the v3-shaped attribute object out of a Next data payload.

    Returns None when the payload is not a usable video page (error body,
    missing props, or a shape we do not understand).
    """
    if not isinstance(payload, dict):
        return None
    props = payload.get("pageProps")
    if not isinstance(props, dict):
        return None
    medium = props.get("medium")
    if not isinstance(medium, dict):
        return None
    attributes = medium.get("attributes")
    if not isinstance(attributes, dict) or "id" not in attributes:
        return None
    return attributes
