"""Unit tests for the v3 response normalizer (pure logic, no I/O)."""

from crawler.pexels_v3 import (
    build_search_url,
    build_seed_url,
    normalize_video,
    request_headers,
)

FIXTURE = {
    "id": 10395606,
    "slug": "sunlight-seen-through-leaves",
    "title": "Sunlight Seen through Leaves",
    "description": "Calm morning scene of grass silhouetted against a sunrise sky.",
    "tags": ["nature", "sunrise"],
    "duration": 20,
    "width": 1080,
    "height": 1920,
    "fps": 24.0,
    "aspect_ratio": 0.5625,
    "license": "Pexels",
    "user": {"first_name": "Dmitry", "last_name": "Marchenkov", "username": "electrotrack"},
    "created_at": "2021-11-30T10:38:43.000Z",
    "video": {
        "download_link": "https://www.pexels.com/download/video/10395606/",
        "thumbnail": {
            "large": "https://images.pexels.com/videos/10395606/pexels-photo-10395606.jpeg"
        },
        "video_files": [
            {
                "quality": "hd",
                "width": 1080,
                "height": 1920,
                "fps": 24.0,
                "link": "https://videos.pexels.com/video-files/10395606/10395606-hd_1080_1920_24fps.mp4",
            }
        ],
    },
}


def test_normalize_core_fields():
    row = normalize_video(FIXTURE)
    assert row["pexels_id"] == 10395606
    assert row["orientation"] == "portrait"
    assert row["tags"] == ["nature", "sunrise"]
    assert row["user_name"] == "Dmitry Marchenkov"
    assert row["thumbnail_url"].endswith(".jpeg")
    assert row["video_files"][0]["quality"] == "hd"
    assert row["download_link"].endswith("/download/video/10395606/")
    assert row["raw"] == FIXTURE


def test_orientation_variants():
    assert normalize_video({**FIXTURE, "width": 1920, "height": 1080})["orientation"] == "landscape"
    assert normalize_video({**FIXTURE, "width": 100, "height": 100})["orientation"] == "square"
    assert normalize_video({**FIXTURE, "width": None})["orientation"] is None


def test_thumbnail_string_variant():
    attrs = {**FIXTURE, "video": {"thumbnail": "https://example.com/t.jpg"}}
    assert normalize_video(attrs)["thumbnail_url"] == "https://example.com/t.jpg"


def test_thumbnail_missing_falls_back_to_preview():
    attrs = {**FIXTURE, "video": {"preview_src": "https://example.com/p.mp4"}}
    assert normalize_video(attrs)["thumbnail_url"] == "https://example.com/p.mp4"


def test_missing_video_block_is_tolerated():
    row = normalize_video({**FIXTURE, "video": None})
    assert row["video_files"] == []
    assert row["thumbnail_url"] is None


def test_url_builders():
    assert "&seed=" not in build_seed_url(None)
    quoted = build_seed_url("2026-10-01T13:15:28.000Z")
    assert "&seed=2026-10-01T13%3A15%3A28.000Z" in quoted
    search = build_search_url("panda eating", 2)
    assert "query=panda%20eating" in search
    assert "page=2" in search
    headers = request_headers("secret-value")
    assert headers == {"secret-key": "secret-value", "x-client-type": "react"}
