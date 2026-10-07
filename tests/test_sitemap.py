"""Unit tests for sitemap parsing and spooling helpers (pure logic, no I/O)."""

from crawler.sitemap import parse_index, parse_query_shard, parse_video_shard

INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://www.pexels.com/sitemaps/en-US/video-sitemap1.xml.gz</loc><lastmod>2026-10-04T20:05:21Z</lastmod></sitemap>
  <sitemap><loc>https://www.pexels.com/sitemaps/en-US/video-sitemap2.xml.gz</loc></sitemap>
</sitemapindex>"""

VIDEO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.pexels.com/video/typing-on-keyboard-of-macbook-pro-852057/</loc><lastmod>2021-05-04T10:00:00Z</lastmod></url>
  <url><loc>https://www.pexels.com/video/bikers-and-carriages-driving-on-street-852038/</loc></url>
  <url><loc>https://www.pexels.com/video/4k-drone-shot-12345/</loc><lastmod>2026-01-02T03:04:05Z</lastmod></url>
  <url><loc>https://www.pexels.com/search/videos/ocean/</loc></url>
</urlset>"""

QUERY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.pexels.com/search/videos/ocean%20waves/</loc><lastmod>2026-09-25T09:59:51Z</lastmod></url>
  <url><loc>https://www.pexels.com/search/videos/%60friendly'/</loc></url>
  <url><loc>https://www.pexels.com/video/some-video-123/</loc></url>
</urlset>"""


def test_parse_index_keeps_shards_in_order():
    shards = parse_index(INDEX_XML)
    assert [shard.loc for shard in shards] == [
        "https://www.pexels.com/sitemaps/en-US/video-sitemap1.xml.gz",
        "https://www.pexels.com/sitemaps/en-US/video-sitemap2.xml.gz",
    ]
    assert shards[0].lastmod == "2026-10-04T20:05:21Z"
    assert shards[1].lastmod is None


def test_parse_video_shard_extracts_slug_and_id():
    rows = parse_video_shard(VIDEO_XML)
    assert [(row.pexels_id, row.slug) for row in rows] == [
        (852057, "typing-on-keyboard-of-macbook-pro"),
        (852038, "bikers-and-carriages-driving-on-street"),
        (12345, "4k-drone-shot"),
    ]
    assert rows[0].lastmod == "2021-05-04T10:00:00Z"
    assert rows[1].lastmod is None


def test_parse_query_shard_decodes_terms_and_skips_videos():
    rows = parse_query_shard(QUERY_XML)
    assert rows == [("ocean waves", "2026-09-25T09:59:51Z"), ("`friendly'", None)]
