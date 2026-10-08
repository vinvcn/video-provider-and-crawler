"""Unit tests for the Next.js data route helpers (pure logic, no I/O)."""

from crawler.nextdata import DATA_URL_TEMPLATE, build_data_url, extract_attributes

PAYLOAD = {
    "pageProps": {
        "id": "40083338",
        "medium": {
            "id": "40083338",
            "type": "video",
            "attributes": {"id": 40083338, "slug": "wildebeest-migration-across-river"},
        },
    },
    "__lang": "en-US",
}


def test_build_data_url_matches_template():
    url = build_data_url("QdeoVNfzoQpFEtMbcyw4_", "some-slug", 852038)
    assert url == (
        "https://www.pexels.com/_next/data/QdeoVNfzoQpFEtMbcyw4_/en-us/video/some-slug-852038.json"
    )
    assert DATA_URL_TEMPLATE.format(build="b", slug="s", id=9).endswith("/en-us/video/s-9.json")


def test_extract_attributes_returns_v3_shape():
    attributes = extract_attributes(PAYLOAD)
    assert attributes is not None
    assert attributes["id"] == 40083338
    assert attributes["slug"] == "wildebeest-migration-across-river"


def test_extract_attributes_rejects_unusable_payloads():
    assert extract_attributes(None) is None
    assert extract_attributes("nope") is None
    assert extract_attributes({}) is None
    assert extract_attributes({"pageProps": {"medium": {}}}) is None
    assert extract_attributes({"pageProps": {"medium": {"attributes": {"slug": "x"}}}}) is None
