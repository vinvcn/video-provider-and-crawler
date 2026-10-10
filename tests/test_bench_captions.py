"""Captions: key-pool rotation, response parsing, text-source resolution."""

from pathlib import Path

import httpx
import pytest
from conftest import material_row, write_materials

from bench import strategies
from bench.captions import CaptionError, GemmaCaptionClient, parse_caption
from bench.captions import Captions as CaptionsCorpus
from bench.judge import Pacer

GEMMA_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "* Main subject: Great Wall of China.\n"},
                    {"text": "* Setting: mountainous landscape at golden hour.\n"},
                ]
            }
        }
    ]
}


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_parse_caption_joins_parts_and_strips_markdown():
    caption = parse_caption(GEMMA_RESPONSE)
    assert caption == (
        "Main subject: Great Wall of China. Setting: mountainous landscape at golden hour."
    )
    assert parse_caption({"candidates": []}) == ""


def test_caption_client_rotates_keys_on_server_errors():
    seen_keys = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers.get("X-goog-api-key", "")
        seen_keys.append(key)
        if len(seen_keys) == 1:
            return httpx.Response(500, json={"error": {"code": 500}})
        return httpx.Response(200, json=GEMMA_RESPONSE)

    client = GemmaCaptionClient(
        keys=["key-a", "key-b"], model="gemma-4-31b-it", rpm=0, transport=_transport(handler)
    )
    caption = client.caption_image(b"fake-jpeg-bytes")
    client.close()
    assert caption.startswith("Main subject: Great Wall")
    assert seen_keys == ["key-a", "key-b"]


def test_caption_client_raises_after_pool_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": 500}})

    client = GemmaCaptionClient(
        keys=["key-a", "key-b"], model="gemma-4-31b-it", rpm=0, transport=_transport(handler)
    )
    with pytest.raises(CaptionError, match="failed after 4 attempts"):
        client.caption_image(b"fake-jpeg-bytes")
    client.close()


def test_caption_client_rejects_empty_key_pool():
    with pytest.raises(CaptionError, match="at least one"):
        GemmaCaptionClient(keys=[], model="x", rpm=0)


def test_resolve_texts_raw_and_caption(bench_root: Path):
    rows = [material_row(1, "Ocean Waves at Sunset", tags=("ocean",))]
    materials = write_materials(bench_root, "v1", rows)
    captions = CaptionsCorpus(
        version="c1",
        texts={1: "Surfers ride a wave at dusk."},
        content_hash="abc123",
        manifest={"materials_version": "v1"},
    )

    raw = strategies.resolve_texts(materials, None, "raw")
    assert raw[1].startswith("Ocean Waves at Sunset")
    cap = strategies.resolve_texts(materials, captions, "caption")
    assert cap[1] == "Surfers ride a wave at dusk."
    # docs without a caption stay retrievable by filters only
    cap_missing = strategies.resolve_texts(
        materials,
        CaptionsCorpus(version="c1", texts={}, content_hash="abc", manifest={}),
        "caption",
    )
    assert cap_missing[1] == ""

    with pytest.raises(SystemExit):
        strategies.resolve_texts(materials, None, "caption")
    with pytest.raises(SystemExit):
        strategies.resolve_texts(materials, None, "subtitles")


def test_bm25_over_captions_only(bench_root: Path):
    rows = [
        material_row(1, "Boring Raw Title", tags=("rawtag",)),
        material_row(2, "Ocean Waves at Sunset", tags=("ocean", "waves")),
    ]
    materials = write_materials(bench_root, "v1", rows)
    texts = {1: "A dog runs through shallow ocean waves.", 2: "A city skyline at night."}
    arm = strategies.Bm25Strategy("bm25-query", materials, idf_side="query", texts=texts)
    # the caption tokens drive retrieval, not the raw titles
    result = arm.search("dog waves", k=2, hard={})
    assert result.hits[0][0] == 1
    result_raw_only = arm.search("rawtag", k=2, hard={})
    assert result_raw_only.hits == []


def test_strategy_config_distinguishes_text_sources():
    raw_spec = strategies.strategy_config("bm25-query", None, text_source="raw")
    cap_spec = strategies.strategy_config(
        "bm25-query", None, text_source="caption", captions_version="v1"
    )
    assert raw_spec["text_source"] == "raw"
    assert cap_spec["captions_version"] == "v1"
    assert strategies.config_hash(raw_spec) != strategies.config_hash(cap_spec)


def test_hash_dense_over_captions(bench_root: Path):
    rows = [material_row(1, "Whatever", tags=("x",)), material_row(2, "Else", tags=("y",))]
    materials = write_materials(bench_root, "v1", rows)
    texts = {1: "misty forest morning", 2: ""}
    arm = strategies.build_hash_dense(materials, texts)
    result = arm.search("misty forest", k=2, hard={})
    assert result.hits[0][0] == 1
    # empty-caption docs surface only at the tail with (near-)zero scores
    tail = arm.search("misty forest", k=2, hard={})
    assert all(doc_id in (1, 2) for doc_id, _ in tail.hits)


def test_pacer_import_unchanged():
    # Pacer is shared between judge and caption clients; make sure it still exists
    assert Pacer(per_minute=0).wait() is None
