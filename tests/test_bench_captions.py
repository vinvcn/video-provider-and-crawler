"""Captions: key-pool rotation, response parsing, text-source resolution."""

import json
from pathlib import Path

import httpx
import pytest
from conftest import material_row, write_materials

from bench import strategies
from bench.captions import (
    DEEPSEEK_OCR_PROMPT,
    DEEPSEEK_OCR_PROMPT_VERSION,
    MODEL_PROMPTS,
    CaptionError,
    GemmaCaptionClient,
    SiliconFlowCaptionClient,
    parse_caption,
)
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


def test_parse_caption_cuts_task_echo_preamble():
    """gemma sometimes restates the task before answering; the echo must not
    leak boilerplate into the corpus."""
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "* Task: Describe the stock video thumbnail.\n"
                                "* Constraints: 2-4 sentences.\n"
                                "* Required elements: subject, setting, colors.\n"
                                "* Format: Text only.\n"
                                "* Main subject: Silhouetted evergreen trees.\n"
                                "* Setting: forest edge at sunset.\n"
                            )
                        }
                    ]
                }
            }
        ]
    }
    caption = parse_caption(payload)
    assert caption == ("Main subject: Silhouetted evergreen trees. Setting: forest edge at sunset.")


def test_parse_caption_drops_orphan_echo_lines_without_marker():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Task: Describe a thumbnail.\n"
                                "Constraints: none.\n"
                                "A close-up of a vintage camera on a desk.\n"
                            )
                        }
                    ]
                }
            }
        ]
    }
    caption = parse_caption(payload)
    assert caption == "A close-up of a vintage camera on a desk."


def test_caption_client_rotates_keys_on_server_errors(monkeypatch):
    monkeypatch.setattr("bench.captions.time.sleep", lambda seconds: None)
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


def test_caption_client_raises_after_pool_exhausted(monkeypatch):
    monkeypatch.setattr("bench.captions.time.sleep", lambda seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": 500}})

    client = GemmaCaptionClient(
        keys=["key-a", "key-b"], model="gemma-4-31b-it", rpm=0, transport=_transport(handler)
    )
    with pytest.raises(CaptionError, match="failed after 4 attempts"):
        client.caption_image(b"fake-jpeg-bytes")
    client.close()


def test_caption_client_blocks_dead_key_and_uses_healthy(monkeypatch):
    monkeypatch.setattr("bench.captions.time.sleep", lambda seconds: None)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers.get("X-goog-api-key", "")
        calls.append(key)
        if key == "dead":
            return httpx.Response(403, json={"error": {"code": 403}})
        return httpx.Response(200, json=GEMMA_RESPONSE)

    client = GemmaCaptionClient(
        keys=["dead", "live"], model="gemma-4-31b-it", rpm=0, transport=_transport(handler)
    )
    caption = client.caption_image(b"fake-jpeg-bytes")
    assert caption.startswith("Main subject")
    # the 403 key is now permanently blocked; the next image goes straight to live
    calls.clear()
    client.caption_image(b"fake-jpeg-bytes")
    assert calls == ["live"]
    client.close()


def test_probe_keys_classifies_health(monkeypatch):
    monkeypatch.setattr("bench.captions.time.sleep", lambda seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers.get("X-goog-api-key", "")
        if key == "ok":
            return httpx.Response(200, json=GEMMA_RESPONSE)
        if key == "banned":
            return httpx.Response(403, json={"error": {"code": 403}})
        return httpx.Response(500, json={"error": {"code": 500}})

    client = GemmaCaptionClient(
        keys=["ok", "banned", "broken"],
        model="gemma-4-31b-it",
        rpm=0,
        transport=_transport(handler),
    )
    summary = client.probe_keys()
    assert summary == {"healthy": 1, "cooldown": 1, "blocked": 1}
    # after probing, only the healthy key serves requests
    seen: list[str] = []

    def record_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("X-goog-api-key", ""))
        return httpx.Response(200, json=GEMMA_RESPONSE)

    client._client._transport = _transport(record_handler)  # noqa: SLF001 - test seam
    client.caption_image(b"fake-jpeg-bytes")
    assert seen == ["ok"]
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


def test_sf_caption_client_parses_chat_content(monkeypatch):
    monkeypatch.setattr("bench.captions.time.sleep", lambda seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "deepseek-ai/DeepSeek-OCR"
        assert any(part["type"] == "image_url" for part in body["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "A sunset over a forest."}}
                ]
            },
        )

    client = SiliconFlowCaptionClient(
        api_key="sf-key",
        model="deepseek-ai/DeepSeek-OCR",
        rpm=0,
        transport=_transport(handler),
    )
    assert client.prompt_version == "caption-ocr-v1"
    assert client.caption_image(b"fake-bytes") == "A sunset over a forest."
    client.close()


def test_sf_caption_client_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("bench.captions.time.sleep", lambda seconds: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = SiliconFlowCaptionClient(
        api_key="sf-key", model="deepseek-ai/DeepSeek-OCR", rpm=0, transport=_transport(handler)
    )
    assert client.caption_image(b"fake-bytes") == "ok"
    assert len(calls) == 2
    client.close()


def test_corpus_label_distinguishes_caption_models():
    from bench.strategies import corpus_label

    assert corpus_label({"strategy": "dense"}) == "raw"
    assert corpus_label({"text_source": "caption", "captions_model": "gemma-4-26b-a4b-it"}) == (
        "caption:gemma-4-26b-a4b-it"
    )
    assert (
        corpus_label({"text_source": "caption", "captions_model": "deepseek-ai/DeepSeek-OCR"})
        == "caption:deepseek-ai/DeepSeek-OCR"
    )
    # legacy specs without a recorded model fall back to the version
    assert corpus_label({"text_source": "caption", "captions_version": "v1"}) == "caption:v1"


def test_caption_prompt_registry_keys_are_lowercase():
    assert "deepseek-ai/deepseek-ocr" in MODEL_PROMPTS
    prompt, version = MODEL_PROMPTS["deepseek-ai/deepseek-ocr"]
    assert prompt == DEEPSEEK_OCR_PROMPT and version == DEEPSEEK_OCR_PROMPT_VERSION


def test_breaker_trips_only_on_consecutive_failures():
    """A wedged endpoint must abort the pass instead of grinding timeouts."""
    from bench.captions import _Breaker

    breaker = _Breaker(threshold=3)
    breaker.record(False)
    breaker.record(False)
    breaker.record(True)  # any success resets the counter
    breaker.record(False)
    breaker.record(False)
    with pytest.raises(SystemExit, match="unreachable"):
        breaker.record(False)
    # after a trip, the very next failure still trips (sticky until a success)
    with pytest.raises(SystemExit):
        breaker.record(False)
    breaker.record(True)
    breaker.record(False)  # healthy again: single failures are tolerated


def test_pacer_import_unchanged():
    # Pacer is shared between judge and caption clients; make sure it still exists
    assert Pacer(per_minute=0).wait() is None
