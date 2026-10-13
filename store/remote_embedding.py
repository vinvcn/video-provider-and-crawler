"""Remote dense embedding provider: OpenAI-compatible POST {base}/embeddings.

Speaks the same surface as SiliconFlow / DashScope compatible-mode / vLLM
(`bench/embedding_client.py` proven this shape in the three evaluation rounds).
Config comes from VPC_EMBED_* environment variables (already populated in the
repo's .env by the bench work): BASE_URL, API_KEY, MODEL, optional DIM and
BATCH. Network goes through the environment proxy (httpx trust_env).

Only `dense_text` is supported; the sparse arm of the hybrid is the local BM25
provider (store/bm25.py), so the two production passes never collide.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from store.embedding import EmbeddingSpec

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024


class EmbeddingEndpointError(RuntimeError):
    """The embeddings endpoint kept failing after the configured retries."""


class RemoteEmbeddingProvider:
    """Dense-text embeddings from an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        batch: int | None = None,
        retries: int = 3,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ["VPC_EMBED_BASE_URL"]).rstrip("/")
        self.model_name = model or os.environ.get("VPC_EMBED_MODEL", DEFAULT_MODEL)
        self.dim = dim or int(os.environ.get("VPC_EMBED_DIM", "") or DEFAULT_DIM)
        self.retries = retries
        self.total_tokens = 0
        self.spec = EmbeddingSpec(
            model_id=f"{self.model_name}@{self.dim}",
            dim=self.dim,
            distance="cosine",
            max_batch=batch or int(os.environ.get("VPC_EMBED_BATCH", "16")),
            supports=frozenset({"dense_text"}),
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key or os.environ['VPC_EMBED_API_KEY']}"},
            timeout=httpx.Timeout(timeout),
            trust_env=True,
            transport=transport,
        )

    def dense_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in spec.max_batch chunks; returns vectors in order."""
        if not texts:
            return []
        if len(texts) > self.spec.max_batch:
            raise ValueError("batch larger than spec.max_batch")
        vectors = self._embed_batch(list(texts))
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingEndpointError(
                    f"endpoint returned {len(vector)} dims, expected {self.dim}"
                )
        return vectors

    def dense_image(self, images: Sequence[bytes]) -> list[list[float]]:
        raise NotImplementedError("remote text provider is text-only")

    def sparse_text(self, texts: Sequence[str]) -> list[dict[int, float]]:
        raise NotImplementedError("remote text provider is dense-only")

    def close(self) -> None:
        self._client.close()

    def _embed_batch(self, chunk: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model_name, "input": chunk}
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = self._client.post("/embeddings", json=body)
                if response.status_code == 200:
                    payload = response.json()
                    data = sorted(payload["data"], key=lambda item: item["index"])
                    self.total_tokens += int(payload.get("usage", {}).get("total_tokens", 0))
                    return [item["embedding"] for item in data]
                if response.status_code != 429 and response.status_code < 500:
                    raise EmbeddingEndpointError(
                        f"embeddings endpoint rejected the request: "
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(2**attempt)
        raise EmbeddingEndpointError(
            f"embeddings failed after {self.retries + 1} attempts: {last_error}"
        )
