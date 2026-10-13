"""OpenAI-compatible embeddings client for the dense benchmark arms.

Any endpoint speaking POST {base}/embeddings with {model, input[, dimensions]}
works (OpenAI, DashScope compatible-mode, vLLM, ...). Network goes through the
environment proxy (httpx trust_env); retries cover transport errors, 429 and 5xx.
"""

from __future__ import annotations

import time

import httpx

from bench.env import EmbedConfig


class EmbeddingError(RuntimeError):
    """Raised when the embeddings endpoint keeps failing after retries."""


class EmbeddingClient:
    """Batched embeddings with retry/backoff and token accounting."""

    def __init__(self, config: EmbedConfig, max_retries: int = 3) -> None:
        self.config = config
        self.max_retries = max_retries
        self.total_tokens = 0
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(120.0),
            trust_env=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of `config.batch`; returns vectors in order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch):
            chunk = texts[start : start + self.config.batch]
            vectors.extend(self._embed_batch(chunk))
        return vectors

    def close(self) -> None:
        self._client.close()

    def _embed_batch(self, chunk: list[str]) -> list[list[float]]:
        body: dict[str, object] = {"model": self.config.model, "input": chunk}
        if self.config.dim:
            body["dimensions"] = self.config.dim
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post("/embeddings", json=body)
                if response.status_code in (429,) or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                elif response.status_code == 200:
                    payload = response.json()
                    data = sorted(payload["data"], key=lambda item: item["index"])
                    self.total_tokens += int(payload.get("usage", {}).get("total_tokens", 0))
                    return [item["embedding"] for item in data]
                else:
                    raise EmbeddingError(
                        f"embeddings endpoint rejected the request: "
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(2**attempt)
        raise EmbeddingError(
            f"embeddings failed after {self.max_retries + 1} attempts: {last_error}"
        )
