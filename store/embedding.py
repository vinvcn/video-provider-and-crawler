"""Embedding seam (Phase 2): provider contract and canonical embedding text.

The seam keeps the model choice open (docs/handoff-2026-10-07.md §5): dense and
sparse backends implement the same small protocol, corpus and query go through
the same provider, and each row records the model id that produced its vector so
two models can coexist while a swap is evaluated.

Only the offline-testable half lives here: the contract, the canonical text
builder, and a deterministic local hashing provider for pipeline tests. Concrete
backends (DashScope tongyi / BGE-M3 / BM25) and the DB writer arrive with the
Phase-2 implementation ticket.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

DENSE_MODALITIES = ("dense_text", "dense_image")
KNOWN_MODALITIES = frozenset({"dense_text", "dense_image", "sparse_text"})

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EmbeddingSpec:
    """What a provider guarantees: identity, dimensionality, metric, batch cap."""

    model_id: str
    dim: int = 0
    distance: str = "cosine"
    max_batch: int = 16
    supports: frozenset[str] = field(default_factory=lambda: frozenset({"dense_text"}))

    def __post_init__(self) -> None:
        unknown = set(self.supports) - KNOWN_MODALITIES
        if unknown:
            raise ValueError(f"unknown modality: {sorted(unknown)}")
        if not self.supports:
            raise ValueError("a provider must support at least one modality")
        if self.distance not in ("cosine", "l2", "ip"):
            raise ValueError(f"unsupported distance: {self.distance}")
        if self.max_batch <= 0:
            raise ValueError("max_batch must be positive")
        if self.dim <= 0 and any(mod in self.supports for mod in DENSE_MODALITIES):
            raise ValueError("dense providers need a positive dim")
        if self.dim < 0:
            raise ValueError("dim cannot be negative")


class EmbeddingProvider(Protocol):
    """Dense text/image and sparse text embedding for corpus and query alike."""

    spec: EmbeddingSpec

    def dense_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts into `spec.dim`-dimensional dense vectors."""
        ...

    def dense_image(self, images: Sequence[bytes]) -> list[list[float]]:
        """Embed a batch of images into the same dense space as `dense_text`."""
        ...

    def sparse_text(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Embed a batch of texts into sparse `{token_id: weight}` maps."""
        ...


def build_embed_text(row: Mapping[str, Any]) -> str:
    """Canonical embedding text for one stock_videos row (title, description, tags).

    The same function must produce the corpus side and (with the same wording
    rules) the query side; keep it whitespace-normalised and stable.
    """
    parts = [
        str(row.get("title") or "").strip(),
        str(row.get("description") or "").strip(),
    ]
    tags = [str(tag).strip() for tag in (row.get("tags") or [])]
    tags = [tag for tag in tags if tag]
    if tags:
        parts.append(", ".join(tags))
    text = " — ".join(part for part in parts if part)
    return _WS_RE.sub(" ", text).strip()


@dataclass
class HashEmbeddingProvider:
    """Deterministic local provider (no network): exercises the pipeline offline.

    Not for retrieval quality — it is a hashing bag-of-words so tests and dry
    runs can move data through the seam without API keys.
    """

    spec: EmbeddingSpec = field(
        default_factory=lambda: EmbeddingSpec(
            model_id="local-hash-64",
            dim=64,
            supports=frozenset({"dense_text", "sparse_text"}),
        )
    )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def dense_text(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) > self.spec.max_batch:
            raise ValueError("batch larger than spec.max_batch")
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.spec.dim
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                vector[int.from_bytes(digest, "big") % self.spec.dim] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            out.append([value / norm for value in vector])
        return out

    def sparse_text(self, texts: Sequence[str]) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
                token_id = int.from_bytes(digest, "big")
                counts[token_id] = counts.get(token_id, 0.0) + 1.0
            out.append(counts)
        return out

    def dense_image(self, images: Sequence[bytes]) -> list[list[float]]:
        raise NotImplementedError("hash provider is text-only")
