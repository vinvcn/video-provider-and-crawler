"""Strategy arms for the benchmark: BM25 (both idf modes), dense, RRF hybrid.

Every arm implements the same contract: `search(query_text, k, hard)` returning
ranked (doc_id, score) hits over the frozen materials set, honouring the
query's hard constraints (in-memory filtering), with deterministic tie-breaks
by ascending doc_id so identical inputs always produce identical runs.

The arms run exact search (10k rows): no ANN index is involved. Comparing ANN
structures is a later, larger-corpus question (docs/phase2-index-design §4).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from bench import util
from bench.embedding_client import EmbeddingClient
from bench.materials import Materials
from store.bm25 import Bm25Model
from store.embedding import HashEmbeddingProvider

if TYPE_CHECKING:  # runtime import would create a cycle: captions -> judge -> run -> strategies
    from bench.captions import Captions

RRF_K = 60
RRF_ARM_DEPTH = 100
BM25_DEFAULTS = {"k1": 1.2, "b": 0.75}

STRATEGY_IDS = ("bm25-both", "bm25-query", "hash-dense", "dense", "rrf")
TEXT_SOURCES = ("raw", "caption")


def resolve_texts(
    materials: Materials, captions: Captions | None, text_source: str
) -> dict[int, str]:
    """Per-doc retrieval text for one source (caption docs may be missing -> '')."""
    if text_source not in TEXT_SOURCES:
        raise SystemExit(f"unknown text source {text_source!r}; expected one of {TEXT_SOURCES}")
    if text_source == "raw":
        return {doc_id: materials.rows[doc_id].embed_text for doc_id in materials.ids}
    if captions is None:
        raise SystemExit("text source 'caption' needs a built captions version")
    return {doc_id: captions.texts.get(doc_id, "") for doc_id in materials.ids}


@dataclass(frozen=True)
class SearchResult:
    """Ranked hits for one query plus timing attribution."""

    hits: list[tuple[int, float]]
    search_ms: float
    embed_ms: float


class Strategy(Protocol):
    """Contract every benchmark arm fulfils."""

    spec_id: str

    def search(self, query_text: str, k: int, hard: dict) -> SearchResult: ...


def _rank_hits(scored: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Top-k by (score desc, doc_id asc) — the deterministic tie-break."""
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, score) for doc_id, score in ordered[:k]]


def _allowed_ids(materials: Materials, hard: dict) -> set[int] | None:
    """doc_ids passing the hard constraints; None when unconstrained."""
    if not hard:
        return None
    return set(materials.filter_ids(hard))


class Bm25Strategy:
    """Local BM25 arm; `idf_side` selects legacy ('both') or textbook ('query')."""

    def __init__(
        self, spec_id: str, materials: Materials, idf_side: str, texts: dict[int, str]
    ) -> None:
        self.spec_id = spec_id
        self.materials = materials
        self.doc_ids = list(materials.ids)
        self.model = Bm25Model(idf_side=idf_side, **BM25_DEFAULTS).fit(
            texts[doc_id] for doc_id in self.doc_ids
        )
        self._postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for doc_id in self.doc_ids:
            for term_id, weight in self.model.doc_weights(texts[doc_id]).items():
                self._postings[term_id].append((doc_id, weight))

    def search(self, query_text: str, k: int, hard: dict) -> SearchResult:
        started = time.perf_counter()
        allowed = _allowed_ids(self.materials, hard)
        scored: dict[int, float] = {}
        for term_id, q_weight in self.model.query_weights(query_text).items():
            for doc_id, d_weight in self._postings.get(term_id, ()):
                if allowed is not None and doc_id not in allowed:
                    continue
                scored[doc_id] = scored.get(doc_id, 0.0) + q_weight * d_weight
        hits = _rank_hits(scored, k)
        return SearchResult(
            hits=hits,
            search_ms=(time.perf_counter() - started) * 1000.0,
            embed_ms=0.0,
        )


class QueryEmbedder(Protocol):
    """Query-side embedding (normalized), shared by dense arms."""

    def embed_query(self, text: str) -> np.ndarray: ...


class HashQueryEmbedder:
    """Query embedding via the offline HashEmbeddingProvider."""

    def __init__(self) -> None:
        self._provider = HashEmbeddingProvider()

    def embed_query(self, text: str) -> np.ndarray:
        [vector] = self._provider.dense_text([text])
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr


class ApiQueryEmbedder:
    """Query embedding through the user-provided endpoint (with prefix)."""

    def __init__(self, client: EmbeddingClient, prefix: str) -> None:
        self._client = client
        self._prefix = prefix

    def embed_query(self, text: str) -> np.ndarray:
        [vector] = self._client.embed([self._prefix + text])
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row (zero vectors stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


class DenseStrategy:
    """Exact cosine over a dense matrix (hash provider or API embeddings)."""

    def __init__(
        self,
        spec_id: str,
        materials: Materials,
        matrix: np.ndarray,
        query_embedder: QueryEmbedder,
    ) -> None:
        self.spec_id = spec_id
        self.materials = materials
        self.doc_ids = list(materials.ids)
        self.matrix = matrix
        self._query_embedder = query_embedder

    def search(self, query_text: str, k: int, hard: dict) -> SearchResult:
        embed_started = time.perf_counter()
        query_vector = self._query_embedder.embed_query(query_text)
        embed_ms = (time.perf_counter() - embed_started) * 1000.0
        started = time.perf_counter()
        scores = self.matrix @ query_vector
        if hard:
            allowed = _allowed_ids(self.materials, hard)
            keep = np.zeros(len(self.doc_ids), dtype=bool)
            for position, doc_id in enumerate(self.doc_ids):
                keep[position] = doc_id in allowed  # type: ignore[operator]
            scores = np.where(keep, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        hits = [
            (int(self.doc_ids[index]), float(scores[index]))
            for index in order
            if scores[index] != -np.inf
        ]
        return SearchResult(
            hits=hits,
            search_ms=(time.perf_counter() - started) * 1000.0,
            embed_ms=embed_ms,
        )


class RrfStrategy:
    """RRF fusion of two arms (default: bm25-query + dense), k=60."""

    def __init__(self, spec_id: str, arms: list[Strategy]) -> None:
        self.spec_id = spec_id
        self.arms = arms

    def search(self, query_text: str, k: int, hard: dict) -> SearchResult:
        depth = max(RRF_ARM_DEPTH, k)
        fused: dict[int, float] = {}
        search_ms = 0.0
        embed_ms = 0.0
        for arm in self.arms:
            result = arm.search(query_text, depth, hard)
            search_ms += result.search_ms
            embed_ms += result.embed_ms
            for rank, (doc_id, _) in enumerate(result.hits, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        hits = _rank_hits(fused, k)
        return SearchResult(hits=hits, search_ms=search_ms, embed_ms=embed_ms)


def build_hash_dense(materials: Materials, texts: dict[int, str]) -> DenseStrategy:
    """Offline smoke arm: hashing bag-of-words vectors over the chosen text source."""
    provider = HashEmbeddingProvider()
    doc_ids = list(materials.ids)
    vectors: list[list[float]] = []
    for start in range(0, len(doc_ids), provider.spec.max_batch):
        chunk = doc_ids[start : start + provider.spec.max_batch]
        vectors.extend(provider.dense_text([texts[doc_id] for doc_id in chunk]))
    matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
    return DenseStrategy("hash-dense", materials, matrix, HashQueryEmbedder())


def _embeddings_cache_dir(
    root: Path, model: str, dim: int | None, materials: Materials, text_key: str
) -> Path:
    key = f"{model.replace('/', '_')}-{dim or 'na'}-{text_key}-{materials.content_hash[:16]}"
    return root / "embeddings" / key


def build_api_dense(
    materials: Materials,
    client: EmbeddingClient,
    root: Path,
    texts: dict[int, str],
    text_key: str = "raw",
) -> DenseStrategy:
    """Dense arm over the user endpoint; materials vectors cached on disk."""
    doc_ids = list(materials.ids)
    cache_dir = _embeddings_cache_dir(
        root, client.config.model, client.config.dim, materials, text_key=text_key
    )
    vectors_path = cache_dir / "vectors.npy"
    tokens_before = client.total_tokens
    if not vectors_path.is_file():
        ordered_texts = [texts[doc_id] for doc_id in doc_ids]
        client.total_tokens = 0
        vectors = client.embed(ordered_texts)
        # docs with no text (missing caption) get a zero vector: findable by
        # filters, never by similarity — the fail-open rule for missing values.
        for position, text in enumerate(ordered_texts):
            if not text:
                vectors[position] = [0.0] * len(vectors[position])
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, np.asarray(vectors, dtype=np.float32))
        util.write_json(
            cache_dir / "meta.json",
            {
                "model": client.config.model,
                "dim": client.config.dim,
                "dim_effective": int(len(vectors[0])),
                "text_key": text_key,
                "materials_version": materials.version,
                "materials_hash": materials.content_hash,
                "tokens": client.total_tokens,
            },
        )
    matrix = _normalize_rows(np.load(vectors_path).astype(np.float32))
    strategy = DenseStrategy(
        "dense", materials, matrix, ApiQueryEmbedder(client, client.config.query_prefix)
    )
    # Cost accounting: what this run actually spent on the materials side
    # (0 on a cache hit), plus the effective dimension of the cached matrix.
    strategy.materials_tokens_spent = client.total_tokens - tokens_before  # type: ignore[attr-defined]
    strategy.dim_effective = int(matrix.shape[1])  # type: ignore[attr-defined]
    return strategy


def build_strategy(
    spec_id: str,
    materials: Materials,
    root: Path,
    embed_client: EmbeddingClient | None,
    text_source: str = "raw",
    captions: Captions | None = None,
) -> Strategy:
    """Registry: spec_id -> wired arm (validated against STRATEGY_IDS)."""
    if spec_id not in STRATEGY_IDS:
        raise SystemExit(f"unknown strategy {spec_id!r}; expected one of {STRATEGY_IDS}")
    texts = resolve_texts(materials, captions, text_source)
    text_key = "raw" if text_source == "raw" else f"caption-{captions.content_hash[:16]}"
    if spec_id == "bm25-both":
        return Bm25Strategy(spec_id, materials, idf_side="both", texts=texts)
    if spec_id == "bm25-query":
        return Bm25Strategy(spec_id, materials, idf_side="query", texts=texts)
    if spec_id == "hash-dense":
        return build_hash_dense(materials, texts)
    if spec_id == "dense":
        if embed_client is None:
            raise SystemExit("strategy 'dense' needs an embeddings endpoint (.env VPC_EMBED_*)")
        return build_api_dense(materials, embed_client, root, texts, text_key=text_key)
    if embed_client is None:
        raise SystemExit("strategy 'rrf' needs an embeddings endpoint (.env VPC_EMBED_*)")
    bm25 = Bm25Strategy("bm25-query", materials, idf_side="query", texts=texts)
    dense = build_api_dense(materials, embed_client, root, texts, text_key=text_key)
    rrf = RrfStrategy("rrf", [bm25, dense])
    rrf.materials_tokens_spent = getattr(dense, "materials_tokens_spent", 0)  # type: ignore[attr-defined]
    rrf.dim_effective = getattr(dense, "dim_effective", None)  # type: ignore[attr-defined]
    return rrf


def strategy_config(
    spec_id: str,
    embed_client: EmbeddingClient | None,
    text_source: str = "raw",
    captions_version: str | None = None,
) -> dict:
    """Serializable spec (recorded in run manifests; hashed into run ids)."""
    spec: dict[str, object] = {"strategy": spec_id, "text_source": text_source}
    if text_source == "caption":
        spec["captions_version"] = captions_version
    if spec_id in ("bm25-both", "bm25-query"):
        spec["idf_side"] = "both" if spec_id == "bm25-both" else "query"
        spec.update(BM25_DEFAULTS)
    if spec_id == "hash-dense":
        spec["provider"] = "local-hash-768"
    if spec_id in ("dense", "rrf") and embed_client is not None:
        spec["model"] = embed_client.config.model
        spec["dim"] = embed_client.config.dim
        spec["query_prefix"] = bool(embed_client.config.query_prefix)
    if spec_id == "rrf":
        spec.update({"rrf_k": RRF_K, "arm_depth": RRF_ARM_DEPTH, "arms": ["bm25-query", "dense"]})
    return spec


def config_hash(spec: dict) -> str:
    """Stable hash of the resolved strategy configuration."""
    return util.content_hash(spec)
