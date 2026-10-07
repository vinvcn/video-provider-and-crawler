"""Local BM25 sparse provider (Phase 2): zero-cost, deterministic sparse vectors.

Fit once over the corpus text (document frequency and length statistics), then
emit sparse vectors keyed by integer ids so they map straight onto pgvector
`sparsevec`. The fitted model serialises to JSON, so the query side reuses the
exact corpus-side statistics.

This is the documented default sparse backend (docs/handoff-2026-10-07.md §5);
it needs no API key and no network.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from store.embedding import EmbeddingSpec, tokenize

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


@dataclass
class Bm25Model:
    """Fitted BM25 statistics: term ids, document frequencies, length averages."""

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    term_ids: dict[str, int] = field(default_factory=dict)
    df: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0
    total_len: int = 0

    @property
    def avg_len(self) -> float:
        """Average document length in tokens (0 when unfitted)."""
        return (self.total_len / self.n_docs) if self.n_docs else 0.0

    def fit(self, docs: Iterable[str]) -> Bm25Model:
        """Accumulate corpus statistics; returns self so it can be chained."""
        for doc in docs:
            tokens = tokenize(doc)
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1
            self.n_docs += 1
            self.total_len += len(tokens)
        # Frequent terms get low ids: deterministic and nicer to eyeball in psql.
        self.term_ids = {
            term: index + 1
            for index, term in enumerate(sorted(self.df, key=lambda t: (-self.df[t], t)))
        }
        return self

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency (the +1 form, always positive)."""
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def doc_weights(self, text: str) -> dict[int, float]:
        """BM25 term weights for one document (corpus side)."""
        tokens = tokenize(text)
        length = len(tokens)
        if not length:
            return {}
        norm = self.k1 * (
            1.0 - self.b + self.b * (length / self.avg_len if self.avg_len else 1.0)
        )
        weights: dict[int, float] = {}
        for term, tf in Counter(tokens).items():
            term_id = self.term_ids.get(term)
            if term_id is None:
                continue
            weights[term_id] = self.idf(term) * (tf * (self.k1 + 1.0)) / (tf + norm)
        return weights

    def query_weights(self, text: str) -> dict[int, float]:
        """Query-side weights: idf per distinct term (dot product with doc side)."""
        weights: dict[int, float] = {}
        for term in set(tokenize(text)):
            term_id = self.term_ids.get(term)
            if term_id is not None:
                weights[term_id] = self.idf(term)
        return weights

    def save(self, path: Path) -> None:
        """Persist the fitted model (term ids, df, stats) as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "n_docs": self.n_docs,
                    "total_len": self.total_len,
                    "term_ids": self.term_ids,
                    "df": self.df,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Bm25Model:
        """Load a model previously written by `save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            k1=data["k1"],
            b=data["b"],
            n_docs=data["n_docs"],
            total_len=data["total_len"],
            term_ids={str(k): int(v) for k, v in data["term_ids"].items()},
            df={str(k): int(v) for k, v in data["df"].items()},
        )


@dataclass
class Bm25SparseProvider:
    """`EmbeddingProvider` adapter exposing a fitted BM25 model as sparse text."""

    spec: EmbeddingSpec = field(
        default_factory=lambda: EmbeddingSpec(
            model_id="local-bm25",
            dim=0,
            supports=frozenset({"sparse_text"}),
        )
    )
    model: Bm25Model = field(default_factory=Bm25Model)

    def sparse_text(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self.model.doc_weights(text) for text in texts]

    def dense_text(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("BM25 is sparse-only")

    def dense_image(self, images: Sequence[bytes]) -> list[list[float]]:
        raise NotImplementedError("BM25 is sparse-only")
