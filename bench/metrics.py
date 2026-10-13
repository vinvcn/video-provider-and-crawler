"""Pure metric math: nDCG, Recall, MRR, paired bootstrap, Cohen's kappa.

No I/O here — everything is unit-testable against hand-computed cases.
Conventions (docs/bench-harness-design-2026-10-09.md §6):
- graded gain for nDCG  = 2^grade - 1;
- binary relevance     = grade >= 2 (Recall@k, MRR@k denominators);
- pooled nDCG/R        = the query's judged pool is the known universe, so
  Recall is a lower bound and IDCG comes from the pool's ideal ordering;
- unparsed judgments   = grade 0 for scoring, counted separately (never silent).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

METRIC_K = 10
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = "vpc-bench"


def gain(grade: int) -> float:
    """Graded gain: 0 -> 0, 1 -> 1, 2 -> 3, 3 -> 7."""
    return float(2**grade - 1)


def dcg(grades: Sequence[int]) -> float:
    """Discounted cumulative gain for a ranked grade list (position 1 = top)."""
    return sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(ranked_grades: Sequence[int], pool_grades: Sequence[int], k: int = METRIC_K) -> float:
    """nDCG@k against the pool's ideal ordering (0.0 when nothing is relevant)."""
    ideal = sorted(pool_grades, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg(list(ranked_grades)[:k]) / idcg


def n_relevant(pool_grades: Sequence[int]) -> int:
    """Judged-relevant count (grade >= 2) in the pool."""
    return sum(1 for grade in pool_grades if grade is not None and grade >= 2)


def recall_at_k(ranked_grades: Sequence[int], relevant_total: int, k: int) -> float:
    """Pooled recall@k (lower bound); 0.0 when the pool holds no relevant doc."""
    if relevant_total <= 0:
        return 0.0
    hits = sum(1 for grade in list(ranked_grades)[:k] if grade is not None and grade >= 2)
    return hits / relevant_total


def mrr_at_k(ranked_grades: Sequence[int], k: int = METRIC_K) -> float:
    """Reciprocal rank of the first relevant (grade >= 2) doc within top-k."""
    for rank, grade in enumerate(list(ranked_grades)[:k], start=1):
        if grade is not None and grade >= 2:
            return 1.0 / rank
    return 0.0


def bootstrap_paired(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: str = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap CI for mean(score_a - score_b).

    Both sequences must be aligned per query (same order) — benchmark runs are
    naturally paired this way, which makes this far more sensitive than
    unpaired comparison.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired bootstrap needs aligned sequences")
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    observed = sum(diffs) / n if n else 0.0
    if n == 0:
        return {"diff": 0.0, "ci": (0.0, 0.0), "win": "tie", "rounds": rounds}
    rng = random.Random(
        int.from_bytes(hashlib.blake2b(seed.encode(), digest_size=8).digest(), "big")
    )
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lo = means[max(0, int(tail * rounds))]
    hi = means[min(rounds - 1, rounds - 1 - int(tail * rounds))]
    win = "a" if lo > 0.0 else "b" if hi < 0.0 else "tie"
    return {"diff": observed, "ci": (lo, hi), "win": win, "rounds": rounds}


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Unweighted Cohen's kappa over aligned integer ratings; 1.0 when perfect."""
    if len(a) != len(b):
        raise ValueError("kappa needs aligned ratings")
    n = len(a)
    if not n:
        return 0.0
    classes = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    counts_a = {c: sum(1 for x in a if x == c) for c in classes}
    counts_b = {c: sum(1 for y in b if y == c) for c in classes}
    expected = sum(counts_a[c] * counts_b[c] for c in classes) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def confusion_matrix(a: Sequence[int], b: Sequence[int]) -> dict[str, dict[str, int]]:
    """{row: human grade, col: judge grade} counts for aligned ratings."""
    matrix: dict[str, dict[str, int]] = {}
    for x, y in zip(a, b, strict=True):
        matrix.setdefault(str(x), {})[str(y)] = matrix.setdefault(str(x), {}).get(str(y), 0) + 1
    return matrix


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 when empty."""
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    """Median; 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def aggregate(values: Sequence[float]) -> dict:
    """mean/median/min/max summary for one metric over one query set."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (fraction in [0,1]); 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered))))
    return ordered[rank - 1]


def cut_values(record: Mapping[str, object]) -> dict[str, str]:
    """Cut keys one query contributes to (category, lang, source, ...)."""
    facets = record.get("facets") or []
    cuts = {
        "category": str(record.get("category")),
        "lang": str(record.get("lang")),
        "source": str(record.get("source")),
        "difficulty": str(record.get("difficulty")),
        "hard": "yes" if record.get("hard") else "no",
    }
    for facet in facets:
        cuts[f"facet:{facet}"] = "yes"
    return cuts
