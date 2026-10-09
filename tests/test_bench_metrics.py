"""Metrics math: nDCG, Recall, MRR, paired bootstrap, kappa."""

import math

import pytest

from bench import metrics


def test_dcg_discounting():
    # gains: rank1 -> 2^3-1=7, rank2 -> 2^1-1=1 discounted by log2(3)
    assert metrics.dcg([3, 1]) == pytest.approx(7 + 1 / math.log2(3))


def test_ndcg_perfect_ranking():
    assert metrics.ndcg_at_k([3, 2, 1, 0], [3, 2, 1, 0]) == 1.0


def test_ndcg_hand_computed():
    # pool [1,1]: ideal = [1,1]; ranked [1,0]
    idcg = 1 + 1 / math.log2(3)
    assert metrics.ndcg_at_k([1, 0], [1, 1]) == pytest.approx(1 / idcg)


def test_ndcg_zero_when_nothing_relevant():
    assert metrics.ndcg_at_k([0, 0], [0, 0]) == 0.0


def test_ndcg_ignores_beyond_k():
    grades = [3] + [0] * 20
    assert metrics.ndcg_at_k(grades, grades) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(list(reversed(grades)), grades) == 0.0


def test_recall_counts_relevant_within_k():
    ranked = [2, 0, 2, 0, 0, 2]
    assert metrics.recall_at_k(ranked, relevant_total=3, k=3) == pytest.approx(2 / 3)
    assert metrics.recall_at_k(ranked, relevant_total=3, k=6) == 1.0
    assert metrics.recall_at_k(ranked, relevant_total=0, k=3) == 0.0


def test_mrr_first_relevant():
    assert metrics.mrr_at_k([0, 1, 2, 0]) == pytest.approx(1 / 3)
    assert metrics.mrr_at_k([3, 0, 0]) == 1.0
    assert metrics.mrr_at_k([0, 0, 0]) == 0.0
    # grade 1 is marginal, not relevant for binary metrics
    assert metrics.mrr_at_k([1, 2]) == pytest.approx(1 / 2)


def test_bootstrap_detects_clear_winner():
    a = [0.9] * 30
    b = [0.5] * 30
    result = metrics.bootstrap_paired(a, b, rounds=200)
    assert result["win"] == "a"
    assert result["ci"][0] > 0


def test_bootstrap_tie_for_identical_runs():
    scores = [0.4, 0.9, 0.1, 0.7]
    result = metrics.bootstrap_paired(scores, scores, rounds=200)
    assert result["diff"] == 0.0
    assert result["ci"][0] <= 0.0 <= result["ci"][1]
    assert result["win"] == "tie"


def test_bootstrap_is_deterministic():
    a = [0.9, 0.1, 0.5, 0.6, 0.2]
    b = [0.4, 0.2, 0.3, 0.9, 0.8]
    first = metrics.bootstrap_paired(a, b, rounds=100)
    second = metrics.bootstrap_paired(a, b, rounds=100)
    assert first["ci"] == second["ci"]


def test_bootstrap_rejects_misaligned():
    with pytest.raises(ValueError):
        metrics.bootstrap_paired([1.0, 2.0], [1.0])


def test_kappa_perfect_and_zero():
    assert metrics.cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    # observed agreement 0.5 == chance agreement 0.5 -> kappa 0
    assert metrics.cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_kappa_known_value():
    # 10 pairs, 8 agree; chance agreement 0.5 -> kappa = (0.8 - 0.5) / 0.5 = 0.6
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    assert metrics.cohen_kappa(a, b) == pytest.approx(0.6)


def test_confusion_matrix_counts():
    matrix = metrics.confusion_matrix([0, 1, 1], [1, 1, 0])
    assert matrix == {"0": {"1": 1}, "1": {"1": 1, "0": 1}}


def test_aggregate_helpers():
    assert metrics.mean([1, 2, 3]) == 2.0
    assert metrics.median([1, 9, 2]) == 2
    assert metrics.aggregate([])["n"] == 0
    assert metrics.percentile([1, 2, 3, 4], 0.5) == 2
    assert metrics.percentile([], 0.95) == 0.0


def test_cut_values_keys():
    cuts = metrics.cut_values(
        {
            "category": "concept",
            "lang": "zh",
            "source": "designed",
            "difficulty": "hard",
            "hard": {"orientation": "portrait"},
            "facets": ["mood", "lighting"],
        }
    )
    assert cuts["lang"] == "zh"
    assert cuts["hard"] == "yes"
    assert cuts["facet:mood"] == "yes"
    assert cuts["category"] == "concept"
