"""Materials sampling: quota algorithm, strata keys, deterministic selection."""

import pytest

from bench.materials import compute_quotas, stratum_key


def test_stratum_key_boundaries():
    assert stratum_key("landscape", 1920, 3) == ("landscape", "sd", "lt5")
    assert stratum_key("landscape", 1920, 5) == ("landscape", "sd", "b5_15")
    assert stratum_key("landscape", 1920, 15) == ("landscape", "sd", "b5_15")
    assert stratum_key("landscape", 1920, 16) == ("landscape", "sd", "b16_60")
    assert stratum_key("landscape", 1920, 60) == ("landscape", "sd", "b16_60")
    assert stratum_key("portrait", 3840, 61) == ("portrait", "uhd", "gt60")
    assert stratum_key(None, None, None) == ("unknown", "sd", "unknown")
    assert stratum_key("weird", 3839, 4) == ("unknown", "sd", "lt5")
    assert stratum_key("weird", 3839, 10) == ("unknown", "sd", "b5_15")


def test_quotas_sum_exactly_when_capacity_allows():
    cells = {("a",): 600, ("b",): 300, ("c",): 100}
    quotas = compute_quotas(cells, n_total=1_000, floor=100)
    assert sum(quotas.values()) == 1_000
    assert quotas[("a",)] == 600
    assert quotas[("b",)] == 300
    assert quotas[("c",)] == 100


def test_quotas_give_rare_cells_their_floor():
    cells = {("big",): 9_000, ("rare",): 40}
    quotas = compute_quotas(cells, n_total=1_000, floor=100)
    assert quotas[("rare",)] == 40  # all it has
    assert quotas[("big",)] == 960
    assert sum(quotas.values()) == 1_000


def test_quotas_when_floors_exceed_budget_scale_down():
    cells = {("a",): 500, ("b",): 5}
    quotas = compute_quotas(cells, n_total=100, floor=100)
    assert sum(quotas.values()) == 100
    assert quotas[("a",)] == 95
    assert quotas[("b",)] == 5


def test_quotas_respect_capacity_and_redistribute():
    cells = {("tiny",): 5, ("huge",): 995}
    quotas = compute_quotas(cells, n_total=1_000, floor=100)
    assert quotas[("tiny",)] == 5
    assert quotas[("huge",)] == 995
    assert sum(quotas.values()) == 1_000


def test_quotas_never_exceed_cell_rows():
    cells = {("a",): 3, ("b",): 7, ("c",): 10_000}
    quotas = compute_quotas(cells, n_total=500, floor=100)
    for key, rows in cells.items():
        assert quotas[key] <= rows
    assert sum(quotas.values()) == 500


def test_quotas_are_deterministic():
    cells = {("a",): 123, ("b",): 456, ("c",): 789}
    first = compute_quotas(cells, n_total=100, floor=10)
    second = compute_quotas(cells, n_total=100, floor=10)
    assert first == second


def test_quotas_empty_and_zero_inputs():
    assert compute_quotas({}, n_total=100, floor=10) == {}
    assert compute_quotas({("a",): 50}, n_total=0, floor=10) == {("a",): 0}


def test_quotas_large_proportional_part():
    cells = {("head",): 100_000, ("tail",): 10}
    quotas = compute_quotas(cells, n_total=1_000, floor=100)
    assert quotas[("tail",)] == 10
    assert quotas[("head",)] == 990
    assert sum(quotas.values()) == 1_000


@pytest.mark.parametrize(
    ("n_total", "expected"),
    [
        (1, {("a",): 1, ("b",): 0}),
        (2, {("a",): 1, ("b",): 1}),
    ],
)
def test_quotas_tiny_budgets(n_total, expected):
    assert compute_quotas({("a",): 5, ("b",): 5}, n_total=n_total, floor=100) == expected
