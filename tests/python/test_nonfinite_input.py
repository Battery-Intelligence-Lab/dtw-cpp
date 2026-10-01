"""FX-15: the bindings that take series reject NaN and +-inf, naming the series
and position; finite answers are unchanged.

A property test over random finite pairs with one NaN, +inf or -inf injected at a
random position of x or y. ``dtwcpp.distance.dtw``, ``soft_dtw_gradient`` and
``compute_distance_matrix`` must raise ``dtwcpp.InvalidInput`` saying
"<series>[<position>] is <value>"; under a missing-data strategy NaN is a
missing value and only +-inf is refused. Before FX-15 each of these returned NaN,
the unreachable double max or an ordinary-looking number. Each variant's scan is
C++'s (tests/unit/core/unit_test_nonfinite_input.cpp).

Oracles: the injected position for the diagnostic, and pure-Python recurrences
(standard, ZeroCost and AROW DTW) for the answers, which the kernels must match
bit for bit.
"""

import math
import re

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _dtwcpp_core as core

TRIALS = 120
POISONS = ((float("nan"), "NaN"), (math.inf, "+inf"), (-math.inf, "-inf"))


def _pairs(seed):
    rng = np.random.default_rng(seed)
    for _ in range(TRIALS):
        n, m = (int(v) for v in rng.integers(2, 13, size=2))
        band = int(rng.integers(-1, max(n, m) + 1))
        yield rng, rng.uniform(-10, 10, n), rng.uniform(-10, 10, m), band


# name -> (call(x, y, band), NaN is a missing value)
ENTRY_POINTS = {
    "distance.dtw": (lambda x, y, b: dtwcpp.distance.dtw(x, y, band=b), False),
    "soft_dtw_gradient": (lambda x, y, b: core.soft_dtw_gradient(x, y, 1.0), False),
    "distance.dtw zero_cost": (
        lambda x, y, b: dtwcpp.distance.dtw(x, y, band=b, missing_strategy="zero_cost"),
        True),
    "distance.dtw arow": (
        lambda x, y, b: dtwcpp.distance.dtw(x, y, band=b, missing_strategy="arow"), True),
    "compute_distance_matrix": (
        lambda x, y, b: core.compute_distance_matrix(
            [x.tolist(), y.tolist()], b, "l1"), False),
}
if core.gpu_available() and core.gpu_info().startswith("Metal:"):
    # Before FX-15 Metal returned finite numbers for some non-finite input.
    ENTRY_POINTS["compute_distance_matrix_metal"] = (
        lambda x, y, b: core.compute_distance_matrix_metal(
            [x.tolist(), y.tolist()], band=b), False)


def _dtw_oracle(x, y, band, nan_costs_zero=False):
    """Standard L1 DTW, the textbook recurrence: min(diag, up, left) + cost.

    With ``nan_costs_zero`` a pair holding a NaN costs 0 (ZeroCost missing data).
    """
    n, m = len(x), len(y)
    big = np.finfo(np.float64).max
    if band >= 0 and abs(n - m) > band:
        return big  # no warping path: the kernels' finite sentinel
    cost = [[math.inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band >= 0 and abs(i - j) > band:
                continue
            local = abs(float(x[i - 1]) - float(y[j - 1]))
            if nan_costs_zero and math.isnan(local):
                local = 0.0
            best = min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])
            cost[i][j] = best + local
    return cost[n][m]


def _arow_oracle(x, y, band):
    """DTW-AROW (Yurtman et al. 2023): a pair holding a NaN costs nothing and is
    reached only diagonally; a first-row or first-column cell carries its one
    predecessor."""
    n, m = len(x), len(y)
    if band >= 0 and abs(n - m) > band:
        return np.finfo(np.float64).max
    cost = [[math.inf] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if band >= 0 and abs(i - j) > band:
                continue
            local = abs(float(x[i]) - float(y[j]))
            missing = math.isnan(local)
            if i == 0 and j == 0:
                cost[i][j] = 0.0 if missing else local
            elif i == 0 or j == 0:
                before = cost[i][j - 1] if i == 0 else cost[i - 1][j]
                cost[i][j] = before if missing else before + local
            elif missing:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                best = min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])
                cost[i][j] = best + local
    return cost[n - 1][m - 1]


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_nonfinite_input_raises_naming_series_and_position(name):
    call, nan_is_missing = ENTRY_POINTS[name]
    checked = 0
    for trial, (rng, x, y, band) in enumerate(_pairs(15)):
        value, label = POISONS[trial % 3]
        if nan_is_missing and label == "NaN":
            continue
        in_x = (trial // 3) % 2 == 0
        target = x if in_x else y
        index = int(rng.integers(0, target.size))
        target[index] = value
        series = "x" if in_x else "y"
        if name.startswith("compute_distance_matrix"):
            series = f"series[{0 if in_x else 1}]"
        expected = f"{series}[{index}] is {label}"
        with pytest.raises(dtwcpp.InvalidInput, match=re.escape(expected)):
            call(x, y, band)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize(
    "name", sorted(n for n, (_, nan_ok) in ENTRY_POINTS.items() if nan_ok))
def test_missing_data_distances_read_nan_as_missing(name):
    call, _ = ENTRY_POINTS[name]
    for trial, (rng, x, y, band) in enumerate(_pairs(16)):
        target = x if trial % 2 == 0 else y
        target[int(rng.integers(0, target.size))] = math.nan
        got = call(x, y, band)
        if "arow" in name:
            want = _arow_oracle(x, y, band)
        else:
            want = _dtw_oracle(x, y, band, nan_costs_zero=True)
        assert got.hex() == want.hex(), (x, y, band)


@pytest.mark.parametrize(
    "name",
    ["distance.dtw", "distance.dtw zero_cost", "distance.dtw arow"])
def test_finite_input_matches_the_recurrence_bit_for_bit(name):
    call, _ = ENTRY_POINTS[name]
    for _, x, y, band in _pairs(17):
        got = call(x, y, band)
        want = _dtw_oracle(x, y, band)
        assert got.hex() == want.hex(), (x, y, band)


def test_finite_matrix_matches_the_recurrence_bit_for_bit():
    for _, x, y, band in _pairs(18):
        want = _dtw_oracle(x, y, band)
        got = core.compute_distance_matrix([x, y], band, "l1")
        assert got[0, 1].hex() == want.hex(), (x, y, band)
        assert got[1, 0].hex() == want.hex(), (x, y, band)
