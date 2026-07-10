"""Phase 8 M36: raw metric tokens and variant/missing cross-products."""

import re

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _dtwcpp_core as core


UNKNOWN_METRIC = (
    "Unknown metric 'bogus'. Expected one of: "
    "l1, squared_euclidean, sqeuclidean."
)
CROSS_PRODUCT = (
    "Non-Standard DTW variants require MissingStrategy::Error."
)

X = np.array([0.0, 3.0], dtype=np.float64)
Y = np.array([0.0, 1.0], dtype=np.float64)


@pytest.mark.parametrize(
    "call",
    [
        lambda: core.dtw_distance(X, Y, -1, "bogus"),
        lambda: core.dtw_distance_missing(X, Y, -1, "bogus"),
        lambda: core.dtw_arow_distance(X, Y, -1, "bogus"),
        lambda: core.compute_distance_matrix([X.tolist(), Y.tolist()], -1,
                                             "bogus", False),
    ],
)
def test_raw_metric_tokens_reject_unknown_instead_of_running_l1(call):
    with pytest.raises(dtwcpp.InvalidInput,
                       match=f"^{re.escape(UNKNOWN_METRIC)}$"):
        call()


def test_raw_metric_aliases_keep_exact_fingerprints():
    l1 = core.dtw_distance(X, Y, -1, "l1")
    squared = core.dtw_distance(X, Y, -1, "squared_euclidean")
    alias = core.dtw_distance(X, Y, -1, "sqeuclidean")
    assert l1 == pytest.approx(2.0, abs=0.0)
    assert squared == pytest.approx(4.0, abs=0.0)
    assert alias == pytest.approx(squared, abs=0.0)


@pytest.mark.parametrize(
    ("variant", "strategy"),
    [
        (dtwcpp.DTWVariant.DDTW, dtwcpp.MissingStrategy.ZeroCost),
        (dtwcpp.DTWVariant.WDTW, dtwcpp.MissingStrategy.AROW),
        (dtwcpp.DTWVariant.ADTW, dtwcpp.MissingStrategy.Interpolate),
        (dtwcpp.DTWVariant.SoftDTW, dtwcpp.MissingStrategy.ZeroCost),
        (dtwcpp.DTWVariant.MSM, dtwcpp.MissingStrategy.AROW),
        (dtwcpp.DTWVariant.TWE, dtwcpp.MissingStrategy.Interpolate),
    ],
)
def test_problem_binding_rejects_variant_missing_cross_product(variant, strategy):
    problem = dtwcpp.Problem("m36")
    problem.set_data([[0.0], [0.0, 0.0]], ["x", "y"])
    params = dtwcpp.DTWVariantParams()
    params.variant = variant
    problem.set_variant_params(params)
    with pytest.raises(dtwcpp.InvalidInput,
                       match=f"^{re.escape(CROSS_PRODUCT)}$"):
        problem.missing_strategy = strategy


def test_high_level_dispatch_rejects_before_substitute_call(monkeypatch):
    def substitute(*_args, **_kwargs):
        raise AssertionError("Standard missing-data substitute executed")

    monkeypatch.setattr(dtwcpp.distance, "missing", substitute)
    with pytest.raises(ValueError, match="requires variant='standard'"):
        dtwcpp.distance.dtw(
            X, Y, variant="adtw", missing_strategy="zero_cost",
        )


def test_accepted_standard_missing_and_nonstandard_error_fingerprints():
    nan = float("nan")
    x = np.array([0.0, nan, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    assert dtwcpp.distance.dtw(
        x, y, variant="standard", missing_strategy="zero_cost"
    ) == pytest.approx(dtwcpp.distance.missing(x, y), abs=0.0)
    assert dtwcpp.distance.dtw(
        X, Y, variant="adtw", penalty=0.75, missing_strategy="error"
    ) == pytest.approx(dtwcpp.distance.adtw(X, Y, penalty=0.75), abs=0.0)
