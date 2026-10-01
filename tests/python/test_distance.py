"""dtwcpp.distance.dtw, the one distance entry, against values worked out by hand.

Every variant x metric x missing-data strategy that C++ computes (Standard with
every metric and strategy, DDTW with every metric, the other variants with L1)
on a pair whose distance is derived beside it, with each variant's parameter
away from its default; then one row per refusal of core::validate and of the
input scan. The values without a derivation here are tests/unit/core/test_dtw.cpp's
hand values.
"""

import math
import re

import pytest

import dtwcpp

NAN = math.nan

# x = [0, NaN, 2] against y = [0, 9, 9, 0]; the end cell |2 - 0| = 2 (squared 4) is on
# every path. Zero cost: the NaN row takes both 9s for nothing: 2 (4). AROW enters a NaN
# cell only diagonally, so the NaN takes one 9 and the 2 the other: 7 + 2 = 9 (49 + 4).
# Interpolate fills [0, 1, 2]; the 1 meets the first 0 and the 2 both 9s: 1 + 7 + 7 + 2
# = 17 (1 + 49 + 49 + 4 = 103).
GAP_X, GAP_Y = [0, NAN, 2], [0, 9, 9, 0]

ACCEPTED = [
    ({}, [1, 2, 3], [3, 4, 5, 6, 7], 13),
    ({"metric": "squared_euclidean"}, [1, 2, 3], [3, 4, 5, 6, 7], 35),
    ({"band": 2}, [0, 1, 0, 2, 0], [0, 0, 0, 0, 0, 0, 2], 5),
    ({"missing_strategy": "zero_cost"}, GAP_X, GAP_Y, 2),
    ({"missing_strategy": "zero_cost", "metric": "squared_euclidean"}, GAP_X, GAP_Y, 4),
    ({"missing_strategy": "arow"}, GAP_X, GAP_Y, 9),
    ({"missing_strategy": "arow", "metric": "squared_euclidean"}, GAP_X, GAP_Y, 53),
    ({"missing_strategy": "interpolate"}, GAP_X, GAP_Y, 17),
    ({"missing_strategy": "interpolate", "metric": "squared_euclidean"}, GAP_X, GAP_Y, 103),
    # The slopes of [0, 1, 2] are 1 and those of [0, 3, 6] are 3 throughout (the ends
    # copy the inner one): three cells at |1 - 3| = 2, squared 4.
    ({"variant": "ddtw"}, [0, 1, 2], [0, 3, 6], 6),
    ({"variant": "ddtw", "metric": "squared_euclidean"}, [0, 1, 2], [0, 3, 6], 12),
    ({"variant": "wdtw", "wdtw_g": 0.0}, [1, 2, 3], [3, 4, 5, 6, 7], 6.5),
    ({"variant": "adtw", "adtw_penalty": 2.0}, [0, 1], [0, 0, 1], 2),
    ({"variant": "softdtw", "sdtw_gamma": 0.7}, [0, 0], [0, 0], -0.7 * math.log(3)),
    ({"variant": "msm", "msm_c": 0.5}, [0, 2], [1], 2.5),
    ({"variant": "twe", "twe_nu": 0.05, "twe_lambda": 0.6}, [0, 0], [0], 0.65),
]


@pytest.mark.parametrize(("settings", "x", "y", "want"), ACCEPTED)
def test_distance_matches_the_hand_value(settings, x, y, want):
    assert dtwcpp.distance.dtw(x, y, **settings) == pytest.approx(want, rel=1e-15)


REFUSED = [
    ({"variant": "bogus"}, [0], [0],
     "unknown variant 'bogus'. Valid: standard, ddtw, wdtw, adtw, softdtw, msm, twe."),
    ({"wdtw_g": -1.0}, [0], [0], "WDTW g must be finite and non-negative."),
    ({"variant": "ddtw", "missing_strategy": "zero_cost"}, [0], [0],
     "Non-Standard DTW variants require MissingStrategy::Error."),
    ({"variant": "wdtw", "metric": "squared_euclidean"}, [0], [0],
     "metric SquaredL2 is implemented for Standard DTW and DDTW only, but variant = wdtw "
     "was requested. Use metric L1 for this configuration."),
    ({}, GAP_X, GAP_Y, "x[1] is NaN"),
    ({"missing_strategy": "zero_cost"}, GAP_X, [0, 9, math.inf, 0], "y[2] is +inf"),
]


@pytest.mark.parametrize(("settings", "x", "y", "message"), REFUSED)
def test_distance_refuses_what_no_kernel_computes(settings, x, y, message):
    with pytest.raises(dtwcpp.InvalidInput, match=re.escape(message)):
        dtwcpp.distance.dtw(x, y, **settings)
