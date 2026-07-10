"""Phase 8 M48: Python Problem semantic setters are transactional."""

import re

import numpy as np
import pytest

import dtwcpp


CROSS_PRODUCT = "Non-Standard DTW variants require MissingStrategy::Error."
WHOLE_ROUTES = (
    "missing_property",
    "variant_property",
    "variant_enum_method",
    "variant_params_method",
)


def _problem():
    problem = dtwcpp.Problem("m48-python")
    problem.set_data([[0.0], [1.0]], ["x", "y"])
    return problem


def _params(variant=dtwcpp.DTWVariant.Standard, penalty=1.0):
    params = dtwcpp.DTWVariantParams()
    params.variant = variant
    params.adtw_penalty = penalty
    return params


def _snapshot(problem):
    params = problem.variant_params
    return (
        params.variant,
        params.wdtw_g,
        params.adtw_penalty,
        params.sdtw_gamma,
        params.msm_c,
        params.twe_nu,
        params.twe_lambda,
        params.mv_mode,
        problem.missing_strategy,
    )


def _inject(problem, sentinel):
    expected = np.array([[0.0, sentinel], [sentinel, 0.0]])
    problem.set_distance_matrix(expected)
    assert problem.is_distance_matrix_filled()
    np.testing.assert_array_equal(problem.distance_matrix(), expected)
    return expected


def _reject_whole_route(route, sentinel=123.0):
    problem = _problem()
    if route == "missing_property":
        problem.set_variant_params(_params(dtwcpp.DTWVariant.ADTW, 2.0))
        mutate = lambda: setattr(
            problem, "missing_strategy", dtwcpp.MissingStrategy.ZeroCost
        )
    else:
        problem.missing_strategy = dtwcpp.MissingStrategy.ZeroCost
        candidate = _params(dtwcpp.DTWVariant.ADTW, 7.5)
        if route == "variant_property":
            mutate = lambda: setattr(problem, "variant_params", candidate)
        elif route == "variant_enum_method":
            mutate = lambda: problem.set_variant(dtwcpp.DTWVariant.ADTW)
        else:
            mutate = lambda: problem.set_variant_params(candidate)

    before_state = _snapshot(problem)
    before_matrix = _inject(problem, sentinel)
    with pytest.raises(
        dtwcpp.InvalidInput, match=f"^{re.escape(CROSS_PRODUCT)}$"
    ):
        mutate()
    return problem, before_state, before_matrix


@pytest.mark.parametrize("route", WHOLE_ROUTES)
@pytest.mark.parametrize("observable", ("selector", "filled", "matrix"))
def test_rejected_whole_property_and_method_routes_preserve_state_and_cache(
    route, observable
):
    """Independent observables distinguish prevalidation from partial rollback."""
    problem, before_state, before_matrix = _reject_whole_route(route)
    if observable == "selector":
        assert _snapshot(problem) == before_state
    elif observable == "filled":
        assert problem.is_distance_matrix_filled()
    else:
        np.testing.assert_array_equal(problem.distance_matrix(), before_matrix)


@pytest.mark.parametrize("route", WHOLE_ROUTES)
def test_accepted_noop_routes_preserve_exact_state_and_cache(route):
    problem = _problem()
    if route == "missing_property":
        problem.set_variant_params(_params(dtwcpp.DTWVariant.ADTW, 2.0))
        mutate = lambda: setattr(
            problem, "missing_strategy", dtwcpp.MissingStrategy.Error
        )
    else:
        problem.missing_strategy = dtwcpp.MissingStrategy.ZeroCost
        same = _params()
        if route == "variant_property":
            mutate = lambda: setattr(problem, "variant_params", same)
        elif route == "variant_enum_method":
            mutate = lambda: problem.set_variant(dtwcpp.DTWVariant.Standard)
        else:
            mutate = lambda: problem.set_variant_params(same)

    before_state = _snapshot(problem)
    before_matrix = _inject(problem, 321.0)
    mutate()
    assert _snapshot(problem) == before_state
    assert problem.is_distance_matrix_filled()
    np.testing.assert_array_equal(problem.distance_matrix(), before_matrix)


@pytest.mark.parametrize("route", WHOLE_ROUTES)
def test_accepted_valid_routes_publish_and_invalidate(route):
    problem = _problem()
    _inject(problem, 654.0)
    if route == "missing_property":
        problem.missing_strategy = dtwcpp.MissingStrategy.ZeroCost
        assert problem.missing_strategy == dtwcpp.MissingStrategy.ZeroCost
    elif route == "variant_property":
        problem.variant_params = _params(dtwcpp.DTWVariant.ADTW, 2.0)
        assert problem.variant_params.variant == dtwcpp.DTWVariant.ADTW
    elif route == "variant_enum_method":
        problem.set_variant(dtwcpp.DTWVariant.ADTW)
        assert problem.variant_params.variant == dtwcpp.DTWVariant.ADTW
    else:
        problem.set_variant_params(_params(dtwcpp.DTWVariant.ADTW, 2.0))
        assert problem.variant_params.adtw_penalty == pytest.approx(2.0)
    assert not problem.is_distance_matrix_filled()


@pytest.mark.parametrize("observable", ("filled", "matrix"))
def test_nested_variant_write_rejects_at_guard_without_destroying_cache(observable):
    """Nested writes bypass the setter; the next compute boundary must be atomic."""
    problem = _problem()
    problem.missing_strategy = dtwcpp.MissingStrategy.ZeroCost
    before_matrix = _inject(problem, 987.0)

    problem.variant_params.variant = dtwcpp.DTWVariant.ADTW
    with pytest.raises(
        dtwcpp.InvalidInput, match=f"^{re.escape(CROSS_PRODUCT)}$"
    ):
        problem.fill_distance_matrix()
    assert problem.variant_params.variant == dtwcpp.DTWVariant.ADTW

    # Raw nested state is caller-owned. Restore it without invoking a setter so
    # the assertion observes whether the rejecting boundary touched the cache.
    problem.variant_params.variant = dtwcpp.DTWVariant.Standard
    if observable == "filled":
        assert problem.is_distance_matrix_filled()
    else:
        np.testing.assert_array_equal(problem.distance_matrix(), before_matrix)


def test_nested_valid_write_still_reconciles_at_compute_boundary():
    problem = _problem()
    before_matrix = _inject(problem, 999.0)
    problem.variant_params.variant = dtwcpp.DTWVariant.ADTW

    after = problem.distance_matrix()
    assert problem.variant_params.variant == dtwcpp.DTWVariant.ADTW
    assert problem.is_distance_matrix_filled()
    assert not np.array_equal(after, before_matrix)
