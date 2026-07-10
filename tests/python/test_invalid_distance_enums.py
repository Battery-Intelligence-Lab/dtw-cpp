"""M47 preregistration for binding-side selector conversion and publication."""

import re

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _dtwcpp_core as core


INVALID_INT32_VALUES = (-1, None, -(2**31), 2**31 - 1)

ENUM_DOMAINS = (
    ("DTWVariant", core.DTWVariant, 7),
    ("MissingStrategy", core.MissingStrategy, 4),
    ("MetricType", core.MetricType, 3),
    ("MVMode", core.MVMode, 2),
    ("DistanceMatrixStrategy", core.DistanceMatrixStrategy, 5),
    ("StoragePolicy", core.StoragePolicy, 3),
    ("LowerBoundStrategy", core.LowerBoundStrategy, 7),
)


@pytest.mark.parametrize("name,enum_type,first_above", ENUM_DOMAINS)
@pytest.mark.parametrize("raw_template", INVALID_INT32_VALUES)
def test_unknown_enum_construction_remains_rejected(
    name, enum_type, first_above, raw_template
):
    del name
    raw = first_above if raw_template is None else raw_template
    with pytest.raises(ValueError):
        enum_type(raw)


PROPERTY_ROUTES = (
    ("variant", "variant", 7),
    ("mv_mode", "mv_mode", 2),
    ("missing_strategy", "missing_strategy", 4),
    ("distance_strategy", "distance_strategy", 5),
    ("lb_strategy", "lb_strategy", 7),
    ("storage_policy", "storage_policy", 3),
)


def _property_owner(route):
    if route in {"variant", "mv_mode"}:
        return core.DTWVariantParams()
    return core.Problem(f"m47_python_{route}")


@pytest.mark.parametrize("route,attribute,first_above", PROPERTY_ROUTES)
@pytest.mark.parametrize("raw_template", INVALID_INT32_VALUES)
def test_raw_integer_enum_properties_remain_typed(
    route, attribute, first_above, raw_template
):
    raw = first_above if raw_template is None else raw_template
    owner = _property_owner(route)
    with pytest.raises(TypeError):
        setattr(owner, attribute, raw)


@pytest.mark.parametrize("raw", (-1, 3, -(2**31), 2**31 - 1))
def test_invalid_cuda_precision_property_is_rejected_transactionally(raw):
    problem = core.Problem("m47_python_cuda_precision")
    problem.set_data([[0.0], [1.0]], ["x", "y"])
    expected = np.array([[0.0, 123.0], [123.0, 0.0]])
    problem.set_distance_matrix(expected)

    candidate = core.CUDASettings()
    with pytest.raises(
        dtwcpp.InvalidInput,
        match=f"^{re.escape('Invalid CUDA precision value.')}$",
    ):
        candidate.precision = raw
    assert candidate.precision == 0

    with pytest.raises(
        dtwcpp.InvalidInput,
        match=f"^{re.escape('Invalid CUDA precision value.')}$",
    ):
        problem.cuda_settings.precision = raw

    assert problem.cuda_settings.precision == 0
    assert problem.is_distance_matrix_filled()
    np.testing.assert_array_equal(problem.distance_matrix(), expected)
