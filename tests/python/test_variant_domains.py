"""Phase 8 M34: binding-level DTW variant parameter domains."""

import re

import numpy as np
import pytest

import dtwcpp


X = np.array([0.0], dtype=np.float64)
Y = np.array([0.0, 0.0], dtype=np.float64)


def test_soft_dtw_gradient_raises_the_typed_domain_error():
    with pytest.raises(dtwcpp.InvalidInput,
                       match=r"^Soft-DTW gamma must be finite and positive\.$"):
        dtwcpp.soft_dtw_gradient(X, Y, gamma=-1.0)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("wdtw_g", -1.0, "WDTW g must be finite and non-negative."),
        ("adtw_penalty", -1.0, "ADTW penalty must be finite and non-negative."),
        ("sdtw_gamma", 0.0, "Soft-DTW gamma must be finite and positive."),
        ("msm_c", 0.0, "MSM c must be finite and positive."),
        ("twe_nu", 0.0, "TWE nu must be finite and positive."),
        ("twe_lambda", np.nan, "TWE lambda must be finite and positive."),
    ],
)
def test_variant_parameter_properties_reject_invalid_state(field, bad, message):
    params = dtwcpp.DTWVariantParams()
    with pytest.raises(dtwcpp.InvalidInput, match=f"^{re.escape(message)}$"):
        setattr(params, field, bad)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variant": "wdtw", "wdtw_g": -1.0},
         "WDTW g must be finite and non-negative."),
        ({"variant": "adtw", "adtw_penalty": -1.0},
         "ADTW penalty must be finite and non-negative."),
        ({"variant": "msm", "msm_c": 0.0},
         "MSM c must be finite and positive."),
        ({"variant": "twe", "twe_nu": 0.0},
         "TWE nu must be finite and positive."),
        ({"variant": "twe", "twe_lambda": np.inf},
         "TWE lambda must be finite and positive."),
    ],
)
def test_estimator_rejects_domains_before_touching_input(kwargs, message):
    model = dtwcpp.DTWClustering(**kwargs)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model.fit(object())
