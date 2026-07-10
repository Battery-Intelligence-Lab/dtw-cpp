"""Shared Python-side validation for serializable DTW variant parameters."""

from __future__ import annotations

import numpy as np


def _finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        return value
    return value


def normalize_variant_parameters(
    *, wdtw_g, adtw_penalty, msm_c, twe_nu, twe_lambda,
):
    """Normalize the Python/HPC parameter envelope using the C++ domains."""
    values = {
        "wdtw_g": _finite_real("wdtw_g", wdtw_g),
        "adtw_penalty": _finite_real("adtw_penalty", adtw_penalty),
        "msm_c": _finite_real("msm_c", msm_c),
        "twe_nu": _finite_real("twe_nu", twe_nu),
        "twe_lambda": _finite_real("twe_lambda", twe_lambda),
    }

    if not np.isfinite(values["wdtw_g"]) or values["wdtw_g"] < 0.0:
        raise ValueError("WDTW g must be finite and non-negative.")
    if (not np.isfinite(values["adtw_penalty"])
            or values["adtw_penalty"] < 0.0):
        raise ValueError("ADTW penalty must be finite and non-negative.")
    if not np.isfinite(values["msm_c"]) or values["msm_c"] <= 0.0:
        raise ValueError("MSM c must be finite and positive.")
    if not np.isfinite(values["twe_nu"]) or values["twe_nu"] <= 0.0:
        raise ValueError("TWE nu must be finite and positive.")
    if (not np.isfinite(values["twe_lambda"])
            or values["twe_lambda"] <= 0.0):
        raise ValueError("TWE lambda must be finite and positive.")
    return values
