"""
@file distance.py
@brief Additive distance namespace for the Python API.
"""

from __future__ import annotations

import numpy as np

from dtwcpp._dtwcpp_core import (
    adtw_distance as _adtw_distance_raw,
    ddtw_distance as _ddtw_distance_raw,
    dtw_arow_distance as _dtw_arow_distance_raw,
    dtw_distance as _dtw_distance_raw,
    dtw_distance_missing as _dtw_distance_missing_raw,
    soft_dtw_distance as _soft_dtw_distance_raw,
    wdtw_distance as _wdtw_distance_raw,
)


def _as_array(x):
    return np.asarray(x, dtype=np.float64)


def standard(x, y, band=-1, metric="l1"):
    """Standard DTW distance."""
    return _dtw_distance_raw(_as_array(x), _as_array(y), band, metric)


def ddtw(x, y, band=-1):
    """Derivative DTW distance."""
    xa = _as_array(x)
    ya = _as_array(y)
    return _ddtw_distance_raw(list(xa), list(ya), band)


def wdtw(x, y, band=-1, g=0.05):
    """Weighted DTW distance."""
    xa = _as_array(x)
    ya = _as_array(y)
    return _wdtw_distance_raw(list(xa), list(ya), band, g)


def adtw(x, y, band=-1, penalty=1.0):
    """Amerced DTW distance."""
    xa = _as_array(x)
    ya = _as_array(y)
    return _adtw_distance_raw(list(xa), list(ya), band, penalty)


def soft_dtw(x, y, gamma=1.0):
    """Soft-DTW distance."""
    xa = _as_array(x)
    ya = _as_array(y)
    return _soft_dtw_distance_raw(list(xa), list(ya), gamma)


def missing(x, y, band=-1, metric="l1"):
    """Zero-cost DTW with missing values."""
    return _dtw_distance_missing_raw(_as_array(x), _as_array(y), band, metric)


def arow(x, y, band=-1, metric="l1"):
    """DTW-AROW distance."""
    return _dtw_arow_distance_raw(_as_array(x), _as_array(y), band, metric)


def dtw(
    x,
    y,
    *,
    variant="standard",
    band=-1,
    metric="l1",
    g=0.05,
    penalty=1.0,
    gamma=1.0,
    missing_strategy="error",
):
    """Convenience dispatcher for DTW-family distances.

    This is a convenience layer for interactive use and examples. For tight
    loops, prefer the explicit `distance.ddtw`, `distance.wdtw`, etc. entry
    points to avoid repeated dispatch and argument normalization.
    """
    variant_key = variant.strip().lower().replace("-", "_")
    missing_key = missing_strategy.strip().lower().replace("-", "_")

    if missing_key != "error":
        if variant_key != "standard":
            raise ValueError(
                "missing_strategy dispatch currently requires variant='standard'. "
                "Use Problem/DTWClustering for combined variant + missing-data workflows."
            )
        if missing_key in {"zero_cost", "missing"}:
            return missing(x, y, band=band, metric=metric)
        if missing_key == "arow":
            return arow(x, y, band=band, metric=metric)
        if missing_key == "interpolate":
            raise ValueError(
                "missing_strategy='interpolate' is available through Problem/DTWClustering, "
                "not the standalone distance dispatcher."
            )
        raise ValueError(f"Unknown missing_strategy '{missing_strategy}'.")

    if variant_key in {"standard", "dtw"}:
        return standard(x, y, band=band, metric=metric)
    if variant_key == "ddtw":
        return ddtw(x, y, band=band)
    if variant_key == "wdtw":
        return wdtw(x, y, band=band, g=g)
    if variant_key == "adtw":
        return adtw(x, y, band=band, penalty=penalty)
    if variant_key in {"softdtw", "soft_dtw"}:
        return soft_dtw(x, y, gamma=gamma)

    raise ValueError(
        f"Unknown variant '{variant}'. Expected one of: "
        "standard, ddtw, wdtw, adtw, soft_dtw."
    )


__all__ = [
    "adtw",
    "arow",
    "ddtw",
    "dtw",
    "missing",
    "soft_dtw",
    "standard",
    "wdtw",
]
