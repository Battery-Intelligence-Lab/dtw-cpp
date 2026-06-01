"""Prototype preprocessing pipeline for ZOH-corrupted power-signal time series.

Implements the domain reviewer's recommended sequence:
    1. strip_idle      — drop leading/trailing samples where |v| < threshold
    2. decimate_zoh    — run-length-decode: keep first sample of each constant run
    3. sg_smooth       — Savitzky-Golay (window=5, poly=2) for quantisation noise
    4. derivative      — np.diff for event-based shape representation
    5. z_normalize     — per-series zero-mean unit-std (uses dtwcpp if available)

The default `power_signal(s)` chains all five with defaults that match the
Kasper sample. Each step is pure-Python (numpy + scipy.signal), no C++ rebuild
needed for the prototype.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter


def strip_idle(s: Sequence[float], threshold: float = 10.0) -> np.ndarray:
    """Drop leading and trailing samples where |s| < threshold.

    Returns an empty array if the whole series is below threshold.
    """
    a = np.asarray(s, dtype=np.float64)
    active = np.abs(a) >= threshold
    if not active.any():
        return a[:0]
    lo, hi = np.argmax(active), len(a) - np.argmax(active[::-1])
    return a[lo:hi]


def decimate_zoh(s: Sequence[float]) -> np.ndarray:
    """Run-length-decode: keep the first sample of every constant-value run.

    Reduces samples corrupted by zero-order-hold logging back to one sample per
    unique value transition. Safe for empty input.
    """
    a = np.asarray(s, dtype=np.float64)
    if a.size == 0:
        return a
    # Keep sample i if it differs from sample i-1; always keep sample 0.
    keep = np.empty(a.size, dtype=bool)
    keep[0] = True
    keep[1:] = a[1:] != a[:-1]
    return a[keep]


def sg_smooth(s: Sequence[float], window: int = 5, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing. If len(s) < window, returns input unchanged."""
    a = np.asarray(s, dtype=np.float64)
    if a.size < window:
        return a
    # window must be odd, > poly
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(a, window_length=w, polyorder=poly)


def derivative(s: Sequence[float]) -> np.ndarray:
    """First difference (np.diff). Returns empty for inputs of length < 2."""
    a = np.asarray(s, dtype=np.float64)
    if a.size < 2:
        return a[:0]
    return np.diff(a)


def z_normalize(s: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """Per-series z-normalisation; falls back to dtwcpp.z_normalize if installed."""
    try:
        import dtwcpp
        return np.asarray(dtwcpp.z_normalize(list(map(float, s))), dtype=np.float64)
    except Exception:
        a = np.asarray(s, dtype=np.float64)
        m = a.mean()
        sd = a.std()
        if sd < eps:
            return a - m
        return (a - m) / sd


def power_signal(
    s: Sequence[float],
    idle_threshold: float = 10.0,
    sg_window: int = 5,
    sg_poly: int = 2,
    take_derivative: bool = True,
    znorm: bool = True,
) -> np.ndarray:
    """Apply the full Kasper-style preprocessing chain. Order matters."""
    a = strip_idle(s, idle_threshold)
    a = decimate_zoh(a)
    a = sg_smooth(a, sg_window, sg_poly)
    if take_derivative:
        a = derivative(a)
    if znorm:
        a = z_normalize(a)
    return a


def describe(orig: Sequence[float], processed: np.ndarray) -> dict:
    """Length + simple stats before/after, for QA."""
    o = np.asarray(orig, dtype=np.float64)
    return {
        "len_in": int(o.size),
        "len_out": int(processed.size),
        "reduction": 1.0 - processed.size / max(o.size, 1),
        "in_mean": float(o.mean()),
        "in_std": float(o.std()),
        "out_mean": float(processed.mean()) if processed.size else 0.0,
        "out_std": float(processed.std()) if processed.size else 0.0,
    }
