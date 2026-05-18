"""Preprocessing helpers for time-series clustering with DTW.

Use this module when your raw signal has artefacts that confuse DTW:
zero-order-hold (ZOH) staircases from sub-sampled logging, leading/trailing
idle plateaus, or magnitude differences that you want to ignore in favour of
shape.

The recommended chain for ZOH-corrupted, idle-padded telemetry (e-bike or
small-EV traction power, similar logger outputs) is exposed via
:func:`power_signal`:

    1. :func:`strip_idle`    — drop leading/trailing samples below a threshold
    2. :func:`decimate_zoh`  — keep only the first sample of each constant run
    3. :func:`sg_smooth`     — Savitzky-Golay smoothing (optional, needs scipy)
    4. :func:`derivative`    — first difference for event-based shape
    5. :func:`z_normalize`   — zero-mean, unit-std (uses C++ ``z_normalize``)

Use the individual functions if you want a different combination.

**When to use this vs the raw signal:** the chain is intended for *shape*
clustering. If your goal is to cluster by *magnitude / energy intensity*, do
NOT apply this chain — raw DTW (or a feature-based method on summary
statistics) is the correct tool. See ``test_preprocess.py`` for examples.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dtwcpp._dtwcpp_core import z_normalize as _z_normalize_cpp


def strip_idle(s: Sequence[float], threshold: float = 10.0) -> np.ndarray:
    """Drop leading and trailing samples whose absolute value is below threshold.

    Returns an empty array if the entire series is below threshold.
    """
    a = np.asarray(s, dtype=np.float64)
    active = np.abs(a) >= threshold
    if not active.any():
        return a[:0]
    lo = int(np.argmax(active))
    hi = len(a) - int(np.argmax(active[::-1]))
    return a[lo:hi]


def decimate_zoh(s: Sequence[float]) -> np.ndarray:
    """Run-length-decode: keep the first sample of every constant-value run.

    Removes the zero-order-hold (ZOH) repeats that appear when a higher-rate
    signal is logged at lower rate and held between samples. The first sample
    is always kept.
    """
    a = np.asarray(s, dtype=np.float64)
    if a.size == 0:
        return a
    keep = np.empty(a.size, dtype=bool)
    keep[0] = True
    keep[1:] = a[1:] != a[:-1]
    return a[keep]


def sg_smooth(s: Sequence[float], window: int = 5, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing. Requires *scipy* (optional dependency).

    If ``len(s) < window`` the input is returned unchanged. ``window`` is
    rounded up to the next odd value internally.
    """
    a = np.asarray(s, dtype=np.float64)
    if a.size < window:
        return a
    try:
        from scipy.signal import savgol_filter
    except ImportError as err:
        raise ImportError(
            "sg_smooth requires scipy. Install it with: uv add scipy"
        ) from err
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(a, window_length=w, polyorder=poly)


def derivative(s: Sequence[float]) -> np.ndarray:
    """First difference (``np.diff``). Returns empty for inputs of length < 2."""
    a = np.asarray(s, dtype=np.float64)
    if a.size < 2:
        return a[:0]
    return np.diff(a)


def z_normalize(s: Sequence[float]) -> np.ndarray:
    """Per-series z-normalisation (zero mean, unit std). Wraps C++ ``z_normalize``."""
    return np.asarray(_z_normalize_cpp(list(map(float, s))), dtype=np.float64)


def power_signal(
    s: Sequence[float],
    idle_threshold: float = 10.0,
    sg_window: int = 0,
    sg_poly: int = 2,
    take_derivative: bool = True,
    znorm: bool = True,
) -> np.ndarray:
    """Apply the recommended ZOH-power preprocessing chain.

    Parameters
    ----------
    s :
        Raw series as a 1-D sequence of floats.
    idle_threshold :
        Drop leading/trailing samples with ``|s| < idle_threshold``. Set to
        ``0.0`` to skip the strip step.
    sg_window :
        Savitzky-Golay window length. ``0`` skips smoothing (default). Requires
        scipy when nonzero.
    sg_poly :
        SG polynomial order. Ignored when ``sg_window == 0``.
    take_derivative :
        Apply first difference after smoothing. Default ``True``.
    znorm :
        Apply per-series z-normalisation as the final step. Default ``True``.

    Returns
    -------
    np.ndarray
        Processed series; length may be shorter than the input.
    """
    a = strip_idle(s, idle_threshold) if idle_threshold > 0 else np.asarray(s, dtype=np.float64)
    a = decimate_zoh(a)
    if sg_window > 0:
        a = sg_smooth(a, sg_window, sg_poly)
    if take_derivative:
        a = derivative(a)
    if znorm and a.size > 0:
        a = z_normalize(a)
    return a


__all__ = [
    "strip_idle",
    "decimate_zoh",
    "sg_smooth",
    "derivative",
    "z_normalize",
    "power_signal",
]
