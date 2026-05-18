"""Summary-feature extraction for the feature-based clustering baseline.

When DTW finds only weak structure (low silhouette, conflicting partitions
across methods, lots of singletons), summary features + a vanilla clustering
algorithm often outperform DTW because the signal lives in aggregate
statistics rather than time-warped shape. :func:`summarise` returns an
``(N, K)`` matrix you can hand straight to ``sklearn.cluster.KMeans`` or any
other Euclidean clusterer.

The default feature set is chosen to be informative across power-like
telemetry signals: ``mean, std, max, min, idle_fraction, length, abs_sum``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _mean(s: np.ndarray) -> float:
    return float(s.mean()) if s.size else 0.0


def _std(s: np.ndarray) -> float:
    return float(s.std()) if s.size else 0.0


def _max(s: np.ndarray) -> float:
    return float(s.max()) if s.size else 0.0


def _min(s: np.ndarray) -> float:
    return float(s.min()) if s.size else 0.0


def _idle_fraction(s: np.ndarray, threshold: float = 10.0) -> float:
    return float(np.mean(np.abs(s) < threshold)) if s.size else 0.0


def _length(s: np.ndarray) -> float:
    return float(s.size)


def _abs_sum(s: np.ndarray) -> float:
    return float(np.abs(s).sum()) if s.size else 0.0


DEFAULT_FEATURES: dict[str, Callable[[np.ndarray], float]] = {
    "mean":          _mean,
    "std":           _std,
    "max":           _max,
    "min":           _min,
    "idle_fraction": _idle_fraction,
    "length":        _length,
    "abs_sum":       _abs_sum,
}


def summarise(
    series: Sequence[Sequence[float]],
    features: dict[str, Callable[[np.ndarray], float]] | None = None,
    *,
    standardise: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute per-series summary features.

    Parameters
    ----------
    series :
        Iterable of 1-D sequences (variable length OK).
    features :
        Optional override of ``{name: callable(series) -> float}``. Pass a
        subset of :data:`DEFAULT_FEATURES` to drop features, or extend with
        your own. ``None`` uses :data:`DEFAULT_FEATURES`.
    standardise :
        Z-score each column (column mean 0, std 1) before returning. Default
        ``True`` because most Euclidean clusterers assume unit-scale features.
        Columns with zero variance are returned as zero columns.

    Returns
    -------
    matrix : ndarray of shape (N, K)
        N = len(series), K = len(features).
    names : list[str]
        Feature names in column order.
    """
    feats = features if features is not None else DEFAULT_FEATURES
    names = list(feats.keys())
    if not series:
        return np.zeros((0, len(names))), names

    arrs = [np.asarray(s, dtype=np.float64) for s in series]
    mat = np.empty((len(arrs), len(names)), dtype=np.float64)
    for j, (_, fn) in enumerate(feats.items()):
        for i, a in enumerate(arrs):
            mat[i, j] = fn(a)

    if standardise:
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)
        mat = np.where(sd > 0, (mat - mu) / np.where(sd > 0, sd, 1.0), 0.0)

    return mat, names


__all__ = ["DEFAULT_FEATURES", "summarise"]
