"""Sanity-check helpers for clustering results.

Catches the silent failure modes that bit us on the Kasper analysis:

- **Singleton clusters** that mean the solver is fitting noise.
- **Idle-dominated medoids** that mean the medoid carries no information.
- **Cluster labels that are a proxy for series length**, so DTW just measured
  duration rather than shape.
- **Degenerate hierarchical splits** like ``[N-1, 1]`` where one outlier got
  peeled off.

All checks are read-only; they return data, never modify the Problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_int_array(x) -> np.ndarray:
    return np.asarray(x, dtype=int)


def cluster_sizes(labels: Sequence[int]) -> list[int]:
    """Return per-cluster member counts in label order (zero-padded to max+1)."""
    lab = _as_int_array(labels)
    if lab.size == 0:
        return []
    return np.bincount(lab, minlength=int(lab.max()) + 1).tolist()


def medoid_idle_fractions(
    series: Sequence[Sequence[float]],
    medoid_indices: Sequence[int],
    threshold: float = 10.0,
) -> list[float]:
    """Fraction of |value| < threshold for each medoid series."""
    out: list[float] = []
    for m in medoid_indices:
        s = np.asarray(series[m], dtype=np.float64)
        out.append(float(np.mean(np.abs(s) < threshold)) if s.size else 0.0)
    return out


def within_between_length_ratio(
    series_lengths: Sequence[int],
    labels: Sequence[int],
) -> float:
    """Within-cluster vs between-cluster variance of series length.

    Returns ``var_within / var_between`` weighted by cluster size. Ratio < 0.5
    suggests cluster labels are a length proxy — i.e. the clustering is just
    measuring how long each ride is, not its shape.
    """
    L = np.asarray(series_lengths, dtype=np.float64)
    lab = _as_int_array(labels)
    if L.size == 0:
        return 1.0
    overall_mean = L.mean()
    within_sum = 0.0
    between_sum = 0.0
    for c in np.unique(lab):
        mask = lab == c
        n = int(mask.sum())
        within_sum += n * L[mask].var()
        between_sum += n * (L[mask].mean() - overall_mean) ** 2
    within = within_sum / L.size
    between = between_sum / L.size
    return float(within / (between + 1e-12))


def diagnose_clusters(
    series: Sequence[Sequence[float]],
    labels: Sequence[int],
    medoid_indices: Sequence[int] | None = None,
    *,
    singleton_size: int = 3,
    idle_threshold: float = 10.0,
    idle_medoid_fraction: float = 0.5,
    length_ratio_floor: float = 0.5,
    degenerate_share: float = 0.95,
) -> dict[str, Any]:
    """Run all sanity checks on a clustering result.

    Parameters
    ----------
    series :
        Original (un-preprocessed) input series — needed for idle / length
        checks against the real signal.
    labels :
        Cluster label per series.
    medoid_indices :
        Indices of medoid series for each cluster. Optional — pass ``None`` for
        hierarchical or kMeans results that don't expose medoids.
    singleton_size :
        Clusters with ``<= singleton_size`` members raise the singleton flag.
    idle_threshold :
        Value below which a sample counts as idle, for the medoid-idle check.
    idle_medoid_fraction :
        A medoid with idle fraction above this is flagged.
    length_ratio_floor :
        Within-vs-between length variance ratio below this is flagged
        (clusters look like a length proxy).
    degenerate_share :
        If a single cluster holds more than this share of all series, the
        partition is flagged as degenerate (e.g. hierarchical's ``[136, 1]``).

    Returns
    -------
    dict
        Keys: ``sizes``, ``min_size``, ``any_singleton``, ``medoid_idle``,
        ``max_medoid_idle``, ``any_idle_medoid``, ``length_ratio``,
        ``length_dominated``, ``largest_cluster_share``, ``degenerate``, and
        ``flags`` (list of strings).
    """
    sizes = cluster_sizes(labels)
    if not sizes:
        return {
            "sizes": [], "min_size": 0, "any_singleton": False,
            "medoid_idle": [], "max_medoid_idle": 0.0, "any_idle_medoid": False,
            "length_ratio": 1.0, "length_dominated": False,
            "largest_cluster_share": 0.0, "degenerate": True,
            "flags": ["empty"],
        }

    flags: list[str] = []
    min_size = min(sizes)
    any_singleton = min_size <= singleton_size
    if any_singleton:
        flags.append(f"singleton: smallest cluster has {min_size} members")

    if medoid_indices is not None:
        medoid_idle = medoid_idle_fractions(series, medoid_indices, idle_threshold)
        max_idle = max(medoid_idle) if medoid_idle else 0.0
        any_idle = max_idle > idle_medoid_fraction
        if any_idle:
            flags.append(
                f"idle-medoid: a medoid is {max_idle:.0%} idle "
                f"(>{idle_medoid_fraction:.0%})"
            )
    else:
        medoid_idle = []
        max_idle = 0.0
        any_idle = False

    lens = [len(s) for s in series]
    L_arr = np.asarray(lens, dtype=np.float64)
    if L_arr.var() == 0:
        # All series same length — length cannot dominate, check is meaningless.
        ratio = 1.0
        length_dom = False
    else:
        ratio = within_between_length_ratio(lens, labels)
        length_dom = ratio < length_ratio_floor
        if length_dom:
            flags.append(
                f"length-dominated: within/between length-variance ratio = "
                f"{ratio:.2f} (<{length_ratio_floor})"
            )

    n_total = sum(sizes)
    share = max(sizes) / n_total if n_total else 0.0
    degenerate = share > degenerate_share
    if degenerate:
        flags.append(
            f"degenerate: largest cluster holds {share:.0%} of all series "
            f"(>{degenerate_share:.0%})"
        )

    return {
        "sizes": sizes,
        "min_size": min_size,
        "any_singleton": any_singleton,
        "medoid_idle": medoid_idle,
        "max_medoid_idle": max_idle,
        "any_idle_medoid": any_idle,
        "length_ratio": ratio,
        "length_dominated": length_dom,
        "largest_cluster_share": share,
        "degenerate": degenerate,
        "flags": flags,
    }


__all__ = [
    "cluster_sizes",
    "medoid_idle_fractions",
    "within_between_length_ratio",
    "diagnose_clusters",
]
