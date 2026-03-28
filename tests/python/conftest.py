"""Shared fixtures for DTWC++ Python binding tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """Return a small dataset: 10 series of length 20."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 20))


@pytest.fixture
def well_separated_data():
    """Return 3 well-separated clusters of 5 series each (15 total, length 20).

    Cluster 0: baseline ~ 0
    Cluster 1: baseline ~ 50
    Cluster 2: baseline ~ 100
    """
    rng = np.random.default_rng(123)
    length = 20
    series_per_cluster = 5
    clusters = []
    for offset in [0.0, 50.0, 100.0]:
        for _ in range(series_per_cluster):
            clusters.append(offset + rng.standard_normal(length) * 0.5)
    return np.array(clusters)
