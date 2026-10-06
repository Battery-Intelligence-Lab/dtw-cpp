"""
@file conftest.py
@brief Shared fixtures for DTWC++ Python binding tests.
@author Volkan Kumtepeli
"""

import os

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


@pytest.fixture
def highspy_route():
    """Run a case only where method="mip" solves with highspy: an extension that links no HiGHS
    (two HiGHS builds in one process crash) and an importable highspy, the mip extra. Elsewhere the
    case skips; with DTWC_REQUIRE_HIGHSPY set it fails instead, as a missing DTWC_CL_PATH file does.
    The Python CI job sets it, so a missing extra cannot leave that job green."""
    import dtwcpp

    if dtwcpp.HIGHS_AVAILABLE:
        reason = "this extension links HiGHS, so method 'mip' never reaches highspy"
    else:
        try:
            import highspy  # noqa: F401
            return
        except ImportError as error:
            reason = f"highspy cannot be imported ({error}); it is the mip extra"
    if os.environ.get("DTWC_REQUIRE_HIGHSPY"):
        pytest.fail(f"DTWC_REQUIRE_HIGHSPY is set, and {reason}", pytrace=False)
    pytest.skip(reason)
