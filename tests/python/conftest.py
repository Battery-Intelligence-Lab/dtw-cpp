"""
@file conftest.py
@brief Shared fixtures for DTWC++ Python binding tests.
@author Volkan Kumtepeli
"""

import os

import pytest


@pytest.fixture(scope="session")
def dtwc_cl():
    """The dtwc_cl the CLI cases run: the one DTWC_CL_PATH names. Unset,
    they skip; set to anything but the binary, they fail where they run it."""
    path = os.environ.get("DTWC_CL_PATH")
    if path is None:
        pytest.skip("DTWC_CL_PATH names no dtwc_cl")
    return path


@pytest.fixture
def highspy_route():
    """Run a case only where method="mip" solves with highspy: an extension that links no HiGHS
    (two HiGHS builds in one process crash) and an importable highspy, a dependency of dtwcpp.
    Elsewhere the case skips; with DTWC_REQUIRE_HIGHSPY set it fails instead, as a missing
    DTWC_CL_PATH file does. The Python CI job sets it, so a missing highspy cannot leave that job
    green."""
    import dtwcpp

    if dtwcpp.HIGHS_AVAILABLE:
        reason = "this extension links HiGHS, so method 'mip' never reaches highspy"
    else:
        try:
            import highspy  # noqa: F401
            return
        except ImportError as error:
            reason = f"highspy cannot be imported ({error}); dtwcpp requires it: reinstall dtwcpp"
    if os.environ.get("DTWC_REQUIRE_HIGHSPY"):
        pytest.fail(f"DTWC_REQUIRE_HIGHSPY is set, and {reason}", pytrace=False)
    pytest.skip(reason)
