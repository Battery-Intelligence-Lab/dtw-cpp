"""Keep the cibuildwheel artifact gate executable under ordinary pytest."""

import pytest

import dtwcpp
from dtwcpp._wheel_smoke import run


def test_wheel_smoke_on_highs_build():
    if not dtwcpp.HIGHS_AVAILABLE:
        pytest.skip("developer extension was built without HiGHS")
    run()
