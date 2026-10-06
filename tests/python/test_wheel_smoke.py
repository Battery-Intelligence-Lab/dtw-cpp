"""Keep the cibuildwheel artifact gate executable under ordinary pytest."""

import pytest

import dtwcpp
from dtwcpp._wheel_smoke import run


def test_wheel_smoke_on_the_wheel_build():
    if dtwcpp.HIGHS_AVAILABLE:
        pytest.skip("developer extension links HiGHS; the wheel does not")
    pytest.importorskip("highspy")  # the mip extra
    run()
