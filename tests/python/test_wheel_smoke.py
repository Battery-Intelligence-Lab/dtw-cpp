## @file test_wheel_smoke.py
## @brief Keep the cibuildwheel artifact gate executable under ordinary pytest.
## @author Volkan Kumtepeli
## @date 10 Jul 2026
"""Keep the cibuildwheel artifact gate executable under ordinary pytest."""

from dtwcpp._wheel_smoke import run


def test_wheel_smoke_on_the_wheel_build(highspy_route):  # the wheel solves its MIP with highspy
    run()
