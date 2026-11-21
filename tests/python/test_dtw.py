"""
@file test_dtw.py
@brief Single-element and NumPy inputs of dtwcpp.distance.dtw (the hand-value oracles live in test_distance.py).
@author Volkan Kumtepeli
"""

import numpy as np
import pytest

import dtwcpp


class TestDTWDistance:
    """Tests for dtwcpp.distance.dtw."""

    def test_single_element(self):
        """DTW of single-element series."""
        assert dtwcpp.distance.dtw([5.0], [3.0]) == pytest.approx(2.0)

    def test_numpy_input(self):
        """DTW accepts numpy arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        assert dtwcpp.distance.dtw(x, y) == pytest.approx(9.0)

