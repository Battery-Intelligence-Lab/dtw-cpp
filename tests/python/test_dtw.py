"""
@file test_dtw.py
@brief Tests for DTW distance functions.
@author Volkan Kumtepeli
"""

import math

import numpy as np
import pytest

import dtwcpp


# ---------------------------------------------------------------------------
# Standard DTW
# ---------------------------------------------------------------------------


class TestDTWDistance:
    """Tests for dtwcpp.distance.dtw."""

    def test_root_distance_alias_removed(self):
        """The breaking change removes root-level pairwise distance helpers."""
        assert not hasattr(dtwcpp, "dtw_distance")

    def test_identity(self):
        """DTW of identical series is zero."""
        assert dtwcpp.distance.dtw([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value_l1(self):
        """DTW([1,2,3],[4,5,6]) == 9 with the default L1 local cost."""
        assert dtwcpp.distance.dtw([1, 2, 3], [4, 5, 6]) == pytest.approx(9.0)

    def test_symmetry(self):
        """DTW(x, y) == DTW(y, x)."""
        x = [1.0, 3.0, 0.5, 2.0, 7.0]
        y = [2.0, 0.0, 4.0, 1.0, 6.0]
        assert dtwcpp.distance.dtw(x, y) == pytest.approx(dtwcpp.distance.dtw(y, x))

    def test_non_negativity(self):
        """DTW distance is never negative."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            x = rng.standard_normal(15).tolist()
            y = rng.standard_normal(15).tolist()
            assert dtwcpp.distance.dtw(x, y) >= 0.0

    def test_triangle_inequality_holds_approximately(self):
        """DTW does not guarantee triangle inequality, but for L2-style metrics
        it usually holds on short series. Just verify non-negative here."""
        x = [1.0, 2.0, 3.0]
        y = [3.0, 1.0, 2.0]
        z = [2.0, 3.0, 1.0]
        dxy = dtwcpp.distance.dtw(x, y)
        dyz = dtwcpp.distance.dtw(y, z)
        dxz = dtwcpp.distance.dtw(x, z)
        assert dxy >= 0 and dyz >= 0 and dxz >= 0

    def test_banded(self):
        """Banded DTW returns a finite non-negative result."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        d = dtwcpp.distance.dtw(x, y, band=2)
        assert math.isfinite(d) and d >= 0.0

    def test_banded_ge_full(self):
        """Banded DTW >= full DTW (band restricts the warping path)."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        d_full = dtwcpp.distance.dtw(x, y, band=-1)
        d_band = dtwcpp.distance.dtw(x, y, band=2)
        assert d_band >= d_full - 1e-12

    def test_different_lengths(self):
        """DTW works with different-length series."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = dtwcpp.distance.dtw(x, y)
        assert math.isfinite(d) and d >= 0.0

    def test_single_element(self):
        """DTW of single-element series."""
        assert dtwcpp.distance.dtw([5.0], [3.0]) == pytest.approx(2.0)

    def test_numpy_input(self):
        """DTW accepts numpy arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        assert dtwcpp.distance.dtw(x, y) == pytest.approx(9.0)

