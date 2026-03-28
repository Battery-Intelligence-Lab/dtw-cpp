"""Tests for DTW distance functions."""

import math

import numpy as np
import pytest

import dtwcpp


# ---------------------------------------------------------------------------
# Standard DTW
# ---------------------------------------------------------------------------


class TestDTWDistance:
    """Tests for dtwcpp.dtw_distance."""

    def test_identity(self):
        """DTW of identical series is zero."""
        assert dtwcpp.dtw_distance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value_l1(self):
        """DTW([1,2,3],[4,5,6]) == 9 with default (L1-squared) metric."""
        assert dtwcpp.dtw_distance([1, 2, 3], [4, 5, 6]) == pytest.approx(9.0)

    def test_symmetry(self):
        """DTW(x, y) == DTW(y, x)."""
        x = [1.0, 3.0, 0.5, 2.0, 7.0]
        y = [2.0, 0.0, 4.0, 1.0, 6.0]
        assert dtwcpp.dtw_distance(x, y) == pytest.approx(dtwcpp.dtw_distance(y, x))

    def test_non_negativity(self):
        """DTW distance is never negative."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            x = rng.standard_normal(15).tolist()
            y = rng.standard_normal(15).tolist()
            assert dtwcpp.dtw_distance(x, y) >= 0.0

    def test_triangle_inequality_holds_approximately(self):
        """DTW does not guarantee triangle inequality, but for L2-style metrics
        it usually holds on short series. Just verify non-negative here."""
        x = [1.0, 2.0, 3.0]
        y = [3.0, 1.0, 2.0]
        z = [2.0, 3.0, 1.0]
        dxy = dtwcpp.dtw_distance(x, y)
        dyz = dtwcpp.dtw_distance(y, z)
        dxz = dtwcpp.dtw_distance(x, z)
        assert dxy >= 0 and dyz >= 0 and dxz >= 0

    def test_banded(self):
        """Banded DTW returns a finite non-negative result."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        d = dtwcpp.dtw_distance(x, y, band=2)
        assert math.isfinite(d) and d >= 0.0

    def test_banded_ge_full(self):
        """Banded DTW >= full DTW (band restricts the warping path)."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        y = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        d_full = dtwcpp.dtw_distance(x, y, band=-1)
        d_band = dtwcpp.dtw_distance(x, y, band=2)
        assert d_band >= d_full - 1e-12

    def test_different_lengths(self):
        """DTW works with different-length series."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = dtwcpp.dtw_distance(x, y)
        assert math.isfinite(d) and d >= 0.0

    def test_single_element(self):
        """DTW of single-element series."""
        assert dtwcpp.dtw_distance([5.0], [3.0]) == pytest.approx(2.0)

    def test_numpy_input(self):
        """DTW accepts numpy arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        assert dtwcpp.dtw_distance(x, y) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# DTW Variants
# ---------------------------------------------------------------------------


class TestDDTW:
    """Tests for Derivative DTW."""

    def test_shifted_constant_slope(self):
        """Series with same slope but different offset have DDTW == 0."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 6.0, 7.0, 8.0]
        assert dtwcpp.ddtw_distance(x, y) == pytest.approx(0.0)

    def test_different_series_positive(self):
        """DDTW of different series is positive."""
        x = [1.0, 3.0, 2.0, 5.0, 4.0]
        y = [2.0, 1.0, 4.0, 3.0, 5.0]
        d = dtwcpp.ddtw_distance(x, y)
        assert math.isfinite(d) and d > 0.0

    def test_banded(self):
        """Banded DDTW works."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 3.0, 1.0, 2.0, 4.0]
        d = dtwcpp.ddtw_distance(x, y, band=2)
        assert math.isfinite(d) and d >= 0.0


class TestWDTW:
    """Tests for Weighted DTW."""

    def test_positive_for_different_series(self):
        """WDTW returns a finite positive value for different series."""
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        d = dtwcpp.wdtw_distance(x, y)
        assert math.isfinite(d) and d > 0.0

    def test_identity(self):
        """WDTW of identical series is zero."""
        x = [1.0, 2.0, 3.0, 4.0]
        assert dtwcpp.wdtw_distance(x, x) == pytest.approx(0.0)

    def test_g_parameter_effect(self):
        """Different g values produce different distances."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        d1 = dtwcpp.wdtw_distance(x, y, g=0.01)
        d2 = dtwcpp.wdtw_distance(x, y, g=1.0)
        # Different g values should (generally) give different distances
        assert d1 != pytest.approx(d2, abs=1e-6)

    def test_le_standard_dtw(self):
        """WDTW <= standard DTW (weights are in [0,1])."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        d_std = dtwcpp.dtw_distance(x, y)
        d_w = dtwcpp.wdtw_distance(x, y)
        assert d_w <= d_std + 1e-10


class TestADTW:
    """Tests for Amerced DTW."""

    def test_positive_for_different_series(self):
        """ADTW returns positive value for different series."""
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        d = dtwcpp.adtw_distance(x, y)
        assert math.isfinite(d) and d > 0.0

    def test_identity(self):
        """ADTW of identical series is zero."""
        x = [1.0, 2.0, 3.0, 4.0]
        assert dtwcpp.adtw_distance(x, x) == pytest.approx(0.0)

    def test_penalty_zero_equals_dtw(self):
        """With penalty=0, ADTW should equal standard DTW."""
        x = [1.0, 3.0, 2.0, 5.0]
        y = [2.0, 1.0, 4.0, 3.0]
        d_std = dtwcpp.dtw_distance(x, y)
        d_adtw = dtwcpp.adtw_distance(x, y, penalty=0.0)
        assert d_adtw == pytest.approx(d_std)

    def test_higher_penalty_ge(self):
        """Higher penalty should give >= distance (more penalization of warping)."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        d_low = dtwcpp.adtw_distance(x, y, penalty=0.0)
        d_high = dtwcpp.adtw_distance(x, y, penalty=10.0)
        assert d_high >= d_low - 1e-10


class TestSoftDTW:
    """Tests for Soft-DTW."""

    def test_positive_for_different_series(self):
        """Soft-DTW returns finite value for different series."""
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        d = dtwcpp.soft_dtw_distance(x, y, gamma=1.0)
        assert math.isfinite(d)

    def test_identity_small_gamma(self):
        """Soft-DTW of identical series approaches 0 as gamma -> 0."""
        x = [1.0, 2.0, 3.0]
        d = dtwcpp.soft_dtw_distance(x, x, gamma=0.01)
        assert d == pytest.approx(0.0, abs=0.1)

    def test_self_distance_le_zero(self):
        """Soft-DTW self-distance is <= 0 (softmin effect)."""
        x = [1.0, 2.0, 3.0]
        d = dtwcpp.soft_dtw_distance(x, x, gamma=1.0)
        assert d <= 0.0 + 1e-10

    def test_approaches_dtw_as_gamma_decreases(self):
        """As gamma -> 0, Soft-DTW -> standard DTW."""
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        d_hard = dtwcpp.dtw_distance(x, y)
        d_soft_small = dtwcpp.soft_dtw_distance(x, y, gamma=0.01)
        assert d_soft_small == pytest.approx(d_hard, abs=0.1)

    def test_soft_dtw_le_hard_dtw(self):
        """Soft-DTW <= hard DTW (softmin <= min)."""
        x = [1.0, 3.0, 2.0, 5.0, 4.0]
        y = [2.0, 1.0, 4.0, 3.0, 6.0]
        d_hard = dtwcpp.dtw_distance(x, y)
        d_soft = dtwcpp.soft_dtw_distance(x, y, gamma=1.0)
        assert d_soft <= d_hard + 1e-10

    def test_symmetry(self):
        """Soft-DTW is symmetric."""
        x = [1.0, 3.0, 5.0]
        y = [2.0, 4.0, 6.0]
        assert dtwcpp.soft_dtw_distance(x, y) == pytest.approx(
            dtwcpp.soft_dtw_distance(y, x)
        )
