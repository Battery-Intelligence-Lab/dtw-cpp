"""Tests for dtwcpp.preprocess.

Covers each helper in isolation plus the chained `power_signal` entry point.
Scipy-dependent `sg_smooth` is tested with skipif when scipy is missing.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import dtwcpp
from dtwcpp import preprocess as pp


class TestStripIdle:
    def test_drops_leading_and_trailing(self):
        s = [0.0, 0.0, 50.0, 100.0, -200.0, 0.0, 0.0, 0.0]
        out = pp.strip_idle(s, threshold=10.0)
        np.testing.assert_array_equal(out, [50.0, 100.0, -200.0])

    def test_keeps_when_above_threshold(self):
        s = [50.0, 100.0, -200.0]
        out = pp.strip_idle(s, threshold=10.0)
        np.testing.assert_array_equal(out, s)

    def test_all_below_returns_empty(self):
        out = pp.strip_idle([0.0, 1.0, -2.0, 3.0], threshold=10.0)
        assert out.size == 0

    def test_threshold_uses_absolute_value(self):
        out = pp.strip_idle([0.0, -50.0, 0.0], threshold=10.0)
        np.testing.assert_array_equal(out, [-50.0])

    def test_threshold_zero_keeps_all(self):
        s = [0.0, 1.0, 2.0, 3.0]
        out = pp.strip_idle(s, threshold=0.0)
        np.testing.assert_array_equal(out, s)


class TestDecimateZoh:
    def test_removes_consecutive_duplicates(self):
        s = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0]
        out = pp.decimate_zoh(s)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_preserves_non_repeating(self):
        s = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(pp.decimate_zoh(s), s)

    def test_empty_input(self):
        out = pp.decimate_zoh([])
        assert out.size == 0

    def test_single_element(self):
        out = pp.decimate_zoh([5.0])
        np.testing.assert_array_equal(out, [5.0])

    def test_all_same(self):
        out = pp.decimate_zoh([7.0] * 10)
        np.testing.assert_array_equal(out, [7.0])

    def test_alternating(self):
        s = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0]
        np.testing.assert_array_equal(pp.decimate_zoh(s), [1.0, 2.0, 1.0])


class TestDerivative:
    def test_simple(self):
        np.testing.assert_array_equal(pp.derivative([1.0, 3.0, 6.0]), [2.0, 3.0])

    def test_constant_returns_zeros(self):
        np.testing.assert_array_equal(pp.derivative([5.0, 5.0, 5.0]), [0.0, 0.0])

    def test_short_input_returns_empty(self):
        assert pp.derivative([]).size == 0
        assert pp.derivative([1.0]).size == 0


class TestZNormalize:
    def test_mean_zero_unit_std(self):
        out = pp.z_normalize([10.0, 20.0, 30.0, 40.0])
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std() == pytest.approx(1.0, abs=1e-12)

    def test_returns_numpy_array(self):
        out = pp.z_normalize([1.0, 2.0, 3.0])
        assert isinstance(out, np.ndarray)


_SCIPY = importlib.util.find_spec("scipy") is not None


@pytest.mark.skipif(not _SCIPY, reason="scipy not installed")
class TestSgSmooth:
    def test_preserves_length(self):
        s = np.arange(20, dtype=np.float64)
        out = pp.sg_smooth(s, window=5, poly=2)
        assert out.size == s.size

    def test_short_input_passes_through(self):
        s = [1.0, 2.0, 3.0]   # len < default window 5
        out = pp.sg_smooth(s, window=5, poly=2)
        np.testing.assert_array_equal(out, s)


@pytest.mark.skipif(_SCIPY, reason="scipy IS installed; test only when missing")
class TestSgSmoothNoScipy:
    def test_raises_clear_import_error(self):
        with pytest.raises(ImportError, match="scipy"):
            pp.sg_smooth([1.0] * 10, window=5, poly=2)


class TestPowerSignal:
    def test_chains_strip_decimate_diff_znorm(self):
        # Build: 3 zero pads + ZOH(3) of [10,20,30] + 2 trailing zeros
        s = [0.0, 0.0, 0.0,
             10.0, 10.0, 10.0,
             20.0, 20.0, 20.0,
             30.0, 30.0, 30.0,
             0.0, 0.0]
        out = pp.power_signal(s, idle_threshold=5.0, sg_window=0,
                              take_derivative=True, znorm=True)
        # After strip: 9 samples of [10,10,10,20,20,20,30,30,30]
        # After decimate: [10, 20, 30]
        # After diff: [10, 10]
        # After z-norm: [0, 0] (zero std edge — handled by C++ z_normalize
        # which returns zero-mean but defined output)
        assert out.size == 2
        assert out.mean() == pytest.approx(0.0, abs=1e-12)

    def test_no_derivative_no_znorm(self):
        s = [0.0, 100.0, 100.0, 200.0, 0.0]
        out = pp.power_signal(s, idle_threshold=10.0, sg_window=0,
                              take_derivative=False, znorm=False)
        np.testing.assert_array_equal(out, [100.0, 200.0])

    def test_empty_after_strip_returns_empty(self):
        out = pp.power_signal([0.0, 1.0, 2.0], idle_threshold=10.0)
        assert out.size == 0

    def test_idle_threshold_zero_skips_strip(self):
        s = [0.0, 1.0, 1.0, 2.0]
        out = pp.power_signal(s, idle_threshold=0.0, sg_window=0,
                              take_derivative=False, znorm=False)
        np.testing.assert_array_equal(out, [0.0, 1.0, 2.0])


class TestModuleExport:
    def test_preprocess_attached_to_package(self):
        # Importable as `dtwcpp.preprocess`
        assert hasattr(dtwcpp, "preprocess")
        assert dtwcpp.preprocess is pp

    def test_public_names(self):
        for name in ("strip_idle", "decimate_zoh", "sg_smooth", "derivative",
                     "z_normalize", "power_signal"):
            assert hasattr(pp, name)
