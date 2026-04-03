"""
@file test_utils.py
@brief Tests for utility functions and enums.
@author Volkan Kumtepeli
"""

import math

import numpy as np
import pytest

import dtwcpp


class TestZNormalize:
    """Tests for dtwcpp.z_normalize."""

    def test_mean_zero(self):
        """z-normalized data has mean approximately 0."""
        zn = dtwcpp.z_normalize([10, 20, 30, 40, 50])
        assert np.mean(zn) == pytest.approx(0.0, abs=1e-12)

    def test_stddev_one(self):
        """z-normalized data has population stddev approximately 1."""
        zn = dtwcpp.z_normalize([10, 20, 30, 40, 50])
        assert np.std(zn, ddof=0) == pytest.approx(1.0, abs=1e-12)

    def test_returns_list(self):
        """z_normalize returns a list."""
        zn = dtwcpp.z_normalize([1.0, 2.0, 3.0])
        assert isinstance(zn, list)

    def test_length_preserved(self):
        """Output length matches input length."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert len(dtwcpp.z_normalize(x)) == len(x)

    def test_already_normalized(self):
        """z-normalizing already-normalized data is idempotent."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        zn1 = dtwcpp.z_normalize(x)
        zn2 = dtwcpp.z_normalize(zn1)
        np.testing.assert_allclose(zn1, zn2, atol=1e-12)

    def test_numpy_input(self):
        """z_normalize accepts numpy arrays."""
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        zn = dtwcpp.z_normalize(x)
        assert np.mean(zn) == pytest.approx(0.0, abs=1e-12)


class TestDerivativeTransform:
    """Tests for dtwcpp.derivative_transform."""

    def test_known_values(self):
        """derivative_transform([1,3,5,2,4]) matches expected output."""
        dt = dtwcpp.derivative_transform([1, 3, 5, 2, 4])
        # From smoke test: [2.0, 2.0, 0.75, -1.75, 2.0]
        expected = [2.0, 2.0, 0.75, -1.75, 2.0]
        np.testing.assert_allclose(dt, expected, atol=1e-10)

    def test_constant_series(self):
        """Derivative of a constant series is all zeros."""
        dt = dtwcpp.derivative_transform([5.0, 5.0, 5.0, 5.0])
        np.testing.assert_allclose(dt, [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_length_preserved(self):
        """Output length matches input length."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(dtwcpp.derivative_transform(x)) == len(x)

    def test_returns_list(self):
        """derivative_transform returns a list."""
        dt = dtwcpp.derivative_transform([1.0, 2.0, 3.0])
        assert isinstance(dt, list)


class TestEnums:
    """Tests for enum types."""

    def test_dtw_variant_members(self):
        """DTWVariant has expected members."""
        assert hasattr(dtwcpp.DTWVariant, "Standard")
        assert hasattr(dtwcpp.DTWVariant, "DDTW")
        assert hasattr(dtwcpp.DTWVariant, "WDTW")
        assert hasattr(dtwcpp.DTWVariant, "ADTW")
        assert hasattr(dtwcpp.DTWVariant, "SoftDTW")

    def test_dtw_variant_equality(self):
        """Enum members compare equal to themselves."""
        assert dtwcpp.DTWVariant.Standard == dtwcpp.DTWVariant.Standard
        assert dtwcpp.DTWVariant.WDTW == dtwcpp.DTWVariant.WDTW

    def test_dtw_variant_inequality(self):
        """Different enum members are not equal."""
        assert dtwcpp.DTWVariant.Standard != dtwcpp.DTWVariant.WDTW

    def test_method_enum(self):
        """Method enum has Kmedoids."""
        assert hasattr(dtwcpp.Method, "Kmedoids")

    def test_constraint_type_enum(self):
        """ConstraintType enum has expected members."""
        assert hasattr(dtwcpp.ConstraintType, "NONE")
        assert hasattr(dtwcpp.ConstraintType, "SakoeChibaBand")

    def test_metric_type_enum(self):
        """MetricType enum has expected members."""
        assert hasattr(dtwcpp.MetricType, "L1")
        assert hasattr(dtwcpp.MetricType, "L2")
        assert hasattr(dtwcpp.MetricType, "SquaredL2")

    def test_dtw_variant_params_defaults(self):
        """DTWVariantParams has sensible defaults."""
        params = dtwcpp.DTWVariantParams()
        assert params.variant == dtwcpp.DTWVariant.Standard

    def test_dtw_variant_params_set_fields(self):
        """DTWVariantParams fields can be set."""
        params = dtwcpp.DTWVariantParams()
        params.variant = dtwcpp.DTWVariant.ADTW
        params.adtw_penalty = 2.5
        assert params.variant == dtwcpp.DTWVariant.ADTW
        assert params.adtw_penalty == pytest.approx(2.5)
