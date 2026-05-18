"""Tests for dtwcpp.features.summarise."""

from __future__ import annotations

import numpy as np
import pytest

import dtwcpp
from dtwcpp import features as ft


class TestSummariseBasic:
    def test_default_feature_set(self):
        rng = np.random.default_rng(0)
        series = [rng.standard_normal(10) for _ in range(5)]
        mat, names = ft.summarise(series, standardise=False)
        assert mat.shape == (5, len(ft.DEFAULT_FEATURES))
        assert names == list(ft.DEFAULT_FEATURES.keys())

    def test_known_features(self):
        # One series, exact features
        series = [[0.0, 100.0, -100.0, 0.0]]
        mat, names = ft.summarise(series, standardise=False)
        # Find column indices
        idx = {n: i for i, n in enumerate(names)}
        assert mat[0, idx["mean"]] == pytest.approx(0.0)
        assert mat[0, idx["max"]] == pytest.approx(100.0)
        assert mat[0, idx["min"]] == pytest.approx(-100.0)
        assert mat[0, idx["length"]] == pytest.approx(4.0)
        assert mat[0, idx["abs_sum"]] == pytest.approx(200.0)
        # idle fraction: |0| < 10 and |0| < 10 → 2 out of 4
        assert mat[0, idx["idle_fraction"]] == pytest.approx(0.5)

    def test_variable_length(self):
        series = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0, 50.0]]
        mat, _ = ft.summarise(series, standardise=False)
        idx_length = list(ft.DEFAULT_FEATURES).index("length")
        assert mat[0, idx_length] == 3.0
        assert mat[1, idx_length] == 5.0

    def test_standardise_zero_mean_unit_std(self):
        series = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]
        mat, _ = ft.summarise(series, standardise=True)
        # Each column should have ~0 mean and ~1 std
        np.testing.assert_allclose(mat.mean(axis=0), 0, atol=1e-12)
        sd = mat.std(axis=0)
        # Constant columns (e.g. length all = 2) collapse to zero, not std=1
        nonzero_cols = sd > 1e-9
        np.testing.assert_allclose(sd[nonzero_cols], 1, atol=1e-12)

    def test_constant_column_safe(self):
        # All series same length → length column has zero variance
        series = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mat, names = ft.summarise(series, standardise=True)
        idx = names.index("length")
        np.testing.assert_array_equal(mat[:, idx], [0.0, 0.0])

    def test_empty_input(self):
        mat, names = ft.summarise([])
        assert mat.shape == (0, len(ft.DEFAULT_FEATURES))
        assert names == list(ft.DEFAULT_FEATURES.keys())

    def test_custom_feature_set(self):
        custom = {"mean": np.mean, "max": np.max}
        series = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mat, names = ft.summarise(series, features=custom, standardise=False)
        assert names == ["mean", "max"]
        assert mat.shape == (2, 2)


class TestModuleExport:
    def test_attached(self):
        assert hasattr(dtwcpp, "features")
        assert dtwcpp.features is ft
