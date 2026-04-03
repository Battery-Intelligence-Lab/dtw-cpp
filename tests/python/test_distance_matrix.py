"""
@file test_distance_matrix.py
@brief Tests for DenseDistanceMatrix Python bindings.
@author Volkan Kumtepeli
"""

import math

import numpy as np
import pytest

import dtwcpp


class TestDenseDistanceMatrix:
    """Tests for dtwcpp.DenseDistanceMatrix."""

    def test_construction_size(self):
        """Constructed matrix has correct size."""
        dm = dtwcpp.DenseDistanceMatrix(5)
        assert dm.size == 5

    def test_set_get(self):
        """set(i,j,v) and get(i,j) round-trip correctly."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(1, 2, 3.14)
        assert dm.get(1, 2) == pytest.approx(3.14)

    def test_symmetry_enforcement(self):
        """set(i,j,v) makes get(j,i) == v (symmetric storage)."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(0, 3, 7.5)
        assert dm.get(3, 0) == pytest.approx(7.5)

    def test_to_numpy_shape(self):
        """to_numpy() returns (n, n) array."""
        n = 6
        dm = dtwcpp.DenseDistanceMatrix(n)
        arr = dm.to_numpy()
        assert arr.shape == (n, n)

    def test_to_numpy_values(self):
        """to_numpy() reflects values set via set()."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        dm.set(0, 1, 5.0)
        dm.set(0, 2, 7.0)
        dm.set(1, 2, 3.0)
        arr = dm.to_numpy()
        assert arr[0, 1] == pytest.approx(5.0)
        assert arr[1, 0] == pytest.approx(5.0)
        assert arr[0, 2] == pytest.approx(7.0)
        assert arr[2, 0] == pytest.approx(7.0)
        assert arr[1, 2] == pytest.approx(3.0)
        assert arr[2, 1] == pytest.approx(3.0)

    def test_to_numpy_symmetric(self):
        """The full numpy matrix is symmetric."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(0, 1, 1.0)
        dm.set(0, 2, 2.0)
        dm.set(0, 3, 3.0)
        dm.set(1, 2, 4.0)
        dm.set(1, 3, 5.0)
        dm.set(2, 3, 6.0)
        arr = dm.to_numpy()
        np.testing.assert_array_almost_equal(arr, arr.T)

    def test_to_numpy_zero_copy(self):
        """Modifying the numpy array changes the C++ matrix (zero-copy)."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(0, 1, 1.0)
        arr = dm.to_numpy()
        assert not arr.flags["OWNDATA"], "Expected zero-copy (OWNDATA=False)"
        arr[0, 1] = 42.0
        assert dm.get(0, 1) == pytest.approx(42.0)

    def test_to_numpy_dtype(self):
        """to_numpy() returns float64 array."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        arr = dm.to_numpy()
        assert arr.dtype == np.float64

    def test_resize(self):
        """resize() changes the matrix size."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        assert dm.size == 3
        dm.resize(7)
        assert dm.size == 7

    def test_max(self):
        """max() returns the largest stored value."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        dm.set(0, 1, 10.0)
        dm.set(0, 2, 5.0)
        dm.set(1, 2, 20.0)
        assert dm.max() == pytest.approx(20.0)
