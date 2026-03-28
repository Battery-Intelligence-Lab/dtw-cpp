"""Tests for the Problem class Python bindings."""

import math

import numpy as np
import pytest

import dtwcpp


class TestProblemConstruction:
    """Tests for Problem construction and basic properties."""

    def test_default_construction(self):
        """Problem() default construction works."""
        p = dtwcpp.Problem()
        assert p is not None

    def test_named_construction(self):
        """Problem('name') stores the name."""
        p = dtwcpp.Problem("mytest")
        assert p.name == "mytest"

    def test_default_band(self):
        """Default band is -1 (full DTW)."""
        p = dtwcpp.Problem("test")
        assert p.band == -1


class TestProblemData:
    """Tests for set_data and data access."""

    def test_set_data_list_of_lists(self):
        """set_data works with list-of-lists."""
        p = dtwcpp.Problem("test")
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        p.set_data(data, ["a", "b", "c"])
        assert p.size == 3

    def test_set_data_numpy(self, synthetic_data):
        """set_data works with numpy array rows."""
        p = dtwcpp.Problem("test")
        names = [f"s{i}" for i in range(len(synthetic_data))]
        rows = [row.tolist() for row in synthetic_data]
        p.set_data(rows, names)
        assert p.size == len(synthetic_data)


class TestDistanceMatrix:
    """Tests for fill_distance_matrix and dist_by_ind."""

    def _make_problem(self, data):
        """Helper to create a filled problem."""
        names = [f"s{i}" for i in range(len(data))]
        p = dtwcpp.Problem("test")
        p.set_data(data, names)
        p.fill_distance_matrix()
        return p

    def test_fill_and_query(self):
        """fill_distance_matrix populates all pairs."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
        p = self._make_problem(data)
        assert p.is_distance_matrix_filled()
        # All pairs should be finite
        for i in range(3):
            for j in range(3):
                d = p.dist_by_ind(i, j)
                assert math.isfinite(d), f"dist({i},{j}) is not finite: {d}"

    def test_self_distance_zero(self):
        """dist_by_ind(i, i) == 0 for all i."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        p = self._make_problem(data)
        for i in range(3):
            assert p.dist_by_ind(i, i) == pytest.approx(0.0)

    def test_symmetry(self):
        """dist_by_ind(i, j) == dist_by_ind(j, i)."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.5, 2.5, 3.5]]
        p = self._make_problem(data)
        for i in range(3):
            for j in range(i + 1, 3):
                assert p.dist_by_ind(i, j) == pytest.approx(p.dist_by_ind(j, i))

    def test_known_distance(self):
        """dist_by_ind matches standalone dtw_distance."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        p = self._make_problem(data)
        expected = dtwcpp.dtw_distance([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert p.dist_by_ind(0, 1) == pytest.approx(expected)


class TestBandProperty:
    """Tests for the band property."""

    def test_set_band(self):
        """Band can be set and read back."""
        p = dtwcpp.Problem("test")
        p.band = 5
        assert p.band == 5

    def test_band_affects_distance(self):
        """Setting a tight band can change computed distances."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
        names = ["a", "b"]

        # Full DTW
        p1 = dtwcpp.Problem("full")
        p1.set_data(data, names)
        p1.fill_distance_matrix()
        d_full = p1.dist_by_ind(0, 1)

        # Banded DTW
        p2 = dtwcpp.Problem("banded")
        p2.band = 1
        p2.set_data(data, names)
        p2.fill_distance_matrix()
        d_banded = p2.dist_by_ind(0, 1)

        # Banded should be >= full (or equal for trivial cases)
        assert d_banded >= d_full - 1e-10


class TestVariant:
    """Tests for DTW variant selection on Problem."""

    def test_set_variant_enum(self):
        """set_variant accepts a DTWVariant enum and updates variant_params."""
        p = dtwcpp.Problem("test")
        p.set_variant(dtwcpp.DTWVariant.WDTW)
        assert p.variant_params.variant == dtwcpp.DTWVariant.WDTW

    def test_variant_params_fields(self):
        """variant_params fields can be set via the property."""
        p = dtwcpp.Problem("test")
        p.set_variant(dtwcpp.DTWVariant.WDTW)
        vp = p.variant_params
        vp.wdtw_g = 0.1
        p.variant_params = vp
        assert p.variant_params.wdtw_g == pytest.approx(0.1)

    def test_variant_changes_distances(self):
        """Using WDTW variant produces different distances than standard."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0]]
        names = ["a", "b"]

        # Standard DTW
        p1 = dtwcpp.Problem("standard")
        p1.set_data(data, names)
        p1.fill_distance_matrix()
        d_std = p1.dist_by_ind(0, 1)

        # WDTW
        p2 = dtwcpp.Problem("wdtw")
        p2.set_variant(dtwcpp.DTWVariant.WDTW)
        p2.set_data(data, names)
        p2.fill_distance_matrix()
        d_wdtw = p2.dist_by_ind(0, 1)

        # WDTW should differ from standard
        assert d_wdtw != pytest.approx(d_std, abs=1e-6)
