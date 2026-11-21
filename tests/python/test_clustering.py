"""
@file test_clustering.py
@brief Tests for FastPAM iteration contracts via DTWC++ Python bindings.
@author Volkan Kumtepeli
"""

import pytest

import dtwcpp


class TestFastPamMaxIter:
    """max_iter = 0 is the BUILD-only oracle (the seeded start, no SWAP); a negative
    count has no meaning and the C++ refuses it, so every binding raises the same
    InvalidInput. The series and medoids are those of unit_test_fast_pam.cpp and of
    the MATLAB suite (there 1-based)."""

    _SERIES = [[i, i + 0.01, i - 0.02, i + 0.03] for i in range(8)]

    def _problem(self):
        prob = dtwcpp.Problem("max_iter")
        prob.set_data(self._SERIES, [str(i) for i in range(len(self._SERIES))])
        return prob

    def test_zero_returns_the_seeded_build_medoids(self):
        prob = self._problem()
        build_29 = dtwcpp.fast_pam(prob, 3, max_iter=0, seed=29)
        assert list(build_29.medoid_indices) == [4, 2, 7]
        assert build_29.iterations == 0
        assert build_29.converged is False
        assert list(dtwcpp.fast_pam(prob, 3, max_iter=0, seed=42).medoid_indices) == [6, 2, 5]
        default_seed = dtwcpp.fast_pam(prob, 3, max_iter=0)  # the default seed is 42
        assert list(default_seed.medoid_indices) == [6, 2, 5]
        assert default_seed.iterations == 0
        assert default_seed.converged is False
        swapped = dtwcpp.fast_pam(prob, 3, max_iter=100, seed=29)
        assert list(swapped.medoid_indices) == [4, 1, 7]
        assert (swapped.n_clusters, swapped.n_points, swapped.converged) == (3, 8, True)

    @pytest.mark.parametrize("bad", [-1, -100, -(2**31)])
    def test_a_negative_count_raises_before_it_touches_the_problem(self, bad):
        prob = self._problem()
        prob.set_n_clusters(2)
        with pytest.raises(dtwcpp.InvalidInput, match=rf"fast_pam: max_iter .*got {bad}(?![0-9])"):
            dtwcpp.fast_pam(prob, 3, bad)
        assert list(prob.labels()) == []
        assert prob.n_clusters() == 2
