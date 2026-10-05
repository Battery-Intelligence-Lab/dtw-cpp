"""
@file test_cross_validation.py
@brief Cross-validation tests: verify C++ and Python interfaces give identical results.
@details
The filled matrix against dtwcpp.distance.dtw pair by pair, and DTWClustering
against the seeded FastPAM it wraps.
@author Volkan Kumtepeli
"""

import numpy as np
import pytest
import dtwcpp


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_cluster_data():
    """Well-separated clusters for clustering cross-validation."""
    rng = np.random.RandomState(123)
    cluster_a = [rng.randn(20).tolist() for _ in range(5)]           # near 0
    cluster_b = [(rng.randn(20) + 100).tolist() for _ in range(5)]   # near 100
    cluster_c = [(rng.randn(20) + 200).tolist() for _ in range(5)]   # near 200
    return cluster_a + cluster_b + cluster_c


# ---------------------------------------------------------------------------
# Distance matrix cross-validation
# ---------------------------------------------------------------------------

class TestDistanceMatrixCrossValidation:
    """Verify the filled matrix matches individual dist_by_ind calls."""

    def test_full_matrix_matches_pairwise(self, three_cluster_data):
        series = three_cluster_data
        names = [f"s{i}" for i in range(len(series))]

        prob = dtwcpp.Problem("xval_matrix")
        prob.set_data(series, names)
        prob.band = -1
        prob.fill_distance_matrix()

        n = len(series)
        for i in range(n):
            for j in range(i, n):
                d_matrix = prob.dist_by_ind(i, j)
                d_direct = dtwcpp.distance.dtw(series[i], series[j], band=-1)
                assert d_matrix == pytest.approx(d_direct, abs=1e-12), \
                    f"Matrix[{i},{j}]={d_matrix} != direct={d_direct}"

    def test_symmetry_in_filled_matrix(self, three_cluster_data):
        series = three_cluster_data
        names = [f"s{i}" for i in range(len(series))]

        prob = dtwcpp.Problem("xval_sym")
        prob.set_data(series, names)
        prob.band = 5
        prob.fill_distance_matrix()

        n = len(series)
        for i in range(n):
            for j in range(i + 1, n):
                assert prob.dist_by_ind(i, j) == pytest.approx(
                    prob.dist_by_ind(j, i), abs=1e-15
                )


# ---------------------------------------------------------------------------
# Clustering cross-validation: FastPAM vs DTWClustering sugar
# ---------------------------------------------------------------------------

class TestClusteringCrossValidation:
    """Verify DTWClustering produces the seeded Tier-1 FastPAM result."""

    def test_dtw_clustering_matches_fast_pam(self, three_cluster_data):
        series = three_cluster_data
        names = [f"s{i}" for i in range(len(series))]
        k = 3

        # Raw invocation-local FastPAM via Problem.
        prob = dtwcpp.Problem("xval_pam")
        prob.set_data(series, names)
        prob.band = -1
        result_raw = dtwcpp.fast_pam_seeded(
            prob, k, dtwcpp.DEFAULT_RANDOM_SEED
        )

        # Via DTWClustering sugar
        X = np.array(series)
        clf = dtwcpp.DTWClustering(n_clusters=k, band=-1)
        clf.fit(X)

        assert clf.inertia_ == result_raw.total_cost
        np.testing.assert_array_equal(clf.labels_, result_raw.labels)
        np.testing.assert_array_equal(
            clf.medoid_indices_, result_raw.medoid_indices
        )

    def test_dtw_clustering_restarts_use_distinct_local_seeds(self):
        base = np.array([0.0, 0.01, -0.02, 0.03])
        X = base[None, :] + np.arange(8.0)[:, None]

        one = dtwcpp.DTWClustering(n_clusters=3, n_init=1).fit(X)
        two = dtwcpp.DTWClustering(n_clusters=3, n_init=2).fit(X)

        assert one.inertia_ == 24.0  # seed 42
        assert two.inertia_ == 20.0  # seed 43 improves the retained result
        assert two.inertia_ < one.inertia_

    @pytest.mark.parametrize("setting", [{"max_iter": 0}, {"n_init": 0}])
    def test_dtw_clustering_refuses_no_iteration_and_no_run(self, setting):
        """As MATLAB's DTWClustering: max_iter = 0 would report the initial
        medoids' cost as a clustering, n_init = 0 runs nothing."""
        X = np.arange(16.0).reshape(4, 4)
        with pytest.raises(dtwcpp.InvalidInput, match="must be at least 1"):
            dtwcpp.DTWClustering(n_clusters=2, **setting).fit(X)

    def test_predict_assigns_to_nearest_medoid(self, three_cluster_data):
        series = three_cluster_data
        X = np.array(series)
        k = 3

        clf = dtwcpp.DTWClustering(n_clusters=k, band=-1)
        clf.fit(X)

        # Predict on training data
        labels_predict = clf.predict(X)

        # Each point should be assigned to its nearest medoid
        for i in range(len(series)):
            dists = [dtwcpp.distance.dtw(series[i], list(c), band=-1)
                     for c in clf.cluster_centers_]
            expected_label = int(np.argmin(dists))
            assert labels_predict[i] == expected_label, \
                f"Point {i}: predict={labels_predict[i]}, expected={expected_label}"

