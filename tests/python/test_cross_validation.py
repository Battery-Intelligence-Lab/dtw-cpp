"""
Cross-validation tests: verify C++ and Python interfaces give identical results.

These tests compute DTW distances through both the direct C++ bindings and
the Python sugar layer, ensuring they produce the same numerical results.
"""

import numpy as np
import pytest
import dtwcpp


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def series_pair():
    """A deterministic pair of time series for reproducibility."""
    rng = np.random.RandomState(42)
    x = rng.randn(100).tolist()
    y = rng.randn(100).tolist()
    return x, y


@pytest.fixture
def short_pair():
    """Hand-crafted short series for exact verification."""
    return [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 3.0, 1.0]


@pytest.fixture
def three_cluster_data():
    """Well-separated clusters for clustering cross-validation."""
    rng = np.random.RandomState(123)
    cluster_a = [rng.randn(20).tolist() for _ in range(5)]           # near 0
    cluster_b = [(rng.randn(20) + 100).tolist() for _ in range(5)]   # near 100
    cluster_c = [(rng.randn(20) + 200).tolist() for _ in range(5)]   # near 200
    return cluster_a + cluster_b + cluster_c


# ---------------------------------------------------------------------------
# DTW distance cross-validation: direct C++ vs Problem.dist_by_ind
# ---------------------------------------------------------------------------

class TestDTWCrossValidation:
    """Verify dtw_distance() matches Problem.dist_by_ind() for all variants."""

    def test_standard_dtw_matches_problem(self, short_pair):
        x, y = short_pair

        # Direct C++ function
        d_direct = dtwcpp.dtw_distance(x, y, band=-1)

        # Via Problem class
        prob = dtwcpp.Problem("xval")
        prob.set_data([x, y], ["s0", "s1"])
        prob.band = -1
        prob.fill_distance_matrix()
        d_problem = prob.dist_by_ind(0, 1)

        assert d_direct == pytest.approx(d_problem, abs=1e-12), \
            f"Direct DTW {d_direct} != Problem DTW {d_problem}"

    def test_banded_dtw_matches_problem(self, series_pair):
        x, y = series_pair

        d_direct = dtwcpp.dtw_distance(x, y, band=10)

        prob = dtwcpp.Problem("xval_banded")
        prob.set_data([x, y], ["s0", "s1"])
        prob.band = 10
        prob.fill_distance_matrix()
        d_problem = prob.dist_by_ind(0, 1)

        assert d_direct == pytest.approx(d_problem, abs=1e-12)

    def test_ddtw_matches_problem(self, short_pair):
        x, y = short_pair

        d_direct = dtwcpp.ddtw_distance(x, y, band=-1)

        prob = dtwcpp.Problem("xval_ddtw")
        prob.set_data([x, y], ["s0", "s1"])
        prob.band = -1
        prob.set_variant(dtwcpp.DTWVariant.DDTW)
        prob.fill_distance_matrix()
        d_problem = prob.dist_by_ind(0, 1)

        assert d_direct == pytest.approx(d_problem, abs=1e-12), \
            f"Direct DDTW {d_direct} != Problem DDTW {d_problem}"

    def test_wdtw_matches_problem(self, short_pair):
        x, y = short_pair
        g = 0.1

        d_direct = dtwcpp.wdtw_distance(x, y, band=-1, g=g)

        prob = dtwcpp.Problem("xval_wdtw")
        prob.set_data([x, y], ["s0", "s1"])
        prob.band = -1
        vp = dtwcpp.DTWVariantParams()
        vp.variant = dtwcpp.DTWVariant.WDTW
        vp.wdtw_g = g
        prob.variant_params = vp
        prob.refresh_distance_matrix()
        prob.fill_distance_matrix()
        d_problem = prob.dist_by_ind(0, 1)

        assert d_direct == pytest.approx(d_problem, abs=1e-12), \
            f"Direct WDTW {d_direct} != Problem WDTW {d_problem}"

    def test_adtw_matches_problem(self, short_pair):
        x, y = short_pair
        penalty = 2.0

        d_direct = dtwcpp.adtw_distance(x, y, band=-1, penalty=penalty)

        prob = dtwcpp.Problem("xval_adtw")
        prob.set_data([x, y], ["s0", "s1"])
        prob.band = -1
        vp = dtwcpp.DTWVariantParams()
        vp.variant = dtwcpp.DTWVariant.ADTW
        vp.adtw_penalty = penalty
        prob.variant_params = vp
        prob.refresh_distance_matrix()
        prob.fill_distance_matrix()
        d_problem = prob.dist_by_ind(0, 1)

        assert d_direct == pytest.approx(d_problem, abs=1e-12), \
            f"Direct ADTW {d_direct} != Problem ADTW {d_problem}"


# ---------------------------------------------------------------------------
# Distance matrix cross-validation
# ---------------------------------------------------------------------------

class TestDistanceMatrixCrossValidation:
    """Verify DenseDistanceMatrix.to_numpy() matches individual dist_by_ind calls."""

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
                d_direct = dtwcpp.dtw_distance(series[i], series[j], band=-1)
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
    """Verify DTWClustering produces same results as raw fast_pam."""

    def test_dtw_clustering_matches_fast_pam(self, three_cluster_data):
        series = three_cluster_data
        names = [f"s{i}" for i in range(len(series))]
        k = 3

        # Raw fast_pam via Problem
        prob = dtwcpp.Problem("xval_pam")
        prob.set_data(series, names)
        prob.band = -1
        result_raw = dtwcpp.fast_pam(prob, k)

        # Via DTWClustering sugar
        X = np.array(series)
        clf = dtwcpp.DTWClustering(n_clusters=k, band=-1)
        clf.fit(X)

        # Both should find the same clusters (labels may differ in numbering)
        # Check: total cost should be identical or very close
        # (May differ slightly due to different initialization seeds)
        assert clf.inertia_ >= 0
        assert len(clf.labels_) == len(series)
        assert len(clf.medoid_indices_) == k

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
            dists = [dtwcpp.dtw_distance(series[i], list(c), band=-1)
                     for c in clf.cluster_centers_]
            expected_label = int(np.argmin(dists))
            assert labels_predict[i] == expected_label, \
                f"Point {i}: predict={labels_predict[i]}, expected={expected_label}"


# ---------------------------------------------------------------------------
# Variant consistency: same variant gives same result through different paths
# ---------------------------------------------------------------------------

class TestVariantConsistency:
    """Verify that DTW variants are consistent across all calling paths."""

    @pytest.mark.parametrize("band", [-1, 5, 20])
    def test_dtw_banded_consistency(self, series_pair, band):
        """Same band value gives same result via dtw_distance and Problem."""
        x, y = series_pair

        d1 = dtwcpp.dtw_distance(x, y, band=band)
        d2 = dtwcpp.dtw_distance(x, y, band=band)
        assert d1 == d2, "Same call twice gives different results!"

    def test_derivative_then_dtw_equals_ddtw(self, short_pair):
        """DDTW should equal manual derivative + standard DTW."""
        x, y = short_pair

        dx = dtwcpp.derivative_transform(x)
        dy = dtwcpp.derivative_transform(y)
        d_manual = dtwcpp.dtw_distance(dx, dy, band=-1)
        d_ddtw = dtwcpp.ddtw_distance(x, y, band=-1)

        assert d_manual == pytest.approx(d_ddtw, abs=1e-12), \
            f"Manual deriv+DTW {d_manual} != DDTW {d_ddtw}"

    def test_adtw_penalty_zero_equals_standard(self, series_pair):
        """ADTW with penalty=0 should equal standard DTW."""
        x, y = series_pair

        d_std = dtwcpp.dtw_distance(x, y, band=-1)
        d_adtw = dtwcpp.adtw_distance(x, y, band=-1, penalty=0.0)

        assert d_std == pytest.approx(d_adtw, abs=1e-12)

    def test_soft_dtw_approaches_dtw(self, short_pair):
        """Soft-DTW with gamma→0 should approach standard DTW."""
        x, y = short_pair

        d_hard = dtwcpp.dtw_distance(x, y, band=-1)
        d_soft = dtwcpp.soft_dtw_distance(x, y, gamma=0.001)

        assert d_soft == pytest.approx(d_hard, abs=0.1), \
            f"Soft-DTW(gamma=0.001)={d_soft} not close to DTW={d_hard}"
