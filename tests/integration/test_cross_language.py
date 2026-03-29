"""Cross-language integration tests: verify C++ and Python produce identical results.

These tests exercise the C++ core through the Python bindings, verifying that
different entry points (standalone functions, Problem class, DTWClustering wrapper)
all produce consistent and correct results for the same input data.

This complements tests/python/test_cross_validation.py by adding:
- Missing-data DTW tests (NaN handling)
- compute_distance_matrix standalone function tests
- CLARA clustering tests
- Checkpoint save/load roundtrip
- Metric cross-validation (L1 vs squared_euclidean)
"""

import math
import os
import tempfile

import numpy as np
import pytest

import dtwcpp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_series(n, seed):
    """Deterministic random series for reproducibility."""
    return np.random.default_rng(seed).uniform(-1, 1, n).tolist()


def make_dataset(n_series, length, seed_base=0):
    """Create a list of deterministic random series."""
    return [make_series(length, seed_base + i) for i in range(n_series)]


# ===========================================================================
# DTW distance consistency
# ===========================================================================


class TestDTWMetricConsistency:
    """Verify metric selection produces correct results across entry points."""

    def test_l1_known_value(self):
        """DTW([0,1,2], [0,0,0]) with L1 metric = |0|+|1|+|2| = 3."""
        x = [0.0, 1.0, 2.0]
        y = [0.0, 0.0, 0.0]
        d = dtwcpp.dtw_distance(x, y, metric='l1')
        assert d == pytest.approx(3.0)

    def test_squared_euclidean_known_value(self):
        """DTW([0,1,2], [0,0,0]) with squared_euclidean = 0+1+4 = 5."""
        x = [0.0, 1.0, 2.0]
        y = [0.0, 0.0, 0.0]
        d = dtwcpp.dtw_distance(x, y, metric='squared_euclidean')
        assert d == pytest.approx(5.0)

    def test_l1_vs_sqeuclidean_differ(self):
        """L1 and squared_euclidean metrics produce different distances."""
        x = make_series(50, 42)
        y = make_series(50, 43)
        d_l1 = dtwcpp.dtw_distance(x, y, metric='l1')
        d_sq = dtwcpp.dtw_distance(x, y, metric='squared_euclidean')
        # They should differ for non-trivial series
        assert d_l1 != pytest.approx(d_sq, abs=1e-6)

    def test_metric_symmetry(self):
        """Both metrics preserve symmetry."""
        x = make_series(30, 10)
        y = make_series(30, 11)
        for metric in ['l1', 'squared_euclidean']:
            d_xy = dtwcpp.dtw_distance(x, y, metric=metric)
            d_yx = dtwcpp.dtw_distance(y, x, metric=metric)
            assert d_xy == pytest.approx(d_yx), \
                f"Symmetry violated for metric={metric}"


class TestDTWBandedVsFull:
    """Banded DTW must always be >= full DTW (fewer paths)."""

    def test_banded_ge_full_short(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        d_full = dtwcpp.dtw_distance(x, y, band=-1)
        d_band = dtwcpp.dtw_distance(x, y, band=1)
        assert d_band >= d_full - 1e-12

    def test_banded_ge_full_long(self):
        x = make_series(100, 42)
        y = make_series(100, 43)
        d_full = dtwcpp.dtw_distance(x, y, band=-1)
        for band in [5, 10, 50]:
            d_band = dtwcpp.dtw_distance(x, y, band=band)
            assert d_band >= d_full - 1e-10, \
                f"band={band}: {d_band} < {d_full}"

    def test_large_band_equals_full(self):
        """Band >= n-1 should give same result as full DTW."""
        x = make_series(20, 1)
        y = make_series(20, 2)
        d_full = dtwcpp.dtw_distance(x, y, band=-1)
        d_wide = dtwcpp.dtw_distance(x, y, band=19)
        assert d_wide == pytest.approx(d_full, abs=1e-12)


# ===========================================================================
# Missing-data DTW
# ===========================================================================


class TestDTWMissing:
    """DTW with NaN (missing data) support."""

    def test_no_nan_matches_standard(self):
        """Without NaN, dtw_distance_missing should match dtw_distance."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [4.0, 3.0, 2.0, 1.0]
        d_std = dtwcpp.dtw_distance(x, y)
        d_miss = dtwcpp.dtw_distance_missing(x, y)
        assert d_miss == pytest.approx(d_std)

    def test_nan_reduces_or_equals_distance(self):
        """NaN positions contribute 0 cost, so distance <= full distance."""
        x = [1.0, float('nan'), 3.0]
        y = [1.0, 2.0, 3.0]
        d_miss = dtwcpp.dtw_distance_missing(x, y)
        d_full = dtwcpp.dtw_distance([1.0, 2.0, 3.0], y)
        assert d_miss <= d_full + 1e-10

    def test_all_nan_is_zero(self):
        """All-NaN series should have distance 0 (every pair is missing)."""
        x = [float('nan')] * 5
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = dtwcpp.dtw_distance_missing(x, y)
        assert d == pytest.approx(0.0)

    def test_symmetry_with_nan(self):
        """Missing DTW should be symmetric even with NaN."""
        x = [1.0, float('nan'), 3.0, 4.0]
        y = [2.0, 3.0, float('nan'), 5.0]
        d_xy = dtwcpp.dtw_distance_missing(x, y)
        d_yx = dtwcpp.dtw_distance_missing(y, x)
        assert d_xy == pytest.approx(d_yx)

    def test_missing_banded(self):
        """Missing DTW works with band constraint."""
        x = [1.0, float('nan'), 3.0, 4.0, 5.0]
        y = [2.0, 3.0, 4.0, float('nan'), 6.0]
        d = dtwcpp.dtw_distance_missing(x, y, band=2)
        assert math.isfinite(d) and d >= 0.0


# ===========================================================================
# compute_distance_matrix standalone function
# ===========================================================================


class TestComputeDistanceMatrix:
    """Tests for the standalone compute_distance_matrix function."""

    def test_shape(self):
        data = make_dataset(10, 50)
        dm = dtwcpp.compute_distance_matrix(data)
        assert dm.shape == (10, 10)

    def test_symmetric(self):
        data = make_dataset(8, 30)
        dm = dtwcpp.compute_distance_matrix(data)
        np.testing.assert_allclose(dm, dm.T, atol=1e-15)

    def test_zero_diagonal(self):
        data = make_dataset(6, 40)
        dm = dtwcpp.compute_distance_matrix(data)
        np.testing.assert_allclose(dm.diagonal(), 0.0, atol=1e-15)

    def test_non_negative(self):
        data = make_dataset(8, 30)
        dm = dtwcpp.compute_distance_matrix(data)
        assert np.all(dm >= -1e-15)

    def test_matches_pairwise_dtw_distance(self):
        """Each entry matches a standalone dtw_distance call."""
        data = make_dataset(5, 20)
        dm = dtwcpp.compute_distance_matrix(data)
        for i in range(5):
            for j in range(i + 1, 5):
                expected = dtwcpp.dtw_distance(data[i], data[j])
                assert dm[i, j] == pytest.approx(expected, abs=1e-12), \
                    f"dm[{i},{j}]={dm[i, j]} != dtw_distance={expected}"

    def test_l1_vs_sqeuclidean(self):
        """Different metrics produce different distance matrices."""
        data = make_dataset(5, 50)
        dm_l1 = dtwcpp.compute_distance_matrix(data, metric='l1')
        dm_sq = dtwcpp.compute_distance_matrix(data, metric='squared_euclidean')
        assert not np.allclose(dm_l1, dm_sq)
        # Both should be symmetric with zero diagonal
        np.testing.assert_allclose(dm_l1, dm_l1.T, atol=1e-15)
        np.testing.assert_allclose(dm_sq, dm_sq.T, atol=1e-15)

    def test_banded(self):
        """Banded distance matrix entries >= full distance matrix entries."""
        data = make_dataset(8, 100)
        dm_full = dtwcpp.compute_distance_matrix(data)
        dm_banded = dtwcpp.compute_distance_matrix(data, band=10)
        assert np.all(dm_banded >= dm_full - 1e-10)

    def test_matches_problem_dist_by_ind(self):
        """compute_distance_matrix matches Problem.dist_by_ind for same data."""
        data = make_dataset(6, 30)
        names = [f"s{i}" for i in range(6)]

        dm = dtwcpp.compute_distance_matrix(data)

        prob = dtwcpp.Problem("xval")
        prob.set_data(data, names)
        prob.fill_distance_matrix()

        for i in range(6):
            for j in range(i + 1, 6):
                assert dm[i, j] == pytest.approx(prob.dist_by_ind(i, j), abs=1e-12), \
                    f"Mismatch at ({i},{j}): matrix={dm[i, j]}, problem={prob.dist_by_ind(i, j)}"


# ===========================================================================
# Clustering
# ===========================================================================


class TestFastCLARA:
    """Tests for CLARA scalable clustering."""

    def test_basic_clustering(self):
        """CLARA returns valid labels and medoids."""
        data = make_dataset(30, 50)
        names = [f"s{i}" for i in range(30)]
        prob = dtwcpp.Problem("clara_test")
        prob.set_data(data, names)

        result = dtwcpp.fast_clara(prob, n_clusters=3, sample_size=15, n_samples=3)
        assert result.n_clusters == 3
        assert len(result.labels) == 30
        assert all(0 <= lab < 3 for lab in result.labels)
        assert result.total_cost > 0

    def test_clara_all_labels_used(self):
        """Each cluster label should be assigned to at least one point."""
        # Use well-separated data to ensure all clusters are found
        rng = np.random.default_rng(99)
        data = []
        for offset in [0.0, 50.0, 100.0]:
            for _ in range(10):
                data.append((offset + rng.standard_normal(30) * 0.5).tolist())
        names = [f"s{i}" for i in range(30)]
        prob = dtwcpp.Problem("clara_labels")
        prob.set_data(data, names)

        result = dtwcpp.fast_clara(prob, n_clusters=3, sample_size=20, n_samples=5)
        used_labels = set(result.labels)
        assert len(used_labels) == 3, f"Only {len(used_labels)} labels used: {used_labels}"


class TestDTWClusteringWrapper:
    """Tests for the sklearn-compatible DTWClustering wrapper."""

    def test_fit_predict_basic(self):
        data = np.random.default_rng(42).uniform(-1, 1, (20, 50))
        clust = dtwcpp.DTWClustering(n_clusters=3)
        labels = clust.fit_predict(data)
        assert len(labels) == 20
        assert set(labels).issubset({0, 1, 2})

    def test_reproducible_with_same_data(self):
        """Same input data should produce same clustering."""
        data = np.random.default_rng(42).uniform(-1, 1, (20, 50))
        labels1 = dtwcpp.DTWClustering(n_clusters=3).fit_predict(data)
        labels2 = dtwcpp.DTWClustering(n_clusters=3).fit_predict(data)
        np.testing.assert_array_equal(labels1, labels2)

    def test_predict_consistency(self):
        """predict() on training data gives same labels as fit_predict()."""
        data = np.random.default_rng(42).uniform(-1, 1, (15, 40))
        clust = dtwcpp.DTWClustering(n_clusters=2)
        labels_fit = clust.fit_predict(data)
        labels_pred = clust.predict(data)
        np.testing.assert_array_equal(labels_fit, labels_pred)

    def test_variant_wdtw(self):
        """DTWClustering with WDTW variant runs without error."""
        data = np.random.default_rng(77).uniform(-1, 1, (15, 30))
        clust = dtwcpp.DTWClustering(n_clusters=2, variant="wdtw")
        labels = clust.fit_predict(data)
        assert len(labels) == 15

    def test_variant_ddtw(self):
        """DTWClustering with DDTW variant runs without error."""
        data = np.random.default_rng(77).uniform(-1, 1, (15, 30))
        clust = dtwcpp.DTWClustering(n_clusters=2, variant="ddtw")
        labels = clust.fit_predict(data)
        assert len(labels) == 15

    def test_inertia_non_negative(self):
        """Total within-cluster cost (inertia) is non-negative."""
        data = np.random.default_rng(42).uniform(-1, 1, (20, 30))
        clust = dtwcpp.DTWClustering(n_clusters=3)
        clust.fit(data)
        assert clust.inertia_ >= 0


# ===========================================================================
# DTW variants
# ===========================================================================


class TestVariantsIntegration:
    """Integration tests for DTW variants across calling paths."""

    def test_ddtw_via_function_matches_problem(self):
        """DDTW via standalone function matches Problem with DDTW variant."""
        x = make_series(30, 1)
        y = make_series(30, 2)

        d_func = dtwcpp.ddtw_distance(x, y)

        prob = dtwcpp.Problem("ddtw_xval")
        prob.set_data([x, y], ["s0", "s1"])
        prob.set_variant(dtwcpp.DTWVariant.DDTW)
        prob.fill_distance_matrix()
        d_prob = prob.dist_by_ind(0, 1)

        assert d_func == pytest.approx(d_prob, abs=1e-12)

    def test_wdtw_via_function_matches_problem(self):
        """WDTW via standalone function matches Problem with WDTW variant."""
        x = make_series(30, 3)
        y = make_series(30, 4)
        g = 0.1

        d_func = dtwcpp.wdtw_distance(x, y, g=g)

        prob = dtwcpp.Problem("wdtw_xval")
        prob.set_data([x, y], ["s0", "s1"])
        vp = dtwcpp.DTWVariantParams()
        vp.variant = dtwcpp.DTWVariant.WDTW
        vp.wdtw_g = g
        prob.variant_params = vp
        prob.refresh_distance_matrix()
        prob.fill_distance_matrix()
        d_prob = prob.dist_by_ind(0, 1)

        assert d_func == pytest.approx(d_prob, abs=1e-12)

    def test_adtw_via_function_matches_problem(self):
        """ADTW via standalone function matches Problem with ADTW variant."""
        x = make_series(30, 5)
        y = make_series(30, 6)
        penalty = 1.5

        d_func = dtwcpp.adtw_distance(x, y, penalty=penalty)

        prob = dtwcpp.Problem("adtw_xval")
        prob.set_data([x, y], ["s0", "s1"])
        vp = dtwcpp.DTWVariantParams()
        vp.variant = dtwcpp.DTWVariant.ADTW
        vp.adtw_penalty = penalty
        prob.variant_params = vp
        prob.refresh_distance_matrix()
        prob.fill_distance_matrix()
        d_prob = prob.dist_by_ind(0, 1)

        assert d_func == pytest.approx(d_prob, abs=1e-12)

    def test_soft_dtw_self_distance(self):
        """Soft-DTW self-distance is <= 0 (softmin effect)."""
        x = make_series(20, 7)
        d = dtwcpp.soft_dtw_distance(x, x, gamma=1.0)
        assert d <= 1e-10

    def test_soft_dtw_symmetry(self):
        """Soft-DTW is symmetric."""
        x = make_series(20, 8)
        y = make_series(20, 9)
        d_xy = dtwcpp.soft_dtw_distance(x, y)
        d_yx = dtwcpp.soft_dtw_distance(y, x)
        assert d_xy == pytest.approx(d_yx)


# ===========================================================================
# Checkpoint save/load roundtrip
# ===========================================================================


class TestCheckpointRoundtrip:
    """Distance matrix checkpoint save and reload."""

    def test_save_load_preserves_distances(self):
        """Saved distance matrix can be reloaded and matches original."""
        data = make_dataset(5, 50)
        names = [f"s{i}" for i in range(5)]

        prob = dtwcpp.Problem("ckpt_save")
        prob.set_data(data, names)
        prob.fill_distance_matrix()

        # Record original distances
        original = {}
        for i in range(5):
            for j in range(i + 1, 5):
                original[(i, j)] = prob.dist_by_ind(i, j)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt")
            dtwcpp.save_checkpoint(prob, path)

            # Create new problem and load checkpoint
            prob2 = dtwcpp.Problem("ckpt_load")
            prob2.set_data(data, names)
            loaded = dtwcpp.load_checkpoint(prob2, path)
            assert loaded, "Checkpoint load returned False"
            assert prob2.is_distance_matrix_filled()

            # Verify all distances match
            for (i, j), d_orig in original.items():
                d_loaded = prob2.dist_by_ind(i, j)
                assert d_loaded == pytest.approx(d_orig, abs=1e-12), \
                    f"Checkpoint mismatch at ({i},{j}): {d_loaded} vs {d_orig}"


# ===========================================================================
# End-to-end pipeline
# ===========================================================================


class TestEndToEndPipeline:
    """Full pipeline: data -> distance matrix -> clustering -> scores."""

    def test_full_pipeline(self):
        """Run the complete workflow and verify all outputs are consistent."""
        # Create well-separated clusters
        rng = np.random.default_rng(42)
        data = []
        for offset in [0.0, 50.0, 100.0]:
            for _ in range(5):
                data.append((offset + rng.standard_normal(30) * 0.5).tolist())
        names = [f"s{i}" for i in range(15)]

        # Build Problem
        prob = dtwcpp.Problem("pipeline")
        prob.set_data(data, names)

        # Cluster with FastPAM
        result = dtwcpp.fast_pam(prob, 3)
        assert result.n_clusters == 3
        assert result.n_points == 15
        assert result.total_cost > 0

        # Apply clustering result for scoring
        prob.set_number_of_clusters(3)
        prob.clusters_ind = result.labels
        prob.centroids_ind = result.medoid_indices

        # Evaluate
        sil = dtwcpp.silhouette(prob)
        assert len(sil) == 15
        assert all(-1.0 - 1e-10 <= s <= 1.0 + 1e-10 for s in sil)
        # Well-separated data should have high silhouette
        assert np.mean(sil) > 0.5

        dbi = dtwcpp.davies_bouldin_index(prob)
        assert dbi >= 0.0
        assert dbi < 1.0  # well-separated => low DBI

    def test_standalone_vs_problem_pipeline(self):
        """Standalone compute_distance_matrix produces same clustering input."""
        data = make_dataset(10, 30, seed_base=100)
        names = [f"s{i}" for i in range(10)]

        # Standalone distance matrix
        dm_standalone = dtwcpp.compute_distance_matrix(data)

        # Problem distance matrix
        prob = dtwcpp.Problem("cmp")
        prob.set_data(data, names)
        prob.fill_distance_matrix()

        # Compare all pairs
        for i in range(10):
            for j in range(i + 1, 10):
                assert dm_standalone[i, j] == pytest.approx(
                    prob.dist_by_ind(i, j), abs=1e-12
                )
