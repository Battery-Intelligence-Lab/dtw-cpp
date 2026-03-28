"""Tests for clustering (FastPAM) via DTWC++ Python bindings."""

import numpy as np
import pytest

import dtwcpp


def _make_problem(data, names=None):
    """Helper: create a Problem, set data, and fill the distance matrix."""
    if names is None:
        names = [f"s{i}" for i in range(len(data))]
    p = dtwcpp.Problem("test")
    p.set_data([row.tolist() if hasattr(row, "tolist") else list(row) for row in data], names)
    p.fill_distance_matrix()
    return p


class TestFastPAM:
    """Tests for dtwcpp.fast_pam."""

    def test_returns_clustering_result(self, synthetic_data):
        """fast_pam returns a ClusteringResult."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 3)
        assert isinstance(result, dtwcpp.ClusteringResult)

    def test_labels_range(self, synthetic_data):
        """Labels are in [0, k) and length == n."""
        n = len(synthetic_data)
        k = 3
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, k)
        labels = result.labels
        assert len(labels) == n
        assert all(0 <= lab < k for lab in labels)

    def test_medoid_indices_range(self, synthetic_data):
        """Medoid indices are in [0, n) and length == k."""
        n = len(synthetic_data)
        k = 3
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, k)
        medoids = result.medoid_indices
        assert len(medoids) == k
        assert all(0 <= m < n for m in medoids)

    def test_medoids_are_unique(self, synthetic_data):
        """Each medoid index is distinct."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 3)
        assert len(set(result.medoid_indices)) == len(result.medoid_indices)

    def test_total_cost_non_negative(self, synthetic_data):
        """Total cost is non-negative."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 3)
        assert result.total_cost >= 0.0

    def test_n_clusters_field(self, synthetic_data):
        """n_clusters matches requested k."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 4)
        assert result.n_clusters == 4

    def test_n_points_field(self, synthetic_data):
        """n_points matches dataset size."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 2)
        assert result.n_points == len(synthetic_data)

    def test_converged(self, synthetic_data):
        """Algorithm converges on small datasets."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 2, max_iter=100)
        assert result.converged is True

    def test_k_equals_1_all_same_label(self, synthetic_data):
        """With k=1, all labels should be 0."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 1)
        assert all(lab == 0 for lab in result.labels)

    def test_well_separated_clusters_recovered(self, well_separated_data):
        """Three well-separated clusters should be perfectly recovered."""
        n_per_cluster = 5
        k = 3
        prob = _make_problem(well_separated_data)
        result = dtwcpp.fast_pam(prob, k)
        labels = np.array(result.labels)

        # Each ground-truth cluster should map to exactly one predicted label
        gt_groups = [set(range(i * n_per_cluster, (i + 1) * n_per_cluster)) for i in range(k)]
        pred_groups = [set(np.where(labels == lab)[0]) for lab in range(k)]

        # Every gt group should be a subset of some pred group
        for gt in gt_groups:
            assert any(gt <= pg for pg in pred_groups), (
                f"Ground truth cluster {gt} not contained in any predicted cluster"
            )

    def test_each_medoid_has_its_own_label(self, synthetic_data):
        """Each medoid should carry a distinct label."""
        prob = _make_problem(synthetic_data)
        result = dtwcpp.fast_pam(prob, 3)
        medoid_labels = [result.labels[m] for m in result.medoid_indices]
        assert len(set(medoid_labels)) == 3


class TestSilhouetteAndDBI:
    """Tests for evaluation metrics."""

    def test_silhouette_length(self, well_separated_data):
        """Silhouette scores length matches dataset size."""
        prob = _make_problem(well_separated_data)
        prob.set_number_of_clusters(3)
        prob.cluster()
        sil = dtwcpp.silhouette(prob)
        assert len(sil) == len(well_separated_data)

    def test_silhouette_range(self, well_separated_data):
        """Silhouette scores are in [-1, 1]."""
        prob = _make_problem(well_separated_data)
        prob.set_number_of_clusters(3)
        prob.cluster()
        sil = dtwcpp.silhouette(prob)
        assert all(-1.0 - 1e-10 <= s <= 1.0 + 1e-10 for s in sil)

    def test_well_separated_high_silhouette(self, well_separated_data):
        """Well-separated data should have high average silhouette."""
        prob = _make_problem(well_separated_data)
        prob.set_number_of_clusters(3)
        prob.cluster()
        sil = dtwcpp.silhouette(prob)
        avg = np.mean(sil)
        assert avg > 0.5, f"Expected high silhouette, got {avg}"

    def test_dbi_non_negative(self, well_separated_data):
        """Davies-Bouldin Index is non-negative."""
        prob = _make_problem(well_separated_data)
        prob.set_number_of_clusters(3)
        prob.cluster()
        dbi = dtwcpp.davies_bouldin_index(prob)
        assert dbi >= 0.0

    def test_well_separated_low_dbi(self, well_separated_data):
        """Well-separated data should have low DBI (< 1)."""
        prob = _make_problem(well_separated_data)
        prob.set_number_of_clusters(3)
        prob.cluster()
        dbi = dtwcpp.davies_bouldin_index(prob)
        assert dbi < 1.0, f"Expected low DBI, got {dbi}"
