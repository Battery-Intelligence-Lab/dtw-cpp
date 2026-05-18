"""Tests for dtwcpp.diagnose."""

from __future__ import annotations

import numpy as np
import pytest

import dtwcpp
from dtwcpp import diagnose as dx


class TestClusterSizes:
    def test_basic(self):
        assert dx.cluster_sizes([0, 0, 1, 1, 1, 2]) == [2, 3, 1]

    def test_empty(self):
        assert dx.cluster_sizes([]) == []

    def test_single_cluster(self):
        assert dx.cluster_sizes([0, 0, 0]) == [3]


class TestMedoidIdleFractions:
    def test_active_medoid(self):
        # |v| >= 10 for every sample → idle fraction 0
        series = [[100.0, -200.0, 50.0]]
        out = dx.medoid_idle_fractions(series, [0], threshold=10.0)
        assert out == [0.0]

    def test_fully_idle_medoid(self):
        series = [[0.0, 1.0, -2.0, 5.0]]   # all |v| < 10
        out = dx.medoid_idle_fractions(series, [0], threshold=10.0)
        assert out == [1.0]

    def test_mixed(self):
        series = [[0.0, 100.0, 0.0, 100.0]]   # 50% idle
        out = dx.medoid_idle_fractions(series, [0], threshold=10.0)
        assert out == [0.5]


class TestLengthRatio:
    def test_perfectly_length_sorted_labels(self):
        # Two groups of equal-length series; ratio should be ~0 (perfect)
        lens = [100] * 5 + [200] * 5
        labels = [0] * 5 + [1] * 5
        ratio = dx.within_between_length_ratio(lens, labels)
        assert ratio == pytest.approx(0.0, abs=1e-9)

    def test_no_length_correlation(self):
        # Mixed lengths in each cluster — ratio dominates within
        lens = [100, 200, 100, 200]
        labels = [0, 0, 1, 1]
        ratio = dx.within_between_length_ratio(lens, labels)
        # All variance is within-cluster, none between → ratio is large
        assert ratio > 100   # essentially infinite vs zero between-var

    def test_empty_returns_one(self):
        assert dx.within_between_length_ratio([], []) == 1.0


class TestDiagnoseClusters:
    def test_clean_result_no_flags(self):
        # Two balanced clusters, no idle medoids, no length proxy, no singleton
        series = ([[100.0] * 10] * 20) + ([[-100.0] * 10] * 20)   # 40 series, all len 10
        labels = [0] * 20 + [1] * 20
        med = [0, 20]
        out = dx.diagnose_clusters(series, labels, med)
        assert out["flags"] == []
        assert not out["any_singleton"]
        assert not out["any_idle_medoid"]
        assert not out["degenerate"]

    def test_flags_singleton(self):
        series = [[100.0] * 10] * 10
        labels = [0] * 9 + [1]   # singleton at index 9
        out = dx.diagnose_clusters(series, labels, [0, 9])
        assert out["any_singleton"]
        assert any("singleton" in f for f in out["flags"])

    def test_flags_idle_medoid(self):
        # Medoid 0 is fully idle (all values < threshold)
        series = [[0.0] * 5, [100.0] * 5, [100.0] * 5, [100.0] * 5]
        labels = [0, 1, 1, 1]
        out = dx.diagnose_clusters(series, labels, [0, 1])
        assert out["any_idle_medoid"]
        assert any("idle-medoid" in f for f in out["flags"])

    def test_flags_degenerate_hierarchical_split(self):
        # 19-vs-1 split (matches the Kasper hierarchical pattern)
        series = [[100.0] * 5] * 20
        labels = [0] * 19 + [1]
        out = dx.diagnose_clusters(
            series, labels, medoid_indices=[0, 19],
            singleton_size=0,   # disable singleton flag
            degenerate_share=0.9,
        )
        assert out["degenerate"]
        assert any("degenerate" in f for f in out["flags"])

    def test_no_medoid_indices_ok(self):
        # Hierarchical / kMeans don't expose medoids; should not crash
        series = [[1.0] * 5, [2.0] * 5]
        labels = [0, 1]
        out = dx.diagnose_clusters(series, labels, medoid_indices=None,
                                   singleton_size=0)
        assert out["medoid_idle"] == []
        assert not out["any_idle_medoid"]

    def test_empty_input(self):
        out = dx.diagnose_clusters([], [], [])
        assert "empty" in out["flags"]


class TestModuleExport:
    def test_attached(self):
        assert hasattr(dtwcpp, "diagnose")
        assert dtwcpp.diagnose is dx
