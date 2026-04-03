"""
@file test_cuda.py
@brief Tests for CUDA/GPU functionality in DTWC++ Python bindings.
@author Volkan Kumtepeli
"""
import warnings

import numpy as np
import pytest

import dtwcpp


requires_cuda = pytest.mark.skipif(
    not getattr(dtwcpp, "CUDA_AVAILABLE", False)
    or not dtwcpp.cuda_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Introspection (always runs, even without CUDA)
# ---------------------------------------------------------------------------
class TestCUDAIntrospection:
    def test_cuda_available_is_bool(self):
        assert isinstance(dtwcpp.CUDA_AVAILABLE, bool)

    def test_cuda_available_function_returns_bool(self):
        result = dtwcpp.cuda_available()
        assert isinstance(result, bool)

    def test_cuda_device_info_returns_string(self):
        info = dtwcpp.cuda_device_info()
        assert isinstance(info, str)
        assert len(info) > 0


# ---------------------------------------------------------------------------
# Device parsing and fallback
# ---------------------------------------------------------------------------
class TestDeviceParsing:
    def test_cpu_default(self):
        rng = np.random.default_rng(42)
        series = [list(rng.standard_normal(20)) for _ in range(5)]
        dm = dtwcpp.compute_distance_matrix(series)
        assert isinstance(dm, np.ndarray)
        assert dm.shape == (5, 5)

    def test_cpu_explicit(self):
        rng = np.random.default_rng(42)
        series = [list(rng.standard_normal(20)) for _ in range(5)]
        dm = dtwcpp.compute_distance_matrix(series, device="cpu")
        assert dm.shape == (5, 5)

    def test_invalid_device_raises(self):
        series = [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(ValueError, match="Unknown device"):
            dtwcpp.compute_distance_matrix(series, device="tpu")

    def test_cuda_fallback_warns_when_unavailable(self):
        """When CUDA not available, device='cuda' warns and falls back."""
        if getattr(dtwcpp, "CUDA_AVAILABLE", False) and dtwcpp.cuda_available():
            pytest.skip("CUDA is available; fallback won't trigger")
        series = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm = dtwcpp.compute_distance_matrix(series, device="cuda")
            assert len(w) >= 1
            assert any("Falling back to CPU" in str(x.message) for x in w)
        assert dm.shape == (2, 2)


# ---------------------------------------------------------------------------
# CUDA distance matrix (requires GPU)
# ---------------------------------------------------------------------------
@requires_cuda
class TestCUDADistanceMatrix:
    @pytest.fixture
    def series(self):
        rng = np.random.default_rng(123)
        return [list(rng.standard_normal(50)) for _ in range(10)]

    def test_shape_and_dtype(self, series):
        dm = dtwcpp.compute_distance_matrix(series, device="cuda")
        assert dm.shape == (10, 10)
        assert dm.dtype == np.float64

    def test_symmetric(self, series):
        dm = dtwcpp.compute_distance_matrix(series, device="cuda")
        np.testing.assert_array_almost_equal(dm, dm.T)

    def test_zero_diagonal(self, series):
        dm = dtwcpp.compute_distance_matrix(series, device="cuda")
        np.testing.assert_array_almost_equal(np.diag(dm), 0.0)

    def test_matches_cpu(self, series):
        dm_cpu = dtwcpp.compute_distance_matrix(series, device="cpu")
        dm_gpu = dtwcpp.compute_distance_matrix(series, device="cuda")
        np.testing.assert_array_almost_equal(dm_cpu, dm_gpu, decimal=10)

    def test_device_id_syntax(self, series):
        dm = dtwcpp.compute_distance_matrix(series, device="cuda:0")
        assert dm.shape == (10, 10)

    def test_nonnegative(self, series):
        dm = dtwcpp.compute_distance_matrix(series, device="cuda")
        assert np.all(dm >= 0)


# ---------------------------------------------------------------------------
# CUDA clustering (requires GPU)
# ---------------------------------------------------------------------------
@requires_cuda
class TestCUDAClustering:
    @pytest.fixture
    def well_separated_data(self):
        """3 well-separated clusters of 5 series each."""
        rng = np.random.default_rng(999)
        clusters = []
        for center in [0.0, 10.0, 20.0]:
            for _ in range(5):
                clusters.append(center + rng.standard_normal(30) * 0.1)
        return clusters

    def test_fit_with_cuda(self, well_separated_data):
        model = dtwcpp.DTWClustering(n_clusters=3, device="cuda")
        model.fit(well_separated_data)
        assert hasattr(model, "labels_")
        assert len(model.labels_) == 15

    def test_cuda_matches_cpu_inertia(self, well_separated_data):
        cpu = dtwcpp.DTWClustering(n_clusters=3, device="cpu", n_init=1)
        cpu.fit(well_separated_data)
        gpu = dtwcpp.DTWClustering(n_clusters=3, device="cuda", n_init=1)
        gpu.fit(well_separated_data)
        # Inertia should match since distance matrices are identical
        assert cpu.inertia_ == pytest.approx(gpu.inertia_, rel=1e-6)

    def test_cuda_variant_error(self, well_separated_data):
        """device='cuda' with non-standard variant should raise."""
        model = dtwcpp.DTWClustering(n_clusters=3, variant="ddtw", device="cuda")
        with pytest.raises(ValueError, match="only supports variant='standard'"):
            model.fit(well_separated_data)


# ---------------------------------------------------------------------------
# DTWClustering device parameter (always runs)
# ---------------------------------------------------------------------------
class TestDTWClusteringDevice:
    def test_default_device_is_cpu(self):
        model = dtwcpp.DTWClustering(n_clusters=2)
        assert model.device == "cpu"

    def test_device_parameter_stored(self):
        model = dtwcpp.DTWClustering(n_clusters=2, device="cuda")
        assert model.device == "cuda"
