"""
@file test_device.py
@brief Tests for friendly device names ('gpu'/'hpc') and the global device() setter.
@author Volkan Kumtepeli
"""
import warnings

import numpy as np
import pytest

import dtwcpp


@pytest.fixture(autouse=True)
def _reset_device():
    """Each test starts and ends with the default device restored to 'cpu'."""
    dtwcpp.device("cpu")
    yield
    dtwcpp.device("cpu")


def _series(n=5, length=20, seed=42):
    rng = np.random.default_rng(seed)
    return [list(rng.standard_normal(length)) for _ in range(n)]


class TestGpuAlias:
    def test_gpu_alias_resolves_like_cuda(self):
        """device='gpu' is an alias for 'cuda' (with CPU fallback when no GPU)."""
        dm = dtwcpp.compute_distance_matrix(_series(), device="gpu")
        assert dm.shape == (5, 5)

    def test_gpu_matches_cpu_when_no_gpu(self):
        """With no GPU present, 'gpu' falls back and matches the CPU matrix."""
        if dtwcpp.CUDA_AVAILABLE and dtwcpp.cuda_available():
            pytest.skip("GPU present; CPU-fallback path not exercised")
        np.testing.assert_array_almost_equal(
            dtwcpp.compute_distance_matrix(_series(), device="gpu"),
            dtwcpp.compute_distance_matrix(_series(), device="cpu"),
        )

    def test_gpu_is_case_insensitive(self):
        assert dtwcpp.compute_distance_matrix(_series(), device="GPU").shape == (5, 5)


class TestGlobalDevice:
    def test_default_device_is_cpu(self):
        assert dtwcpp.get_device() == "cpu"

    def test_set_and_get_device(self):
        dtwcpp.device("gpu")
        assert dtwcpp.get_device() == "gpu"

    def test_device_call_with_no_arg_returns_current(self):
        dtwcpp.device("gpu")
        assert dtwcpp.device() == "gpu"

    def test_invalid_device_rejected(self):
        with pytest.raises(ValueError, match="Unknown device"):
            dtwcpp.device("tpu")

    def test_global_default_used_when_device_unset(self):
        """compute_distance_matrix with no device= uses the global default."""
        dtwcpp.device("gpu")                              # -> cuda -> CPU fallback here
        dm = dtwcpp.compute_distance_matrix(_series())    # no explicit device
        assert dm.shape == (5, 5)

    def test_explicit_device_overrides_global(self):
        """An explicit device= argument wins over the global default."""
        dtwcpp.device("gpu")
        with warnings.catch_warnings():
            warnings.simplefilter("error")                # explicit cpu must not warn
            dm = dtwcpp.compute_distance_matrix(_series(), device="cpu")
        assert dm.shape == (5, 5)


class TestHpcDevice:
    def test_hpc_is_a_valid_global_device(self):
        dtwcpp.device("hpc")
        assert dtwcpp.get_device() == "hpc"

    def test_compute_distance_matrix_rejects_hpc(self):
        """'hpc' is an execution location, not a local compute backend."""
        with pytest.raises(ValueError, match="hpc"):
            dtwcpp.compute_distance_matrix(_series(), device="hpc")
