"""
@file test_device.py
@brief Tests for friendly device names ('gpu'/'hpc') and the global device() setter.
@author Volkan Kumtepeli
"""
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
        """device='gpu' selects an available CUDA or Metal backend."""
        if not ((dtwcpp.CUDA_AVAILABLE and dtwcpp.cuda_available()) or
                (dtwcpp.METAL_AVAILABLE and dtwcpp.metal_available())):
            pytest.skip("no GPU available")
        dm = dtwcpp.compute_distance_matrix(_series(), device="gpu")
        assert dm.shape == (5, 5)

    def test_gpu_fails_loudly_when_no_gpu(self):
        """With no GPU present, 'gpu' raises instead of changing the backend."""
        if ((dtwcpp.CUDA_AVAILABLE and dtwcpp.cuda_available()) or
                (dtwcpp.METAL_AVAILABLE and dtwcpp.metal_available())):
            pytest.skip("GPU present; unavailable-device path not exercised")
        with pytest.raises(dtwcpp.DeviceError, match="not silently fall back"):
            dtwcpp.compute_distance_matrix(_series(), device="gpu")

    def test_gpu_is_case_insensitive(self):
        if not ((dtwcpp.CUDA_AVAILABLE and dtwcpp.cuda_available()) or
                (dtwcpp.METAL_AVAILABLE and dtwcpp.metal_available())):
            with pytest.raises(dtwcpp.DeviceError):
                dtwcpp.compute_distance_matrix(_series(), device="GPU")
        else:
            assert dtwcpp.compute_distance_matrix(_series(), device="GPU").shape == (5, 5)


class TestGlobalDevice:
    def test_default_device_is_cpu(self):
        assert dtwcpp.get_device() == "cpu"

    def test_set_and_get_device(self):
        if not ((dtwcpp.CUDA_AVAILABLE and dtwcpp.cuda_available()) or
                (dtwcpp.METAL_AVAILABLE and dtwcpp.metal_available())):
            with pytest.raises(dtwcpp.DeviceError):
                dtwcpp.device("gpu")
            assert dtwcpp.get_device() == "cpu"
        else:
            dtwcpp.device("gpu")
            assert dtwcpp.get_device() == "gpu"

    def test_device_call_with_no_arg_returns_current(self):
        dtwcpp.device("cpu")
        assert dtwcpp.device() == "cpu"

    def test_invalid_device_rejected(self):
        with pytest.raises(ValueError, match="Unknown device"):
            dtwcpp.device("tpu")

    @pytest.mark.parametrize("name", ["cuda:", "cuda:abc", "cuda:-1"])
    def test_invalid_cuda_ordinal_rejected_cleanly(self, name):
        with pytest.raises(ValueError, match="CUDA device ordinal"):
            dtwcpp.device(name)

    def test_non_string_device_rejected_cleanly(self):
        with pytest.raises(ValueError, match="device must be a string"):
            dtwcpp.device(1)

    def test_global_default_used_when_device_unset(self):
        """compute_distance_matrix with no device= uses the global default."""
        dtwcpp.device("cpu")
        assert dtwcpp.compute_distance_matrix(_series()).shape == (5, 5)

    def test_explicit_device_overrides_global(self):
        """An explicit device= argument wins over the global default."""
        dtwcpp.device("cpu")
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
