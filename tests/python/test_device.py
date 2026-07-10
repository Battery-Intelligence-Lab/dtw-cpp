"""
@file test_device.py
@brief Tests for friendly device names ('gpu'/'hpc') and the global device() setter.
@author Volkan Kumtepeli
"""
import numpy as np
import pytest

import dtwcpp


_GPU_ALIAS_CANDIDATES = tuple(
    f"{prefix}{suffix}"
    for prefix in ("gpu", "cuda", "GPU", "CUDA")
    for suffix in (
        "",
        ":0",
        ":07",
        ":2147483647",
        ":",
        ":-1",
        ":+1",
        ": 1",
        ":1 ",
        ":1.0",
        ":1_0",
        ":2147483648",
        ":999999999999999999999999999999999999",
        ":\N{ARABIC-INDIC DIGIT ONE}",
    )
) + (
    " gpu:3 ",                 # C++ trim characters around a valid alias
    "\tCUDA:3\r\n",
    "\vgpu:3",                 # Python strip() characters C++ does not trim
    "gpu:3\v",
    "\fgpu",
    "gpu\N{NO-BREAK SPACE}",
)


@pytest.fixture(autouse=True)
def _reset_device():
    """Each test starts and ends with the default device restored to 'cpu'."""
    dtwcpp.device("cpu")
    yield
    dtwcpp.device("cpu")


def _series(n=5, length=20, seed=42):
    rng = np.random.default_rng(seed)
    return [list(rng.standard_normal(length)) for _ in range(n)]


def _cpp_gpu_alias_result(name):
    """Ask the live C++ Env whether *name* belongs to its GPU alias grammar.

    A CPU-only build raises the GPU-not-built DeviceError even for a syntactically
    valid alias, so that error is an accepted parse result.  Unknown-name errors
    are rejected parse results.  This keeps C++ as the acceptance oracle instead
    of copying its valid-alias list into the Python test.
    """
    registry = dtwcpp.env()
    try:
        registry.set_device(name)
    except dtwcpp.DeviceError as exc:
        message = str(exc)
        if message.startswith("[dtwc] unknown device '"):
            return False, message, None
        if message.startswith("[dtwc] device='gpu' requested"):
            return True, message, None
        raise AssertionError(f"Unexpected C++ device error for {name!r}: {message}") from exc
    else:
        assert registry.device() == dtwcpp.Device.GPU
        return True, None, registry.device_index()
    finally:
        registry.set_device("cpu")


class TestGpuAlias:
    @pytest.mark.parametrize("name", _GPU_ALIAS_CANDIDATES)
    def test_python_gpu_alias_grammar_matches_live_cpp(self, name):
        """Python accepts/rejects the same GPU spellings as Env::set_device."""
        cpp_accepts, cpp_error, cpp_ordinal = _cpp_gpu_alias_result(name)

        if not cpp_accepts:
            with pytest.raises(dtwcpp.DeviceError) as py_error:
                dtwcpp._parse_device(name)
            assert str(py_error.value) == cpp_error
            return

        backend, ordinal = dtwcpp._parse_device(name)
        normalized = name.strip(" \t\r\n").lower()
        assert backend == normalized.partition(":")[0]
        assert ordinal == (int(normalized.partition(":")[2]) if ":" in normalized else 0)
        if cpp_ordinal is not None:
            assert ordinal == cpp_ordinal

    @pytest.mark.parametrize(
        ("cuda_available", "metal_available", "expected_backend"),
        [(True, True, "cuda"), (False, True, "metal")],
    )
    def test_gpu_ordinal_survives_backend_resolution(
        self, monkeypatch, cuda_available, metal_available, expected_backend
    ):
        """The friendly alias keeps N after resolving to either GPU backend."""
        monkeypatch.setattr(dtwcpp, "CUDA_AVAILABLE", cuda_available)
        monkeypatch.setattr(dtwcpp, "cuda_available", lambda: cuda_available)
        monkeypatch.setattr(dtwcpp, "METAL_AVAILABLE", metal_available)
        monkeypatch.setattr(dtwcpp, "metal_available", lambda: metal_available)

        assert dtwcpp._resolve_device("gpu:7") == (expected_backend, 7)

    def test_explicit_cuda_never_falls_back_to_metal(self, monkeypatch):
        monkeypatch.setattr(dtwcpp, "CUDA_AVAILABLE", False)
        monkeypatch.setattr(dtwcpp, "cuda_available", lambda: False)
        monkeypatch.setattr(dtwcpp, "METAL_AVAILABLE", True)
        monkeypatch.setattr(dtwcpp, "metal_available", lambda: True)

        with pytest.raises(dtwcpp.DeviceError, match="CUDA was not compiled in"):
            dtwcpp._resolve_device("cuda:7")

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
        with pytest.raises(dtwcpp.DeviceError, match="unknown device"):
            dtwcpp.device("tpu")
        assert dtwcpp.get_device() == "cpu"

    @pytest.mark.parametrize("name", ["gpu:", "gpu:abc", "gpu:-1", "cuda:", "cuda:abc", "cuda:-1"])
    def test_invalid_gpu_ordinal_rejected_cleanly(self, name):
        with pytest.raises(dtwcpp.DeviceError, match="unknown device"):
            dtwcpp.device(name)
        assert dtwcpp.get_device() == "cpu"

    def test_non_string_device_rejected_cleanly(self):
        with pytest.raises(dtwcpp.InvalidInput, match="device must be a string"):
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
