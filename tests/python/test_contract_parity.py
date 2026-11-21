"""
@file test_contract_parity.py
@brief API contract tests for the Python surface: diagnostics, error hierarchy,
       device/hybrid refusals, and public signature defaults.
@author Volkan Kumtepeli
"""
import inspect
import re
import warnings

import pytest

import dtwcpp


def test_the_diagnostics_return_the_cpp_report_fields():
    """The binding copies each C++ report into a dict by hand; what the fields say
    is tests/unit/test_test_api.cpp's."""
    assert (set(dtwcpp.test.parallelisation()), set(dtwcpp.test.gpu())) == (
        {"available", "max_threads", "threads_engaged", "pass", "reason"},
        {"available", "backend", "device_name", "validated", "pass", "reason", "hybrid", "hybrid_reason"})


def test_problem_cluster_size_is_the_v1_method():
    """v1.0.0 bound ``cluster_size`` as a method (python/py_main.cpp), so
    ``prob.cluster_size()`` must keep working, silently, and equal ``n_clusters()``."""
    prob = dtwcpp.Problem("v1_cluster_size")
    prob.set_n_clusters(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert prob.cluster_size() == 3
    assert prob.n_clusters() == 3


# ===========================================================================
# Error taxonomy — class hierarchy (dual-catch contract)
# ===========================================================================
def test_error_hierarchy():
    assert issubclass(dtwcpp.DtwcError, Exception)
    assert issubclass(dtwcpp.InvalidInput, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.InvalidInput, ValueError)
    # The one sub-leaf: dtwc::UndefinedScore derives from dtwc::InvalidInput, so
    # `except InvalidInput` still catches it.
    assert issubclass(dtwcpp.UndefinedScore, dtwcpp.InvalidInput)
    assert issubclass(dtwcpp.SolverError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.SolverError, RuntimeError)
    assert issubclass(dtwcpp.DeviceError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.DeviceError, RuntimeError)
    assert issubclass(dtwcpp.IOError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.IOError, OSError)


# A live C++ site raises its typed leaf, so `except ValueError` catches the bad
# dendrogram and `except OSError` the missing file.
def test_cpp_bad_input_raises_invalid_input():
    prob = dtwcpp.Problem("bad_input")
    prob.set_data([[0.0, 1.0], [1.0, 2.0], [5.0, 6.0]], ["a", "b", "c"])
    with pytest.raises(dtwcpp.InvalidInput, match="does not match Problem size"):
        dtwcpp.cut_dendrogram(dtwcpp.Dendrogram(), prob, 1)


def test_cpp_file_failure_raises_io_error(tmp_path):
    prob = dtwcpp.Problem("missing_file")
    with pytest.raises(dtwcpp.IOError, match="Cannot open file for reading"):
        prob.read_distance_matrix(tmp_path / "missing.csv")


# ===========================================================================
# MIPSettings — lr_max_nodes
# ===========================================================================
def test_mip_settings_lr_max_nodes_roundtrip():
    """lr_max_nodes reaches the C++ Problem and is reported back."""
    s = dtwcpp.MIPSettings()
    assert s.lr_max_nodes == 2000000
    s.lr_max_nodes = 12345
    assert "lr_max_nodes=12345" in repr(s)

    prob = dtwcpp.Problem("lr_nodes")
    prob.mip_settings = s
    assert prob.mip_settings.lr_max_nodes == 12345


def test_mip_settings_lr_max_nodes_below_one_is_rejected():
    """validate_mip_settings runs before LR-core consumes the node cap."""
    prob = dtwcpp.Problem("lr_nodes_invalid")
    prob.set_data([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], ["a", "b", "c"])
    prob.set_n_clusters(2)
    s = dtwcpp.MIPSettings()
    s.lr_max_nodes = 0
    prob.mip_settings = s
    prob.method = dtwcpp.Method.LRCore
    with pytest.raises(dtwcpp.InvalidInput, match="lr_max_nodes"):
        prob.cluster()


# ===========================================================================
# Device "cpu+gpu" and gpu_share: the same words as C++, dtwc_cl and MATLAB
# ===========================================================================
_SERIES = [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1], [5.0, 6.0, 7.0], [5.1, 6.1, 7.1]]


def test_cpu_gpu_is_a_device_word_and_refused_as_cpp_refuses_it():
    """dtwcpp.device, Problem(device=) and cluster(device=) read "cpu+gpu" by the C++
    grammar. Where this build cannot run it, each raises C++'s DeviceError, the hybrid
    refusal, from "device 'cpu+gpu'" on as dtwc::device and Problem::set_device word it;
    where it can, each accepts it. C++ is the oracle: the build decides."""
    core = dtwcpp._dtwcpp_core
    assert dtwcpp._parse_device(" CPU+GPU:2 ") == ("cpu+gpu", 2)
    assert core.device_to_string(core.Device.CPU_GPU) == "cpu+gpu"

    def refusal(call):
        try:
            call()
        except dtwcpp.DeviceError as exc:
            text = str(exc)
            assert "device 'cpu+gpu' splits FastCLARA's assignment pass and the distance matrix's fill" in text
            return text[text.index("device 'cpu+gpu'"):]
        return None

    try:
        oracle = refusal(lambda: core.device("cpu+gpu"))
    finally:
        core.device("cpu")
    try:
        assert refusal(lambda: dtwcpp.device("cpu+gpu")) == oracle
    finally:
        dtwcpp.device("cpu")
    assert refusal(lambda: dtwcpp.Problem("hybrid", device="cpu+gpu")) == oracle
    assert refusal(lambda: dtwcpp.Problem("hybrid").set_device("cpu+gpu:1")) == oracle
    # test.gpu() says of the hybrid what the build says: a CUDA build with a device, else why not.
    report = dtwcpp.test.gpu()
    if report["backend"] != "cuda":  # a CUDA build selects it and refuses a missing device at the run, as for gpu
        assert refusal(lambda: dtwcpp.cluster(_SERIES, 2, device="cpu+gpu", method="pam")) == oracle
    if report["hybrid"]:
        assert oracle is None and report["hybrid_reason"] == ""
    else:
        assert report["hybrid_reason"] != ""


def test_gpu_share_is_one_key_with_cpps_range_and_device_refusals():
    """gpu_share (cluster key, Problem.set_gpu_share) reaches C++: a value outside [-1] and
    [0, 1] and a forced one on a device other than cpu+gpu are InvalidInput in C++'s words,
    the same as dtwc_cl's --gpu-share and MATLAB's GpuShare."""
    out_of_range = "the share must be -1 (measured) or in [0, 1]; got 1.500000."
    prob = dtwcpp.Problem("share")
    with pytest.raises(dtwcpp.InvalidInput, match=re.escape("Problem::set_gpu_share: " + out_of_range)):
        prob.set_gpu_share(1.5)
    with pytest.raises(dtwcpp.InvalidInput, match=re.escape(out_of_range)):
        dtwcpp.cluster(_SERIES, 2, gpu_share=1.5)

    refused = ("gpu_share splits the distance matrix's fill and FastCLARA's pass of device cpu+gpu, "
               "but the device is cpu, which would ignore it; leave gpu_share at -1.")
    with pytest.raises(dtwcpp.InvalidInput, match=re.escape("run: " + refused)):
        dtwcpp.cluster(_SERIES, 2, device="cpu", method="onebatch", gpu_share=0.5)
    prob.set_data(_SERIES)
    prob.set_n_clusters(2)
    prob.set_gpu_share(0.5)
    with pytest.raises(dtwcpp.InvalidInput, match=re.escape(
            "Problem::fill_distance_matrix: gpu_share splits the fill of device cpu+gpu, but the device is cpu")):
        prob.fill_distance_matrix()
    prob.set_gpu_share(-1)  # measured again: the Problem runs
    assert len(prob.cluster().labels) == 4


def test_a_forced_gpu_share_reaches_the_cpp_split():
    """On a build that runs cpu+gpu (CUDA, a device present) a forced share is C++'s split: the run
    report's fill row says the share was forced, as dtwc_cl -v does. Elsewhere the key's
    refusals above are what reaches C++."""
    if not dtwcpp.test.gpu()["hybrid"]:
        pytest.skip("cpu+gpu needs a CUDA build with a device: " + dtwcpp.test.gpu()["hybrid_reason"])
    result = dtwcpp.cluster(_SERIES, 2, method="pam", device="cpu+gpu", gpu_precision="fp64", gpu_share=0.5)
    assert "share 0.5, forced" in " ".join(result.report.split())  # the report wraps its notes


# ===========================================================================
# Introspectable defaults
# ===========================================================================
def test_cluster_signature_defaults():
    """cluster(data, k, **keys); a key not given takes dtwc_cl's default,
    the C++ Config's: method 'auto', band -1, max_iter 100."""
    assert list(inspect.signature(dtwcpp.cluster).parameters) == ["data", "k", "keys"]
    config = dtwcpp._dtwcpp_core.Config()
    assert (config.method, config.band, config.max_iter) == ("auto", -1, 100)


def test_load_signature_defaults():
    """load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None)."""
    p = inspect.signature(dtwcpp.load).parameters
    assert p["skip_cols"].default == 0
    assert p["skip_rows"].default == 0
    assert p["delimiter"].default is None
    assert p["name"].default is None


def test_dtwclustering_constructor_param_set():
    """The shared estimator parameter set and defaults."""
    p = inspect.signature(dtwcpp.DTWClustering.__init__).parameters
    expected = {"n_clusters", "variant", "band", "max_iter", "n_init", "wdtw_g",
                "adtw_penalty", "missing_strategy", "metric", "device"}
    assert expected.issubset(set(p)), expected - set(p)
    assert p["metric"].default == "l1"
    assert p["device"].default is None

