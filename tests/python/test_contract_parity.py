"""
@file test_contract_parity.py
@brief Cross-language API-contract parity gate for the Python surface.

Asserts that every symbol named in the **Python column** of the FROZEN API
contract exists on the ``dtwcpp`` module with its exact canonical name (and, where
introspectable, its documented default values).

Contract pinned: docs/api-contract-2.0.md — STATUS: FROZEN 2026-07-07.
Sections consumed: §1 (Tier-1), §2.1/§2.2 (Problem), §2.4 (scores), §2.5
(algorithms), §2.6 (distance), §2.7 (checkpoint), §5 (error taxonomy), §6 (device).

Why the contract's Python column is HARD-CODED here (not parsed at test time):
the column lives inside prose markdown tables carrying provenance tags
(``[new bind]``), footnote markers (``†``/``‡``), and multi-name cells — a parser
would be brittle and could *silently pass on a mis-parse*, which is worse than a
maintained explicit list. ``_SURFACE`` below is auditable line-by-line against the
FROZEN doc, and drift in either direction (a dropped binding or a renamed symbol)
fails loudly. Re-pin the version string above when the contract changes.

Drives the LIVE public surface: `import dtwcpp` and its `dtwcpp.distance`
submodule — i.e. exactly what a user imports.
"""
import inspect
import warnings

import pytest

import dtwcpp


# ---------------------------------------------------------------------------
# The Python surface the contract promises: one table, one row per name.
# The key says which object the name is looked up on (see _OWNER).
# ---------------------------------------------------------------------------
_SURFACE = {
    "module": [
        # §1 Tier-1 + top-level classes / functions
        "device", "load", "cluster", "plot",
        "Dataset", "Result",
        "DTWClustering", "compute_distance_matrix",
        "DEFAULT_RANDOM_SEED",
        # §6 device
        "device_to_string", "Device", "gpu_available", "gpu_info",
        # §5 error taxonomy
        "DtwcError", "InvalidInput", "SolverError", "DeviceError", "IOError",
        # enums, structs, Tier-2 classes: the types behind the contract's fields and
        # arguments (several have no table row of their own)
        "Method", "Solver", "MetricType", "DTWVariant",
        "MissingStrategy", "GpuPrecision", "Linkage",
        "DTWVariantParams", "MIPSettings", "Data",
        "DendrogramStep", "Dendrogram", "HierarchicalOptions",
        "CLARAOptions", "ClusteringResult", "Problem",
        # §2.5 algorithm free functions
        "fast_pam", "fast_clara", "build_dendrogram", "cut_dendrogram",
        # §2.7 checkpoint
        "CheckpointOptions", "save_checkpoint", "load_checkpoint",
        # utils: public exports the contract does not tabulate
        "derivative_transform", "z_normalize", "soft_dtw_gradient",
        # §2.4 scores
        "silhouette", "davies_bouldin", "dunn", "inertia", "calinski_harabasz",
        "adjusted_rand", "normalized_mutual_info",
    ],
    # §2.6 distance namespace
    "distance": ["dtw"],
    # §2.1/§2.2 Problem canonical setters / accessors / methods
    "Problem": [
        # config setters (§2.1)
        "set_n_clusters", "set_method", "set_band", "set_max_iter",
        "set_n_repetitions", "set_variant", "set_variant_params", "set_distance",
        "set_solver",
        "set_data", "set_result", "set_device", "set_gpu_precision", "set_random_seed",
        # config attributes (§2.1)
        "method", "max_iter", "n_repetitions", "band", "variant_params",
        "missing_strategy", "random_seed",
        "mip_settings", "verbose", "name", "output_folder",
        "clusters_ind", "centroids_ind", "checkpoint",
        # read accessors (§2.2)
        "size", "n_clusters", "labels", "medoids", "series", "series_name",
        "centroid_of", "is_distance_matrix_filled", "max_distance", "dist_by_ind",
        # distance-matrix methods (§2.2)
        "fill_distance_matrix", "refresh_distance_matrix", "read_distance_matrix",
        "print_distance_matrix", "write_distance_matrix", "distance_matrix",
        "set_distance_matrix", "use_mmap_distance_matrix",
        # clustering (§2.2)
        "cluster", "find_total_cost", "assign_clusters", "calculate_medoids",
        # I/O (§2.2)
        "print_clusters", "write_clusters", "write_medoid_members", "write_silhouettes",
    ],
    # §1.4 Result members: labels/medoids/cost/device are set in __init__
    # (instance attributes), so the owner is an instance; score/save/plot are methods.
    "Result": ["labels", "medoids", "score", "distance_matrix", "save", "plot",
               "cost", "device"],
    # §2.1 MIP settings fields
    "MIPSettings": [
        "mip_gap", "time_limit_sec", "warm_start", "numeric_focus", "mip_focus",
        "verbose_solver", "lr_max_nodes",
    ],
}
_ROWS = [pytest.param(kind, name, id=f"{kind}.{name}")
         for kind, names in _SURFACE.items() for name in names]

_RESULT = dtwcpp.Result([0, 1], device="cpu", elapsed_s=0.0, k=1, n_series=2)
_OWNER = {
    "module": lambda: dtwcpp,
    "distance": lambda: dtwcpp.distance,
    "Problem": lambda: dtwcpp.Problem("parity"),
    "Result": lambda: _RESULT,
    "MIPSettings": dtwcpp.MIPSettings,
}

# Names 2.0 development coined and renamed before any release. None is an alias:
# a removed name raises AttributeError and never resolves to something else (§4).
_REMOVED = [
    pytest.param(dtwcpp, "get_device", id="dtwcpp.get_device"),
    pytest.param(dtwcpp, "ClusterResult", id="dtwcpp.ClusterResult"),
    pytest.param(dtwcpp.Problem, "set_number_of_clusters",
                 id="Problem.set_number_of_clusters"),
    pytest.param(dtwcpp.Problem, "n_repetition", id="Problem.n_repetition"),
    pytest.param(dtwcpp.Problem, "distance_matrix_numpy",
                 id="Problem.distance_matrix_numpy"),
    pytest.param(dtwcpp.Problem, "set_distance_matrix_from_numpy",
                 id="Problem.set_distance_matrix_from_numpy"),
    pytest.param(_RESULT, "medoid_indices", id="Result.medoid_indices"),
]


# ===========================================================================
# Symbol existence
# ===========================================================================
@pytest.mark.parametrize("kind, name", _ROWS)
def test_contract_name_exists(kind, name):
    """Every contract Python-column name is present with its exact spelling."""
    assert hasattr(_OWNER[kind](), name), f"{kind}.{name} missing (api-contract-2.0)"


@pytest.mark.parametrize("owner, name", _REMOVED)
def test_removed_name_is_gone(owner, name):
    assert not hasattr(owner, name), f"{owner!r}.{name} must not exist (§4)"


def test_the_diagnostics_return_the_cpp_report_fields():
    """The binding copies each C++ report into a dict by hand; what the fields say
    is tests/unit/test_test_api.cpp's."""
    assert (set(dtwcpp.test.parallelisation()), set(dtwcpp.test.gpu())) == (
        {"available", "max_threads", "threads_engaged", "pass", "reason"},
        {"available", "backend", "device_name", "validated", "pass", "reason"})


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
# §5 error taxonomy — class hierarchy (dual-catch contract)
# ===========================================================================
def test_error_hierarchy():
    assert issubclass(dtwcpp.DtwcError, Exception)
    assert issubclass(dtwcpp.InvalidInput, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.InvalidInput, ValueError)
    # The one sub-leaf: dtwc::UndefinedScore derives from dtwc::InvalidInput, so
    # `except InvalidInput` still catches it (§6).
    assert issubclass(dtwcpp.UndefinedScore, dtwcpp.InvalidInput)
    assert issubclass(dtwcpp.SolverError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.SolverError, RuntimeError)
    assert issubclass(dtwcpp.DeviceError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.DeviceError, RuntimeError)
    assert issubclass(dtwcpp.IOError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.IOError, OSError)


# GT-4: a live C++ site raises the §5 leaf. Both used to be bare
# std::runtime_error, so Python saw RuntimeError: `except ValueError` missed
# the bad dendrogram and `except OSError` missed the missing file.
def test_cpp_bad_input_raises_invalid_input():
    prob = dtwcpp.Problem("gt4")
    prob.set_data([[0.0, 1.0], [1.0, 2.0], [5.0, 6.0]], ["a", "b", "c"])
    with pytest.raises(dtwcpp.InvalidInput, match="does not match Problem size"):
        dtwcpp.cut_dendrogram(dtwcpp.Dendrogram(), prob, 1)


def test_cpp_file_failure_raises_io_error(tmp_path):
    prob = dtwcpp.Problem("gt4")
    with pytest.raises(dtwcpp.IOError, match="Cannot open file for reading"):
        prob.read_distance_matrix(tmp_path / "missing.csv")


# ===========================================================================
# MIPSettings — §2.1 lr_max_nodes
# ===========================================================================
def test_mip_settings_lr_max_nodes_roundtrip():
    """§2.1: lr_max_nodes reaches the C++ Problem and is reported back."""
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
# Introspectable defaults (§1.2/§1.3/§1.5/§2.6)
# ===========================================================================
def test_cluster_signature_defaults():
    """§1.3: cluster(data, k, **keys); a key not given takes dtwc_cl's default,
    the C++ Config's: method 'auto', band -1, max_iter 100."""
    assert list(inspect.signature(dtwcpp.cluster).parameters) == ["data", "k", "keys"]
    config = dtwcpp._dtwcpp_core.Config()
    assert (config.method, config.band, config.max_iter) == ("auto", -1, 100)


def test_load_signature_defaults():
    """§1.2: load(source, *, skip_cols=0, skip_rows=0, delimiter=None, name=None)."""
    p = inspect.signature(dtwcpp.load).parameters
    assert p["skip_cols"].default == 0
    assert p["skip_rows"].default == 0
    assert p["delimiter"].default is None
    assert p["name"].default is None


def test_dtwclustering_constructor_param_set():
    """§1.5: the shared estimator param set, incl. the newly-added metric + device."""
    p = inspect.signature(dtwcpp.DTWClustering.__init__).parameters
    expected = {"n_clusters", "variant", "band", "max_iter", "n_init", "wdtw_g",
                "adtw_penalty", "missing_strategy", "metric", "device"}
    assert expected.issubset(set(p)), expected - set(p)
    assert p["metric"].default == "l1"     # Python GAINS metric (§1.5)
    assert p["device"].default is None

