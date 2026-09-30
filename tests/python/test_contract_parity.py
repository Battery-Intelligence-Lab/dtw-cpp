"""
@file test_contract_parity.py
@brief Cross-language API-contract parity gate for the Python surface.

Asserts that every symbol named in the **Python column** of the FROZEN API
contract exists on the ``dtwcpp`` module with its exact canonical name (and, where
introspectable, its documented default values).

Contract pinned: docs/api-contract-2.0.md — STATUS: FROZEN 2026-07-07.
Sections consumed: §1 (Tier-1), §2.1/§2.2 (Problem), §2.4 (scores), §2.5
(algorithms), §2.6 (distance), §5 (error taxonomy), §6 (device).

Why the contract's Python column is HARD-CODED here (not parsed at test time):
the column lives inside prose markdown tables carrying provenance tags
(``[new bind]``), footnote markers (``†``/``‡``), and multi-name cells — a parser
would be brittle and could *silently pass on a mis-parse*, which is worse than a
maintained explicit list. The list below is auditable line-by-line against the
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
# §1 Tier-1 + top-level classes / functions
# ---------------------------------------------------------------------------
_TIER1 = [
    "device", "load", "cluster", "plot",
    "Dataset", "Result",
    "DTWClustering", "compute_distance_matrix",
]

# ---------------------------------------------------------------------------
# §6 device
# ---------------------------------------------------------------------------
_ENV = ["device_to_string", "Device"]

# ---------------------------------------------------------------------------
# §5 error taxonomy
# ---------------------------------------------------------------------------
_ERRORS = ["DtwcError", "InvalidInput", "SolverError", "DeviceError", "IOError"]

# ---------------------------------------------------------------------------
# Enums, structs, Tier-2 classes
# ---------------------------------------------------------------------------
_ENUMS = [
    "Method", "Solver", "ConstraintType", "MetricType", "DTWVariant",
    "MissingStrategy", "DistanceMatrixStrategy", "GpuPrecision", "Linkage",
]
_STRUCTS = [
    "DTWVariantParams", "MIPSettings", "CUDASettings", "Data",
    "DendrogramStep", "Dendrogram", "HierarchicalOptions",
    "CLARAOptions", "ClusteringResult", "Problem",
    "CheckpointOptions",
]

# ---------------------------------------------------------------------------
# §2.5 algorithm free functions + checkpoint + utils
# ---------------------------------------------------------------------------
_ALGOS = ["fast_pam", "fast_clara", "build_dendrogram", "cut_dendrogram"]
_CHECKPOINT = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointOptions",
]
_UTILS = ["derivative_transform", "z_normalize", "soft_dtw_gradient"]

# ---------------------------------------------------------------------------
# §2.4 scores
# ---------------------------------------------------------------------------
_SCORES_CANON = [
    "silhouette", "davies_bouldin", "dunn", "inertia", "calinski_harabasz",
    "adjusted_rand", "normalized_mutual_info",
]

# ---------------------------------------------------------------------------
# §2.6 distance namespace
# ---------------------------------------------------------------------------
_DISTANCE = ["standard", "ddtw", "wdtw", "adtw", "soft_dtw", "missing", "arow", "dtw"]

# ---------------------------------------------------------------------------
# §2.1/§2.2 Problem canonical setters / accessors / methods
# ---------------------------------------------------------------------------
_PROBLEM_CANON = [
    # config setters (§2.1)
    "set_n_clusters", "set_method", "set_band", "set_max_iter",
    "set_n_repetitions", "set_variant", "set_variant_params", "set_solver",
    "set_data",
    # config attributes (§2.1)
    "method", "max_iter", "n_repetitions", "band", "variant_params",
    "missing_strategy", "distance_strategy",
    "cuda_settings", "mip_settings", "verbose", "name", "output_folder",
    "clusters_ind", "centroids_ind",
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
]
# Names 2.0 development coined and renamed before any release. None is an alias:
# a removed name raises AttributeError and never resolves to something else (§4).
_RESULT = dtwcpp.Result([0, 1], device="cpu", elapsed_s=0.0, k=1, n_series=2)
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
# Module-level symbol existence
# ===========================================================================
@pytest.mark.parametrize(
    "name",
    _TIER1 + _ENV + _ERRORS + _ENUMS + _STRUCTS + _ALGOS + _CHECKPOINT
    + _UTILS + _SCORES_CANON,
)
def test_module_symbol_exists(name):
    """Every contract Python-column module symbol is present with its exact name."""
    assert hasattr(dtwcpp, name), f"dtwcpp.{name} missing (contract §1/§2/§5/§6)"


def test_distance_dtw_distance_is_not_public():
    """The pairwise helper is namespaced under dtwcpp.distance, not the root
    (pinned separately in test_dtw.py); the root alias stays removed."""
    assert not hasattr(dtwcpp, "dtw_distance")


# ===========================================================================
# §2.6 distance submodule
# ===========================================================================
@pytest.mark.parametrize("name", _DISTANCE)
def test_distance_symbol_exists(name):
    assert hasattr(dtwcpp.distance, name), f"dtwcpp.distance.{name} missing (§2.6)"


# ===========================================================================
# §2.1/§2.2 Problem surface
# ===========================================================================
@pytest.mark.parametrize("name", _PROBLEM_CANON)
def test_problem_canonical_member_exists(name):
    p = dtwcpp.Problem("parity")
    assert hasattr(p, name), f"Problem.{name} missing (contract §2.1/§2.2)"


@pytest.mark.parametrize("owner, name", _REMOVED)
def test_removed_name_is_gone(owner, name):
    assert not hasattr(owner, name), f"{owner!r}.{name} must not exist (§4)"


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
# §1.4 Result members
# ===========================================================================
@pytest.mark.parametrize(
    "name", ["labels", "medoids", "score", "save", "plot", "cost", "device"])
def test_result_member_exists(name):
    # labels/medoids/cost/device are set in __init__ (instance attrs), so check an
    # instance; score/save/plot are methods on the class.
    res = dtwcpp.Result([0, 1], device="cpu", elapsed_s=0.0, k=1, n_series=2)
    assert hasattr(res, name), f"Result.{name} missing (contract §1.4)"


# ===========================================================================
# MIPSettings — §2.1 fields
# ===========================================================================
@pytest.mark.parametrize(
    "name",
    ["mip_gap", "time_limit_sec", "warm_start", "numeric_focus", "mip_focus",
     "verbose_solver", "lr_max_nodes"],
)
def test_mip_settings_field_exists(name):
    assert hasattr(dtwcpp.MIPSettings(), name), f"MIPSettings.{name} missing (§2.1)"


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
    """§1.3: cluster(data, k, *, method='pam', band=-1, device=None, max_iter=100)."""
    sig = inspect.signature(dtwcpp.cluster)
    p = sig.parameters
    assert p["method"].default == "pam"
    assert p["band"].default == -1
    assert p["device"].default is None
    assert p["max_iter"].default == 100


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


def test_distance_standard_signature_defaults():
    """§2.6: standard(x, y, band=-1, metric='l1')."""
    p = inspect.signature(dtwcpp.distance.standard).parameters
    assert p["band"].default == -1
    assert p["metric"].default == "l1"


def test_distance_dispatcher_signature():
    """§2.6: dtw dispatcher is keyword-only variant/band/metric/g/penalty/gamma."""
    p = inspect.signature(dtwcpp.distance.dtw).parameters
    assert p["variant"].default == "standard"
    assert p["metric"].default == "l1"
    assert p["g"].default == 0.05
    assert p["penalty"].default == 1.0
    assert p["gamma"].default == 1.0
