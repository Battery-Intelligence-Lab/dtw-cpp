"""
@file test_contract_parity.py
@brief Cross-language API-contract parity gate for the Python surface.

Asserts that every symbol named in the **Python column** of the FROZEN API
contract exists on the ``dtwcpp`` module with its exact canonical name (and, where
introspectable, its documented default values).

Contract pinned: docs/api-contract-2.0.md — STATUS: FROZEN 2026-07-07.
Sections consumed: §1 (Tier-1), §2.1/§2.2 (Problem), §2.4 (scores), §2.5
(algorithms), §2.6 (distance), §5 (error taxonomy), §6 (device/Env).

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

import pytest

import dtwcpp


# ---------------------------------------------------------------------------
# §1 Tier-1 + top-level classes / functions
# ---------------------------------------------------------------------------
_TIER1 = [
    "device", "get_device", "load", "cluster", "plot",
    "Dataset", "Result", "ClusterResult",           # ClusterResult = deprecated alias (§4)
    "DTWClustering", "compute_distance_matrix",
]

# ---------------------------------------------------------------------------
# §6 device / Env registry
# ---------------------------------------------------------------------------
_ENV = ["Env", "env", "device_to_string", "Device"]

# ---------------------------------------------------------------------------
# §5 error taxonomy
# ---------------------------------------------------------------------------
_ERRORS = ["DtwcError", "InvalidInput", "SolverError", "DeviceError", "IOError"]

# ---------------------------------------------------------------------------
# Enums, structs, Tier-2 classes
# ---------------------------------------------------------------------------
_ENUMS = [
    "Method", "Solver", "ConstraintType", "MetricType", "DTWVariant",
    "MissingStrategy", "DistanceMatrixStrategy", "StoragePolicy",
    "LowerBoundStrategy", "Linkage",
]
_STRUCTS = [
    "DTWVariantParams", "MIPSettings", "CUDASettings", "DenseDistanceMatrix",
    "Data", "DendrogramStep", "Dendrogram", "HierarchicalOptions",
    "CLARANSOptions", "CLARAOptions", "ClusteringResult", "Problem",
    "CheckpointOptions",
]

# ---------------------------------------------------------------------------
# §2.5 algorithm free functions + checkpoint + utils
# ---------------------------------------------------------------------------
_ALGOS = ["fast_pam", "fast_clara", "clarans", "build_dendrogram", "cut_dendrogram"]
_CHECKPOINT = ["save_checkpoint", "load_checkpoint", "CheckpointOptions"]
_UTILS = ["derivative_transform", "z_normalize", "soft_dtw_gradient"]

# ---------------------------------------------------------------------------
# §2.4 scores — canonical names + retained deprecated aliases (§4)
# ---------------------------------------------------------------------------
_SCORES_CANON = [
    "silhouette", "davies_bouldin", "dunn", "inertia", "calinski_harabasz",
    "adjusted_rand", "normalized_mutual_info",
]
_SCORES_DEPRECATED = [
    "davies_bouldin_index", "dunn_index", "calinski_harabasz_index",
    "adjusted_rand_index", "normalized_mutual_information",
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
    "set_data", "set_view_data",
    # config attributes (§2.1)
    "method", "max_iter", "n_repetitions", "band", "variant_params",
    "missing_strategy", "distance_strategy", "lb_strategy", "storage_policy",
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
# Deprecated Problem names that must still RESOLVE one cycle (§4 "nothing silently
# disappears") — canonical is the documented one, the old spelling still works.
_PROBLEM_DEPRECATED = [
    "set_number_of_clusters", "distance_matrix_numpy",
    "set_distance_matrix_from_numpy", "n_repetition", "cluster_size",
]


# ===========================================================================
# Module-level symbol existence
# ===========================================================================
@pytest.mark.parametrize(
    "name",
    _TIER1 + _ENV + _ERRORS + _ENUMS + _STRUCTS + _ALGOS + _CHECKPOINT
    + _UTILS + _SCORES_CANON + _SCORES_DEPRECATED,
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


@pytest.mark.parametrize("name", _PROBLEM_DEPRECATED)
def test_problem_deprecated_alias_still_resolves(name):
    """§4: deprecated names survive one cycle; nothing silently disappears."""
    p = dtwcpp.Problem("parity")
    assert hasattr(p, name), f"Problem.{name} should still resolve one cycle (§4)"


# ===========================================================================
# §5 error taxonomy — class hierarchy (dual-catch contract)
# ===========================================================================
def test_error_hierarchy():
    assert issubclass(dtwcpp.DtwcError, Exception)
    assert issubclass(dtwcpp.InvalidInput, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.InvalidInput, ValueError)
    assert issubclass(dtwcpp.SolverError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.SolverError, RuntimeError)
    assert issubclass(dtwcpp.DeviceError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.DeviceError, RuntimeError)
    assert issubclass(dtwcpp.IOError, dtwcpp.DtwcError)
    assert issubclass(dtwcpp.IOError, OSError)


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


def test_result_is_clusterresult_alias():
    assert dtwcpp.Result is dtwcpp.ClusterResult


# ===========================================================================
# MIPSettings — §2.1 benders / max_benders_iter [new bind]
# ===========================================================================
@pytest.mark.parametrize(
    "name",
    ["mip_gap", "time_limit_sec", "warm_start", "numeric_focus", "mip_focus",
     "verbose_solver", "benders", "max_benders_iter"],
)
def test_mip_settings_field_exists(name):
    assert hasattr(dtwcpp.MIPSettings(), name), f"MIPSettings.{name} missing (§2.1)"


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
    """§1.2: load(source, *, skip_cols=0, delimiter=None, name=None)."""
    p = inspect.signature(dtwcpp.load).parameters
    assert p["skip_cols"].default == 0
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
