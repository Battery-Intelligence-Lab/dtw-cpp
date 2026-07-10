"""
@file __init__.py
@brief DTWC++ — Fast Dynamic Time Warping and Clustering.
@author Volkan Kumtepeli
"""

from dtwcpp._dtwcpp_core import (
    # Enums
    Method,
    Solver,
    ConstraintType,
    MetricType,
    DTWVariant,
    MissingStrategy,
    DistanceMatrixStrategy,
    StoragePolicy,
    LowerBoundStrategy,
    Linkage,
    Device,
    # Structs
    DTWVariantParams,
    ClusteringResult,
    DenseDistanceMatrix,
    Data,
    MIPSettings,
    CUDASettings,
    DendrogramStep,
    Dendrogram,
    HierarchicalOptions,
    CLARANSOptions,
    OneBatchWeighting,
    OneBatchPAMOptions,
    OneBatchPAMStats,
    BarycenterMethod,
    BarycenterOptions,
    BarycenterClusteringOptions,
    BarycenterClusteringResult,
    # Classes
    Problem,
    Env,
    # Arrow C Data / stream ingest (Task 5.7 — zero-copy, no pyarrow)
    data_from_arrow_c_array,
    # Error taxonomy (api-contract-2.0.md §5)
    DtwcError,
    InvalidInput,
    SolverError,
    DeviceError,
    IOError,
    # Device registry (api-contract-2.0.md §6)
    env,
    device_to_string,
    # DTW functions (raw C++ bindings — require numpy arrays)
    dtw_distance as _dtw_distance_raw,
    ddtw_distance as _ddtw_distance_raw,
    wdtw_distance as _wdtw_distance_raw,
    adtw_distance as _adtw_distance_raw,
    soft_dtw_distance as _soft_dtw_distance_raw,
    soft_dtw_gradient,
    dtw_distance_missing as _dtw_distance_missing_raw,
    dtw_arow_distance as _dtw_arow_distance_raw,
    # Algorithms
    fast_pam,
    fast_pam_seeded,
    fast_clara,
    one_batch_pam,
    one_batch_pam_with_stats,
    dtw_barycenter,
    barycenter_kmeans,
    CLARAOptions,
    clarans,
    build_dendrogram,
    cut_dendrogram,
    # Scores (canonical 2.0 names)
    silhouette,
    davies_bouldin,
    dunn,
    inertia,
    calinski_harabasz,
    adjusted_rand,
    normalized_mutual_info,
    # Scores (deprecated aliases, kept one cycle — api-contract §4)
    davies_bouldin_index,
    dunn_index,
    calinski_harabasz_index,
    adjusted_rand_index,
    normalized_mutual_information,
    # Utils
    derivative_transform,
    z_normalize,
    # Distance matrix
    compute_distance_matrix as _compute_distance_matrix_cpu,
    # Checkpointing
    save_checkpoint,
    load_checkpoint,
    CheckpointOptions,
)

from dtwcpp._dtwcpp_core import (
    CUDA_AVAILABLE,
    cuda_available,
    cuda_device_info,
    compute_distance_matrix_cuda as _compute_distance_matrix_cuda,
    compute_lb_keogh_cuda,
    METAL_AVAILABLE,
    metal_available,
    metal_device_info,
    compute_distance_matrix_metal as _compute_distance_matrix_metal,
    OPENMP_AVAILABLE,
    openmp_max_threads,
    MPI_AVAILABLE,
    HIGHS_AVAILABLE,
    system_info as _system_info_raw,
    __version__,
)

_CXX_INT_MAX = (1 << 31) - 1
_CXX_TRIM_CHARS = " \t\r\n"


def _unknown_device(name):
    """Raise the frozen §6 device-name error used by ``dtwc::Env``."""
    raise DeviceError(
        f"[dtwc] unknown device '{name}'. Valid devices: "
        "cpu, gpu, gpu:N (aliases cuda, cuda:N), hpc."
    )


def _parse_device(device):
    """Parse the C++ ``Env`` device grammar into ``(backend, ordinal)``."""
    if not isinstance(device, str):
        raise InvalidInput(f"device must be a string, got {type(device).__name__}")

    raw = device.strip(_CXX_TRIM_CHARS)
    normalized = raw.lower()
    if normalized == "cpu":
        return ("cpu", 0)

    backend, separator, ordinal_text = normalized.partition(":")
    if backend in ("gpu", "cuda"):
        if not separator:
            return (backend, 0)
        # Match Env::set_device exactly: ASCII decimal digits only, with the
        # value representable by its C++ ``int`` device_index_ field.  Python's
        # int() is deliberately not the grammar oracle because it also accepts
        # whitespace, signs, underscores, and arbitrary-size integers.
        if (not ordinal_text or not ordinal_text.isascii()
                or not ordinal_text.isdigit()):
            _unknown_device(raw)
        device_id = int(ordinal_text)
        if device_id > _CXX_INT_MAX:
            _unknown_device(raw)
        return (backend, device_id)

    if normalized == "hpc":
        return ("hpc", 0)           # execution location, not a local compute backend
    _unknown_device(raw)


def _resolve_device(device):
    """Resolve a requested device, failing loudly when it cannot be used."""
    backend, device_id = _parse_device(device)
    if backend == "gpu":
        if CUDA_AVAILABLE and cuda_available():
            return ("cuda", device_id)
        if METAL_AVAILABLE and metal_available():
            return ("metal", device_id)
        compiled = CUDA_AVAILABLE or METAL_AVAILABLE
        detail = ("no compatible GPU device was detected"
                  if compiled else "this build has no GPU backend compiled in")
        raise DeviceError(
            f"[dtwc] device='gpu' requested but {detail}. "
            "This request will not silently fall back to CPU."
        )
    if backend == "cuda":
        if not CUDA_AVAILABLE:
            raise DeviceError(
                "[dtwc] device='cuda' requested but CUDA was not compiled in. "
                "Rebuild with -DDTWC_ENABLE_CUDA=ON. This request will not "
                "silently fall back to CPU."
            )
        if not cuda_available():
            raise DeviceError(
                "[dtwc] device='cuda' requested but no CUDA GPU was detected. "
                "This request will not silently fall back to CPU."
            )
    return (backend, device_id)


_DEFAULT_DEVICE = "cpu"


def _sync_env(name):
    """Mirror the Python device selection into the shared
    ``dtwc::Env`` registry (api-contract-2.0.md §6 — "device() delegates to Env").

    ``cpu``/``gpu``/``cuda`` sync after operational validation. ``hpc`` is NOT
    eagerly validated here — its ``.env``/SSH credential check runs at offload
    submit time, not at ``device()`` set-time, so declaring ``device("hpc")`` never
    blocks on the network.
    """
    base = name.split(":", 1)[0]
    if base in ("cpu", "gpu", "cuda"):
        env().set_device(name)


def device(device=None):
    """Get or set the global default device, PyTorch-style.

    Call with no argument to read the current default; pass a name to set it.
    Accepts ``"cpu"``, ``"gpu"``, ``"gpu:N"``, ``"cuda"``, ``"cuda:N"``, or
    ``"hpc"``. The friendly name is stored verbatim (e.g. ``"gpu"``) and resolved per call, and
    the selection is mirrored into the shared ``dtwc::Env`` registry (§6). An
    explicit ``device=`` argument always overrides this global default.

    Examples
    --------
    >>> dtwcpp.device("gpu")     # subsequent ops require an available GPU
    'gpu'
    >>> dtwcpp.device()          # read the current default
    'gpu'
    """
    global _DEFAULT_DEVICE
    if device is None:
        return _DEFAULT_DEVICE
    backend, _ = _parse_device(device)     # type + syntax validation
    normalized = device.strip().lower()
    if backend != "hpc":
        _resolve_device(normalized)        # operational validation; no fallback
    _sync_env(normalized)                  # mirror into dtwc::Env (shared source of truth)
    _DEFAULT_DEVICE = normalized           # update only after successful validation
    return _DEFAULT_DEVICE


def get_device():
    """Return the current global default device string."""
    return _DEFAULT_DEVICE


def compute_distance_matrix(series, band=-1, metric="l1", use_pruning=True, *, device=None):
    """Compute pairwise DTW distance matrix.

    Parameters
    ----------
    series : list of list of float
        Input time series.
    band : int, default=-1
        Sakoe-Chiba band width (-1 = full DTW).
    metric : str, default='l1'
        Distance metric: 'l1' or 'squared_euclidean'.
    use_pruning : bool, default=True
        Use LB_Keogh pruning (CPU only).
    device : str or None, default=None
        Computation device: 'cpu', 'gpu', 'gpu:N', 'cuda', or 'cuda:N'. ``None`` uses
        the global default set via :func:`device` (itself 'cpu' unless changed).
        ``'hpc'`` is rejected here — it offloads the whole job, not just the
        matrix; use the high-level clustering path instead.

    Returns
    -------
    numpy.ndarray of shape (N, N)
    """
    _valid_metrics = {"l1", "squared_euclidean", "sqeuclidean"}
    if metric not in _valid_metrics:
        raise ValueError(
            f"Unknown metric '{metric}'. Expected one of: {sorted(_valid_metrics)}"
        )

    if device is None:
        device = _DEFAULT_DEVICE
    backend, device_id = _resolve_device(device)
    if backend == "hpc":
        raise ValueError(
            "device='hpc' offloads the entire clustering job to a cluster and is "
            "not a local compute backend. Use DTWClustering(device='hpc').fit(X) "
            "or examples/python/09_device_clustering.py hpc."
        )
    if backend == "cuda":
        use_squared_l2 = metric in ("squared_euclidean", "sqeuclidean")
        return _compute_distance_matrix_cuda(
            series, band=band, use_squared_l2=use_squared_l2,
            device_id=device_id, verbose=False,
        )
    if backend == "metal":
        use_squared_l2 = metric in ("squared_euclidean", "sqeuclidean")
        return _compute_distance_matrix_metal(
            series, band=band, use_squared_l2=use_squared_l2, verbose=False,
        )
    return _compute_distance_matrix_cpu(series, band, metric, use_pruning)


# Pure-Python sklearn-compatible layer
from dtwcpp._clustering import DTWClustering
from dtwcpp.sklearn import DTWCKMedoids

# Unified high-level interface: device() -> load() -> cluster() -> result.plot()
from dtwcpp._api import Dataset, load, cluster, Result, ClusterResult, plot

# Pure-Python I/O utilities (CSV always available; HDF5/Parquet optional)
from dtwcpp.io import (
    save_dataset_csv,
    load_dataset_csv,
    save_dataset_hdf5,
    load_dataset_hdf5,
    save_dataset_parquet,
    load_dataset_parquet,
)

from . import distance
from . import preprocess
from . import diagnose
from . import features
from . import test

def check_system():
    """Print a diagnostic summary of available DTWC++ backends.

    Usage::

        import dtwcpp
        dtwcpp.check_system()
    """
    import sys
    _utf8 = sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower()
    _ok = "\u2705" if _utf8 else "[OK]"
    _no = "\u274c" if _utf8 else "[--]"

    print("DTWC++ System Check")
    print("=" * 40)

    # OpenMP
    if OPENMP_AVAILABLE:
        print(f"  {_ok} OpenMP: {openmp_max_threads()} threads")
    else:
        print(f"  {_no} OpenMP: not available")
        print("     Rebuild with OpenMP support enabled.")
        print("     CMake: compiler should support /openmp (MSVC) or -fopenmp (GCC/Clang)")

    # CUDA
    if CUDA_AVAILABLE:
        if cuda_available():
            print(f"  {_ok} CUDA:   {cuda_device_info(0)}")
        else:
            print(f"  {_no} CUDA:   compiled but no GPU detected")
            print("     Check nvidia-smi and CUDA driver installation.")
    else:
        print(f"  {_no} CUDA:   not compiled")
        print("     Rebuild with: cmake -DDTWC_ENABLE_CUDA=ON ...")

    # Metal (Apple GPU) — metal_available/metal_device_info are bound in the core
    # extension; surface them here for parity with the MATLAB check_system report.
    if METAL_AVAILABLE:
        if metal_available():
            print(f"  {_ok} Metal:  {metal_device_info()}")
        else:
            print(f"  {_no} Metal:  compiled but no GPU detected")
    else:
        print(f"  {_no} Metal:  not compiled (macOS only)")

    # MPI
    if MPI_AVAILABLE:
        print(f"  {_ok} MPI:    available")
    else:
        print(f"  {_no} MPI:    not compiled")
        print("     Rebuild with: cmake -DDTWC_ENABLE_MPI=ON ...")
        print("     Windows: install MS-MPI SDK from microsoft.com/mpi")

    print("=" * 40)


__all__ = [
    "Method", "Solver", "ConstraintType", "MetricType", "DTWVariant",
    "MissingStrategy", "DistanceMatrixStrategy", "StoragePolicy",
    "LowerBoundStrategy", "Linkage", "Device",
    "DTWVariantParams", "ClusteringResult", "DenseDistanceMatrix", "Data",
    "MIPSettings", "CUDASettings", "DendrogramStep", "Dendrogram",
    "HierarchicalOptions", "CLARANSOptions", "OneBatchWeighting",
    "OneBatchPAMOptions", "OneBatchPAMStats",
    "BarycenterMethod", "BarycenterOptions", "BarycenterClusteringOptions",
    "BarycenterClusteringResult",
    "Problem", "Env", "env", "device_to_string", "data_from_arrow_c_array",
    "DtwcError", "InvalidInput", "SolverError", "DeviceError", "IOError",
    "soft_dtw_gradient",
    "fast_pam", "fast_pam_seeded", "fast_clara", "CLARAOptions", "one_batch_pam",
    "one_batch_pam_with_stats",
    "dtw_barycenter", "barycenter_kmeans",
    "clarans", "build_dendrogram", "cut_dendrogram",
    # Scores (canonical 2.0 names)
    "silhouette", "davies_bouldin", "dunn", "inertia", "calinski_harabasz",
    "adjusted_rand", "normalized_mutual_info",
    # Scores (deprecated aliases, §4)
    "davies_bouldin_index", "dunn_index", "calinski_harabasz_index",
    "adjusted_rand_index", "normalized_mutual_information",
    "derivative_transform", "z_normalize",
    "compute_distance_matrix",
    "device", "get_device",
    "Dataset", "load", "cluster", "Result", "ClusterResult", "plot",
    "distance",
    "CUDA_AVAILABLE", "cuda_available", "cuda_device_info", "compute_lb_keogh_cuda",
    "METAL_AVAILABLE", "metal_available", "metal_device_info",
    "OPENMP_AVAILABLE", "openmp_max_threads", "HIGHS_AVAILABLE",
    "MPI_AVAILABLE",
    "check_system",
    "save_checkpoint", "load_checkpoint", "CheckpointOptions",
    "DTWClustering", "DTWCKMedoids",
    "save_dataset_csv", "load_dataset_csv",
    "save_dataset_hdf5", "load_dataset_hdf5",
    "save_dataset_parquet", "load_dataset_parquet",
    "preprocess", "diagnose", "features", "test",
]
