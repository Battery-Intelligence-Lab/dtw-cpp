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
    # Classes
    Problem,
    Env,
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
    fast_clara,
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
    OPENMP_AVAILABLE,
    openmp_max_threads,
    MPI_AVAILABLE,
    system_info as _system_info_raw,
)

__version__ = "2.0.0"

def _parse_device(device):
    """Parse PyTorch-style device string (case-insensitive). Returns (backend, device_id)."""
    if not isinstance(device, str):
        raise ValueError(f"device must be a string, got {type(device).__name__}")
    device = device.strip().lower()
    if device == "cpu":
        return ("cpu", 0)
    if device == "gpu":
        return ("cuda", 0)          # friendly alias; falls back to CPU if no GPU
    if device == "cuda" or device.startswith("cuda:"):
        parts = device.split(":", 1)
        device_id = int(parts[1]) if len(parts) > 1 else 0
        return ("cuda", device_id)
    if device == "hpc":
        return ("hpc", 0)           # execution location, not a local compute backend
    raise ValueError(
        f"Unknown device '{device}'. Expected 'cpu', 'gpu', 'cuda', 'cuda:N', or 'hpc'."
    )


def _resolve_device(device):
    """Parse device and fall back to CPU if CUDA unavailable."""
    import warnings
    backend, device_id = _parse_device(device)
    if backend == "cuda":
        if not CUDA_AVAILABLE:
            warnings.warn(
                "device='cuda' requested but CUDA was not compiled in. "
                "Falling back to CPU. Rebuild with -DDTWC_ENABLE_CUDA=ON.",
                RuntimeWarning, stacklevel=3,
            )
            return ("cpu", 0)
        if not cuda_available():
            warnings.warn(
                "device='cuda' requested but no CUDA GPU detected. "
                "Falling back to CPU.",
                RuntimeWarning, stacklevel=3,
            )
            return ("cpu", 0)
    return (backend, device_id)


_DEFAULT_DEVICE = "cpu"


def _sync_env(name):
    """Best-effort mirror of the Python device selection into the shared
    ``dtwc::Env`` registry (api-contract-2.0.md §6 — "device() delegates to Env").

    ``cpu`` always syncs; ``gpu``/``cuda`` sync only when a GPU backend is compiled
    in (otherwise ``Env`` raises :class:`DeviceError` per the no-silent-fallback
    rule, and the Tier-1 path keeps its documented CPU fallback, §1.1). ``hpc`` is
    NOT eagerly validated here — its ``.env``/SSH credential check runs at offload
    submit time, not at ``device()`` set-time, so declaring ``device("hpc")`` never
    blocks on the network.
    """
    base = name.split(":", 1)[0]
    if base in ("cpu", "gpu", "cuda"):
        try:
            env().set_device(name)
        except DeviceError:
            pass  # keep the documented Tier-1 gpu->cpu fallback (§1.1)


def device(device=None):
    """Get or set the global default device, PyTorch-style.

    Call with no argument to read the current default; pass a name to set it.
    Accepts ``"cpu"``, ``"gpu"``, ``"cuda"``, ``"cuda:N"``, or ``"hpc"``. The
    friendly name is stored verbatim (e.g. ``"gpu"``) and resolved per call, and
    the selection is mirrored into the shared ``dtwc::Env`` registry (§6). An
    explicit ``device=`` argument always overrides this global default.

    Examples
    --------
    >>> dtwcpp.device("gpu")     # subsequent ops default to GPU (CPU fallback)
    'gpu'
    >>> dtwcpp.device()          # read the current default
    'gpu'
    """
    global _DEFAULT_DEVICE
    if device is None:
        return _DEFAULT_DEVICE
    _parse_device(device)                 # validate; raises ValueError on unknown
    _DEFAULT_DEVICE = device.strip().lower()
    _sync_env(_DEFAULT_DEVICE)            # mirror into dtwc::Env (shared source of truth)
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
        Computation device: 'cpu', 'gpu', 'cuda', or 'cuda:N'. ``None`` uses
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
    return _compute_distance_matrix_cpu(series, band, metric, use_pruning)


# Pure-Python sklearn-compatible layer
from dtwcpp._clustering import DTWClustering

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
    "HierarchicalOptions", "CLARANSOptions",
    "Problem", "Env", "env", "device_to_string",
    "DtwcError", "InvalidInput", "SolverError", "DeviceError", "IOError",
    "soft_dtw_gradient",
    "fast_pam", "fast_clara", "CLARAOptions",
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
    "OPENMP_AVAILABLE", "openmp_max_threads",
    "MPI_AVAILABLE",
    "check_system",
    "save_checkpoint", "load_checkpoint", "CheckpointOptions",
    "DTWClustering",
    "save_dataset_csv", "load_dataset_csv",
    "save_dataset_hdf5", "load_dataset_hdf5",
    "save_dataset_parquet", "load_dataset_parquet",
    "preprocess", "diagnose", "features",
]
