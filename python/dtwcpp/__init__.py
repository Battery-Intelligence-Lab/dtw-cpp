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
    GpuPrecision,
    Linkage,
    Device,
    # Structs
    DTWVariantParams,
    ClusteringResult,
    Data,
    MIPSettings,
    CUDASettings,
    DendrogramStep,
    Dendrogram,
    HierarchicalOptions,
    OneBatchPAMOptions,
    BarycenterMethod,
    BarycenterOptions,
    BarycenterClusteringOptions,
    BarycenterClusteringResult,
    # Classes
    Problem,
    # Arrow C Data / stream ingest (Task 5.7 — zero-copy, no pyarrow)
    data_from_arrow_c_array,
    # Error taxonomy (api-contract-2.0.md §5)
    DtwcError,
    InvalidInput,
    UndefinedScore,
    SolverError,
    DeviceError,
    IOError,
    DEFAULT_RANDOM_SEED,
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
    dtw_barycenter,
    barycenter_kmeans,
    CLARAOptions,
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

from dtwcpp._dtwcpp_core import device as _core_device
# The one device grammar, dtwc::detail::parse_device (§6.1): a name becomes
# (canonical name, GPU ordinal), so "CUDA:3" -> ("gpu", 3).
from dtwcpp._dtwcpp_core import parse_device as _parse_device

from dtwcpp._dtwcpp_core import (
    CUDA_AVAILABLE,
    cuda_available,
    cuda_device_info,
    compute_distance_matrix_cuda as _compute_distance_matrix_cuda,
    METAL_AVAILABLE,
    metal_available,
    metal_device_info,
    compute_distance_matrix_metal as _compute_distance_matrix_metal,
    OPENMP_AVAILABLE,
    openmp_max_threads,
    HIGHS_AVAILABLE,
    system_info as _system_info_raw,
    __version__,
)


def _resolve_device(device):
    """Resolve a requested device to ``(backend, ordinal)``, failing loudly.

    ``backend`` is ``"cpu"``, ``"cuda"``, ``"metal"`` or ``"hpc"``. The name is
    parsed by the C++ grammar, where ``cuda`` is a spelling of ``gpu`` (§6.1):
    either selects this build's live GPU, and raises ``DeviceError`` without one.
    """
    if not isinstance(device, str):
        raise InvalidInput(f"device must be a string, got {type(device).__name__}")
    if device.strip().lower() in _HPC_NAMES:
        return ("hpc", 0)
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
    return (backend, device_id)


# ``hpc`` / ``hpc:gpu`` submit a whole run to a SLURM cluster, which only
# Python and slurm_remote.sh do (C++ refuses the name), so Python records the
# selection itself; the ``.env`` file and the SSH login are checked at submit
# time, so declaring the device never blocks on the network. Every local
# selection lives in C++ (``dtwc::device``), so there is no Python copy of it
# to drift from.
_HPC_NAMES = ("hpc", "hpc:gpu")
_HPC_SELECTED = ""


def _current_device():
    """Canonical name of the active default device."""
    return _HPC_SELECTED or _core_device()


def _hpc_remote_device(name):
    """The device the cluster job computes on: ``cuda`` for ``hpc:gpu``."""
    return "cuda" if name.strip().lower() == "hpc:gpu" else "cpu"


def device(device=None):
    """Get or set the global default device, PyTorch-style.

    Call with no argument to read the current default; pass a name to set it.
    Accepts ``"cpu"``, ``"gpu"``, ``"gpu:N"``, ``"cuda"``, ``"cuda:N"``,
    ``"hpc"`` or ``"hpc:gpu"``. A local selection is stored in C++ (§6) and the
    CANONICAL name that ``dtwc::device()`` reports is returned, so
    ``"cuda:0"`` comes back as ``"gpu"`` exactly as it does in C++ and MATLAB.
    An explicit ``device=`` argument always overrides this global default.

    Examples
    --------
    >>> dtwcpp.device("cuda:0")  # subsequent ops require an available GPU
    'gpu'
    >>> dtwcpp.device()          # read the current default
    'gpu'
    """
    global _HPC_SELECTED
    if device is None:
        return _current_device()
    backend, _ = _resolve_device(device)   # the C++ grammar, then the GPU; no fallback
    if backend == "hpc":
        _HPC_SELECTED = device.strip().lower()   # credentials checked at submit time
        return _HPC_SELECTED
    canonical = _core_device(device)       # dtwc::device(): sets it, canonicalises
    _HPC_SELECTED = ""                     # update only after successful validation
    return canonical


def compute_distance_matrix(series, band=-1, metric="l1", *, device=None):
    """Compute pairwise DTW distance matrix.

    Parameters
    ----------
    series : list of list of float
        Input time series.
    band : int, default=-1
        Sakoe-Chiba band width (-1 = full DTW).
    metric : str, default='l1'
        Distance metric: 'l1' or 'squared_euclidean'.
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
        device = _current_device()
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
    return _compute_distance_matrix_cpu(series, band, metric)


# Pure-Python sklearn-compatible layer
from dtwcpp._clustering import DTWClustering
from dtwcpp.sklearn import DTWCKMedoids

# Unified high-level interface: device() -> load() -> cluster() -> result.plot()
from dtwcpp._api import Dataset, load, cluster, Result, plot

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

    print("=" * 40)


__all__ = [
    "Method", "Solver", "ConstraintType", "MetricType", "DTWVariant",
    "MissingStrategy", "DistanceMatrixStrategy", "GpuPrecision",
    "Linkage", "Device",
    "DTWVariantParams", "ClusteringResult", "Data",
    "MIPSettings", "CUDASettings", "DendrogramStep", "Dendrogram",
    "HierarchicalOptions",
    "OneBatchPAMOptions",
    "BarycenterMethod", "BarycenterOptions", "BarycenterClusteringOptions",
    "BarycenterClusteringResult",
    "Problem", "device_to_string", "data_from_arrow_c_array",
    "DtwcError", "InvalidInput", "UndefinedScore", "SolverError", "DeviceError",
    "IOError",
    "DEFAULT_RANDOM_SEED",
    "soft_dtw_gradient",
    "fast_pam", "fast_pam_seeded", "fast_clara", "CLARAOptions", "one_batch_pam",
    "dtw_barycenter", "barycenter_kmeans",
    "build_dendrogram", "cut_dendrogram",
    # Scores (canonical 2.0 names)
    "silhouette", "davies_bouldin", "dunn", "inertia", "calinski_harabasz",
    "adjusted_rand", "normalized_mutual_info",
    "derivative_transform", "z_normalize",
    "compute_distance_matrix",
    "device",
    "Dataset", "load", "cluster", "Result", "plot",
    "distance",
    "CUDA_AVAILABLE", "cuda_available", "cuda_device_info",
    "METAL_AVAILABLE", "metal_available", "metal_device_info",
    "OPENMP_AVAILABLE", "openmp_max_threads", "HIGHS_AVAILABLE",
    "check_system",
    "save_checkpoint", "load_checkpoint",
    "CheckpointOptions",
    "DTWClustering", "DTWCKMedoids",
    "save_dataset_csv", "load_dataset_csv",
    "save_dataset_hdf5", "load_dataset_hdf5",
    "save_dataset_parquet", "load_dataset_parquet",
    "preprocess", "diagnose", "features", "test",
]
