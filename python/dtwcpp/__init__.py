"""
@file __init__.py
@brief DTWC++ — Fast Dynamic Time Warping and Clustering.
@author Volkan Kumtepeli
"""

from dtwcpp._dtwcpp_core import (
    # Enums
    Method,
    Solver,
    MetricType,
    DTWVariant,
    MissingStrategy,
    GpuPrecision,
    Linkage,
    Device,
    # Structs
    DTWVariantParams,
    ClusteringResult,
    Data,
    MIPSettings,
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
    soft_dtw_gradient,
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
    gpu_available,
    gpu_info,
    OPENMP_AVAILABLE,
    openmp_max_threads,
    HIGHS_AVAILABLE,
    __version__,
)


def _resolve_device(device):
    """Resolve a requested device to ``(backend, ordinal)``, failing loudly.

    ``backend`` is ``"cpu"``, ``"gpu"`` or ``"hpc"``. The name is parsed by the
    C++ grammar, where ``cuda`` is a spelling of ``gpu`` (§6.1); ``gpu`` needs a
    GPU this build's backend finds, and raises ``DeviceError`` without one.
    """
    if not isinstance(device, str):
        raise InvalidInput(f"device must be a string, got {type(device).__name__}")
    if device.strip().lower() in _HPC_NAMES:
        return ("hpc", 0)
    backend, device_id = _parse_device(device)
    if backend == "gpu" and not gpu_available():
        raise DeviceError(
            f"[dtwc] device='gpu' requested but no GPU is available ({gpu_info()}). "
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

    Raises
    ------
    InvalidInput
        For an unknown metric, a NaN or +-inf value, an empty series, a band below
        -1, or a band narrower than the length difference between the shortest and
        longest series (no warping path fits that pair); the same checks as
        ``Problem.fill_distance_matrix``, which computes the matrix on every device.
    """
    if device is None:
        device = _current_device()
    backend, _ = _resolve_device(device)
    if backend == "hpc":
        raise ValueError(
            "device='hpc' offloads the entire clustering job to a cluster and is "
            "not a local compute backend. Use DTWClustering(device='hpc').fit(X) "
            "or examples/python/09_device_clustering.py hpc."
        )
    # A Problem's fill on `device`: the CPU, or for 'gpu' this build's backend
    # (CUDA, else Metal) and the GPU index of `device`, never the CPU.
    prob = Problem("compute_distance_matrix", device=device)
    prob.set_distance(band=band, metric=metric)
    prob.set_data(series, [str(i) for i in range(len(series))])
    return prob.distance_matrix()


# The sklearn-compatible estimator
from dtwcpp._clustering import DTWClustering

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

# v1.0.0's Problem file writers, written in Python (the compiled core writes no files).
from dtwcpp import io as _io
Problem.write_clusters = _io.write_clusters
Problem.write_silhouettes = _io.write_silhouettes
Problem.write_medoid_members = _io.write_medoid_members
Problem.write_distance_matrix = _io.write_distance_matrix

from . import distance
from . import preprocess
from . import diagnose
from . import features
from . import test

__all__ = [
    "Method", "Solver", "MetricType", "DTWVariant",
    "MissingStrategy", "GpuPrecision",
    "Linkage", "Device",
    "DTWVariantParams", "ClusteringResult", "Data",
    "MIPSettings", "DendrogramStep", "Dendrogram",
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
    "gpu_available", "gpu_info",
    "OPENMP_AVAILABLE", "openmp_max_threads", "HIGHS_AVAILABLE",
    "save_checkpoint", "load_checkpoint",
    "CheckpointOptions",
    "DTWClustering",
    "save_dataset_csv", "load_dataset_csv",
    "save_dataset_hdf5", "load_dataset_hdf5",
    "save_dataset_parquet", "load_dataset_parquet",
    "preprocess", "diagnose", "features", "test",
]
