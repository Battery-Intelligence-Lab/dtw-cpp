"""DTWC++ — Fast Dynamic Time Warping and Clustering."""

import numpy as _np

from dtwcpp._dtwcpp_core import (
    # Enums
    Method,
    Solver,
    ConstraintType,
    MetricType,
    DTWVariant,
    # Structs
    DTWVariantParams,
    ClusteringResult,
    DenseDistanceMatrix,
    Data,
    # Classes
    Problem,
    # DTW functions (raw C++ bindings — require numpy arrays)
    dtw_distance as _dtw_distance_raw,
    ddtw_distance,
    wdtw_distance,
    adtw_distance,
    soft_dtw_distance,
    soft_dtw_gradient,
    dtw_distance_missing as _dtw_distance_missing_raw,
    # Algorithms
    fast_pam,
    fast_clara,
    CLARAOptions,
    # Scores
    silhouette,
    davies_bouldin_index,
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
)

__version__ = "1.0.0"


# Wrappers that accept both lists and numpy arrays (the C++ nb::ndarray
# binding only accepts numpy arrays; these auto-convert for convenience).
def dtw_distance(x, y, band=-1, metric="l1"):
    """Compute DTW distance between two time series.

    Accepts lists or numpy arrays. metric: 'l1' (default) or 'squared_euclidean'.
    band=-1 for full DTW, band>0 for Sakoe-Chiba banded DTW.
    """
    return _dtw_distance_raw(
        _np.asarray(x, dtype=_np.float64),
        _np.asarray(y, dtype=_np.float64),
        band, metric)


def dtw_distance_missing(x, y, band=-1, metric="l1"):
    """DTW distance with missing data support (NaN = missing).

    Accepts lists or numpy arrays. NaN values contribute zero cost.
    """
    return _dtw_distance_missing_raw(
        _np.asarray(x, dtype=_np.float64),
        _np.asarray(y, dtype=_np.float64),
        band, metric)

def _parse_device(device):
    """Parse PyTorch-style device string (case-insensitive). Returns (backend, device_id)."""
    if not isinstance(device, str):
        raise ValueError(f"device must be a string, got {type(device).__name__}")
    device = device.strip().lower()
    if device == "cpu":
        return ("cpu", 0)
    if device == "cuda" or device.startswith("cuda:"):
        parts = device.split(":", 1)
        device_id = int(parts[1]) if len(parts) > 1 else 0
        return ("cuda", device_id)
    raise ValueError(f"Unknown device '{device}'. Expected 'cpu', 'cuda', or 'cuda:N'.")


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


def compute_distance_matrix(series, band=-1, metric="l1", use_pruning=True, *, device="cpu"):
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
    device : str, default='cpu'
        Computation device: 'cpu', 'cuda', or 'cuda:N'.

    Returns
    -------
    numpy.ndarray of shape (N, N)
    """
    _valid_metrics = {"l1", "squared_euclidean", "sqeuclidean"}
    if metric not in _valid_metrics:
        raise ValueError(
            f"Unknown metric '{metric}'. Expected one of: {sorted(_valid_metrics)}"
        )

    backend, device_id = _resolve_device(device)
    if backend == "cuda":
        use_squared_l2 = metric in ("squared_euclidean", "sqeuclidean")
        return _compute_distance_matrix_cuda(
            series, band=band, use_squared_l2=use_squared_l2,
            device_id=device_id, verbose=False,
        )
    return _compute_distance_matrix_cpu(series, band, metric, use_pruning)


# Pure-Python sklearn-compatible layer
from dtwcpp._clustering import DTWClustering

# Pure-Python I/O utilities (CSV always available; HDF5/Parquet optional)
from dtwcpp.io import (
    save_dataset_csv,
    load_dataset_csv,
    save_dataset_hdf5,
    load_dataset_hdf5,
    save_dataset_parquet,
    load_dataset_parquet,
)

__all__ = [
    "Method", "Solver", "ConstraintType", "MetricType", "DTWVariant",
    "DTWVariantParams", "ClusteringResult", "DenseDistanceMatrix", "Data",
    "Problem",
    "dtw_distance", "ddtw_distance", "wdtw_distance", "adtw_distance",
    "soft_dtw_distance", "soft_dtw_gradient", "dtw_distance_missing",
    "fast_pam", "fast_clara", "CLARAOptions",
    "silhouette", "davies_bouldin_index",
    "derivative_transform", "z_normalize",
    "compute_distance_matrix",
    "CUDA_AVAILABLE", "cuda_available", "cuda_device_info", "compute_lb_keogh_cuda",
    "save_checkpoint", "load_checkpoint", "CheckpointOptions",
    "DTWClustering",
    "save_dataset_csv", "load_dataset_csv",
    "save_dataset_hdf5", "load_dataset_hdf5",
    "save_dataset_parquet", "load_dataset_parquet",
]
