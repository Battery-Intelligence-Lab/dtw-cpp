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
    compute_distance_matrix,
    # Checkpointing
    save_checkpoint,
    load_checkpoint,
    CheckpointOptions,
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
    "save_checkpoint", "load_checkpoint", "CheckpointOptions",
    "DTWClustering",
    "save_dataset_csv", "load_dataset_csv",
    "save_dataset_hdf5", "load_dataset_hdf5",
    "save_dataset_parquet", "load_dataset_parquet",
]
