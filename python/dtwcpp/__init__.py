"""DTWC++ — Fast Dynamic Time Warping and Clustering."""

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
    # DTW functions
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    adtw_distance,
    soft_dtw_distance,
    soft_dtw_gradient,
    dtw_distance_missing,
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
)

__version__ = "1.0.0"

# Pure-Python sklearn-compatible layer
from dtwcpp._clustering import DTWClustering

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
    "DTWClustering",
]
