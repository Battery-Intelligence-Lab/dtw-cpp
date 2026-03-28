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
    # Algorithms
    fast_pam,
    # Scores
    silhouette,
    davies_bouldin_index,
    # Utils
    derivative_transform,
    z_normalize,
)

__version__ = "1.0.0"

__all__ = [
    "Method", "Solver", "ConstraintType", "MetricType", "DTWVariant",
    "DTWVariantParams", "ClusteringResult", "DenseDistanceMatrix", "Data",
    "Problem",
    "dtw_distance", "ddtw_distance", "wdtw_distance", "adtw_distance",
    "soft_dtw_distance", "soft_dtw_gradient",
    "fast_pam", "silhouette", "davies_bouldin_index",
    "derivative_transform", "z_normalize",
]
