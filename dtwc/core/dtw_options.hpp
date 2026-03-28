/**
 * @file dtw_options.hpp
 * @brief Configuration struct for DTW computation.
 *
 * @details Encapsulates all options that control DTW distance calculation,
 * including constraint type, band width, normalization, early abandoning,
 * lower-bound pruning, missing data strategy, and metric selection.
 *
 * @date 28 Mar 2026
 */

#pragma once

namespace dtwc::core {

/// Type of warping path constraint applied to the DTW cost matrix.
enum class ConstraintType {
  None,                 ///< No constraint (full matrix).
  SakoeChibaBand,       ///< Sakoe-Chiba band constraint.
  ItakuraParallelogram  ///< Itakura parallelogram constraint.
};

/// Pointwise distance metric used in DTW accumulation.
enum class MetricType {
  L1,         ///< Absolute difference |a - b|.
  L2,         ///< Squared difference (sqrt applied to final result).
  SquaredL2   ///< Squared difference (no final sqrt).
};

/// Strategy for handling missing (NaN) values in time series.
enum class MissingStrategy {
  Error,       ///< Throw an error if NaN is encountered.
  AROW,        ///< Adaptive Recursive Optimal Warping.
  ZeroCost,    ///< Treat NaN comparisons as zero cost.
  ZeroCostNorm,///< Zero cost with path-length normalization.
  Interpolate  ///< Interpolate over missing values before DTW.
};

/// Configuration options for a DTW computation.
struct DTWOptions {
  ConstraintType constraint = ConstraintType::None;
  int band_width = -1;                  ///< Band width for Sakoe-Chiba (-1 = unconstrained).
  bool normalize_by_path = false;       ///< Divide final cost by warping path length.
  double early_abandon_threshold = -1.0;///< Abandon if cost exceeds this (-1 = disabled).
  bool use_lb_keogh = false;            ///< Use LB_Keogh lower bound pruning.
  bool use_lb_kim = false;              ///< Use LB_Kim lower bound pruning.
  MissingStrategy missing = MissingStrategy::Error;
  MetricType metric = MetricType::L1;
  bool z_normalize = false;            ///< Z-normalize series before DTW.
};

} // namespace dtwc::core
