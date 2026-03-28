/**
 * @file dtw_options.hpp
 * @brief Runtime DTW configuration: constraint type, metric selection, etc.
 *
 * @details DTWOptions bundles every knob that can be set at runtime for the
 *          binding-friendly (non-template) DTW entry point.
 *
 * @date 28 Mar 2026
 */

#pragma once

namespace dtwc::core {

/// Warping-path constraint type.
enum class ConstraintType
{
  None,            ///< Unconstrained (full cost matrix)
  SakoeChibaBand   ///< Sakoe-Chiba band constraint
};

/// Runtime metric selector (used by the non-template dtw_runtime entry point).
enum class MetricType
{
  L1,         ///< |a - b|
  L2,         ///< sqrt((a-b)^2) -- same as L1 for scalars
  SquaredL2   ///< (a - b)^2
};

/// Runtime DTW configuration.
struct DTWOptions
{
  ConstraintType constraint = ConstraintType::None;
  MetricType metric = MetricType::L1;
  int band_width = -1;  ///< Band width for Sakoe-Chiba; -1 means unconstrained
};

} // namespace dtwc::core
