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

/// DTW algorithm variant.
enum class DTWVariant
{
  Standard,  ///< Classic DTW with min(3 neighbors) recurrence
  DDTW,      ///< Derivative DTW: derivative preprocessing + standard DTW
  WDTW,      ///< Weighted DTW: position-dependent weight w(|i-j|) on local cost
  ADTW,      ///< Amerced DTW: penalty on non-diagonal (horizontal/vertical) steps
  SoftDTW    ///< Soft-DTW: softmin replaces min (differentiable, Cuturi & Blondel 2017)
};

/// Variant-specific parameters.
struct DTWVariantParams
{
  DTWVariant variant = DTWVariant::Standard;
  double wdtw_g = 0.05;       ///< WDTW: logistic weight steepness (Jeong et al. 2011)
  double adtw_penalty = 1.0;  ///< ADTW: penalty for non-diagonal steps
  double sdtw_gamma = 1.0;    ///< Soft-DTW: smoothing parameter (lower = closer to hard DTW)
};

/// Runtime DTW configuration.
struct DTWOptions
{
  ConstraintType constraint = ConstraintType::None;
  MetricType metric = MetricType::L1;
  int band_width = -1;  ///< Band width for Sakoe-Chiba; -1 means unconstrained
  DTWVariantParams variant_params;  ///< Variant selection and parameters
};

} // namespace dtwc::core
