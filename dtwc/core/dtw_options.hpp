/**
 * @file dtw_options.hpp
 * @brief Runtime DTW configuration: constraint type, metric selection, etc.
 *
 * @details DTWOptions bundles every knob that can be set at runtime for the
 *          binding-friendly (non-template) DTW entry point.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "../base/names.hpp"
#include "../base/settings.hpp" // DEFAULT_BAND

#include <cstddef>

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

/// L2 has no name: no front end offers it.
inline constexpr Name<MetricType> metric_names[]{
  { "l1", MetricType::L1 },
  { "squared_euclidean", MetricType::SquaredL2 },
  { "sqeuclidean", MetricType::SquaredL2 },
  { "l2sq", MetricType::SquaredL2 },
};

/// DTW algorithm variant.
enum class DTWVariant
{
  Standard,  ///< Classic DTW with min(3 neighbors) recurrence
  DDTW,      ///< Derivative DTW: derivative preprocessing + standard DTW
  WDTW,      ///< Weighted DTW: position-dependent weight w(|i-j|) on local cost
  ADTW,      ///< Amerced DTW: penalty on non-diagonal (horizontal/vertical) steps
  SoftDTW,   ///< Soft-DTW: softmin replaces min (differentiable, Cuturi & Blondel 2017)
  MSM,       ///< Move-Split-Merge (Stefan et al. 2013): metric elastic distance, univariate/unbanded
  TWE        ///< Time Warp Edit (Marteau 2009): stiffness ν + edit penalty λ, univariate/unbanded
};

inline constexpr Name<DTWVariant> variant_names[]{
  { "standard", DTWVariant::Standard },
  { "ddtw", DTWVariant::DDTW },
  { "wdtw", DTWVariant::WDTW },
  { "adtw", DTWVariant::ADTW },
  { "softdtw", DTWVariant::SoftDTW },
  { "soft-dtw", DTWVariant::SoftDTW },
  { "msm", DTWVariant::MSM },
  { "twe", DTWVariant::TWE },
};

/// Multivariate combination mode (Shokoohi-Yekta et al., DMKD 2017).
/// Only meaningful when Data::ndim > 1; ignored for univariate series.
enum class MVMode
{
  Dependent,   ///< DTW_D: one warping path, per-cell cost summed over channels (default; existing behaviour)
  Independent  ///< DTW_I: independent per-channel univariate DTW, distances summed
};

inline constexpr Name<MVMode> mv_mode_names[]{
  { "dependent", MVMode::Dependent },
  { "independent", MVMode::Independent },
};

/// Strategy for handling missing data (NaN values) in time series.
enum class MissingStrategy
{
  Error,        ///< Throw if NaN encountered (default, backward-compatible)
  ZeroCost,     ///< Zero local cost for NaN pairs (existing warping_missing.hpp behavior)
  AROW,         ///< DTW-AROW: one-to-one diagonal-only alignment for missing positions
  Interpolate   ///< Linear interpolation preprocessing, then standard DTW
};

inline constexpr Name<MissingStrategy> missing_strategy_names[]{
  { "error", MissingStrategy::Error },
  { "zero_cost", MissingStrategy::ZeroCost },
  { "zero-cost", MissingStrategy::ZeroCost },
  { "zerocost", MissingStrategy::ZeroCost },
  { "arow", MissingStrategy::AROW },
  { "interpolate", MissingStrategy::Interpolate },
};

/// Variant-specific parameters.
struct DTWVariantParams
{
  DTWVariant variant = DTWVariant::Standard;
  double wdtw_g = 0.05;       ///< WDTW: finite logistic steepness, >= 0
  double adtw_penalty = 1.0;  ///< ADTW: finite non-diagonal step penalty, >= 0
  double sdtw_gamma = 1.0;    ///< Soft-DTW: finite smoothing parameter, > 0
  double msm_c = 1.0;         ///< MSM: finite split/merge cost, > 0
  double twe_nu = 0.001;      ///< TWE: stiffness ν, > 0 (Marteau 2009; aeon default 0.001)
  double twe_lambda = 1.0;    ///< TWE: finite edit penalty λ, > 0 (aeon default 1.0)
  MVMode mv_mode = MVMode::Dependent;  ///< Multivariate mode (ndim>1); default keeps DTW_D

  bool operator==(const DTWVariantParams &) const = default;
};

/// Everything that decides what a distance between two series means. A Problem
/// holds one, changed only through its setters, and takes `ndim` from its series.
struct DistanceConfig
{
  DTWVariantParams variant{};                        ///< Variant, its parameters and the multivariate mode
  MetricType metric = MetricType::L1;                ///< Pointwise cost
  MissingStrategy missing = MissingStrategy::Error;  ///< How NaN values are treated
  int band = settings::DEFAULT_BAND;                 ///< Sakoe-Chiba half-width; -1 = full DTW
  std::size_t ndim = 1;                              ///< Channels per time step

  bool operator==(const DistanceConfig &) const = default;
};

/// Runtime DTW configuration.
struct DTWOptions
{
  ConstraintType constraint = ConstraintType::None;
  MetricType metric = MetricType::L1;
  int band = -1;  ///< Band width for Sakoe-Chiba; -1 means unconstrained
  DTWVariantParams variant_params;  ///< Variant selection and parameters
  MissingStrategy missing_strategy = MissingStrategy::Error;  ///< How to handle NaN values
};

} // namespace dtwc::core
