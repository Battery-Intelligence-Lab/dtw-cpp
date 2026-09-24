/**
 * @file distance.hpp
 * @brief Additive distance facade for the public DTWC++ API.
 *
 * @details These wrappers provide a stable `dtwc::distance::*` namespace that
 * mirrors the user-facing Python and MATLAB distance surfaces without removing
 * the existing free functions. The individual functions remain the preferred
 * choice in tight loops; the dispatcher overload is a convenience layer for
 * one-off calls and examples.
 *
 * This namespace is the checked boundary (warping.hpp explains the layering):
 * every function here validates its parameters, then checks x and y before any
 * distance work, in O(n + m), throwing InvalidInput that names the series and
 * position of a NaN or ±inf.
 * `missing`, `arow` and the ZeroCost / AROW / Interpolate strategies read NaN
 * as a missing value and reject only ±inf. The unchecked per-pair wrappers
 * they call are unchanged.
 */

#pragma once

#include "base/error.hpp"
#include "base/settings.hpp"
#include "core/distance_semantics.hpp"
#include "core/dtw_options.hpp"
#include "core/variant_validation.hpp"
#include "core/msm.hpp"
#include "core/twe.hpp"
#include "base/missing_utils.hpp"
#include "soft_dtw.hpp"
#include "warping.hpp"
#include "warping_adtw.hpp"
#include "warping_ddtw.hpp"
#include "warping_missing.hpp"
#include "warping_missing_arow.hpp"
#include "warping_wdtw.hpp"

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dtwc::detail {

/// WDTW, ADTW, Soft-DTW, MSM and TWE take no metric: their kernels compute an
/// L1 cost, so asking one of them for another metric is InvalidInput, never the
/// L1 distance. Standard and DDTW pass the metric to their kernels, as do the
/// missing-data strategies, which run with Standard only. Call after
/// validate_metric_type. Problem::set_metric is stricter: its resolver passes
/// the metric to the Standard kernels only.
inline void require_metric_supported(core::DTWVariant variant,
                                     core::MetricType metric,
                                     std::string_view where)
{
  if (metric == core::MetricType::L1) return;
  const char *name = nullptr;
  switch (variant) {
  case core::DTWVariant::WDTW: name = "WDTW"; break;
  case core::DTWVariant::ADTW: name = "ADTW"; break;
  case core::DTWVariant::SoftDTW: name = "SoftDTW"; break;
  case core::DTWVariant::MSM: name = "MSM"; break;
  case core::DTWVariant::TWE: name = "TWE"; break;
  default: return; // Standard and DDTW take the metric.
  }
  throw InvalidInput(
    std::string(where) + ": metric "
    + (metric == core::MetricType::SquaredL2 ? "SquaredL2" : "L2")
    + " is implemented for Standard DTW and DDTW only, but variant = " + name
    + " was requested. Use metric L1 for this configuration.");
}

} // namespace dtwc::detail

namespace dtwc::distance {

template <typename T = dtwc::settings::default_data_t>
T dtw(std::span<const T> x, std::span<const T> y,
      int band = settings::DEFAULT_BAND,
      core::MetricType metric = core::MetricType::L1)
{
  core::validate_metric_type(metric);
  dtwc::detail::require_finite<T>(x, y, "distance::dtw");
  return dtwBanded<T>(x, y, band, static_cast<T>(-1), metric);
}

template <typename T = dtwc::settings::default_data_t>
T ddtw(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       core::MetricType metric = core::MetricType::L1)
{
  core::validate_metric_type(metric);
  dtwc::detail::require_finite<T>(x, y, "distance::ddtw");
  return ddtwBanded<T>(x, y, band, metric);
}

template <typename T = dtwc::settings::default_data_t>
T wdtw(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       T g = static_cast<T>(0.05))
{
  core::validate_wdtw_g(g);
  dtwc::detail::require_finite<T>(x, y, "distance::wdtw");
  return wdtwBanded<T>(x, y, band, g);
}

template <typename T = dtwc::settings::default_data_t>
T adtw(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       T penalty = static_cast<T>(1.0))
{
  core::validate_adtw_penalty(penalty);
  dtwc::detail::require_finite<T>(x, y, "distance::adtw");
  return adtwBanded<T>(x, y, band, penalty);
}

template <typename T = dtwc::settings::default_data_t>
T soft_dtw(std::span<const T> x, std::span<const T> y,
           T gamma = static_cast<T>(1.0))
{
  core::validate_sdtw_gamma(gamma);
  dtwc::detail::require_finite<T>(x, y, "distance::soft_dtw");
  return dtwc::soft_dtw<T>(x, y, gamma);
}

template <typename T = dtwc::settings::default_data_t>
T missing(std::span<const T> x, std::span<const T> y,
          int band = settings::DEFAULT_BAND,
          core::MetricType metric = core::MetricType::L1)
{
  core::validate_metric_type(metric);
  dtwc::detail::require_finite<T>(x, y, "distance::missing", /*nan_is_missing=*/true);
  return dtwMissing_banded<T>(x, y, band, static_cast<T>(-1), metric);
}

template <typename T = dtwc::settings::default_data_t>
T arow(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       core::MetricType metric = core::MetricType::L1)
{
  core::validate_metric_type(metric);
  dtwc::detail::require_finite<T>(x, y, "distance::arow", /*nan_is_missing=*/true);
  return dtwAROW_banded<T>(x, y, band, metric);
}

/// Move-Split-Merge distance (Stefan et al. 2013), univariate/unbanded.
template <typename T = dtwc::settings::default_data_t>
T msm(std::span<const T> x, std::span<const T> y, T c = static_cast<T>(1.0))
{
  core::validate_msm_c(c);
  dtwc::detail::require_finite<T>(x, y, "distance::msm");
  return core::msm_distance<T>(x, y, c);
}

/// Time Warp Edit distance (Marteau 2009), univariate/unbanded.
template <typename T = dtwc::settings::default_data_t>
T twe(std::span<const T> x, std::span<const T> y,
      T nu = static_cast<T>(0.001), T lambda = static_cast<T>(1.0))
{
  core::validate_twe_nu(nu);
  core::validate_twe_lambda(lambda);
  dtwc::detail::require_finite<T>(x, y, "distance::twe");
  return core::twe_distance<T>(x, y, nu, lambda);
}

/// The variant dispatcher. `metric` reaches Standard, DDTW and the missing-data
/// strategies; WDTW, ADTW, Soft-DTW, MSM and TWE throw InvalidInput for a
/// metric other than L1 (require_metric_supported).
template <typename T = dtwc::settings::default_data_t>
T dtw(std::span<const T> x, std::span<const T> y,
      const core::DTWVariantParams &params,
      int band = settings::DEFAULT_BAND,
      core::MetricType metric = core::MetricType::L1,
      core::MissingStrategy missing_strategy = core::MissingStrategy::Error)
{
  core::validate_variant_params(params);
  core::validate_variant_missing_semantics(params, missing_strategy);
  core::validate_metric_type(metric);
  dtwc::detail::require_metric_supported(params.variant, metric, "distance::dtw");
  // Each branch ends in a checked function above, which scans the input.
  // Interpolate first rejects ±inf, which interpolate_linear() would spread
  // into the gaps it fills.
  switch (missing_strategy) {
  case core::MissingStrategy::ZeroCost:
    return missing<T>(x, y, band, metric);

  case core::MissingStrategy::AROW:
    return arow<T>(x, y, band, metric);

  case core::MissingStrategy::Interpolate: {
    dtwc::detail::require_finite<T>(x, y, "distance::dtw", /*nan_is_missing=*/true);
    auto xi = has_missing(x) ? interpolate_linear(x) : std::vector<T>(x.begin(), x.end());
    auto yi = has_missing(y) ? interpolate_linear(y) : std::vector<T>(y.begin(), y.end());
    return dtw<T>(std::span<const T>{xi}, std::span<const T>{yi}, band, metric);
  }

  case core::MissingStrategy::Error:
    break;
  default:
    core::validate_missing_strategy(missing_strategy);
    throw std::logic_error("distance::dtw: unreachable MissingStrategy");
  }

  switch (params.variant) {
  case core::DTWVariant::DDTW:
    return ddtw<T>(x, y, band, metric);
  case core::DTWVariant::WDTW:
    return wdtw<T>(x, y, band, static_cast<T>(params.wdtw_g));
  case core::DTWVariant::ADTW:
    return adtw<T>(x, y, band, static_cast<T>(params.adtw_penalty));
  case core::DTWVariant::SoftDTW:
    return soft_dtw<T>(x, y, static_cast<T>(params.sdtw_gamma));
  case core::DTWVariant::MSM:
    return msm<T>(x, y, static_cast<T>(params.msm_c));
  case core::DTWVariant::TWE:
    return twe<T>(x, y, static_cast<T>(params.twe_nu),
                  static_cast<T>(params.twe_lambda));
  case core::DTWVariant::Standard:
    return dtw<T>(x, y, band, metric);
  default:
    core::validate_dtw_variant(params.variant);
    throw std::logic_error("distance::dtw: unreachable DTWVariant");
  }
}

template <typename T = dtwc::settings::default_data_t>
T dtw(const std::vector<T> &x, const std::vector<T> &y,
      int band = settings::DEFAULT_BAND,
      core::MetricType metric = core::MetricType::L1)
{
  return dtw<T>(std::span<const T>{x}, std::span<const T>{y}, band, metric);
}

template <typename T = dtwc::settings::default_data_t>
T dtw(const std::vector<T> &x, const std::vector<T> &y,
      const core::DTWVariantParams &params,
      int band = settings::DEFAULT_BAND,
      core::MetricType metric = core::MetricType::L1,
      core::MissingStrategy missing_strategy = core::MissingStrategy::Error)
{
  return dtw<T>(
    std::span<const T>{x}, std::span<const T>{y},
    params, band, metric, missing_strategy);
}

template <typename T = dtwc::settings::default_data_t>
T ddtw(const std::vector<T> &x, const std::vector<T> &y,
       int band = settings::DEFAULT_BAND,
       core::MetricType metric = core::MetricType::L1)
{
  return ddtw<T>(std::span<const T>{x}, std::span<const T>{y}, band, metric);
}

template <typename T = dtwc::settings::default_data_t>
T wdtw(const std::vector<T> &x, const std::vector<T> &y,
       int band = settings::DEFAULT_BAND,
       T g = static_cast<T>(0.05))
{
  return wdtw<T>(std::span<const T>{x}, std::span<const T>{y}, band, g);
}

template <typename T = dtwc::settings::default_data_t>
T adtw(const std::vector<T> &x, const std::vector<T> &y,
       int band = settings::DEFAULT_BAND,
       T penalty = static_cast<T>(1.0))
{
  return adtw<T>(std::span<const T>{x}, std::span<const T>{y}, band, penalty);
}

template <typename T = dtwc::settings::default_data_t>
T soft_dtw(const std::vector<T> &x, const std::vector<T> &y,
           T gamma = static_cast<T>(1.0))
{
  return soft_dtw<T>(std::span<const T>{x}, std::span<const T>{y}, gamma);
}

template <typename T = dtwc::settings::default_data_t>
T msm(const std::vector<T> &x, const std::vector<T> &y, T c = static_cast<T>(1.0))
{
  return msm<T>(std::span<const T>{x}, std::span<const T>{y}, c);
}

template <typename T = dtwc::settings::default_data_t>
T twe(const std::vector<T> &x, const std::vector<T> &y,
      T nu = static_cast<T>(0.001), T lambda = static_cast<T>(1.0))
{
  return twe<T>(std::span<const T>{x}, std::span<const T>{y}, nu, lambda);
}

template <typename T = dtwc::settings::default_data_t>
T missing(const std::vector<T> &x, const std::vector<T> &y,
          int band = settings::DEFAULT_BAND,
          core::MetricType metric = core::MetricType::L1)
{
  return missing<T>(std::span<const T>{x}, std::span<const T>{y}, band, metric);
}

template <typename T = dtwc::settings::default_data_t>
T arow(const std::vector<T> &x, const std::vector<T> &y,
       int band = settings::DEFAULT_BAND,
       core::MetricType metric = core::MetricType::L1)
{
  return arow<T>(std::span<const T>{x}, std::span<const T>{y}, band, metric);
}

} // namespace dtwc::distance

