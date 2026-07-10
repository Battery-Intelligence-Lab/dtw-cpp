/**
 * @file distance.hpp
 * @brief Additive distance facade for the public DTWC++ API.
 *
 * @details These wrappers provide a stable `dtwc::distance::*` namespace that
 * mirrors the user-facing Python and MATLAB distance surfaces without removing
 * the existing free functions. The individual functions remain the preferred
 * choice in tight loops; the dispatcher overload is a convenience layer for
 * one-off calls and examples.
 */

#pragma once

#include "settings.hpp"
#include "core/distance_semantics.hpp"
#include "core/dtw_options.hpp"
#include "core/variant_validation.hpp"
#include "core/msm.hpp"
#include "core/twe.hpp"
#include "missing_utils.hpp"
#include "soft_dtw.hpp"
#include "warping.hpp"
#include "warping_adtw.hpp"
#include "warping_ddtw.hpp"
#include "warping_missing.hpp"
#include "warping_missing_arow.hpp"
#include "warping_wdtw.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace dtwc::distance {

template <typename T = dtwc::settings::default_data_t>
T dtw(std::span<const T> x, std::span<const T> y,
      int band = settings::DEFAULT_BAND,
      core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<T>(x, y, band, static_cast<T>(-1), metric);
}

template <typename T = dtwc::settings::default_data_t>
T ddtw(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       core::MetricType metric = core::MetricType::L1)
{
  return ddtwBanded<T>(x, y, band, metric);
}

template <typename T = dtwc::settings::default_data_t>
T wdtw(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       T g = static_cast<T>(0.05))
{
  return wdtwBanded<T>(x, y, band, g);
}

template <typename T = dtwc::settings::default_data_t>
T adtw(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       T penalty = static_cast<T>(1.0))
{
  return adtwBanded<T>(x, y, band, penalty);
}

template <typename T = dtwc::settings::default_data_t>
T soft_dtw(std::span<const T> x, std::span<const T> y,
           T gamma = static_cast<T>(1.0))
{
  return dtwc::soft_dtw<T>(x, y, gamma);
}

template <typename T = dtwc::settings::default_data_t>
T missing(std::span<const T> x, std::span<const T> y,
          int band = settings::DEFAULT_BAND,
          core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_banded<T>(x, y, band, static_cast<T>(-1), metric);
}

template <typename T = dtwc::settings::default_data_t>
T arow(std::span<const T> x, std::span<const T> y,
       int band = settings::DEFAULT_BAND,
       core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_banded<T>(x, y, band, metric);
}

template <typename T = dtwc::settings::default_data_t>
T dtw(std::span<const T> x, std::span<const T> y,
      const core::DTWVariantParams &params,
      int band = settings::DEFAULT_BAND,
      core::MetricType metric = core::MetricType::L1,
      core::MissingStrategy missing_strategy = core::MissingStrategy::Error)
{
  core::validate_variant_params(params);
  core::validate_variant_missing_semantics(params, missing_strategy);
  switch (missing_strategy) {
  case core::MissingStrategy::ZeroCost:
    return missing<T>(x, y, band, metric);

  case core::MissingStrategy::AROW:
    return arow<T>(x, y, band, metric);

  case core::MissingStrategy::Interpolate: {
    auto xi = has_missing(x) ? interpolate_linear(x) : std::vector<T>(x.begin(), x.end());
    auto yi = has_missing(y) ? interpolate_linear(y) : std::vector<T>(y.begin(), y.end());
    return dtw<T>(std::span<const T>{xi}, std::span<const T>{yi}, band, metric);
  }

  case core::MissingStrategy::Error:
  default:
    break;
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
    return core::msm_distance<T>(x, y, static_cast<T>(params.msm_c));
  case core::DTWVariant::TWE:
    return core::twe_distance<T>(x, y, static_cast<T>(params.twe_nu),
                                 static_cast<T>(params.twe_lambda));
  case core::DTWVariant::Standard:
  default:
    return dtw<T>(x, y, band, metric);
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

/// Move-Split-Merge distance (Stefan et al. 2013), univariate/unbanded.
template <typename T = dtwc::settings::default_data_t>
T msm(std::span<const T> x, std::span<const T> y, T c = static_cast<T>(1.0))
{
  return core::msm_distance<T>(x, y, c);
}

template <typename T = dtwc::settings::default_data_t>
T msm(const std::vector<T> &x, const std::vector<T> &y, T c = static_cast<T>(1.0))
{
  return core::msm_distance<T>(std::span<const T>{x}, std::span<const T>{y}, c);
}

/// Time Warp Edit distance (Marteau 2009), univariate/unbanded.
template <typename T = dtwc::settings::default_data_t>
T twe(std::span<const T> x, std::span<const T> y,
      T nu = static_cast<T>(0.001), T lambda = static_cast<T>(1.0))
{
  return core::twe_distance<T>(x, y, nu, lambda);
}

template <typename T = dtwc::settings::default_data_t>
T twe(const std::vector<T> &x, const std::vector<T> &y,
      T nu = static_cast<T>(0.001), T lambda = static_cast<T>(1.0))
{
  return core::twe_distance<T>(std::span<const T>{x}, std::span<const T>{y}, nu, lambda);
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

