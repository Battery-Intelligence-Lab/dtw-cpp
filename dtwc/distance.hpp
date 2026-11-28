/**
 * @file distance.hpp
 * @brief Additive distance facade for the public DTWC++ API.
 *
 * @details These wrappers provide a stable `dtwc::distance::*` namespace that
 * mirrors the user-facing Python and MATLAB distance surfaces without removing
 * the existing free functions. The individual functions remain the preferred
 * choice in tight loops; the dispatcher overload is a convenience layer for
 * one-off calls and examples. A series is any contiguous range of float or double
 * (core::SeriesPair): a std::vector, a std::span, a std::array or a C array, and the
 * type of the distance is its element type.
 *
 * This namespace is the checked boundary (warping.hpp explains the layering):
 * every function here checks its configuration (core::validate), then x and y
 * before any distance work, in O(n + m), throwing InvalidInput that names an
 * empty series, or the series and position of a NaN or ±inf.
 * `missing`, `arow` and the ZeroCost / AROW / Interpolate strategies read NaN
 * as a missing value and reject only ±inf. The unchecked per-pair wrappers
 * they call are unchanged.
 *
 * Band semantics: `band < -1` is invalid and throws `InvalidInput` via
 * `core::validate`. `band == -1` means unconstrained full DTW. When
 * `band >= 0 && band < |m - n|` (infeasible Sakoe-Chiba window for
 * unequal-length series), `dtwc::distance::*` returns `std::numeric_limits<T>::max()`
 * (`DBL_MAX` for `double`, `FLT_MAX` for `float`).
 *
 * @author Volkan Kumtepeli
 * @date 14 Apr 2026
 */

#pragma once

#include "base/settings.hpp"
#include "cdist.hpp"
#include "core/dtw_options.hpp"
#include "core/msm.hpp"
#include "core/series.hpp"
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
#include <type_traits>
#include <vector>

namespace dtwc::distance {

/// @brief Compute checked banded DTW distance between two series.
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series.
/// @param y Second time series.
/// @param band Sakoe-Chiba half-width (-1 = unconstrained).
/// @param metric Pointwise distance metric (default: L1).
/// @return DTW distance, or `numeric_limits<T>::max()` for an infeasible band.
/// @throws InvalidInput on invalid band/metric, empty series, or non-finite values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> dtw(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                            core::MetricType metric = core::MetricType::L1)
{
  using T = core::series_value_t<X>;
  core::validate({ .metric = metric, .band = band }, std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::dtw");
  return dtwBanded(x, y, band, static_cast<T>(-1), metric);
}

/// @brief Compute checked Derivative DTW (DDTW) distance between two series.
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series.
/// @param y Second time series.
/// @param band Sakoe-Chiba half-width (-1 = unconstrained).
/// @param metric Pointwise distance metric (default: L1).
/// @return DDTW distance, or `numeric_limits<T>::max()` for an infeasible band.
/// @throws InvalidInput on invalid band/metric, empty series, or non-finite values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> ddtw(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                             core::MetricType metric = core::MetricType::L1)
{
  using T = core::series_value_t<X>;
  core::validate({ .variant = { .variant = core::DTWVariant::DDTW }, .metric = metric, .band = band },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::ddtw");
  return ddtwBanded(x, y, band, metric);
}

/// @brief Compute checked Weighted DTW (WDTW) distance between two series.
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series.
/// @param y Second time series.
/// @param band Sakoe-Chiba half-width (-1 = unconstrained).
/// @param g Logistic weight steepness parameter (>= 0).
/// @return WDTW distance, or `numeric_limits<T>::max()` for an infeasible band.
/// @throws InvalidInput on invalid band or @p g, empty series, or non-finite values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> wdtw(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                             core::series_value_t<X> g = 0.05)
{
  using T = core::series_value_t<X>;
  core::validate({ .variant = { .variant = core::DTWVariant::WDTW, .wdtw_g = g }, .band = band },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::wdtw");
  return wdtwBanded(x, y, band, g);
}

/// @brief Compute checked Amerced DTW (ADTW) distance between two series.
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series.
/// @param y Second time series.
/// @param band Sakoe-Chiba half-width (-1 = unconstrained).
/// @param penalty Additive penalty for non-diagonal warping steps (>= 0).
/// @return ADTW distance, or `numeric_limits<T>::max()` for an infeasible band.
/// @throws InvalidInput on invalid band or @p penalty, empty series, or non-finite values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> adtw(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                             core::series_value_t<X> penalty = 1.0)
{
  using T = core::series_value_t<X>;
  core::validate({ .variant = { .variant = core::DTWVariant::ADTW, .adtw_penalty = penalty }, .band = band },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::adtw");
  return adtwBanded(x, y, band, penalty);
}

/// @brief Compute checked Soft-DTW distance (Cuturi & Blondel 2017) between two series.
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series.
/// @param y Second time series.
/// @param gamma Smoothing parameter (> 0).
/// @return Soft-DTW distance (may be negative when @p gamma > 0).
/// @throws InvalidInput on non-positive/non-finite @p gamma, empty series, or non-finite values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> soft_dtw(const X &x, const Y &y, core::series_value_t<X> gamma = 1.0)
{
  using T = core::series_value_t<X>;
  core::validate({ .variant = { .variant = core::DTWVariant::SoftDTW, .sdtw_gamma = gamma } },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::soft_dtw");
  return dtwc::soft_dtw(x, y, gamma);
}

/// @brief Compute checked NaN-aware DTW distance with zero local cost on missing values.
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series (NaN marks missing samples).
/// @param y Second time series (NaN marks missing samples).
/// @param band Sakoe-Chiba half-width (-1 = unconstrained).
/// @param metric Pointwise distance metric (default: L1).
/// @return Missing-data DTW distance, or `numeric_limits<T>::max()` for an infeasible band.
/// @throws InvalidInput on invalid band/metric, empty series, or +/-inf values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> missing(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                                core::MetricType metric = core::MetricType::L1)
{
  using T = core::series_value_t<X>;
  core::validate({ .metric = metric, .missing = core::MissingStrategy::ZeroCost, .band = band },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::missing", /*nan_is_missing=*/true);
  return dtwMissing_banded(x, y, band, static_cast<T>(-1), metric);
}

/// @brief Compute checked DTW-AROW distance (diagonal-only alignment across NaN values).
/// @tparam X First series type (core::SeriesPair).
/// @tparam Y Second series type matching @p X (core::SeriesPair).
/// @param x First time series (NaN marks missing samples).
/// @param y Second time series (NaN marks missing samples).
/// @param band Sakoe-Chiba half-width (-1 = unconstrained).
/// @param metric Pointwise distance metric (default: L1).
/// @return DTW-AROW distance, or `numeric_limits<T>::max()` for an infeasible band.
/// @throws InvalidInput on invalid band/metric, empty series, or +/-inf values.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> arow(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                             core::MetricType metric = core::MetricType::L1)
{
  using T = core::series_value_t<X>;
  core::validate({ .metric = metric, .missing = core::MissingStrategy::AROW, .band = band },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::arow", /*nan_is_missing=*/true);
  return dtwAROW_banded(x, y, band, metric);
}

/// Move-Split-Merge distance (Stefan et al. 2013), univariate/unbanded.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> msm(const X &x, const Y &y, core::series_value_t<X> c = 1.0)
{
  using T = core::series_value_t<X>;
  core::validate({ .variant = { .variant = core::DTWVariant::MSM, .msm_c = c } }, std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::msm");
  return core::msm_distance(x, y, c);
}

/// Time Warp Edit distance (Marteau 2009), univariate/unbanded.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> twe(const X &x, const Y &y, core::series_value_t<X> nu = 0.001,
                            core::series_value_t<X> lambda = 1.0)
{
  using T = core::series_value_t<X>;
  core::validate({ .variant = { .variant = core::DTWVariant::TWE, .twe_nu = nu, .twe_lambda = lambda } },
                 std::is_same_v<T, float>);
  dtwc::detail::require_finite<T>(x, y, "distance::twe");
  return core::twe_distance(x, y, nu, lambda);
}

/// The variant dispatcher. `metric` reaches Standard, DDTW and the missing-data
/// strategies; WDTW, ADTW, Soft-DTW, MSM and TWE throw InvalidInput for a
/// metric other than L1 (core::validate).
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> dtw(const X &x, const Y &y, const core::DTWVariantParams &params,
                            int band = settings::DEFAULT_BAND,
                            core::MetricType metric = core::MetricType::L1,
                            core::MissingStrategy missing_strategy = core::MissingStrategy::Error)
{
  using T = core::series_value_t<X>;
  core::validate({ params, metric, missing_strategy, band }, std::is_same_v<T, float>);
  // Each branch ends in a checked function above, which scans the input.
  // Interpolate first rejects ±inf, which interpolate_linear_into() would
  // spread into the gaps it fills.
  switch (missing_strategy) {
  case core::MissingStrategy::ZeroCost:
    return missing(x, y, band, metric);

  case core::MissingStrategy::AROW:
    return arow(x, y, band, metric);

  case core::MissingStrategy::Interpolate: {
    dtwc::detail::require_finite<T>(x, y, "distance::dtw", /*nan_is_missing=*/true);
    std::vector<T> x_buffer, y_buffer;
    return dtw(interpolate_linear_into<T>(x, x_buffer), interpolate_linear_into<T>(y, y_buffer), band, metric);
  }

  case core::MissingStrategy::Error:
    break;
  }

  switch (params.variant) {
  case core::DTWVariant::DDTW:
    return ddtw(x, y, band, metric);
  case core::DTWVariant::WDTW:
    return wdtw(x, y, band, static_cast<T>(params.wdtw_g));
  case core::DTWVariant::ADTW:
    return adtw(x, y, band, static_cast<T>(params.adtw_penalty));
  case core::DTWVariant::SoftDTW:
    return soft_dtw(x, y, static_cast<T>(params.sdtw_gamma));
  case core::DTWVariant::MSM:
    return msm(x, y, static_cast<T>(params.msm_c));
  case core::DTWVariant::TWE:
    return twe(x, y, static_cast<T>(params.twe_nu), static_cast<T>(params.twe_lambda));
  case core::DTWVariant::Standard:
    return dtw(x, y, band, metric);
  }
  throw std::logic_error("distance::dtw: unreachable DTWVariant");
}

using dtwc::cdist;

/// Cross-collection DTW distance matrix over vectors of series (zero-copy view Data).
template <typename T = dtwc::settings::default_data_t>
std::vector<double> cdist(const std::vector<std::vector<T>> &X,
                          const std::vector<std::vector<T>> &Y,
                          const core::DistanceConfig &distance = {},
                          Device device = Device::CPU,
                          int device_index = 0,
                          GpuPrecision gpu_precision = GpuPrecision::Auto,
                          double gpu_share = -1.0,
                          bool fallback = false)
{
  static_assert(std::is_same_v<T, data_t> || std::is_same_v<T, float>,
                "distance::cdist supports double (data_t) and float series.");
  const std::size_t ndim = distance.ndim;
  auto make_view = [ndim](const std::vector<std::vector<T>> &vecs,
                          std::vector<std::string> &names_buf) {
    names_buf.resize(vecs.size());
    std::vector<std::span<const T>> spans(vecs.size());
    std::vector<std::string_view> name_views(vecs.size());
    for (std::size_t i = 0; i < vecs.size(); ++i) {
      names_buf[i] = std::to_string(i);
      spans[i] = std::span<const T>(vecs[i]);
      name_views[i] = names_buf[i];
    }
    return Data(std::move(spans), std::move(name_views), ndim);
  };
  std::vector<std::string> x_names, y_names;
  const Data X_data = make_view(X, x_names);
  const Data Y_data = make_view(Y, y_names);
  return dtwc::cdist(X_data, Y_data, distance, device, device_index, gpu_precision, gpu_share, fallback);
}

} // namespace dtwc::distance

