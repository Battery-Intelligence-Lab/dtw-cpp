/**
 * @file dtw_dispatch.cpp
 * @brief Implementation of resolve_dtw_fn<T> (see dtw_dispatch.hpp).
 */

#include "dtw_dispatch.hpp"

#include "../Problem.hpp"
#include "../missing_utils.hpp"      // has_missing, interpolate_linear
#include "../soft_dtw.hpp"           // soft_dtw
#include "../warping.hpp"            // dtwBanded, dtwBanded_mv
#include "../warping_adtw.hpp"       // adtwBanded, adtwBanded_mv
#include "../warping_ddtw.hpp"       // ddtwBanded, derivative_transform_mv_inplace
#include "../warping_missing.hpp"    // dtwMissing_banded, dtwMissing_banded_mv
#include "../warping_missing_arow.hpp" // dtwAROW_banded
#include "../warping_wdtw.hpp"       // wdtwBanded, wdtwBanded_mv, wdtw_weights
#include "dtw_options.hpp"           // DTWVariant, MissingStrategy

#include <algorithm>
#include <limits>
#include <span>
#include <vector>

namespace dtwc::core {

namespace {

// ----------------------------------------------------------------------------
// Missing-strategy lambdas. These are strategy-specific; the variant axis is
// suppressed (e.g. ZeroCost always runs the NaN-aware kernel, regardless of
// variant_params.variant — matches the pre-refactor behaviour).
// ----------------------------------------------------------------------------

template <typename T>
auto make_zero_cost(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  if (p.data.ndim > 1) {
    return [&p](std::span<const T> x, std::span<const T> y) -> double {
      const auto ndim = p.data.ndim;
      return static_cast<double>(dtwMissing_banded_mv<T>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, p.band));
    };
  }
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    return static_cast<double>(dtwMissing_banded<T>(x, y, p.band));
  };
}

template <typename T>
auto make_interpolate(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    auto xi = has_missing(x) ? interpolate_linear(x) : std::vector<T>(x.begin(), x.end());
    auto yi = has_missing(y) ? interpolate_linear(y) : std::vector<T>(y.begin(), y.end());
    return static_cast<double>(dtwBanded<T>(xi, yi, p.band));
  };
}

template <typename T>
auto make_arow(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  // AROW always uses the scalar path regardless of data.ndim — MV AROW is a
  // genuinely different recurrence (see Phase 3 deferred work). Preserved
  // verbatim from pre-refactor behaviour.
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    return static_cast<double>(dtwAROW_banded<T>(x, y, p.band));
  };
}

// ----------------------------------------------------------------------------
// Variant lambdas (for MissingStrategy::Error or unsupported strategies).
// ----------------------------------------------------------------------------

template <typename T>
auto make_standard(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  if (p.data.ndim > 1) {
    return [&p](std::span<const T> x, std::span<const T> y) -> double {
      const auto ndim = p.data.ndim;
      return static_cast<double>(dtwBanded_mv<T>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, p.band));
    };
  }
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    return static_cast<double>(dtwBanded<T>(x, y, p.band));
  };
}

template <typename T>
auto make_ddtw(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  if (p.data.ndim > 1) {
    return [&p](std::span<const T> x, std::span<const T> y) -> double {
      const auto ndim = p.data.ndim;
      thread_local std::vector<T> dx, dy;
      derivative_transform_mv_inplace(x, ndim, dx);
      derivative_transform_mv_inplace(y, ndim, dy);
      return static_cast<double>(dtwBanded_mv<T>(
        dx.data(), dx.size() / ndim, dy.data(), dy.size() / ndim, ndim, p.band));
    };
  }
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    return static_cast<double>(ddtwBanded<T>(x, y, p.band));
  };
}

// WDTW f64 path: reads the Problem-local, lock-free cache populated serially
// by refresh_variant_caches() before parallel fill.
inline auto make_wdtw_f64(const Problem &p)
  -> std::function<double(std::span<const data_t>, std::span<const data_t>)>
{
  if (p.data.ndim > 1) {
    return [&p](std::span<const data_t> x, std::span<const data_t> y) -> double {
      const auto ndim = p.data.ndim;
      const auto x_steps = x.size() / ndim;
      const auto y_steps = y.size() / ndim;
      if (x_steps == 0 || y_steps == 0) return std::numeric_limits<double>::max();
      const auto max_dev = std::max(x_steps, y_steps) - std::size_t{1};
      auto it = p.wdtw_weights_cache().find(max_dev);
      if (it == p.wdtw_weights_cache().end()) {
        // Cache miss (e.g. DBA centroid with novel length). Serial-only fallback.
        const auto g = static_cast<data_t>(p.variant_params.wdtw_g);
        auto w = wdtw_weights<data_t>(static_cast<int>(max_dev), g);
        return static_cast<double>(
          wdtwBanded_mv<data_t>(x.data(), x_steps, y.data(), y_steps, ndim, w, p.band));
      }
      return static_cast<double>(
        wdtwBanded_mv<data_t>(x.data(), x_steps, y.data(), y_steps, ndim, it->second, p.band));
    };
  }
  return [&p](std::span<const data_t> x, std::span<const data_t> y) -> double {
    const auto max_dev = std::max(x.size(), y.size());
    auto it = p.wdtw_weights_cache().find(max_dev);
    if (it == p.wdtw_weights_cache().end()) {
      const auto g = static_cast<data_t>(p.variant_params.wdtw_g);
      auto w = wdtw_weights<data_t>(static_cast<int>(max_dev), g);
      return static_cast<double>(wdtwBanded<data_t>(x, y, w, p.band));
    }
    return static_cast<double>(wdtwBanded<data_t>(x, y, it->second, p.band));
  };
}

// WDTW f32 path: the Problem-local cache is populated in data_t; a per-call
// materialisation to float would add hot-path overhead. f32 WDTW is an
// uncommon configuration (the primary f32 user is fast_clara's Parquet
// chunk reader which in practice runs Standard DTW). Route through the
// warping_wdtw overload that takes `g` directly; it uses its own
// thread-safe internal cache at `detail::cached_wdtw_weights<float>`.
inline auto make_wdtw_f32(const Problem &p)
  -> std::function<double(std::span<const float>, std::span<const float>)>
{
  const auto g = static_cast<float>(p.variant_params.wdtw_g);
  if (p.data.ndim > 1) {
    return [&p, g](std::span<const float> x, std::span<const float> y) -> double {
      const auto ndim = p.data.ndim;
      return static_cast<double>(wdtwBanded_mv<float>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, p.band, g));
    };
  }
  return [&p, g](std::span<const float> x, std::span<const float> y) -> double {
    return static_cast<double>(wdtwBanded<float>(x.data(), x.size(), y.data(), y.size(), p.band, g));
  };
}

template <typename T>
auto make_adtw(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  if (p.data.ndim > 1) {
    return [&p](std::span<const T> x, std::span<const T> y) -> double {
      const auto ndim = p.data.ndim;
      return static_cast<double>(adtwBanded_mv<T>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, p.band,
        static_cast<T>(p.variant_params.adtw_penalty)));
    };
  }
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    return static_cast<double>(adtwBanded<T>(
      x, y, p.band, static_cast<T>(p.variant_params.adtw_penalty)));
  };
}

template <typename T>
auto make_soft_dtw(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  // Soft-DTW has no MV kernel and ignores `band` (full-matrix only).
  // Pre-refactor behaviour: flat-vector treatment for MV. Preserved here.
  return [&p](std::span<const T> x, std::span<const T> y) -> double {
    return static_cast<double>(soft_dtw<T>(x, y, static_cast<T>(p.variant_params.sdtw_gamma)));
  };
}

// ----------------------------------------------------------------------------
// Per-T variant dispatch (WDTW has two specialisations for the weights cache).
// ----------------------------------------------------------------------------

template <typename T>
auto make_wdtw(const Problem &p)
  -> std::function<double(std::span<const T>, std::span<const T>)>;

template <>
inline auto make_wdtw<data_t>(const Problem &p)
  -> std::function<double(std::span<const data_t>, std::span<const data_t>)>
{
  return make_wdtw_f64(p);
}

template <>
inline auto make_wdtw<float>(const Problem &p)
  -> std::function<double(std::span<const float>, std::span<const float>)>
{
  return make_wdtw_f32(p);
}

} // unnamed namespace

// ----------------------------------------------------------------------------
// resolve_dtw_fn<T>
// ----------------------------------------------------------------------------

template <typename T>
std::function<double(std::span<const T>, std::span<const T>)>
resolve_dtw_fn(const Problem &p)
{
  // Missing-data strategies override variant dispatch — pre-refactor behaviour.
  switch (p.missing_strategy) {
  case MissingStrategy::ZeroCost:    return make_zero_cost<T>(p);
  case MissingStrategy::Interpolate: return make_interpolate<T>(p);
  case MissingStrategy::AROW:        return make_arow<T>(p);
  case MissingStrategy::Error:
  default:                           break; // fall through to variant switch
  }

  switch (p.variant_params.variant) {
  case DTWVariant::DDTW:    return make_ddtw<T>(p);
  case DTWVariant::WDTW:    return make_wdtw<T>(p);
  case DTWVariant::ADTW:    return make_adtw<T>(p);
  case DTWVariant::SoftDTW: return make_soft_dtw<T>(p);
  case DTWVariant::Standard:
  default:                  return make_standard<T>(p);
  }
}

// Explicit instantiations.
template std::function<double(std::span<const data_t>, std::span<const data_t>)>
resolve_dtw_fn<data_t>(const Problem &);
template std::function<double(std::span<const float>, std::span<const float>)>
resolve_dtw_fn<float>(const Problem &);

} // namespace dtwc::core
