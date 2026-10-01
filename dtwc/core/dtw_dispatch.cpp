/**
 * @file dtw_dispatch.cpp
 * @brief validate(DistanceConfig) and resolve_dtw_fn<T> (see dtw_dispatch.hpp).
 */

#include "dtw_dispatch.hpp"

#include "../Data.hpp"
#include "../base/missing_utils.hpp"      // has_missing, interpolate_linear
#include "../warping.hpp"            // dtwBanded, dtwBanded_mv
#include "../warping_adtw.hpp"       // adtwBanded, adtwBanded_mv
#include "../warping_ddtw.hpp"       // ddtwBanded, derivative_transform_mv_inplace
#include "../warping_missing.hpp"    // dtwMissing_banded, dtwMissing_banded_mv
#include "../warping_missing_arow.hpp" // dtwAROW_banded
#include "../warping_wdtw.hpp"       // wdtwBanded, wdtwBanded_mv, wdtw_weights
#include "dtw_cost.hpp"              // SpanMVAROW*Cost
#include "dtw_kernel.hpp"            // dtw_kernel_banded, AROWCell
#include "dtw_options.hpp"           // DistanceConfig, variant_names
#include "msm.hpp"                   // msm_distance
#include "public_distance.hpp"       // normalize_public_distance
#include "twe.hpp"                   // twe_distance
#include "../base/error.hpp"              // InvalidInput

#include <stdexcept>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace dtwc::core {

namespace {

// Every function below captures what it reads by value: the resolved function
// outlives and outmoves the configuration it was built from. validate() has
// accepted the configuration, so none of them checks it again.

// ----------------------------------------------------------------------------
// Missing-strategy lambdas. These are strategy-specific; the variant axis is
// suppressed (e.g. ZeroCost always runs the NaN-aware kernel, regardless of
// the variant — matches the pre-refactor behaviour).
// ----------------------------------------------------------------------------

template <typename T>
auto make_zero_cost(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  const int band = c.band;
  const MetricType metric = c.metric;
  if (c.ndim > 1) {
    return [band, metric, ndim = c.ndim](std::span<const T> x, std::span<const T> y) -> double {
      return normalize_public_distance(dtwMissing_banded_mv<T>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, band, T(-1), metric));
    };
  }
  return [band, metric](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(dtwMissing_banded<T>(x, y, band, T(-1), metric));
  };
}

// Univariate: validate() refuses it on ndim > 1.
template <typename T>
auto make_interpolate(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  return [band = c.band, metric = c.metric](std::span<const T> x, std::span<const T> y) -> double {
    auto xi = has_missing(x) ? interpolate_linear(x) : std::vector<T>(x.begin(), x.end());
    auto yi = has_missing(y) ? interpolate_linear(y) : std::vector<T>(y.begin(), y.end());
    return normalize_public_distance(dtwBanded<T>(xi, yi, band, T(-1), metric));
  };
}

template <typename T>
auto make_arow(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  // AROW (Yurtman 2023) via the unified kernel: a NaN-propagating cost and
  // AROWCell, which carries the diagonal predecessor when the cost is NaN.
  const int band = c.band;
  const MetricType metric = c.metric;
  if (c.ndim > 1) {
    // Per-channel skip for the cost; AROW only when a pair has no comparable
    // channel, so one channel is scalar AROW. L1 or squared L2: validate()
    // refuses L2, which has no multivariate AROW cost.
    return [band, metric, ndim = c.ndim](std::span<const T> x, std::span<const T> y) -> double {
      const auto x_steps = x.size() / ndim;
      const auto y_steps = y.size() / ndim;
      const bool swap = x_steps > y_steps;
      const T* a_data = swap ? y.data() : x.data();
      const T* b_data = swap ? x.data() : y.data();
      const auto a_steps = swap ? y_steps : x_steps;
      const auto b_steps = swap ? x_steps : y_steps;
      if (metric == MetricType::SquaredL2)
        return normalize_public_distance(dtw_kernel_banded<T>(
          a_steps, b_steps, band, SpanMVAROWSquaredL2Cost<T>{a_data, b_data, ndim}, AROWCell{}));
      return normalize_public_distance(dtw_kernel_banded<T>(
        a_steps, b_steps, band, SpanMVAROWL1Cost<T>{a_data, b_data, ndim}, AROWCell{}));
    };
  }
  return [band, metric](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(dtwAROW_banded<T>(x, y, band, metric));
  };
}

// ----------------------------------------------------------------------------
// Variant lambdas (for MissingStrategy::Error or unsupported strategies).
// ----------------------------------------------------------------------------

// Standard, DDTW and the missing-data strategies take the metric; validate()
// refuses another one for WDTW, ADTW, Soft-DTW, MSM and TWE, whose kernels
// compute L1.
template <typename T>
auto make_standard(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  const int band = c.band;
  const MetricType metric = c.metric;
  if (c.ndim > 1) {
    return [band, metric, ndim = c.ndim](std::span<const T> x, std::span<const T> y) -> double {
      return normalize_public_distance(dtwBanded_mv<T>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, band,
        T(-1), metric));
    };
  }
  return [band, metric](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(dtwBanded<T>(x, y, band, T(-1), metric));
  };
}

template <typename T>
auto make_ddtw(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  const int band = c.band;
  const MetricType metric = c.metric;
  if (c.ndim > 1) {
    return [band, metric, ndim = c.ndim](std::span<const T> x, std::span<const T> y) -> double {
      thread_local std::vector<T> dx, dy;
      derivative_transform_mv_inplace(x, ndim, dx);
      derivative_transform_mv_inplace(y, ndim, dy);
      return normalize_public_distance(dtwBanded_mv<T>(
        dx.data(), dx.size() / ndim, dy.data(), dy.size() / ndim, ndim, band, T(-1), metric));
    };
  }
  return [band, metric](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(ddtwBanded<T>(x, y, band, metric));
  };
}

// WDTW f64 path: the weights of every max_dev the series' lengths can give,
// built here, serially, and held by the function, so parallel callers only
// read them. max_dev = max(len_x, len_y) - 1 univariate (the canonical
// wdtwBanded(x, y, band, g) convention, Jeong et al. 2011: weights are indexed
// by |i-j| in [0, m-1]), max(steps_x, steps_y) - 1 multivariate.
inline auto make_wdtw_f64(const DistanceConfig &c, const Data &data)
  -> std::function<double(std::span<const data_t>, std::span<const data_t>)>
{
  const int band = c.band;
  const auto g = static_cast<data_t>(c.variant.wdtw_g);
  const std::size_t ndim = c.ndim;
  std::unordered_map<std::size_t, std::vector<data_t>> weights;
  for (std::size_t i = 0; i < data.size(); ++i) {
    const std::size_t steps = data.series_flat_size(i) / ndim;
    if (steps == 0) continue;
    weights.try_emplace(steps - 1, wdtw_weights<data_t>(static_cast<int>(steps - 1), g));
  }
  if (ndim > 1) {
    return [band, g, ndim, weights = std::move(weights)](
             std::span<const data_t> x, std::span<const data_t> y) -> double {
      const auto x_steps = x.size() / ndim;
      const auto y_steps = y.size() / ndim;
      if (x_steps == 0 || y_steps == 0) return std::numeric_limits<double>::max();
      const auto max_dev = std::max(x_steps, y_steps) - std::size_t{1};
      auto it = weights.find(max_dev);
      if (it == weights.end()) {
        // A length the series do not have (e.g. a DBA centroid): weights for this pair.
        auto w = wdtw_weights<data_t>(static_cast<int>(max_dev), g);
        return normalize_public_distance(
          wdtwBanded_mv<data_t>(x.data(), x_steps, y.data(), y_steps, ndim, w, band));
      }
      return normalize_public_distance(
        wdtwBanded_mv<data_t>(x.data(), x_steps, y.data(), y_steps, ndim, it->second, band));
    };
  }
  return [band, g, weights = std::move(weights)](
           std::span<const data_t> x, std::span<const data_t> y) -> double {
    const auto max_len = std::max(x.size(), y.size());
    if (max_len == 0)
      return std::numeric_limits<double>::max();
    const auto max_dev = max_len - 1;
    auto it = weights.find(max_dev);
    if (it == weights.end()) {
      auto w = wdtw_weights<data_t>(static_cast<int>(max_dev), g);
      return normalize_public_distance(wdtwBanded<data_t>(x, y, w, band));
    }
    return normalize_public_distance(wdtwBanded<data_t>(x, y, it->second, band));
  };
}

// WDTW f32 path: a per-call materialisation of the f64 weights to float would
// add hot-path overhead. f32 WDTW is an uncommon configuration (the primary f32
// user is fast_clara's Parquet chunk reader which in practice runs Standard
// DTW). Route through the warping_wdtw overload that takes `g` directly; it
// uses its own thread-local cache at `detail::cached_wdtw_weights<float>`.
inline auto make_wdtw_f32(const DistanceConfig &c)
  -> std::function<double(std::span<const float>, std::span<const float>)>
{
  const int band = c.band;
  const auto g = static_cast<float>(c.variant.wdtw_g);
  if (c.ndim > 1) {
    return [band, g, ndim = c.ndim](std::span<const float> x, std::span<const float> y) -> double {
      return normalize_public_distance(wdtwBanded_mv<float>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, band, g));
    };
  }
  return [band, g](std::span<const float> x, std::span<const float> y) -> double {
    return normalize_public_distance(
      wdtwBanded<float>(x.data(), x.size(), y.data(), y.size(), band, g));
  };
}

template <typename T>
auto make_adtw(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  const int band = c.band;
  const auto penalty = static_cast<T>(c.variant.adtw_penalty);
  if (c.ndim > 1) {
    return [band, penalty, ndim = c.ndim](std::span<const T> x, std::span<const T> y) -> double {
      return normalize_public_distance(adtwBanded_mv<T>(
        x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, band, penalty));
    };
  }
  return [band, penalty](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(adtwBanded<T>(x, y, band, penalty));
  };
}

template <typename T>
auto make_soft_dtw(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  // Soft-DTW (Cuturi & Blondel 2017) via the unified full-matrix kernel +
  // SoftCell (log-sum-exp with max-subtract stabilisation). Cross-validated
  // bit-for-bit against the legacy soft_dtw() on equal/different-length,
  // identical, and swap-symmetric inputs across gamma {0.1..10.0}
  // (unit_test_soft_dtw.cpp [phase3]).
  //
  // Univariate (validate() refuses ndim > 1), like soft_dtw_gradient() and
  // distance::soft_dtw. The band is intentionally ignored: soft-DTW is a full
  // O(n·m) recurrence here.
  return [gamma = static_cast<T>(c.variant.sdtw_gamma)](std::span<const T> x,
                                                         std::span<const T> y) -> double {
    const bool swap = x.size() > y.size();
    const auto a = swap ? y : x;
    const auto b = swap ? x : y;
    SpanL1Cost<T> cost{a.data(), b.data()};
    SoftCell<T> cell{gamma};
    return normalize_public_distance(
      dtw_kernel_full<T, SpanL1Cost<T>, SoftCell<T>>(
        a.size(), b.size(), cost, cell));
  };
}

template <typename T>
auto make_wdtw(const DistanceConfig &c, const Data &data)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  if constexpr (std::is_same_v<T, data_t>) return make_wdtw_f64(c, data);
  else                                     return make_wdtw_f32(c);
}

// MSM / TWE. Univariate (validate() refuses ndim > 1) and unbanded: the band is
// intentionally ignored, these are full O(n·m) elastic metrics here.
template <typename T>
auto make_msm(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  const T cost = static_cast<T>(c.variant.msm_c);
  return [cost](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(msm_distance<T>(x, y, cost));
  };
}

template <typename T>
auto make_twe(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  const T nu  = static_cast<T>(c.variant.twe_nu);
  const T lam = static_cast<T>(c.variant.twe_lambda);
  return [nu, lam](std::span<const T> x, std::span<const T> y) -> double {
    return normalize_public_distance(twe_distance<T>(x, y, nu, lam));
  };
}

// Independent multivariate DTW (DTW_I; Shokoohi-Yekta et al., DMKD 2017).
// Runs an independent univariate DTW per channel and sums (dtw_independent_mv).
// Standard DTW with MissingStrategy::Error only (validate() refuses the rest);
// resolve_dtw_fn takes this path only when ndim > 1.
template <typename T>
auto make_independent(const DistanceConfig &c)
  -> std::function<double(std::span<const T>, std::span<const T>)>
{
  return [band = c.band, metric = c.metric, ndim = c.ndim](std::span<const T> x,
                                                           std::span<const T> y) -> double {
    return normalize_public_distance(dtw_independent_mv<T>(
      x.data(), x.size() / ndim, y.data(), y.size() / ndim, ndim, band, metric));
  };
}

} // unnamed namespace

// ----------------------------------------------------------------------------
// validate
// ----------------------------------------------------------------------------

void validate(const DistanceConfig &c, bool f32)
{
  const auto &p = c.variant;
  // Each parameter, the variant that reads it, and whether 0 is in its domain:
  // WDTW's g = 0 (constant half weights) and ADTW's penalty = 0 (Standard DTW)
  // are valid limits.
  const struct
  {
    DTWVariant variant;
    const char *name;
    double value;
    bool zero_allowed;
  } parameters[]{
    { DTWVariant::WDTW, "WDTW g", p.wdtw_g, true },
    { DTWVariant::ADTW, "ADTW penalty", p.adtw_penalty, true },
    { DTWVariant::SoftDTW, "Soft-DTW gamma", p.sdtw_gamma, false },
    { DTWVariant::MSM, "MSM c", p.msm_c, false },
    { DTWVariant::TWE, "TWE nu", p.twe_nu, false },
    { DTWVariant::TWE, "TWE lambda", p.twe_lambda, false },
  };
  // Every parameter, the inactive ones too: a later change of variant must not
  // activate a value outside its domain.
  for (const auto &q : parameters)
    if (!std::isfinite(q.value) || q.value < 0 || (q.value == 0 && !q.zero_allowed))
      throw InvalidInput(std::string(q.name)
                         + (q.zero_allowed ? " must be finite and non-negative." : " must be finite and positive."));

  if (p.variant != DTWVariant::Standard && c.missing != MissingStrategy::Error)
    throw InvalidInput("Non-Standard DTW variants require MissingStrategy::Error.");

  // A float32 kernel reads the active parameters as floats: one that became 0
  // or inf there would change the recurrence (1/gamma = inf turns Soft-DTW into
  // NaN, the matrix's "not computed"). The range test comes first: casting a
  // value beyond float's range is undefined.
  if (f32)
    for (const auto &q : parameters)
      if (q.variant == p.variant
          && (q.value > std::numeric_limits<float>::max() || (q.value != 0 && static_cast<float>(q.value) == 0)))
        throw InvalidInput(std::string(q.name)
                           + " cannot be represented in float32 without becoming zero or non-finite.");

  if (c.ndim > 1) {
    if (p.mv_mode == MVMode::Independent) {
      if (p.variant != DTWVariant::Standard)
        throw InvalidInput("Independent multivariate mode is implemented for the Standard DTW variant only in "
                           "this release");
      if (c.missing != MissingStrategy::Error)
        throw InvalidInput("Independent multivariate mode does not support a missing-data strategy in this "
                           "release (set missing_strategy = Error)");
    }
    // No multivariate kernel; interpolation would fill a gap from the
    // neighbouring channel of the interleaved series.
    const char *feature = p.variant == DTWVariant::MSM       ? "MSM distance"
                        : p.variant == DTWVariant::TWE       ? "TWE distance"
                        : p.variant == DTWVariant::SoftDTW   ? "Soft-DTW"
                        : c.missing == MissingStrategy::Interpolate ? "MissingStrategy::Interpolate"
                        : c.missing == MissingStrategy::AROW && c.metric == MetricType::L2
                          ? "metric L2 with MissingStrategy::AROW"
                          : nullptr;
    if (feature) throw InvalidInput(std::string(feature) + " is univariate in this release (ndim must be 1)");
  }

  // WDTW, ADTW, Soft-DTW, MSM and TWE compute an L1 cost: another metric is
  // refused, never answered with the L1 distance.
  if (c.metric != MetricType::L1 && p.variant != DTWVariant::Standard && p.variant != DTWVariant::DDTW)
    throw InvalidInput(std::string("metric ") + (c.metric == MetricType::SquaredL2 ? "SquaredL2" : "L2")
                       + " is implemented for Standard DTW and DDTW only, but variant = "
                       + std::string(name_of(variant_names, p.variant))
                       + " was requested. Use metric L1 for this configuration.");
}

// ----------------------------------------------------------------------------
// resolve_dtw_fn<T>
// ----------------------------------------------------------------------------

template <typename T>
std::function<double(std::span<const T>, std::span<const T>)>
resolve_dtw_fn(const DistanceConfig &c, const Data &data)
{
  // Independent multivariate mode intercepts before every other axis: it is a
  // per-channel decomposition, not a cell-cost or missing-data choice.
  if (c.variant.mv_mode == MVMode::Independent && c.ndim > 1)
    return make_independent<T>(c);

  // Missing-data strategies override variant dispatch — pre-refactor behaviour.
  switch (c.missing) {
  case MissingStrategy::ZeroCost:    return make_zero_cost<T>(c);
  case MissingStrategy::Interpolate: return make_interpolate<T>(c);
  case MissingStrategy::AROW:        return make_arow<T>(c);
  case MissingStrategy::Error:       break; // fall through to variant switch
  }

  switch (c.variant.variant) {
  case DTWVariant::DDTW:    return make_ddtw<T>(c);
  case DTWVariant::WDTW:    return make_wdtw<T>(c, data);
  case DTWVariant::ADTW:    return make_adtw<T>(c);
  case DTWVariant::SoftDTW: return make_soft_dtw<T>(c);
  case DTWVariant::MSM:     return make_msm<T>(c);
  case DTWVariant::TWE:     return make_twe<T>(c);
  case DTWVariant::Standard: return make_standard<T>(c);
  }
  throw std::logic_error("resolve_dtw_fn: unreachable DTWVariant");
}

// Explicit instantiations.
template std::function<double(std::span<const data_t>, std::span<const data_t>)>
resolve_dtw_fn<data_t>(const DistanceConfig &, const Data &);
template std::function<double(std::span<const float>, std::span<const float>)>
resolve_dtw_fn<float>(const DistanceConfig &, const Data &);

} // namespace dtwc::core
