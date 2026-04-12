/**
 * @file warping_missing.hpp
 * @brief DTW with missing data (ZeroCost strategy) — thin wrappers over the
 *        unified core::dtw_kernel_*.
 *
 * @details Missing values (NaN) contribute 0 cost so the warping path can pass
 *          through missing regions without penalty. Recurrence is identical
 *          to Standard DTW:
 *
 *            C(i,j) = cost(x[i], y[j]) + min(C(i-1,j-1), C(i-1,j), C(i,j-1))
 *
 *          where `cost(a, b) = 0` if either is NaN, else the regular L1 /
 *          squared-L2 distance. All entry points route through the unified
 *          kernel with `core::SpanNanAwareL1Cost` (or its MV / SquaredL2
 *          variants) + `core::StandardCell`.
 *
 *          Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023),
 *          "Estimating DTW Distance Between Time Series with Missing Data."
 *          ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

#include "settings.hpp"
#include "warping.hpp"           // transitive dtwFull_L visibility for callers
#include "missing_utils.hpp"     // is_missing
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "core/dtw_options.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace dtwc {

namespace detail {

// ---------------------------------------------------------------------------
// Legacy NaN-aware (a, b) / (a, b, ndim) functors.
// Retained for direct-call call sites and tests that predate the unified
// Cost-functor interface. New code should prefer the index-based Span*
// functors in core/dtw_cost.hpp; the DTW kernel wrappers in this file route
// through those directly.
// ---------------------------------------------------------------------------

struct MissingL1Dist {
  template <typename T>
  T operator()(T a, T b) const {
    if (is_missing(a) || is_missing(b)) return T(0);
    return std::abs(a - b);
  }
};

struct MissingSquaredL2Dist {
  template <typename T>
  T operator()(T a, T b) const {
    if (is_missing(a) || is_missing(b)) return T(0);
    const auto d = a - b;
    return d * d;
  }
};

struct MissingMVL1Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      sum += std::abs(a[d] - b[d]);
    }
    return sum;
  }
};

struct MissingMVSquaredL2Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      const T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};

} // namespace detail

// =========================================================================
//  Scalar missing-data DTW (ZeroCost) — pointer + length entry points
// =========================================================================

template <typename data_t>
data_t dtwMissing_L(const data_t* x, size_t nx, const data_t* y, size_t ny,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  if (metric == core::MetricType::SquaredL2) {
    return core::dtw_kernel_linear<data_t>(
        ns, nl,
        core::SpanNanAwareSquaredL2Cost<data_t>{xs, ys},
        core::StandardCell{}, early_abandon);
  }
  return core::dtw_kernel_linear<data_t>(
      ns, nl,
      core::SpanNanAwareL1Cost<data_t>{xs, ys},
      core::StandardCell{}, early_abandon);
}

template <typename data_t>
data_t dtwMissing(const data_t* x, size_t nx, const data_t* y, size_t ny,
                  core::MetricType metric = core::MetricType::L1)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  if (metric == core::MetricType::SquaredL2) {
    return core::dtw_kernel_full<data_t>(
        ns, nl,
        core::SpanNanAwareSquaredL2Cost<data_t>{xs, ys},
        core::StandardCell{});
  }
  return core::dtw_kernel_full<data_t>(
      ns, nl,
      core::SpanNanAwareL1Cost<data_t>{xs, ys},
      core::StandardCell{});
}

template <typename data_t = double>
data_t dtwMissing_banded(const data_t* x, size_t nx, const data_t* y, size_t ny,
                         int band = settings::DEFAULT_BAND,
                         data_t early_abandon = -1,
                         core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwMissing_L<data_t>(x, nx, y, ny, early_abandon, metric);
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  if (metric == core::MetricType::SquaredL2) {
    return core::dtw_kernel_banded<data_t>(
        ns, nl, band,
        core::SpanNanAwareSquaredL2Cost<data_t>{xs, ys},
        core::StandardCell{}, early_abandon);
  }
  return core::dtw_kernel_banded<data_t>(
      ns, nl, band,
      core::SpanNanAwareL1Cost<data_t>{xs, ys},
      core::StandardCell{}, early_abandon);
}

// -------------------------------------------------------------------------
// Span overloads
// -------------------------------------------------------------------------

template <typename data_t>
data_t dtwMissing_L(std::span<const data_t> x, std::span<const data_t> y,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_L<data_t>(x.data(), x.size(), y.data(), y.size(),
                              early_abandon, metric);
}

template <typename data_t>
data_t dtwMissing(std::span<const data_t> x, std::span<const data_t> y,
                  core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

template <typename data_t = double>
data_t dtwMissing_banded(std::span<const data_t> x, std::span<const data_t> y,
                         int band = settings::DEFAULT_BAND,
                         data_t early_abandon = -1,
                         core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_banded<data_t>(x.data(), x.size(), y.data(), y.size(),
                                   band, early_abandon, metric);
}

// -------------------------------------------------------------------------
// Vector convenience overloads
// -------------------------------------------------------------------------

template <typename data_t>
data_t dtwMissing_L(const std::vector<data_t>& x, const std::vector<data_t>& y,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_L<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y},
                              early_abandon, metric);
}

template <typename data_t>
data_t dtwMissing(const std::vector<data_t>& x, const std::vector<data_t>& y,
                  core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, metric);
}

template <typename data_t = double>
data_t dtwMissing_banded(const std::vector<data_t>& x, const std::vector<data_t>& y,
                         int band = settings::DEFAULT_BAND,
                         data_t early_abandon = -1,
                         core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_banded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y},
                                   band, early_abandon, metric);
}

// =========================================================================
//  Multivariate missing-data DTW (interleaved layout)
// =========================================================================

template <typename data_t = double>
data_t dtwMissing_L_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                       size_t ndim, data_t early_abandon = -1,
                       core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwMissing_L<data_t>(x, nx_steps, y, ny_steps, early_abandon, metric);
  if (nx_steps == 0 || ny_steps == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx_steps == ny_steps) return 0;

  const bool swap = nx_steps > ny_steps;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny_steps : nx_steps;
  const size_t nl = swap ? nx_steps : ny_steps;

  if (metric == core::MetricType::SquaredL2) {
    return core::dtw_kernel_linear<data_t>(
        ns, nl,
        core::SpanMVNanAwareSquaredL2Cost<data_t>{xs, ys, ndim},
        core::StandardCell{}, early_abandon);
  }
  return core::dtw_kernel_linear<data_t>(
      ns, nl,
      core::SpanMVNanAwareL1Cost<data_t>{xs, ys, ndim},
      core::StandardCell{}, early_abandon);
}

/// Multivariate missing-data banded DTW. With the unified kernel, banded MV
/// is now a first-class path (previously fell back to unbanded).
template <typename data_t = double>
data_t dtwMissing_banded_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                            size_t ndim, int band = settings::DEFAULT_BAND,
                            data_t early_abandon = -1,
                            core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwMissing_L_mv<data_t>(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);
  if (ndim == 1) return dtwMissing_banded<data_t>(x, nx_steps, y, ny_steps, band, early_abandon, metric);
  if (nx_steps == 0 || ny_steps == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx_steps == ny_steps) return 0;

  const bool swap = nx_steps > ny_steps;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny_steps : nx_steps;
  const size_t nl = swap ? nx_steps : ny_steps;

  if (metric == core::MetricType::SquaredL2) {
    return core::dtw_kernel_banded<data_t>(
        ns, nl, band,
        core::SpanMVNanAwareSquaredL2Cost<data_t>{xs, ys, ndim},
        core::StandardCell{}, early_abandon);
  }
  return core::dtw_kernel_banded<data_t>(
      ns, nl, band,
      core::SpanMVNanAwareL1Cost<data_t>{xs, ys, ndim},
      core::StandardCell{}, early_abandon);
}

} // namespace dtwc
