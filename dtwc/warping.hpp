/**
 * @file warping.hpp
 * @brief Time warping functions.
 *
 * @details This file contains functions for dynamic time warping, which is a method to
 * compare two temporal sequences that may vary in time or speed. It includes
 * different versions of the algorithm for full, light (L), and banded computations.
 *
 * Each public function accepts an optional core::MetricType parameter (default L1),
 * which picks the Cost functor (core/dtw_cost.hpp) once, outside the kernel;
 * core::run_dtw is the per-pair entry every wrapper shares (empty input, a
 * series against itself, the orientation, the band).
 *
 * Unchecked input. These wrappers, those in warping_*.hpp, soft_dtw() and
 * core::msm_distance / twe_distance are the per-pair layer: the matrix fill
 * calls them once per pair, so they check neither their input nor their
 * parameters. They require finite values (the missing-data wrappers in
 * warping_missing*.hpp also take NaN, as a missing value) and parameters in
 * their domains; anything else comes back as NaN, as the unreachable max() or
 * as an ordinary-looking number, so a caller must check first — once per call
 * or once per fill, never per pair. The checked boundary is dtwc::distance::*
 * (distance.hpp) and soft_dtw_gradient(): each checks its configuration with
 * core::validate and runs detail::require_finite() below once per call.
 * The Python and MATLAB single-pair distance functions call that boundary.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 08 Dec 2022
 */

#pragma once

#include "base/error.hpp"              // for InvalidInput
#include "base/settings.hpp"           // for DEFAULT_BAND
#include "core/dtw_options.hpp"    // for core::MetricType
#include "core/dtw_kernel.hpp"     // unified DTW kernels, orient, run_dtw
#include "core/dtw_cost.hpp"       // Span*Cost functors

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <span>      // for span
#include <string>    // for string, to_string
#include <string_view>

namespace dtwc {

namespace detail {

/// The checked boundary's input test (see the file comment): throws
/// InvalidInput at the first value of `series` that no distance is defined for,
/// naming the series, the position and the fix. NaN passes only when
/// `nan_is_missing` (the missing-data distances); ±inf never does. O(n): run
/// once per public call, never per cell, nor per pair of a fill.
template <typename data_t>
void require_finite(std::span<const data_t> series, std::string_view name,
                    std::string_view where, bool nan_is_missing = false)
{
  for (std::size_t i = 0; i < series.size(); ++i) {
    const data_t value = series[i];
    if (std::isfinite(value) || (nan_is_missing && std::isnan(value))) continue;

    std::string message(where);
    message += ": ";
    message += name;
    message += "[" + std::to_string(i) + "] is ";
    if (std::isnan(value)) {
      message += "NaN. NaN is a missing value only to the missing-data distances "
                 "(missing, arow, or a ZeroCost, AROW or Interpolate missing "
                 "strategy): use one of those, or remove it.";
    } else {
      message += value > 0 ? "+inf" : "-inf";
      message += ". Distances are defined on finite values only: replace or "
                 "remove it (NaN marks a missing value for the missing-data "
                 "distances).";
    }
    throw InvalidInput(message);
  }
}

/// Both series of a pair, x first. Neither may be empty: an empty series has no
/// warping path, which the kernels return as max(), a finite number.
template <typename data_t>
void require_finite(std::span<const data_t> x, std::span<const data_t> y,
                    std::string_view where, bool nan_is_missing = false)
{
  if (x.empty() || y.empty())
    throw InvalidInput(std::string(where) + ": " + (x.empty() ? "x" : "y")
                       + " is empty; every series needs at least one value.");
  require_finite(x, "x", where, nan_is_missing);
  require_finite(y, "y", where, nan_is_missing);
}

} // namespace detail

// =========================================================================
//  Public API â€” pointer + length overloads (zero-copy for bindings)
// =========================================================================

/**
 * @brief Computes the banded DTW distance (pointer + length).
 *
 * @details Uses the canonical fixed Sakoe-Chiba window `|i-j| <= band`.
 *          A non-negative band narrower than `|nx-ny|` contains no endpoint-
 *          preserving path. A negative band requests unconstrained DTW.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param band Fixed diagonal half-width in samples; negative means
 *             unconstrained.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance, or
 *         `numeric_limits<data_t>::max()` for empty input, an infeasible
 *         window, or early abandonment. These cases share the finite sentinel.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded(const data_t* x, size_t nx, const data_t* y, size_t ny,
                 int band = settings::DEFAULT_BAND,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  // A univariate L2 cost is L1: sqrt((a-b)^2) == |a-b|.
  if (metric == core::MetricType::SquaredL2)
    return core::run_dtw<core::SpanSquaredL2Cost>(x, nx, y, ny, band, core::StandardCell{}, early_abandon);
  return core::run_dtw<core::SpanL1Cost>(x, nx, y, ny, band, core::StandardCell{}, early_abandon);
}

/**
 * @brief Computes the linear-space DTW distance (pointer + length).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull_L(const data_t* x, size_t nx, const data_t* y, size_t ny,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<data_t>(x, nx, y, ny, -1, early_abandon, metric);
}

/**
 * @brief Computes the full dynamic time warping distance (pointer + length):
 *        the v1 name of dtwFull_L, the same distance in linear space.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull(const data_t* x, size_t nx, const data_t* y, size_t ny,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwFull_L<data_t>(x, nx, y, ny, data_t(-1), metric);
}

// =========================================================================
//  Public API â€” vector overloads (forward to pointer versions)
// =========================================================================

/// Full DTW, the v1 name of dtwFull_L (span overload).
template <typename data_t>
data_t dtwFull(std::span<const data_t> x, std::span<const data_t> y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwFull<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

/// Linear-space DTW (span overload).
template <typename data_t>
data_t dtwFull_L(std::span<const data_t> x, std::span<const data_t> y,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwFull_L<data_t>(x.data(), x.size(), y.data(), y.size(), early_abandon, metric);
}

/// Banded DTW (span overload).
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded(std::span<const data_t> x, std::span<const data_t> y,
                 int band = settings::DEFAULT_BAND,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<data_t>(x.data(), x.size(), y.data(), y.size(), band, early_abandon, metric);
}

// Vector convenience overloads (vector -> span implicit conversion is non-deduced).
template <typename data_t>
data_t dtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwFull<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, metric);
}

template <typename data_t>
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwFull_L<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, early_abandon, metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 int band = settings::DEFAULT_BAND,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, band, early_abandon, metric);
}

// =========================================================================
//  Public API â€” multivariate (interleaved layout: [t0_f0, t0_f1, ..., t1_f0, ...])
// =========================================================================

/**
 * @brief Banded multivariate DTW (pointer + timestep counts + ndim).
 *
 * @details For ndim==1 delegates to the scalar dtwBanded. For ndim>1 the cost
 *          of a cell sums over the channels (SpanMV*Cost, pointer-stride
 *          indexing); L2 is the Euclidean norm over them.
 *          The fixed window is `|i-j| <= band`; a non-negative band narrower
 *          than `|nx_steps-ny_steps|` admits no endpoint-preserving path.
 *          Input layout is interleaved: x[t * ndim + d] is feature d at timestep t.
 *
 * @tparam data_t Data type of the elements.
 * @param x       Pointer to first series (interleaved).
 * @param nx_steps Number of timesteps in x.
 * @param y       Pointer to second series (interleaved).
 * @param ny_steps Number of timesteps in y.
 * @param ndim    Number of features per timestep.
 * @param band    Fixed Sakoe-Chiba diagonal half-width in timesteps; negative
 *                means unconstrained.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric  Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance, or
 *         `numeric_limits<data_t>::max()` for empty input, an infeasible
 *         window, or early abandonment. These cases share the finite sentinel.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, int band = settings::DEFAULT_BAND,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwBanded(x, nx_steps, y, ny_steps, band, early_abandon, metric);
  if (metric == core::MetricType::SquaredL2)
    return core::run_dtw<core::SpanMVSquaredL2Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{},
                                                    early_abandon, ndim);
  if (metric == core::MetricType::L2)
    return core::run_dtw<core::SpanMVL2Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{},
                                             early_abandon, ndim);
  return core::run_dtw<core::SpanMVL1Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{},
                                           early_abandon, ndim);
}

/**
 * @brief Linear-space multivariate DTW (pointer + timestep counts + ndim):
 *        dtwBanded_mv without a band.
 *
 * @tparam data_t Data type of the elements.
 * @param x       Pointer to first series (interleaved, nx_steps * ndim elements).
 * @param nx_steps Number of timesteps in x.
 * @param y       Pointer to second series (interleaved, ny_steps * ndim elements).
 * @param ny_steps Number of timesteps in y.
 * @param ndim    Number of features per timestep.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric  Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwFull_L_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded_mv<data_t>(x, nx_steps, y, ny_steps, ndim, -1, early_abandon, metric);
}

/**
 * @brief Independent multivariate DTW (DTW_I; Shokoohi-Yekta et al., DMKD 2017).
 *
 * @details Runs an independent univariate DTW on each channel and sums the
 *          per-channel distances: DTW_I = Σ_c DTW(x[:,c], y[:,c]). This differs
 *          from the dependent DTW_D (dtwBanded_mv / dtwFull_L_mv), which forces a
 *          single warping path shared by all channels. Neither mode dominates for
 *          clustering accuracy (Shokoohi-Yekta 2017) — both are provided. With an
 *          additive local cost and the same band, DTW_I ≤ DTW_D always (each
 *          channel is free to pick its own optimal path).
 *
 *          Input layout is interleaved: x[t * ndim + d] is feature d at timestep t
 *          (same as dtwFull_L_mv). Each channel runs dtwBanded (the linear-space
 *          kernel when band < 0). A non-negative band narrower than
 *          `|nx_steps-ny_steps|` returns the single finite no-path sentinel
 *          before channel summation. L1 / SquaredL2 metrics only (per-channel
 *          scalar).
 *
 * @tparam data_t Data type of the elements.
 * @param x       Pointer to first series (interleaved, nx_steps * ndim elements).
 * @param nx_steps Number of timesteps in x.
 * @param y       Pointer to second series (interleaved, ny_steps * ndim elements).
 * @param ny_steps Number of timesteps in y.
 * @param ndim    Number of features per timestep.
 * @param band    Fixed Sakoe-Chiba diagonal half-width in timesteps; negative
 *                means unconstrained.
 * @param metric  Pointwise distance metric (default: L1).
 * @return The summed independent multivariate DTW distance, or
 *         `numeric_limits<data_t>::max()` for empty input or an infeasible
 *         window.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtw_independent_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                          size_t ndim, int band = settings::DEFAULT_BAND,
                          core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwBanded<data_t>(x, nx_steps, y, ny_steps, band, -1, metric);
  if (nx_steps == 0 || ny_steps == 0) return std::numeric_limits<data_t>::max();
  if (band >= 0) {
    const auto min_steps = std::min(nx_steps, ny_steps);
    const auto max_steps = std::max(nx_steps, ny_steps);
    if (max_steps - min_steps > static_cast<size_t>(band))
      return std::numeric_limits<data_t>::max();
  }

  // De-interleave one channel at a time into contiguous scratch, then run the
  // univariate kernel. thread_local buffers keep the parallel matrix fill
  // heap-allocation-free after the first pair per thread.
  thread_local std::vector<data_t> cx, cy;
  cx.resize(nx_steps);
  cy.resize(ny_steps);

  data_t total = data_t(0);
  for (size_t c = 0; c < ndim; ++c) {
    for (size_t t = 0; t < nx_steps; ++t) cx[t] = x[t * ndim + c];
    for (size_t t = 0; t < ny_steps; ++t) cy[t] = y[t * ndim + c];
    total += dtwBanded<data_t>(cx.data(), nx_steps, cy.data(), ny_steps, band, -1, metric);
  }
  return total;
}

} // namespace dtwc

