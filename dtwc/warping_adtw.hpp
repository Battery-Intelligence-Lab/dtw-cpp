/**
 * @file warping_adtw.hpp
 * @brief Amerced Dynamic Time Warping (ADTW) — thin wrappers over the unified DTW kernel.
 *
 * @details ADTW adds a penalty for non-diagonal (horizontal/vertical) warping
 *          steps, discouraging time stretching/compression:
 *
 *            C(i,j) = d(x[i], y[j]) + min(C(i-1,j-1),
 *                                         C(i-1,j) + penalty,
 *                                         C(i,j-1) + penalty)
 *
 *          Implementation-wise, ADTW differs from Standard DTW only in the
 *          cell-recurrence rule — so every ADTW entry point is now a 5-line
 *          wrapper that calls `core::dtw_kernel_*` with `core::ADTWCell<T>`
 *          and an L1 / MV-L1 cost functor. The loop body itself lives in
 *          core/dtw_kernel.hpp (one copy, shared with Standard + WDTW).
 *
 *          Reference: Herrmann, M. & Shifaz, A. (2023), "Amercing: An intuitive
 *          and effective constraint for dynamic time warping." Pattern
 *          Recognition, 137, 109301.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "settings.hpp"
#include "warping.hpp"
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"

#include <cstddef>   // size_t
#include <limits>    // numeric_limits
#include <span>
#include <vector>

namespace dtwc {

// ---------------------------------------------------------------------------
// Scalar ADTW (univariate)
// ---------------------------------------------------------------------------

template <typename data_t>
data_t adtwFull_L(const data_t *x, size_t nx, const data_t *y, size_t ny,
                  data_t penalty, data_t early_abandon = data_t{-1})
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  return core::dtw_kernel_linear<data_t>(
      ns, nl,
      core::SpanL1Cost<data_t>{xs, ys},
      core::ADTWCell<data_t>{penalty},
      early_abandon);
}

template <typename data_t>
data_t adtwFull_L(std::span<const data_t> x, std::span<const data_t> y,
                  data_t penalty, data_t early_abandon = data_t{-1})
{
  return adtwFull_L(x.data(), x.size(), y.data(), y.size(), penalty, early_abandon);
}

template <typename data_t = double>
data_t adtwBanded(const data_t *x, size_t nx, const data_t *y, size_t ny,
                  int band, data_t penalty, data_t early_abandon = data_t{-1})
{
  if (band < 0) return adtwFull_L<data_t>(x, nx, y, ny, penalty, early_abandon);
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  return core::dtw_kernel_banded<data_t>(
      ns, nl, band,
      core::SpanL1Cost<data_t>{xs, ys},
      core::ADTWCell<data_t>{penalty},
      early_abandon);
}

template <typename data_t = double>
data_t adtwBanded(std::span<const data_t> x, std::span<const data_t> y,
                  int band, data_t penalty, data_t early_abandon = data_t{-1})
{
  return adtwBanded<data_t>(x.data(), x.size(), y.data(), y.size(), band, penalty, early_abandon);
}

// vector convenience overloads
template <typename data_t>
data_t adtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  data_t penalty, data_t early_abandon = data_t{-1})
{
  return adtwFull_L<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, penalty, early_abandon);
}

template <typename data_t = double>
data_t adtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t penalty, data_t early_abandon = data_t{-1})
{
  return adtwBanded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, band, penalty, early_abandon);
}

// ---------------------------------------------------------------------------
// Multivariate ADTW (interleaved layout: x[t*ndim + d])
// ---------------------------------------------------------------------------

template <typename data_t = double>
data_t adtwFull_L_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, data_t penalty = 1.0)
{
  if (ndim == 1) return adtwFull_L<data_t>(x, nx_steps, y, ny_steps, penalty);
  if (nx_steps == 0 || ny_steps == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx_steps == ny_steps) return 0;

  const bool swap = nx_steps > ny_steps;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny_steps : nx_steps;
  const size_t nl = swap ? nx_steps : ny_steps;

  return core::dtw_kernel_linear<data_t>(
      ns, nl,
      core::SpanMVL1Cost<data_t>{xs, ys, ndim},
      core::ADTWCell<data_t>{penalty});
}

/// Multivariate ADTW banded. With the unified kernel, banded MV is now a
/// first-class path — no fallback to unbanded.
template <typename data_t = double>
data_t adtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, int band = settings::DEFAULT_BAND, data_t penalty = 1.0)
{
  if (band < 0) return adtwFull_L_mv<data_t>(x, nx_steps, y, ny_steps, ndim, penalty);
  if (ndim == 1) return adtwBanded<data_t>(x, nx_steps, y, ny_steps, band, penalty);
  if (nx_steps == 0 || ny_steps == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx_steps == ny_steps) return 0;

  const bool swap = nx_steps > ny_steps;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny_steps : nx_steps;
  const size_t nl = swap ? nx_steps : ny_steps;

  return core::dtw_kernel_banded<data_t>(
      ns, nl, band,
      core::SpanMVL1Cost<data_t>{xs, ys, ndim},
      core::ADTWCell<data_t>{penalty});
}

} // namespace dtwc
