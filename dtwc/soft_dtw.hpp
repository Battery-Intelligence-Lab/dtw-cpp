/**
 * @file soft_dtw.hpp
 * @brief Soft-DTW: a differentiable variant of Dynamic Time Warping.
 *
 * Soft-DTW replaces the hard min in the DTW recurrence with a differentiable
 * softmin operator, making the distance differentiable w.r.t. input series.
 * As gamma -> 0, Soft-DTW converges to standard DTW.
 *
 * Note: Soft-DTW can be NEGATIVE for identical series when gamma > 0.
 *
 * Input checks (warping.hpp explains the layering): soft_dtw() is the
 * unchecked per-pair value, reached checked through distance::soft_dtw.
 * soft_dtw_gradient() has no such twin and no per-pair
 * caller, so it checks its own input and rejects NaN and ±inf.
 *
 * Reference: Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss
 *            Function for Time-Series"
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "base/error.hpp"
#include "base/settings.hpp"

#include <vector>
#include <span>
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/dtw_kernel.hpp"   // dtw_kernel_linear, SoftCell
#include "core/dtw_cost.hpp"     // SpanL1Cost
#include "core/dtw_options.hpp"  // core::validate
#include "warping.hpp"           // detail::require_finite

namespace dtwc {

namespace detail {

/**
 * Internal soft minimum for callers that already validated gamma once at
 * their public boundary. Keeping this primitive non-throwing avoids repeating
 * the finite-positive domain check in every dynamic-programming cell.
 */
template <typename T>
T softmin_gamma_unchecked(
  T a, T b, T c, const core::detail::SoftGammaScale<T> &scale) noexcept
{
  const T minimum = std::min(a, std::min(b, c));

  // Retain the original reciprocal-multiply operation order for ordinary
  // gamma values. At the valid denormal boundary the reciprocal overflows;
  // normalise the denominator before scaling so -freciprocal-math cannot turn
  // the fallback back into 0 * Inf. frexp(gamma)=fraction*2^exponent.
  if (!scale.finite_reciprocal) {
    return minimum - scale.gamma * std::log(
                                     std::exp(-scale.scaled(a - minimum)) +
                                     std::exp(-scale.scaled(b - minimum)) +
                                     std::exp(-scale.scaled(c - minimum)));
  }
  return minimum - scale.gamma * std::log(
                                   std::exp(-(a - minimum) * scale.inv_gamma) +
                                   std::exp(-(b - minimum) * scale.inv_gamma) +
                                   std::exp(-(c - minimum) * scale.inv_gamma));
}

} // namespace detail

/**
 * @brief Compute Soft-DTW distance between two time series.
 *
 * Uses L1 (absolute difference) as the pointwise cost. Runs dtw_kernel_linear
 * (one rolling column, O(min(n, m)) memory) with `core::SpanL1Cost<T>` +
 * `core::SoftCell<T>{gamma}`. Out-of-bounds predecessors (`maxValue` sentinel)
 * are excluded from the LSE inside `SoftCell::combine`, so first-row/column
 * cells reduce to `predecessor + cost`, the hard accumulation
 * soft_dtw_gradient() uses there.
 *
 * @tparam T Floating point type (default: `settings::default_data_t`, currently `double`).
 * @param x First time series.
 * @param y Second time series.
 * @param gamma Smoothing parameter, > 0 (unchecked). As gamma -> 0, result
 *              converges to standard DTW distance.
 * @return The Soft-DTW distance.
 */
template <typename T = dtwc::settings::default_data_t>
T soft_dtw(std::span<const T> x, std::span<const T> y, T gamma = T(1))
{
  // No shortcut for a series against itself: its Soft-DTW is not 0. An empty
  // series has no path: dtw_kernel_linear returns max().
  const T *xs = x.data(), *ys = y.data();
  std::size_t nx = x.size(), ny = y.size();
  core::orient(xs, nx, ys, ny);
  return core::dtw_kernel_linear<T>(nx, ny, core::SpanL1Cost<T>{ xs, ys }, core::SoftCell<T>{ gamma });
}

/**
 * @brief Compute the gradient of Soft-DTW w.r.t. the first time series.
 *
 * Uses the backward pass from Cuturi & Blondel (2017) to compute the alignment
 * matrix E, then derives the gradient from E and the pointwise cost derivatives.
 *
 * The backward recurrence for E:
 *   E(m-1, n-1) = 1
 *   For each (i,j), E(i,j) accumulates contributions from cells (i',j') where
 *   (i,j) is a predecessor, weighted by the softmin Jacobian.
 *
 * @tparam T Floating point type (default: `settings::default_data_t`, currently `double`).
 * @param x First time series (gradient is w.r.t. this).
 * @param y Second time series.
 * @param gamma Smoothing parameter (must be > 0).
 * @return Gradient vector of size x.size().
 */
template <typename T = dtwc::settings::default_data_t>
std::vector<T> soft_dtw_gradient(std::span<const T> x, std::span<const T> y, T gamma = T(1))
{
  core::validate({ .variant = { .variant = core::DTWVariant::SoftDTW, .sdtw_gamma = gamma } },
                 std::is_same_v<T, float>);

  const std::size_t mx = x.size();
  const std::size_t my = y.size();

  // Refuses an empty x or y too, which x[0] and y[0] below would read past.
  detail::require_finite<T>(x, y, "soft_dtw_gradient");

  // The cost matrix C (forward pass) and the alignment matrix E (backward
  // pass), column-major: the backward pass reads C at every successor and the
  // gradient sums each row of E, so both are kept whole. Grown to the exact
  // size (reserve: resize alone may overshoot), never shrunk: a warmed thread
  // allocates nothing.
  thread_local std::vector<T> c_buf, e_buf;
  for (auto *buffer : { &c_buf, &e_buf })
    if (buffer->size() < mx * my) {
      buffer->reserve(mx * my);
      buffer->resize(mx * my);
    }
  const auto C = [c = c_buf.data(), mx](std::size_t i, std::size_t j) -> T & { return c[i + j * mx]; };
  const auto E = [e = e_buf.data(), mx](std::size_t i, std::size_t j) -> T & { return e[i + j * mx]; };

  // Forward pass: compute cost matrix C
  auto dist = [](T a, T b) -> T { return std::abs(a - b); };
  const core::detail::SoftGammaScale<T> gamma_scale{gamma};

  C(0, 0) = dist(x[0], y[0]);

  for (std::size_t i = 1; i < mx; ++i)
    C(i, 0) = C(i - 1, 0) + dist(x[i], y[0]);

  for (std::size_t j = 1; j < my; ++j)
    C(0, j) = C(0, j - 1) + dist(x[0], y[j]);

  for (std::size_t j = 1; j < my; ++j) {
    for (std::size_t i = 1; i < mx; ++i) {
      C(i, j) = dist(x[i], y[j]) +
                detail::softmin_gamma_unchecked(
                  C(i - 1, j), C(i, j - 1), C(i - 1, j - 1), gamma_scale);
    }
  }

  // Backward pass: compute alignment matrix E.
  //
  // The Jacobian of softmin S = softmin(a,b,c) w.r.t. argument a is:
  //   dS/da = exp((S - a) / gamma)
  //
  // In the DTW recurrence C(i',j') = d(i',j') + S where S = softmin of predecessors,
  // so S = C(i',j') - d(i',j'). For predecessor (i,j) of successor (i',j'):
  //   weight = exp((C(i',j') - d(i',j') - C(i,j)) / gamma)
  //
  // E(i,j) = sum over successors (i',j') of: E(i',j') * weight
  //
  // Special cases: first row/col successors have only one predecessor each,
  // so the weight is 1.0 (the derivative of the identity).
  std::fill_n(e_buf.begin(), mx * my, T{0});
  E(mx - 1, my - 1) = T(1);

  const auto jacobian_weight = [&](T soft, T predecessor) noexcept {
    return std::exp(gamma_scale.scaled(soft - predecessor));
  };

  for (std::size_t j = my; j-- > 0;) {
    for (std::size_t i = mx; i-- > 0;) {
      if (i == mx - 1 && j == my - 1) continue; // already set

      T val = T(0);

      // Successor (i+1, j): (i,j) was "C(i-1,j)" predecessor at cell (i+1,j)
      if (i + 1 < mx) {
        if (j == 0) {
          // First column: C(i+1,0) = C(i,0) + d(...), only one predecessor, weight = 1
          val += E(i + 1, j);
        } else {
          const T S = C(i + 1, j) - dist(x[i + 1], y[j]); // softmin value at successor
          const T w = jacobian_weight(S, C(i, j));
          val += E(i + 1, j) * w;
        }
      }

      // Successor (i, j+1): (i,j) was "C(i,j-1)" predecessor at cell (i,j+1)
      if (j + 1 < my) {
        if (i == 0) {
          // First row: C(0,j+1) = C(0,j) + d(...), only one predecessor, weight = 1
          val += E(i, j + 1);
        } else {
          const T S = C(i, j + 1) - dist(x[i], y[j + 1]);
          const T w = jacobian_weight(S, C(i, j));
          val += E(i, j + 1) * w;
        }
      }

      // Successor (i+1, j+1): (i,j) was "C(i-1,j-1)" predecessor at cell (i+1,j+1)
      if (i + 1 < mx && j + 1 < my) {
        // Diagonal successor only exists for interior cells (i+1>=1 and j+1>=1)
        const T S = C(i + 1, j + 1) - dist(x[i + 1], y[j + 1]);
        const T w = jacobian_weight(S, C(i, j));
        val += E(i + 1, j + 1) * w;
      }

      E(i, j) = val;
    }
  }

  // Gradient w.r.t. x[i]:
  // d/dx[i] soft_dtw = sum_j E(i,j) * d/dx[i] |x[i] - y[j]|
  //                   = sum_j E(i,j) * sign(x[i] - y[j])
  std::vector<T> grad(mx, T(0));
  for (std::size_t i = 0; i < mx; ++i) {
    T g = T(0);
    for (std::size_t j = 0; j < my; ++j) {
      T diff = x[i] - y[j];
      T sign_val = (diff > T(0)) ? T(1) : ((diff < T(0)) ? T(-1) : T(0));
      g += E(i, j) * sign_val;
    }
    grad[i] = g;
  }

  return grad;
}

// Vector convenience overloads (vector -> span implicit conversion is non-deduced).
template <typename T = dtwc::settings::default_data_t>
T soft_dtw(const std::vector<T> &x, const std::vector<T> &y, T gamma = T(1))
{
  return soft_dtw<T>(std::span<const T>{x}, std::span<const T>{y}, gamma);
}

template <typename T = dtwc::settings::default_data_t>
std::vector<T> soft_dtw_gradient(const std::vector<T> &x, const std::vector<T> &y, T gamma = T(1))
{
  return soft_dtw_gradient<T>(std::span<const T>{x}, std::span<const T>{y}, gamma);
}

} // namespace dtwc

