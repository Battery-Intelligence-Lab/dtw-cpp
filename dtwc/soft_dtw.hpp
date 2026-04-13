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
 * Reference: Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss
 *            Function for Time-Series"
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <vector>
#include <span>
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

#include "core/scratch_matrix.hpp"
#include "core/dtw_kernel.hpp"   // dtw_kernel_full, SoftCell
#include "core/dtw_cost.hpp"     // SpanL1Cost

namespace dtwc {

/**
 * @brief Numerically stable softmin of three values using log-sum-exp trick.
 *
 * softmin_gamma(a, b, c, gamma) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
 *
 * Rewritten as: M - gamma * log(exp(-(a-M)/gamma) + exp(-(b-M)/gamma) + exp(-(c-M)/gamma))
 * where M = min(a, b, c), to avoid overflow.
 *
 * @tparam T Floating point type.
 * @param a First value.
 * @param b Second value.
 * @param c Third value.
 * @param gamma Smoothing parameter (must be > 0).
 * @return The soft minimum.
 */
template <typename T>
T softmin_gamma(T a, T b, T c, T gamma)
{
  assert(gamma > T(0) && "softmin_gamma requires gamma > 0");
  const T M = std::min(a, std::min(b, c));
  const T inv_gamma = T(1) / gamma;
  return M - gamma * std::log(
                        std::exp(-(a - M) * inv_gamma) +
                        std::exp(-(b - M) * inv_gamma) +
                        std::exp(-(c - M) * inv_gamma));
}

/**
 * @brief Compute Soft-DTW distance between two time series.
 *
 * Uses L1 (absolute difference) as the pointwise cost. Delegates to the
 * unified DTW kernel via `core::SpanL1Cost<T>` + `core::SoftCell<T>{gamma}`.
 * Out-of-bounds predecessors (`maxValue` sentinel) are excluded from the LSE
 * inside `SoftCell::combine`, so first-row/column cells reduce automatically
 * to `predecessor + cost` — matching the hard-accumulation boundary treatment
 * of the original implementation (cross-validated bit-for-bit in the Phase 3
 * fold and retained here).
 *
 * @tparam T Floating point type (default: double).
 * @param x First time series.
 * @param y Second time series.
 * @param gamma Smoothing parameter (must be > 0). As gamma -> 0, result
 *              converges to standard DTW distance.
 * @return The Soft-DTW distance.
 */
template <typename T = double>
T soft_dtw(std::span<const T> x, std::span<const T> y, T gamma = T(1))
{
  if (gamma <= T(0))
    throw std::invalid_argument("soft_dtw: gamma must be > 0");

  constexpr T maxValue = std::numeric_limits<T>::max();
  if (x.empty() || y.empty()) return maxValue;

  // Unified kernel precondition: n_short <= n_long. L1 + SoftCell are both
  // symmetric under (x, y) swap, so orienting here is a no-op on the result.
  const T* xs = x.data();
  const T* ys = y.data();
  std::size_t n_short = x.size();
  std::size_t n_long  = y.size();
  if (n_short > n_long) {
    std::swap(xs, ys);
    std::swap(n_short, n_long);
  }

  core::SpanL1Cost<T> cost{xs, ys};
  core::SoftCell<T> cell{gamma};
  return core::dtw_kernel_full<T>(n_short, n_long, cost, cell);
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
 * @tparam T Floating point type (default: double).
 * @param x First time series (gradient is w.r.t. this).
 * @param y Second time series.
 * @param gamma Smoothing parameter (must be > 0).
 * @return Gradient vector of size x.size().
 */
template <typename T = double>
std::vector<T> soft_dtw_gradient(std::span<const T> x, std::span<const T> y, T gamma = T(1))
{
  if (gamma <= T(0))
    throw std::invalid_argument("soft_dtw_gradient: gamma must be > 0");

  const auto mx = static_cast<int>(x.size());
  const auto my = static_cast<int>(y.size());

  assert(mx > 0 && my > 0);

  // Forward pass: compute cost matrix C
  thread_local core::ScratchMatrix<T> C;
  C.resize(mx, my);

  auto dist = [](T a, T b) -> T { return std::abs(a - b); };

  C(0, 0) = dist(x[0], y[0]);

  for (int i = 1; i < mx; ++i)
    C(i, 0) = C(i - 1, 0) + dist(x[i], y[0]);

  for (int j = 1; j < my; ++j)
    C(0, j) = C(0, j - 1) + dist(x[0], y[j]);

  for (int j = 1; j < my; ++j) {
    for (int i = 1; i < mx; ++i) {
      C(i, j) = dist(x[i], y[j]) +
                softmin_gamma(C(i - 1, j), C(i, j - 1), C(i - 1, j - 1), gamma);
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
  thread_local core::ScratchMatrix<T> E;
  E.resize(mx, my);
  E.fill(T{0});
  E(mx - 1, my - 1) = T(1);

  const T inv_gamma = T(1) / gamma;

  for (int j = my - 1; j >= 0; --j) {
    for (int i = mx - 1; i >= 0; --i) {
      if (i == mx - 1 && j == my - 1) continue; // already set

      T val = T(0);

      // Successor (i+1, j): (i,j) was "C(i-1,j)" predecessor at cell (i+1,j)
      if (i + 1 < mx) {
        if (j == 0) {
          // First column: C(i+1,0) = C(i,0) + d(...), only one predecessor, weight = 1
          val += E(i + 1, j);
        } else if (i + 1 >= 1) {
          const T S = C(i + 1, j) - dist(x[i + 1], y[j]); // softmin value at successor
          const T w = std::exp((S - C(i, j)) * inv_gamma);
          val += E(i + 1, j) * w;
        }
      }

      // Successor (i, j+1): (i,j) was "C(i,j-1)" predecessor at cell (i,j+1)
      if (j + 1 < my) {
        if (i == 0) {
          // First row: C(0,j+1) = C(0,j) + d(...), only one predecessor, weight = 1
          val += E(i, j + 1);
        } else if (j + 1 >= 1) {
          const T S = C(i, j + 1) - dist(x[i], y[j + 1]);
          const T w = std::exp((S - C(i, j)) * inv_gamma);
          val += E(i, j + 1) * w;
        }
      }

      // Successor (i+1, j+1): (i,j) was "C(i-1,j-1)" predecessor at cell (i+1,j+1)
      if (i + 1 < mx && j + 1 < my) {
        // Diagonal successor only exists for interior cells (i+1>=1 and j+1>=1)
        const T S = C(i + 1, j + 1) - dist(x[i + 1], y[j + 1]);
        const T w = std::exp((S - C(i, j)) * inv_gamma);
        val += E(i + 1, j + 1) * w;
      }

      E(i, j) = val;
    }
  }

  // Gradient w.r.t. x[i]:
  // d/dx[i] soft_dtw = sum_j E(i,j) * d/dx[i] |x[i] - y[j]|
  //                   = sum_j E(i,j) * sign(x[i] - y[j])
  std::vector<T> grad(mx, T(0));
  for (int i = 0; i < mx; ++i) {
    T g = T(0);
    for (int j = 0; j < my; ++j) {
      T diff = x[i] - y[j];
      T sign_val = (diff > T(0)) ? T(1) : ((diff < T(0)) ? T(-1) : T(0));
      g += E(i, j) * sign_val;
    }
    grad[i] = g;
  }

  return grad;
}

// Vector convenience overloads (vector -> span implicit conversion is non-deduced).
template <typename T = double>
T soft_dtw(const std::vector<T> &x, const std::vector<T> &y, T gamma = T(1))
{
  return soft_dtw<T>(std::span<const T>{x}, std::span<const T>{y}, gamma);
}

template <typename T = double>
std::vector<T> soft_dtw_gradient(const std::vector<T> &x, const std::vector<T> &y, T gamma = T(1))
{
  return soft_dtw_gradient<T>(std::span<const T>{x}, std::span<const T>{y}, gamma);
}

} // namespace dtwc
