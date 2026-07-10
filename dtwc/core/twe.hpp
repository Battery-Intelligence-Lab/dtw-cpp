/**
 * @file twe.hpp
 * @brief Time Warp Edit (TWE) elastic distance — standalone O(n·m) DP kernel.
 *
 * @details TWE (Marteau, IEEE TPAMI 2009) is an elastic metric combining time
 *          warping with an edit distance: a stiffness ν penalises matches
 *          proportionally to their index separation (like a soft warping
 *          window) and a constant λ penalises delete/insert. It is one of the
 *          strong elastic distances in the 2024 KAIS clustering evaluation.
 *
 *          Recurrence (matches aeon 1.5.0 `twe_distance`, univariate,
 *          window=None, defaults ν=0.001, λ=1.0). Both series are zero-padded
 *          at the front (x̂₀ = 0, x̂ᵢ = xᵢ₋₁); indices i,j run 1..n over the
 *          padded arrays, d(a,b) = |a−b|:
 *
 *            D[0][0] = 0,  D[0][j>0] = D[i>0][0] = +∞
 *            delete_x = D[i−1][j]   + |x̂ᵢ₋₁ − x̂ᵢ| + (ν+λ)
 *            delete_y = D[i][j−1]   + |ŷⱼ₋₁ − ŷⱼ| + (ν+λ)
 *            match    = D[i−1][j−1] + |x̂ᵢ − ŷⱼ| + |x̂ᵢ₋₁ − ŷⱼ₋₁| + 2ν·|i−j|
 *            D[i][j]  = min(delete_x, delete_y, match)
 *
 *          TWE is symmetric (a metric), so we orient n_short ≤ n_long and keep
 *          a rolling buffer of the (padded) shorter axis — O(min(n,m)) scratch.
 *          The `+∞` padding cells use the `numeric_limits::max()` sentinel with
 *          guarded addition (the build is -ffast-math; true infinities are not
 *          safe, same convention as the DTW kernels).
 *
 *          v1 scope: univariate, unbanded (window=None); MSM/TWE ignore
 *          `Problem::band` (documented in the dispatch + CHANGELOG).
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 4.8
 * @date 2026-07-08
 */

#pragma once

#include "../settings.hpp"
#include "variant_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>   // std::abs(long)
#include <limits>
#include <span>
#include <vector>

namespace dtwc::core {

/// TWE distance (pointer + length). Univariate, unbanded, exact.
template <typename T = dtwc::settings::default_data_t>
T twe_distance(const T* x, std::size_t nx, const T* y, std::size_t ny,
               T nu = T(0.001), T lambda = T(1))
{
  validate_twe_nu(nu);
  validate_twe_lambda(lambda);
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (nx == 0 || ny == 0) return maxValue;
  if (x == y && nx == ny) return T(0);

  // Orient so the rolling buffer spans the shorter axis (TWE is symmetric).
  const T* a = x; const T* b = y; std::size_t na = nx, nb = ny;
  if (nb > na) { std::swap(a, b); std::swap(na, nb); } // a = long (rows), b = short (cols)

  const T del_add = nu + lambda;
  // Padded accessors: index 0 -> 0.0 (the front pad), index k>0 -> series[k-1].
  auto ap = [a](std::size_t k) noexcept { return k == 0 ? T(0) : a[k - 1]; };
  auto bp = [b](std::size_t k) noexcept { return k == 0 ? T(0) : b[k - 1]; };
  auto add = [maxValue](T u, T v) noexcept { return u == maxValue ? maxValue : u + v; };

  // Rolling buffers over the padded shorter axis (size nb+1).
  thread_local std::vector<T> prev_buf, curr_buf;
  prev_buf.assign(nb + 1, maxValue);
  curr_buf.assign(nb + 1, maxValue);
  T* prev = prev_buf.data();
  T* curr = curr_buf.data();

  // Padded row i = 0: D[0][0] = 0, rest +inf. Written into `curr` so the
  // first swap below moves it into `prev` (matches the MSM kernel pattern).
  curr[0] = T(0);

  for (std::size_t i = 1; i <= na; ++i) {
    std::swap(prev, curr);
    curr[0] = maxValue; // D[i][0] = +inf for i > 0
    for (std::size_t j = 1; j <= nb; ++j) {
      const T del_x = add(prev[j],     std::abs(ap(i - 1) - ap(i)) + del_add);
      const T del_y = add(curr[j - 1], std::abs(bp(j - 1) - bp(j)) + del_add);
      const auto idx_pen = static_cast<T>(
          std::abs(static_cast<long long>(i) - static_cast<long long>(j)));
      const T match = add(prev[j - 1],
                          std::abs(ap(i) - bp(j)) + std::abs(ap(i - 1) - bp(j - 1))
                          + T(2) * nu * idx_pen);
      curr[j] = std::min({del_x, del_y, match});
    }
  }
  return curr[nb];
}

template <typename T = dtwc::settings::default_data_t>
T twe_distance(std::span<const T> x, std::span<const T> y,
               T nu = T(0.001), T lambda = T(1))
{
  return twe_distance<T>(x.data(), x.size(), y.data(), y.size(), nu, lambda);
}

template <typename T = dtwc::settings::default_data_t>
T twe_distance(const std::vector<T>& x, const std::vector<T>& y,
               T nu = T(0.001), T lambda = T(1))
{
  return twe_distance<T>(x.data(), x.size(), y.data(), y.size(), nu, lambda);
}

} // namespace dtwc::core
