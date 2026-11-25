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
 *          guarded addition. This avoids overflow from `MAX + cost` and matches
 *          the no-neighbour sentinel convention used by the DTW kernels.
 *
 *          Scope: univariate, unbanded (`band == -1`); `Problem` ignores
 *          positive `band` settings when `DTWVariant::TWE` is selected.
 *
 * @author Volkan Kumtepeli
 * @date 2026-07-08
 */

#pragma once

#include "../base/settings.hpp"
#include "dtw_kernel.hpp" // orient
#include "series.hpp"     // SeriesPair

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace dtwc::core {

namespace detail {

  /// twe_distance's two rolling rows, one pair a T: reached here so that core::release_thread_scratch() can empty them
  /// (detail::dispatch_scratch, core/dtw_dispatch.cpp); two thread_locals returned as references, as detail::lanes_bufs (dtw_kernel.hpp) says why.
  template <typename T>
  std::pair<std::vector<T> &, std::vector<T> &> twe_rows()
  {
    thread_local std::vector<T> prev_buf, curr_buf;
    return { prev_buf, curr_buf };
  }

} // namespace detail

/// TWE distance (pointer + length). Univariate, unbanded, exact; nu, lambda > 0, unchecked.
template <typename T = dtwc::settings::default_data_t>
T twe_distance(const T *x, std::size_t nx, const T *y, std::size_t ny,
               T nu = T(0.001), T lambda = T(1))
{
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (nx == 0 || ny == 0) return maxValue;
  if (x == y && nx == ny) return T(0);

  // The rolling buffer spans the shorter series (TWE is symmetric).
  const T *a = x, *b = y;
  std::size_t na = nx, nb = ny;
  orient(b, nb, a, na); // a = long (rows), b = short (cols); a tie keeps a = x

  const T del_add = nu + lambda;
  const T two_nu = T(2) * nu;
  auto add = [](T u, T v) noexcept { return u == maxValue ? maxValue : u + v; };

  // Rolling buffers over the padded shorter axis (size nb+1).
  // Grown without shrinking so warmed worker threads neither reallocate nor re-zero prev_buf.
  auto [prev_buf, curr_buf] = detail::twe_rows<T>();
  if (prev_buf.size() < nb + 1) prev_buf.resize(nb + 1);
  if (curr_buf.size() < nb + 1) curr_buf.resize(nb + 1);
  T *prev = prev_buf.data();
  T *curr = curr_buf.data();

  // Padded row i = 0: D[0][0] = 0, rest +inf. Written into `curr` so the
  // first swap below moves it into `prev` (matches the MSM kernel pattern).
  curr[0] = T(0);
  for (std::size_t j = 1; j <= nb; ++j) curr[j] = maxValue;

  T ai_prev = T(0);
  for (std::size_t i = 1; i <= na; ++i) {
    std::swap(prev, curr);
    const T ai = a[i - 1];
    const T step_x = std::abs(ai_prev - ai) + del_add;
    T left = maxValue; // D[i][j-1], carried; D[i][0] = +inf for i > 0
    curr[0] = left;
    T bj_prev = T(0);
    for (std::size_t j = 1; j <= nb; ++j) {
      const T bj = b[j - 1];
      const T del_x = add(prev[j], step_x);
      const T del_y = add(left, std::abs(bj_prev - bj) + del_add);
      const std::size_t idx_diff = (i >= j) ? (i - j) : (j - i);
      const T match = add(prev[j - 1],
                          std::abs(ai - bj) + std::abs(ai_prev - bj_prev)
                            + two_nu * static_cast<T>(idx_diff));
      left = std::min(std::min(del_x, del_y), match); // std::min({...}) is a call on the MSVC STL
      curr[j] = left;
      bj_prev = bj;
    }
    ai_prev = ai;
  }
  return curr[nb];
}

/// TWE distance of two series (SeriesPair: a std::vector, std::span, std::array or C array of float or double).
template <typename X, typename Y>
  requires SeriesPair<X, Y>
series_value_t<X> twe_distance(const X &x, const Y &y, series_value_t<X> nu = 0.001, series_value_t<X> lambda = 1)
{
  using T = series_value_t<X>;
  const std::span<const T> xs{ x }, ys{ y };
  return twe_distance<T>(xs.data(), xs.size(), ys.data(), ys.size(), nu, lambda);
}

} // namespace dtwc::core
