/**
 * @file msm.hpp
 * @brief Move-Split-Merge (MSM) elastic distance — standalone O(n·m) DP kernel.
 *
 * @details MSM (Stefan, Athitsos & Das, IEEE TKDE 2013) is an elastic distance
 *          that, unlike DTW, is a true metric and whose edit operations have a
 *          cost invariant to the absolute values being matched. In the 2024
 *          KAIS clustering evaluation (Holder, Middlehurst & Bagnall) MSM was
 *          the single best-performing distance for k-medoids clustering, where
 *          DTW is barely better than Euclidean — hence a quality lever, not a
 *          speed one.
 *
 *          Recurrence (matches aeon 1.5.0 `msm_distance`, univariate,
 *          independent, window=None, default c=1.0):
 *
 *            C(new, a, b) = c                             if a≤new≤b or a≥new≥b
 *                         = c + min(|new−a|, |new−b|)     otherwise
 *
 *            M[0][0] = |x₀−y₀|
 *            M[i][0] = M[i−1][0] + C(xᵢ, xᵢ₋₁, y₀)
 *            M[0][j] = M[0][j−1] + C(yⱼ, x₀, yⱼ₋₁)
 *            M[i][j] = min( M[i−1][j−1] + |xᵢ−yⱼ|,
 *                           M[i−1][j]   + C(xᵢ, xᵢ₋₁, yⱼ),
 *                           M[i][j−1]   + C(yⱼ, xᵢ, yⱼ₋₁) )
 *
 *          MSM is symmetric (a metric), so we orient n_short ≤ n_long and keep
 *          a rolling buffer of the shorter axis — O(min(n,m)) scratch, which
 *          matters for the long-series target (a full O(n·m) matrix would be
 *          hundreds of MB per thread at n≈8k).
 *
 *          v1 scope: univariate, unbanded (window=None). The Sakoe-Chiba band
 *          is NOT applied here — MSM/TWE ignore `Problem::band` (documented in
 *          the dispatch + CHANGELOG). The default build is unbanded anyway.
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
#include <limits>
#include <span>
#include <vector>

namespace dtwc::core {

/// MSM split/merge cost: inserting `nv` between neighbours `a` and `b`.
template <typename T>
inline T msm_cost(T nv, T a, T b, T c) noexcept
{
  if ((a <= nv && nv <= b) || (a >= nv && nv >= b)) return c;
  return c + std::min(std::abs(nv - a), std::abs(nv - b));
}

/// MSM distance (pointer + length). Univariate, unbanded, exact.
template <typename T = dtwc::settings::default_data_t>
T msm_distance(const T* x, std::size_t nx, const T* y, std::size_t ny, T c = T(1))
{
  validate_msm_c(c);
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (nx == 0 || ny == 0) return maxValue;
  if (x == y && nx == ny) return T(0);

  // Orient so the rolling buffer spans the shorter axis (MSM is symmetric).
  const T* a = x; const T* b = y; std::size_t na = nx, nb = ny;
  if (nb > na) { std::swap(a, b); std::swap(na, nb); } // a = long (rows), b = short (cols)

  thread_local std::vector<T> prev_buf, curr_buf;
  prev_buf.resize(nb);
  curr_buf.resize(nb);
  T* prev = prev_buf.data();
  T* curr = curr_buf.data();

  // Row i = 0.
  curr[0] = std::abs(a[0] - b[0]);
  for (std::size_t j = 1; j < nb; ++j)
    curr[j] = curr[j - 1] + msm_cost<T>(b[j], a[0], b[j - 1], c);

  // Rows i = 1 .. na-1.
  for (std::size_t i = 1; i < na; ++i) {
    std::swap(prev, curr);
    curr[0] = prev[0] + msm_cost<T>(a[i], a[i - 1], b[0], c);
    for (std::size_t j = 1; j < nb; ++j) {
      const T move  = prev[j - 1] + std::abs(a[i] - b[j]);
      const T del_a = prev[j]     + msm_cost<T>(a[i], a[i - 1], b[j], c);
      const T del_b = curr[j - 1] + msm_cost<T>(b[j], a[i], b[j - 1], c);
      curr[j] = std::min({move, del_a, del_b});
    }
  }
  return curr[nb - 1];
}

template <typename T = dtwc::settings::default_data_t>
T msm_distance(std::span<const T> x, std::span<const T> y, T c = T(1))
{
  return msm_distance<T>(x.data(), x.size(), y.data(), y.size(), c);
}

template <typename T = dtwc::settings::default_data_t>
T msm_distance(const std::vector<T>& x, const std::vector<T>& y, T c = T(1))
{
  return msm_distance<T>(x.data(), x.size(), y.data(), y.size(), c);
}

} // namespace dtwc::core
