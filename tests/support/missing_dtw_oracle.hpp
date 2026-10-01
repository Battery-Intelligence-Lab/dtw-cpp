/**
 * @file missing_dtw_oracle.hpp
 * @brief Independent oracle for DTW with missing (NaN) values: a plain full-matrix
 *        recurrence per strategy, sharing nothing with dtwc/.
 *
 * @details A pair is missing when x[i] or y[j] is NaN. The local cost is |x - y| (L1) or
 *          (x - y)^2 (SquaredL2) for an observed pair and 0 for a missing one.
 *          - ZeroCost: the standard recurrence, C(i,j) = cost + min(C(i-1,j-1), C(i-1,j), C(i,j-1)).
 *          - AROW (Yurtman et al., ECML-PKDD 2023): a missing pair adds nothing and may only
 *            be entered diagonally, C(i,j) = C(i-1,j-1). In the first row and column there
 *            is one predecessor and no stretching to forbid, so a missing pair there takes it
 *            (the diagonal does not exist) at zero cost.
 *          An empty series, or a band narrower than the length difference, is infinity.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace dtwc::test_support {

enum class MissingRule { ZeroCost, AROW };

inline double missing_dtw_oracle(MissingRule rule, std::span<const double> x, std::span<const double> y,
                                 bool squared = false, int band = -1)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  const std::size_t nx = x.size(), ny = y.size();
  if (nx == 0 || ny == 0) return inf;
  std::vector<double> table(nx * ny, inf);
  const auto at = [&](std::size_t i, std::size_t j) -> double & { return table[i * ny + j]; };
  for (std::size_t i = 0; i < nx; ++i)
    for (std::size_t j = 0; j < ny; ++j) {
      const std::size_t offset = i > j ? i - j : j - i;
      if (band >= 0 && offset > static_cast<std::size_t>(band)) continue;
      const bool missing = std::isnan(x[i]) || std::isnan(y[j]);
      const double diff = x[i] - y[j];
      const double cost = missing ? 0.0 : squared ? diff * diff : std::abs(diff);
      const double diag = i && j ? at(i - 1, j - 1) : inf;
      const double up = i ? at(i - 1, j) : inf;
      const double left = j ? at(i, j - 1) : inf;
      if (i == 0 && j == 0) at(i, j) = cost;
      else if (rule == MissingRule::AROW && missing)
        at(i, j) = std::isfinite(diag) ? diag : std::isfinite(up) ? up : left;
      else at(i, j) = cost + std::min(std::min(diag, up), left);
    }
  return at(nx - 1, ny - 1);
}

/// The library's sentinel for a window with no path is max(); the oracle's is infinity.
inline bool same_distance(double got, double want)
{
  return std::isinf(want) ? got == std::numeric_limits<double>::max() : got == want;
}

/// n steps of small integers, so that every sum is exact and a library result must equal
/// the oracle's bit for bit; the steps whose bit is set in `mask` are NaN.
inline std::vector<double> masked_series(std::size_t n, unsigned long long mask, unsigned salt)
{
  std::vector<double> series(n);
  for (std::size_t i = 0; i < n; ++i)
    series[i] = (mask >> i) & 1 ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>((3 * i + salt) % 7);
  return series;
}

/// Calls f(x, y) for every pair of NaN patterns of the shapes 4x4, 3x5 and 1x4 (every
/// way a missing step can sit at an end, in a gap or alone), then for two longer pairs
/// with gaps of several steps and many more rows than the rolling buffer is wide.
template <class F>
void for_each_masked_pair(F &&f)
{
  constexpr std::pair<unsigned, unsigned> shapes[] = { { 4, 4 }, { 3, 5 }, { 1, 4 } };
  for (const auto [nx, ny] : shapes)
    for (unsigned mx = 0; mx < (1u << nx); ++mx)
      for (unsigned my = 0; my < (1u << ny); ++my) f(masked_series(nx, mx, 1), masked_series(ny, my, 4));
  f(masked_series(40, 0x9E3779B97F4A7C15ull, 1), masked_series(33, 0x00FF00FF0F0F3C3Cull, 4));
  f(masked_series(9, 0x155ull, 2), masked_series(60, 0xFFFF0000FFFF00F0ull, 5));
}

} // namespace dtwc::test_support
