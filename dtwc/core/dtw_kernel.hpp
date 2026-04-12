/**
 * @file dtw_kernel.hpp
 * @brief Unified scalar DTW kernel, parameterised on Cost + Cell policies.
 *
 * @details One kernel each for {Full matrix, Linear-space, Sakoe-Chiba banded}.
 *          Variants (Standard, ADTW, WDTW) pick a Cell policy + Cost functor and
 *          call the same kernel. The wavefront shape, rolling-buffer scratch,
 *          and early-abandon handling live here once — the old per-variant
 *          `_impl` helpers in warping.hpp / warping_adtw.hpp / warping_wdtw.hpp
 *          are collapsed into these three kernels.
 *
 *          Contracts:
 *            Cost: T operator()(size_t short_idx, size_t long_idx) const noexcept
 *                  — pointwise cost at the cell pairing short_idx ∈ [0, n_short)
 *                  with long_idx ∈ [0, n_long). Kernels always pass the short
 *                  index first so wrappers don't need to worry about loop
 *                  orientation.
 *            Cell: T combine(T diag, T up, T left, T cost,
 *                            size_t short_idx, size_t long_idx) const noexcept
 *                  — combines the three DP neighbours plus the pointwise cost.
 *                  Standard: min(diag, up, left) + cost.
 *                  ADTW: min(diag, up+penalty, left+penalty) + cost.
 *                  The "no neighbour" sentinel is std::numeric_limits<T>::max().
 *                  T seed(T cost, size_t short_idx, size_t long_idx) const noexcept
 *                  — value placed at DP cell (0, 0). Default policies return
 *                  `cost` directly. AROW overrides to return 0 when cost is
 *                  NaN (the "missing pair" sentinel) so the diagonal-carry
 *                  doesn't propagate NaN downstream.
 *
 *          @pre n_short <= n_long. Wrappers must swap (x, y) beforehand if
 *               needed and rebuild their Cost functor with the post-swap
 *               orientation (all current Cost functors are symmetric).
 *
 * @date 2026-04-12
 */

#pragma once

#include "scratch_matrix.hpp"

#include <algorithm>    // std::min, std::max
#include <cmath>        // std::ceil, std::floor, std::round
#include <cstddef>      // size_t
#include <limits>       // std::numeric_limits
#include <utility>      // std::pair
#include <vector>

namespace dtwc::core {

// ===========================================================================
// Cell policies
// ===========================================================================

/// Standard DTW recurrence: min(diag, up, left) + cost.
struct StandardCell {
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::min({diag, up, left}) + cost;
  }
  template <typename T>
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return cost;
  }
};

/// ADTW (Amerced DTW): penalty on non-diagonal (horizontal/vertical) steps.
///   dp[i,j] = min(dp[i-1,j-1], dp[i-1,j]+penalty, dp[i,j-1]+penalty) + cost
template <typename T>
struct ADTWCell {
  T penalty;
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::min({diag, up + penalty, left + penalty}) + cost;
  }
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return cost;
  }
};

/// Soft-DTW (Cuturi & Blondel 2017): softmin(diag, up, left) + cost with
/// log-sum-exp stabilisation (M = min of valid predecessors; exponents are
/// subtracted by M before exp/log).
///
/// Sentinel handling: out-of-bounds predecessors (== std::numeric_limits<T>::max())
/// are excluded from the LSE. This preserves the legacy `soft_dtw` boundary
/// semantics — first-row/column cells have exactly one valid predecessor, and
/// the softmin of a single value is that value, so `combine` reduces to
/// `left + cost` (first col) or `up + cost` (first row), matching the hard
/// accumulation used in `soft_dtw.hpp`.
///
/// Soft-DTW is inherently O(n*m) full-matrix (gradient backward pass needs
/// the full C); only `dtw_kernel_full` is meaningful for SoftCell.
template <typename T>
struct SoftCell {
  T gamma;
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    constexpr T maxValue = std::numeric_limits<T>::max();
    T m = maxValue;
    if (diag < m) m = diag;
    if (up   < m) m = up;
    if (left < m) m = left;
    if (m == maxValue) return maxValue;

    const T inv_gamma = T(1) / gamma;
    T acc = T(0);
    if (diag != maxValue) acc += std::exp(-(diag - m) * inv_gamma);
    if (up   != maxValue) acc += std::exp(-(up   - m) * inv_gamma);
    if (left != maxValue) acc += std::exp(-(left - m) * inv_gamma);
    return m - gamma * std::log(acc) + cost;
  }
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return cost;
  }
};

/// AROW (Yurtman 2023): diagonal-carry when the pointwise cost is NaN (the
/// "missing pair" sentinel from a NaN-propagating Cost functor). Otherwise
/// standard min-of-3 + cost. Boundary treatment: when only one predecessor is
/// available (i.e. first row/column), that predecessor is used; at (0,0) with
/// no predecessors, `seed(NaN, 0, 0)` returns 0 — the DP anchor.
struct AROWCell {
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    constexpr T maxValue = std::numeric_limits<T>::max();
    if (std::isnan(cost)) { // NaN -> missing pair.
      if (diag != maxValue) return diag;
      if (up   != maxValue) return up;
      if (left != maxValue) return left;
      return T(0);
    }
    const T m = std::min({diag, up, left});
    return (m == maxValue) ? maxValue : m + cost;
  }
  template <typename T>
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::isnan(cost) ? T(0) : cost;
  }
};

// ===========================================================================
// Shared band-bounds helper
// ===========================================================================

/// Column [lo, hi) range at banded-DTW row `row`, for a band walk with `slope`
/// (= (n_long-1)/(n_short-1)) and `window = max(band, slope/2)`.
inline std::pair<int, int> dtw_band_bounds(double slope, double window, int row) noexcept
{
  const double center = slope * row;
  const int lo = static_cast<int>(std::ceil(std::round(100.0 * (center - window)) / 100.0));
  const int hi = static_cast<int>(std::floor(std::round(100.0 * (center + window)) / 100.0)) + 1;
  return {lo, hi};
}

// ===========================================================================
// Kernel 1: full matrix (O(n*m) space). For reference / correctness.
// ===========================================================================

template <typename T, typename Cost, typename Cell>
T dtw_kernel_full(std::size_t n_short, std::size_t n_long, Cost cost, Cell cell)
{
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (n_short == 0 || n_long == 0) return maxValue;

  thread_local core::ScratchMatrix<T> C;
  C.resize(static_cast<int>(n_short), static_cast<int>(n_long));

  C(0, 0) = cell.seed(cost(0, 0), 0, 0);
  for (std::size_t i = 1; i < n_short; ++i)
    C(static_cast<int>(i), 0) = cell.combine(
        maxValue, C(static_cast<int>(i - 1), 0), maxValue, cost(i, 0), i, 0);
  for (std::size_t j = 1; j < n_long; ++j)
    C(0, static_cast<int>(j)) = cell.combine(
        maxValue, maxValue, C(0, static_cast<int>(j - 1)), cost(0, j), 0, j);

  for (std::size_t j = 1; j < n_long; ++j) {
    for (std::size_t i = 1; i < n_short; ++i) {
      const T diag = C(static_cast<int>(i - 1), static_cast<int>(j - 1));
      const T up   = C(static_cast<int>(i - 1), static_cast<int>(j));
      const T left = C(static_cast<int>(i), static_cast<int>(j - 1));
      C(static_cast<int>(i), static_cast<int>(j)) =
          cell.combine(diag, up, left, cost(i, j), i, j);
    }
  }
  return C(static_cast<int>(n_short - 1), static_cast<int>(n_long - 1));
}

// ===========================================================================
// Kernel 2: linear-space full-band DTW (outer = long, inner = short).
// Rolling buffer of size n_short. Optional early abandon.
// ===========================================================================

template <typename T, typename Cost, typename Cell>
T dtw_kernel_linear(std::size_t n_short, std::size_t n_long,
                    Cost cost, Cell cell, T early_abandon = T(-1))
{
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (n_short == 0 || n_long == 0) return maxValue;

  thread_local static std::vector<T> short_side;
  short_side.resize(n_short);

  // First column (long_idx = 0): accumulate along short axis, no diag/left.
  short_side[0] = cell.seed(cost(0, 0), 0, 0);
  for (std::size_t i = 1; i < n_short; ++i)
    short_side[i] = cell.combine(maxValue, short_side[i - 1], maxValue,
                                 cost(i, 0), i, 0);

  const bool do_early_abandon = (early_abandon >= T(0));

  for (std::size_t j = 1; j < n_long; ++j) {
    T diag = short_side[0];
    // First row of new column: no up, no diag (they'd be from out-of-bounds
    // previous column/row). Only `left` (short_side[0] == dp[0, j-1]) is valid.
    short_side[0] = cell.combine(maxValue, maxValue, short_side[0],
                                 cost(0, j), 0, j);

    T row_min = do_early_abandon ? short_side[0] : T(0);

    for (std::size_t i = 1; i < n_short; ++i) {
      const T old_left = short_side[i - 1]; // dp[i-1, j] — already updated
      const T old_up   = short_side[i];     // dp[i, j-1]
      const T next = cell.combine(diag, old_up, old_left, cost(i, j), i, j);
      diag = old_up;
      short_side[i] = next;
      if (do_early_abandon) row_min = std::min(row_min, next);
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return short_side.back();
}

// ===========================================================================
// Kernel 3: Sakoe-Chiba banded DTW (outer = short, inner = long).
// Rolling column of size n_long plus per-row band bounds. Optional early abandon.
// ===========================================================================

template <typename T, typename Cost, typename Cell>
T dtw_kernel_banded(std::size_t n_short, std::size_t n_long, int band,
                    Cost cost, Cell cell, T early_abandon = T(-1))
{
  constexpr T maxValue = std::numeric_limits<T>::max();
  if (n_short == 0 || n_long == 0) return maxValue;

  const int m_short = static_cast<int>(n_short);
  const int m_long  = static_cast<int>(n_long);

  // Degenerate: series of length 1 — fall back to linear (trivial path).
  if (m_short == 1 || m_long == 1 || m_long <= (band + 1))
    return dtw_kernel_linear<T>(n_short, n_long, cost, cell, early_abandon);

  const double slope  = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto   window = std::max(static_cast<double>(band), slope / 2);

  thread_local std::vector<T> col;
  col.assign(m_long, maxValue);
  thread_local std::vector<int> low_bounds, high_bounds;
  low_bounds.resize(m_short);
  high_bounds.resize(m_short);
  for (int row = 0; row < m_short; ++row) {
    auto [lo, hi] = dtw_band_bounds(slope, window, row);
    low_bounds[row]  = lo;
    high_bounds[row] = hi;
  }
  const bool do_early_abandon = (early_abandon >= T(0));

  // First short-step (j_short = 0): fill col along long axis. Only `left`
  // (col[i-1]) is available — diag and up are out-of-bounds (maxValue).
  col[0] = cell.seed(cost(0, 0), 0, 0);
  {
    const int hi = high_bounds[0];
    for (int i = 1; i < std::min(hi, m_long); ++i) {
      col[i] = cell.combine(maxValue, maxValue, col[i - 1],
                            cost(0, static_cast<std::size_t>(i)),
                            0, static_cast<std::size_t>(i));
    }
  }
  if (do_early_abandon && col[0] > early_abandon) return maxValue;

  for (int j = 1; j < m_short; ++j) {
    const int lo      = low_bounds[j];
    const int hi      = high_bounds[j];
    const int prev_lo = low_bounds[j - 1];
    const int prev_hi = high_bounds[j - 1];
    const int high    = std::min(hi, m_long);
    const int low     = std::max(lo, 0);

    T diag    = maxValue;
    T row_min = do_early_abandon ? maxValue : T(0);

    const int first_row = std::max(low, 1);
    if (first_row - 1 >= std::max(prev_lo, 0)
        && first_row - 1 < std::min(prev_hi, m_long)) {
      diag = col[first_row - 1];
    }

    if (low == 0) {
      // Row 0 of new column: only `left` (col[0] from previous j-step).
      // Also update `diag` to col[0] so the next iteration has a valid diag.
      diag   = col[0];
      col[0] = cell.combine(maxValue, maxValue, col[0],
                            cost(static_cast<std::size_t>(j), 0),
                            static_cast<std::size_t>(j), 0);
      if (do_early_abandon) row_min = col[0];
    }

    // Zero out cells that left the band on the low side (cells in the previous
    // column that don't have corresponding entries in the current band).
    for (int i = std::max(prev_lo, 0); i < std::min(low, std::min(prev_hi, m_long)); ++i)
      col[i] = maxValue;

    for (int i = first_row; i < high; ++i) {
      const T old_up = col[i];                      // dp[j-1, i]
      const T left   = col[i - 1];                  // dp[j, i-1] — already updated
      col[i] = cell.combine(diag, old_up, left,
                            cost(static_cast<std::size_t>(j),
                                 static_cast<std::size_t>(i)),
                            static_cast<std::size_t>(j),
                            static_cast<std::size_t>(i));
      diag = old_up;
      if (do_early_abandon) row_min = std::min(row_min, col[i]);
    }

    // Zero out cells that leave the band on the high side.
    for (int i = std::max(high, std::max(prev_lo, 0));
         i < std::min(prev_hi, m_long); ++i) {
      col[i] = maxValue;
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return col[m_long - 1];
}

} // namespace dtwc::core
