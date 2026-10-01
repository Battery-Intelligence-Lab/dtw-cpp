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
 *          @pre n_short <= n_long: orient() puts a pair that way, and run_dtw()
 *               is the per-pair entry that does it before building the Cost
 *               (every Cost functor is symmetric).
 *
 *          dtw_kernel_lanes computes W equal-length pairs at once and packs
 *          the values itself, so it takes a pointwise distance instead of a
 *          Cost:  T operator()(T a, T b) const.
 *
 * @date 2026-04-12
 */

#pragma once

#include "scratch_matrix.hpp"

#include <algorithm>    // std::min, std::max
#include <array>        // std::array
#include <cmath>        // std::ceil, std::floor, std::round
#include <cstddef>      // size_t
#include <limits>       // std::numeric_limits
#include <utility>      // std::pair
#include <vector>

namespace dtwc::core {

// ===========================================================================
// Cell policies
// ===========================================================================

// The cells nest two-argument std::min, never std::min({...}): the MSVC STL
// compiles the initializer-list form to an out-of-line call per cell
// (__std_min_d). The nesting makes the comparisons min_element makes, so the
// result is unchanged, NaN ordering included.

/// Standard DTW recurrence: min(diag, up, left) + cost.
struct StandardCell {
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::min(std::min(diag, up), left) + cost;
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
    return std::min(std::min(diag, up + penalty), left + penalty) + cost;
  }
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return cost;
  }
};

namespace detail {

/** Precomputed gamma scaling for validated Soft-DTW inner loops. */
template <typename T>
struct SoftGammaScale {
  T gamma;
  T inv_gamma;
  bool finite_reciprocal;
  T gamma_fraction{T(1)};
  int gamma_exponent{0};

  explicit SoftGammaScale(T value) noexcept
    : gamma(value), inv_gamma(T(1) / value),
      finite_reciprocal(std::isfinite(inv_gamma))
  {
    if (!finite_reciprocal)
      gamma_fraction = std::frexp(gamma, &gamma_exponent);
  }

  T scaled(T delta) const noexcept
  {
    return finite_reciprocal
      ? delta * inv_gamma
      : std::scalbn(delta / gamma_fraction, -gamma_exponent);
  }
};

} // namespace detail

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
  detail::SoftGammaScale<T> scale;

  explicit SoftCell(T value) noexcept : gamma(value), scale(value) {}

  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    constexpr T maxValue = std::numeric_limits<T>::max();
    T m = maxValue;
    if (diag < m) m = diag;
    if (up   < m) m = up;
    if (left < m) m = left;
    if (m == maxValue) return maxValue;

    const auto contribution = [&](T predecessor) noexcept {
      return std::exp(-scale.scaled(predecessor - m));
    };
    T acc = T(0);
    if (diag != maxValue) acc += contribution(diag);
    if (up   != maxValue) acc += contribution(up);
    if (left != maxValue) acc += contribution(left);
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
    const T m = std::min(std::min(diag, up), left);
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

/// Canonical Sakoe-Chiba column range [lo, hi) at zero-based row `row`.
/// The adjustment window is |row-column| <= band. Bounds are clamped here so
/// callers never narrow sequence indices to `int` or overflow `row+band+1`.
inline std::pair<std::size_t, std::size_t>
dtw_band_bounds(int band, std::size_t row, std::size_t column_count) noexcept
{
  if (column_count == 0 || row >= column_count)
    return {column_count, column_count};
  if (band < 0) return {0, column_count};

  const auto width = static_cast<std::size_t>(band);
  const auto lo = (row > width) ? row - width : 0;
  // The subtraction is safe because row < column_count. Taking this branch
  // before forming row+width+1 makes the upper clamp overflow-proof.
  const auto hi = (width >= column_count - row - 1)
                    ? column_count
                    : row + width + 1;
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

  // As in the rolling kernels below: C(i-1, j) is carried in a register rather
  // than reloaded right after its store, and the column pointers are taken once
  // per column.
  for (std::size_t j = 1; j < n_long; ++j) {
    const T *prev = &C(0, static_cast<int>(j - 1));
    T *col = &C(0, static_cast<int>(j));
    T up = col[0];
    for (std::size_t i = 1; i < n_short; ++i) {
      const T diag = prev[i - 1];
      const T left = prev[i];
      up = cell.combine(diag, up, left, cost(i, j), i, j); // C(i, j), the next cell's up
      col[i] = up;
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

  thread_local static std::vector<T> short_buf;
  short_buf.resize(n_short);
  T *short_side = short_buf.data(); // hoisted, as in dtw_kernel_full

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
    T left = cell.combine(maxValue, maxValue, short_side[0], cost(0, j), 0, j);
    short_side[0] = left;

    T row_min = do_early_abandon ? left : T(0);

    for (std::size_t i = 1; i < n_short; ++i) {
      const T old_up = short_side[i]; // dp[i, j-1]
      // dp[i, j]; carried as the next cell's left (dp[i-1, j]) instead of reloaded.
      left = cell.combine(diag, old_up, left, cost(i, j), i, j);
      diag = old_up;
      short_side[i] = left;
      if (do_early_abandon) row_min = std::min(row_min, left);
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return short_side[n_short - 1];
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
  if (band < 0)
    return dtw_kernel_linear<T>(n_short, n_long, cost, cell, early_abandon);

  const auto band_width = static_cast<std::size_t>(band);

  // The fixed terminal cell must lie inside |short_idx-long_idx| <= band.
  if (n_long - n_short > band_width) return maxValue;

  // Degenerate length-one path, or a band covering the complete matrix.
  if (n_short == 1 || n_long == 1 || band_width >= n_long - 1)
    return dtw_kernel_linear<T>(n_short, n_long, cost, cell, early_abandon);

  thread_local std::vector<T> col_buf;
  col_buf.assign(n_long, maxValue);
  T *col = col_buf.data(); // hoisted, as in dtw_kernel_full
  thread_local std::vector<std::size_t> low_bounds, high_bounds;
  low_bounds.resize(n_short);
  high_bounds.resize(n_short);
  for (std::size_t row = 0; row < n_short; ++row) {
    auto [lo, hi] = dtw_band_bounds(band, row, n_long);
    low_bounds[row]  = lo;
    high_bounds[row] = hi;
  }
  const bool do_early_abandon = (early_abandon >= T(0));

  // First short-step (j_short = 0): fill col along long axis. Only `left`
  // (col[i-1]) is available — diag and up are out-of-bounds (maxValue).
  col[0] = cell.seed(cost(0, 0), 0, 0);
  {
    const auto hi = high_bounds[0];
    for (std::size_t i = 1; i < hi; ++i) {
      col[i] = cell.combine(maxValue, maxValue, col[i - 1],
                            cost(0, i), 0, i);
    }
  }
  if (do_early_abandon && col[0] > early_abandon) return maxValue;

  for (std::size_t j = 1; j < n_short; ++j) {
    const auto low     = low_bounds[j];
    const auto high    = high_bounds[j];
    const auto prev_lo = low_bounds[j - 1];
    const auto prev_hi = high_bounds[j - 1];

    T diag    = maxValue;
    T row_min = do_early_abandon ? maxValue : T(0);

    const auto first_row = std::max(low, std::size_t{1});
    if (first_row - 1 >= prev_lo && first_row - 1 < prev_hi) {
      diag = col[first_row - 1];
    }

    if (low == 0) {
      // Row 0 of new column: only `left` (col[0] from previous j-step).
      // Also update `diag` to col[0] so the next iteration has a valid diag.
      diag   = col[0];
      col[0] = cell.combine(maxValue, maxValue, col[0], cost(j, 0), j, 0);
      if (do_early_abandon) row_min = col[0];
    }

    // Zero out cells that left the band on the low side (cells in the previous
    // column that don't have corresponding entries in the current band).
    for (std::size_t i = prev_lo; i < std::min(low, prev_hi); ++i)
      col[i] = maxValue;

    T left = col[first_row - 1];                    // dp[j, first_row-1]
    for (std::size_t i = first_row; i < high; ++i) {
      const T old_up = col[i];                      // dp[j-1, i]
      // dp[j, i]; carried as the next cell's left instead of reloaded.
      left = cell.combine(diag, old_up, left, cost(j, i), j, i);
      col[i] = left;
      diag = old_up;
      if (do_early_abandon) row_min = std::min(row_min, left);
    }

    // Zero out cells that leave the band on the high side.
    for (std::size_t i = std::max(high, prev_lo); i < prev_hi; ++i) {
      col[i] = maxValue;
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return col[n_long - 1];
}

// ===========================================================================
// The per-pair entry: orientation, empty input, a series against itself.
// ===========================================================================

/// Orients a pair as the kernels take it: the shorter series first (x on a tie).
template <typename T>
void orient(const T *&x, std::size_t &nx, const T *&y, std::size_t &ny) noexcept
{
  if (nx > ny) {
    std::swap(x, y);
    std::swap(nx, ny);
  }
}

/// DTW of x and y under `cell`, the pointwise cost Cost<T>{x', y', extra...} on
/// the oriented pair (x', y'): what every per-pair wrapper runs. An empty series
/// has no path (max()); a series against itself is 0, the cost of the diagonal
/// path, as no cost or penalty is negative; dtw_kernel_banded takes the band
/// (negative: unconstrained) and the early-abandon threshold.
template <template <typename> class Cost, typename T, typename Cell, typename... Extra>
T run_dtw(const T *x, std::size_t nx, const T *y, std::size_t ny, int band, Cell cell,
          T early_abandon, Extra... extra)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<T>::max();
  if (x == y && nx == ny) return T(0);
  orient(x, nx, y, ny);
  return dtw_kernel_banded<T>(nx, ny, band, Cost<T>{ x, y, extra... }, cell, early_abandon);
}

// ===========================================================================
// Kernel 4: W equal-length pairs in SIMD lanes (outer = y, inner = x).
// The pairs (x, ys[w]) share x. Every lane evaluates dtw_kernel_linear's cells
// with its arithmetic in its order; with a band, dtw_kernel_banded's cells,
// transposed (outer y, not outer x), each from the same three neighbours. So
// each lane is the per-pair result for a Cell that treats `up` and `left`
// alike, as StandardCell does: bit for bit, unless the compiler contracts
// `d * d + m` into an FMA in one kernel and not the other (GCC does by default;
// the two then differ in the last bits). The loop over the lanes is the one that
// vectorises: one cache line of T per DP row, W dependency chains side by side
// where the per-pair kernel runs one.
// ===========================================================================

/// Pairs per dtw_kernel_lanes call: one 64-byte cache line of T.
template <typename T>
inline constexpr std::size_t dtw_lanes = 64 / sizeof(T);

/// DTW between x and each of ys[0 .. dtw_lanes<T>), all n samples long, with the
/// pointwise distance dist(x[i], y[j]); band < 0 is unconstrained, else the
/// Sakoe-Chiba band of dtw_band_bounds.
template <typename T, typename Dist, typename Cell>
std::array<T, dtw_lanes<T>> dtw_kernel_lanes(const T *x, const T *const *ys,
                                             std::size_t n, int band, Dist dist, Cell cell)
{
  constexpr std::size_t W = dtw_lanes<T>;
  constexpr T maxValue = std::numeric_limits<T>::max();
  std::array<T, W> out;
  if (n == 0) {
    out.fill(maxValue);
    return out;
  }

  // [n][W], one cache line per row: the W series interleaved, and the rolling
  // DP column (s[i].v[w] = dp[i, j] of pair w).
  struct alignas(64) Row { T v[W]; };
  thread_local std::vector<Row> y_buf, s_buf;
  if (y_buf.size() < n) {
    y_buf.resize(n);
    s_buf.resize(n);
  }
  Row *Y = y_buf.data();
  Row *s = s_buf.data();
  for (std::size_t w = 0; w < W; ++w) {
    const T *yw = ys[w];
    for (std::size_t t = 0; t < n; ++t) Y[t].v[w] = yw[t];
  }

  // Column 0 (y[0]). The rows above its band hold maxValue: the band only moves
  // up, so a later column first meets such a row as an unreachable `up`.
  const std::size_t hi0 = dtw_band_bounds(band, 0, n).second;
  for (std::size_t w = 0; w < W; ++w) s[0].v[w] = cell.seed(dist(x[0], Y[0].v[w]), 0, 0);
  for (std::size_t i = 1; i < hi0; ++i)
    for (std::size_t w = 0; w < W; ++w)
      s[i].v[w] = cell.combine(maxValue, s[i - 1].v[w], maxValue, dist(x[i], Y[0].v[w]), i, 0);
  for (std::size_t i = hi0; i < n; ++i)
    for (std::size_t w = 0; w < W; ++w) s[i].v[w] = maxValue;

  for (std::size_t j = 1; j < n; ++j) {
    const auto [lo, hi] = dtw_band_bounds(band, j, n);
    // Locals, so the compiler keeps them in registers: without type-based
    // alias analysis (clang's Windows driver) a store to s may alias memory.
    T y[W], diag[W], left[W];
    for (std::size_t w = 0; w < W; ++w) y[w] = Y[j].v[w];
    std::size_t i = lo;
    if (lo == 0) {
      for (std::size_t w = 0; w < W; ++w) {
        diag[w] = s[0].v[w];
        left[w] = cell.combine(maxValue, maxValue, s[0].v[w], dist(x[0], y[w]), 0, j);
        s[0].v[w] = left[w];
      }
      i = 1;
    } else {
      for (std::size_t w = 0; w < W; ++w) {
        diag[w] = s[lo - 1].v[w]; // dp[lo-1, j-1], inside the previous column's band
        left[w] = maxValue;       // dp[lo-1, j], outside this column's band
      }
    }
    for (; i < hi; ++i) {
      const T xi = x[i];
      T *si = s[i].v;
      for (std::size_t w = 0; w < W; ++w) {
        const T up = si[w]; // dp[i, j-1]
        left[w] = cell.combine(diag[w], up, left[w], dist(xi, y[w]), i, j);
        diag[w] = up;
        si[w] = left[w];
      }
    }
  }

  for (std::size_t w = 0; w < W; ++w) out[w] = s[n - 1].v[w];
  return out;
}

} // namespace dtwc::core
