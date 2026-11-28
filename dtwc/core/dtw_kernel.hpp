/**
 * @file dtw_kernel.hpp
 * @brief Unified scalar DTW kernel, parameterised on Cost + Cell policies.
 *
 * @details One kernel each for {Linear-space, Sakoe-Chiba banded}.
 *          Variants (Standard, ADTW, WDTW, Soft-DTW, AROW) pick a Cell policy +
 *          Cost functor and call the same kernel. The wavefront shape,
 *          rolling-buffer scratch, and early-abandon handling live here once.
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
 *          Each kernel copies its Cost into a local before its loops. On Win64 a
 *          by-value Cost wider than 8 bytes arrives as a pointer to the caller's
 *          copy, which the stores to the DP buffer might alias, so a kernel the
 *          compiler does not inline would reload the series pointers through it
 *          in every cell; the local copy stays in registers.
 *
 *          dtw_kernel_lanes computes W pairs at once and packs the values
 *          itself, so it takes a pointwise distance instead of a Cost:
 *          T operator()(T a, T b) const.
 *
 * @author Volkan Kumtepeli
 * @date 12 Apr 2026
 */

#pragma once

#include <algorithm> // std::min, std::max
#include <array>     // std::array
#include <cmath>     // std::ceil, std::floor, std::round, std::fmin
#include <cstddef>   // size_t
#include <limits>    // std::numeric_limits
#include <span>      // std::span
#include <utility>   // std::pair
#include <vector>

namespace dtwc::core {

// ===========================================================================
// Cell policies
// ===========================================================================

// The cells nest two-argument std::min, never std::min({...}): the MSVC STL
// compiles the initializer-list form to an out-of-line call per cell
// (__std_min_d). The nesting makes the comparisons min_element makes, so the
// result is unchanged, NaN ordering included.

#if defined(_MSC_VER)
#define DTWC_LANES_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define DTWC_LANES_RESTRICT __restrict__
#else
#define DTWC_LANES_RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DTWC_KERNEL_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define DTWC_KERNEL_FORCE_INLINE __forceinline
#else
#define DTWC_KERNEL_FORCE_INLINE inline
#endif

/// Standard DTW recurrence: min(diag, up, left) + cost.
struct StandardCell
{
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::min(std::min(diag, up), left) + cost;
  }
  template <typename T>
  T combine_interior(T diag, T up, T left, T cost,
                     std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::min(std::min(diag, up), left) + cost;
  }
  template <typename T>
  T boundary(T prev, T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return prev + cost;
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
struct ADTWCell
{
  T penalty;
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    const T u = up + penalty;
    const T l = left + penalty;
    const T m1 = (u < diag) ? u : diag;
    const T m2 = (l < m1) ? l : m1;
    return m2 + cost;
  }
  T combine_interior(T diag, T up, T left, T cost,
                     std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return combine(diag, up, left, cost, 0, 0);
  }
  T boundary(T prev, T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return (prev + penalty) + cost;
  }
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return cost;
  }
};

/// WDTW cell policy for SIMD lane kernels: broadcasts weights[|short_idx - long_idx|]
/// across all lanes at each (short_idx, long_idx) step.
template <typename T>
struct WDTWCell
{
  const T *DTWC_LANES_RESTRICT weights;
  T row_weight(std::size_t short_idx, std::size_t long_idx) const noexcept
  {
    const std::size_t d = (short_idx > long_idx) ? (short_idx - long_idx) : (long_idx - short_idx);
    return weights[d];
  }
  T combine_w(T diag, T up, T left, T cost, T w) const noexcept
  {
    const T m1 = (up < diag) ? up : diag;
    const T m2 = (left < m1) ? left : m1;
    return m2 + w * cost;
  }
  T boundary_w(T prev, T cost, T w) const noexcept
  {
    return prev + w * cost;
  }
  T combine(T diag, T up, T left, T cost,
            std::size_t short_idx, std::size_t long_idx) const noexcept
  {
    return combine_w(diag, up, left, cost, row_weight(short_idx, long_idx));
  }
  T combine_interior(T diag, T up, T left, T cost,
                     std::size_t short_idx, std::size_t long_idx) const noexcept
  {
    return combine_w(diag, up, left, cost, row_weight(short_idx, long_idx));
  }
  T boundary(T prev, T cost, std::size_t short_idx, std::size_t long_idx) const noexcept
  {
    return boundary_w(prev, cost, row_weight(short_idx, long_idx));
  }
  T seed(T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return weights[0] * cost;
  }
};

namespace detail {

  /** Precomputed gamma scaling for validated Soft-DTW inner loops. */
  template <typename T>
  struct SoftGammaScale
  {
    static constexpr T kCutoff = sizeof(T) == 8 ? T(40.0) : T(20.0);

    T gamma;
    T inv_gamma;
    bool finite_reciprocal;
    T gamma_fraction{ T(1) };
    int gamma_exponent{ 0 };

    explicit SoftGammaScale(T value) noexcept
      : gamma(value), inv_gamma(T(1) / value),
        finite_reciprocal(std::isfinite(inv_gamma))
    {
      if (!finite_reciprocal)
        gamma_fraction = std::frexp(gamma, &gamma_exponent);
    }

    DTWC_KERNEL_FORCE_INLINE T scaled(T delta) const noexcept
    {
      return finite_reciprocal
               ? delta * inv_gamma
               : std::scalbn(delta / gamma_fraction, -gamma_exponent);
    }

    DTWC_KERNEL_FORCE_INLINE T boltzmann(T value, T minimum) const noexcept
    {
      if (value == minimum) return T(1);
      const T s = scaled(value - minimum);
      return s >= kCutoff ? T(0) : std::exp(-s);
    }
  };

} // namespace detail

/// Soft-DTW (Cuturi & Blondel 2017): softmin(diag, up, left) + cost with
/// log-sum-exp stabilisation (M = min of valid predecessors; exponents are
/// subtracted by M before exp/log).
///
/// Sentinel handling: out-of-bounds predecessors (== std::numeric_limits<T>::max())
/// are excluded from the LSE, so first-row/column cells (one valid predecessor)
/// reduce to `up + cost` or `left + cost`, matching `soft_dtw_gradient`.
///
/// The value needs only dtw_kernel_linear's rolling column; soft_dtw_gradient
/// keeps the full matrices its backward pass reads.
template <typename T>
struct SoftCell
{
  T gamma;
  detail::SoftGammaScale<T> scale;

  explicit SoftCell(T value) noexcept : gamma(value), scale(value) {}

  DTWC_KERNEL_FORCE_INLINE T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    constexpr T maxValue = std::numeric_limits<T>::max();
    T m = maxValue;
    if (diag < m) m = diag;
    if (up < m) m = up;
    if (left < m) m = left;
    if (m == maxValue) return maxValue;

    // diag, left, up: on dtw_kernel_linear (up = dp[i, j-1], left = dp[i-1, j])
    // the order soft_dtw has always summed in; another order moves a float32
    // distance by an ulp.
    T acc = T(0);
    if (diag != maxValue) acc += scale.boltzmann(diag, m);
    if (left != maxValue) acc += scale.boltzmann(left, m);
    if (up != maxValue) acc += scale.boltzmann(up, m);
    return (acc == T(1) ? m : m - gamma * std::log(acc)) + cost;
  }
  DTWC_KERNEL_FORCE_INLINE T combine_interior(T diag, T up, T left, T cost,
                     std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    T m = diag;
    if (up < m) m = up;
    if (left < m) m = left;

    T acc = scale.boltzmann(diag, m);
    acc += scale.boltzmann(left, m);
    acc += scale.boltzmann(up, m);
    return (acc == T(1) ? m : m - gamma * std::log(acc)) + cost;
  }
  T boundary(T prev, T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return prev + cost;
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
struct AROWCell
{
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    constexpr T maxValue = std::numeric_limits<T>::max();
    const T m1 = (up < diag) ? up : diag;
    const T m2 = (left < m1) ? left : m1;
    const T non_missing = (m2 == maxValue) ? maxValue : (m2 + cost);
    const T missing_val = (diag != maxValue) ? diag
                        : (up != maxValue)   ? up
                        : (left != maxValue) ? left
                                             : T(0);
    return std::isnan(cost) ? missing_val : non_missing;
  }
  template <typename T>
  T combine_interior(T diag, T up, T left, T cost,
                     std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    const T m1 = (up < diag) ? up : diag;
    const T m2 = (left < m1) ? left : m1;
    return std::isnan(cost) ? diag : (m2 + cost);
  }
  template <typename T>
  T boundary(T prev, T cost, std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::isnan(cost) ? prev : (prev + cost);
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

/// Canonical Sakoe-Chiba range [lo, hi) in [0, column_count) at zero-based index `row`.
/// The adjustment window is |row-column| <= band. Bounds are clamped here so callers never
/// narrow sequence indices to `int` or overflow `row+band+1`. `row` may reach or exceed
/// `column_count` when used along the transposed axis of `dtw_kernel_lanes` (ny[w] > nx).
inline std::pair<std::size_t, std::size_t>
  dtw_band_bounds(int band, std::size_t row, std::size_t column_count) noexcept
{
  if (band < 0) return { 0, column_count }; // every row, of a matrix of any shape
  if (column_count == 0) return { 0, 0 };
  const auto width = static_cast<std::size_t>(band);
  const auto lo = (row > width) ? row - width : 0;
  if (lo >= column_count) return { column_count, column_count };
  // The subtraction is safe when row < column_count. Taking this branch
  // before forming row+width+1 makes the upper clamp overflow-proof.
  const auto hi = (row >= column_count || width >= column_count - row - 1)
                    ? column_count
                    : row + width + 1;
  return { lo, hi };
}

// ===========================================================================
// Kernel 1: linear-space full-band DTW (outer = long, inner = short).
// Rolling buffer of size n_short. Optional early abandon.
// ===========================================================================

namespace detail {

  /// The per-pair kernels' thread_local buffers, one of each a T (every kernel of that T shares it): reached here so
  /// that core::release_thread_scratch() can empty them (detail::dispatch_scratch, core/dtw_dispatch.cpp).
  template <typename T>
  std::vector<T> &linear_short_buf()
  {
    thread_local std::vector<T> short_buf;
    return short_buf;
  }

  template <typename T>
  std::vector<T> &banded_col_buf()
  {
    thread_local std::vector<T> col_buf;
    return col_buf;
  }

  /// dtw_kernel_linear with the early-abandon test known at compile time.
  template <bool Abandon, typename T, typename Cost, typename Cell>
  T dtw_linear(std::size_t n_short, std::size_t n_long, Cost cost_in, Cell cell_in, T early_abandon)
  {
    const Cost cost = cost_in; // in registers: see the file comment
    const Cell cell = cell_in;
    constexpr T maxValue = std::numeric_limits<T>::max();
    if (n_short == 0 || n_long == 0) return maxValue;

    std::vector<T> &short_buf = linear_short_buf<T>();
    if (short_buf.size() < n_short) short_buf.resize(n_short);
    T *short_side = short_buf.data(); // hoisted out of the loops

    // First column (long_idx = 0): accumulate along short axis, no diag/left.
    short_side[0] = cell.seed(cost(0, 0), 0, 0);
    // A column whose minimum exceeds the threshold abandons the pair.
    const auto abandons = [early_abandon](T column_min) { return Abandon && column_min > early_abandon; };

    if (n_short == 1) {
      T acc = short_side[0];
      for (std::size_t j = 1; j < n_long; ++j) {
        // SoftCell uses boundary() directly (one predecessor needs no exp/log);
        // standard cells keep combine()'s min as a barrier against -fassociative-math reassociation.
        if constexpr (std::is_same_v<Cell, SoftCell<T>>)
          acc = cell.boundary(acc, cost(0, j), 0, j);
        else
          acc = cell.combine(maxValue, maxValue, acc, cost(0, j), 0, j);
        if (abandons(acc)) return maxValue;
      }
      return acc;
    }

    for (std::size_t i = 1; i < n_short; ++i)
      short_side[i] = cell.boundary(short_side[i - 1], cost(i, 0), i, 0);

    // Columns j and j + 1 per pass: two chains of min-then-add side by side, where
    // one column is one chain whose latency, not the core, sets the pace. Each cell
    // gets the neighbours, in the same roles, that the one-column loop below gives
    // it, and the pair is abandoned iff some column's minimum exceeds the threshold.
    std::size_t j = 1;
    for (; j + 1 < n_long; j += 2) {
      T diag = short_side[0];
      T left = cell.boundary(short_side[0], cost(0, j), 0, j);
      T next_left = cell.boundary(left, cost(0, j + 1), 0, j + 1);
      short_side[0] = next_left;
      T row_min = left, next_row_min = next_left;

      for (std::size_t i = 1; i < n_short; ++i) {
        const T old_up = short_side[i];                                                     // dp[i, j-1]
        const T here = cell.combine_interior(diag, old_up, left, cost(i, j), i, j);         // dp[i, j]
        next_left = cell.combine_interior(left, here, next_left, cost(i, j + 1), i, j + 1); // dp[i, j+1]
        diag = old_up;
        left = here;
        short_side[i] = next_left;
        if constexpr (Abandon) {
          row_min = std::min(row_min, here);
          next_row_min = std::min(next_row_min, next_left);
        }
      }
      if (abandons(row_min) || abandons(next_row_min)) return maxValue;
    }

    for (; j < n_long; ++j) { // the last column when n_long - 1 is odd; a one-row matrix
      T diag = short_side[0];
      // First row of new column: no up, no diag (they'd be from out-of-bounds
      // previous column/row). Only `left` (short_side[0] == dp[0, j-1]) is valid.
      T left = cell.boundary(short_side[0], cost(0, j), 0, j);
      short_side[0] = left;
      T row_min = left;

      for (std::size_t i = 1; i < n_short; ++i) {
        const T old_up = short_side[i]; // dp[i, j-1]
        // dp[i, j]; carried as the next cell's left (dp[i-1, j]) instead of reloaded.
        left = cell.combine_interior(diag, old_up, left, cost(i, j), i, j);
        diag = old_up;
        short_side[i] = left;
        if constexpr (Abandon) row_min = std::min(row_min, left);
      }
      if (abandons(row_min)) return maxValue;
    }

    return short_side[n_short - 1];
  }

} // namespace detail

/// The early-abandon test is a template argument of the loops, so no build runs
/// the column minimum in the loop that has no threshold: clang unswitches a
/// runtime test only at -O3 (nanobind's -Os build was 1.5x slower for it).
template <typename T, typename Cost, typename Cell>
T dtw_kernel_linear(std::size_t n_short, std::size_t n_long,
                    Cost cost, Cell cell, T early_abandon = T(-1))
{
  return early_abandon >= T(0)
           ? detail::dtw_linear<true, T>(n_short, n_long, cost, cell, early_abandon)
           : detail::dtw_linear<false, T>(n_short, n_long, cost, cell, early_abandon);
}

// ===========================================================================
// Kernel 2: Sakoe-Chiba banded DTW (outer = short, inner = long).
// Rolling column of size n_long. Optional early abandon.
// ===========================================================================

namespace detail {

  /// dtw_kernel_banded with the early-abandon test known at compile time.
  template <bool Abandon, typename T, typename Cost, typename Cell>
  T dtw_banded(std::size_t n_short, std::size_t n_long, int band,
               Cost cost_in, Cell cell_in, T early_abandon)
  {
    const Cost cost = cost_in; // in registers: see the file comment
    const Cell cell = cell_in;
    constexpr T maxValue = std::numeric_limits<T>::max();
    if (n_short == 0 || n_long == 0) return maxValue;
    if (band < 0)
      return dtw_linear<Abandon, T>(n_short, n_long, cost, cell, early_abandon);

    const auto band_width = static_cast<std::size_t>(band);

    // The fixed terminal cell must lie inside |short_idx-long_idx| <= band.
    if (n_long - n_short > band_width) return maxValue;

    // Degenerate length-one path, or a band covering the complete matrix.
    if (n_short == 1 || n_long == 1 || band_width >= n_long - 1)
      return dtw_linear<Abandon, T>(n_short, n_long, cost, cell, early_abandon);

    std::vector<T> &col_buf = banded_col_buf<T>();
    if (col_buf.size() < n_long) col_buf.resize(n_long);
    T *col = col_buf.data(); // hoisted out of the loops
    const auto abandons = [early_abandon](T column_min) { return Abandon && column_min > early_abandon; };

    // First short-step (j_short = 0): fill col along long axis. Only `left`
    // (col[i-1]) is available — diag and up are out-of-bounds (maxValue).
    col[0] = cell.seed(cost(0, 0), 0, 0);
    {
      const auto hi = dtw_band_bounds(band, 0, n_long).second;
      for (std::size_t i = 1; i < hi; ++i) {
        col[i] = cell.boundary(col[i - 1], cost(0, i), 0, i);
      }
      if (hi < n_long) col[hi] = maxValue;
    }
    if (abandons(col[0])) return maxValue;

    for (std::size_t j = 1; j < n_short; ++j) {
      // The band's bounds advance by at most 1 per j: dp[j-1, first_row-1] lies in the previous
      // column's band, and col[high_{j-1}] holds the maxValue boundary sentinel placed above.
      const auto [low, high] = dtw_band_bounds(band, j, n_long);
      const auto first_row = std::max(low, std::size_t{ 1 });
      T diag = col[first_row - 1]; // dp[j-1, first_row-1]
      T row_min = maxValue;

      if (low == 0) {
        // Row 0 of new column: only `left` (col[0] from previous j-step).
        col[0] = cell.boundary(col[0], cost(j, 0), j, 0);
        row_min = col[0];
      }

      // Below the band, dp[j, first_row-1] is unreachable: maxValue.
      T left = (low == 0) ? col[0] : maxValue; // dp[j, first_row-1]
      for (std::size_t i = first_row; i < high; ++i) {
        const T old_up = col[i]; // dp[j-1, i]
        // dp[j, i]; carried as the next cell's left instead of reloaded.
        left = cell.combine(diag, old_up, left, cost(j, i), j, i);
        col[i] = left;
        diag = old_up;
        if constexpr (Abandon) row_min = std::min(row_min, left);
      }
      if (high < n_long) col[high] = maxValue;
      if (abandons(row_min)) return maxValue;
    }

    return col[n_long - 1];
  }

} // namespace detail

/// The early-abandon test is a template argument of the loops, as in
/// dtw_kernel_linear.
template <typename T, typename Cost, typename Cell>
T dtw_kernel_banded(std::size_t n_short, std::size_t n_long, int band,
                    Cost cost, Cell cell, T early_abandon = T(-1))
{
  return early_abandon >= T(0)
           ? detail::dtw_banded<true, T>(n_short, n_long, band, cost, cell, early_abandon)
           : detail::dtw_banded<false, T>(n_short, n_long, band, cost, cell, early_abandon);
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
// Kernel 3: W pairs in SIMD lanes (outer = y, inner = x).
// The pairs (x, ys[w]) share x; without a band each ys[w] has its own length.
// Every lane evaluates the per-pair kernel's cells (dtw_kernel_linear's; with a
// band, dtw_kernel_banded's) with its arithmetic, each from the same three
// neighbours, transposed where that kernel's outer series is x (it puts the
// shorter series inner, x on a tie). So each lane is the per-pair result for a
// Cell that treats `up` and `left` alike, as StandardCell does: bit for bit,
// unless the compiler contracts `d * d + m` into an FMA in one kernel and not
// the other (GCC does by default; the two then differ in the last bits). The
// loop over the lanes is the one that vectorises: W dependency chains side by
// side where the per-pair kernel runs one, each a min then an add per row.
// ===========================================================================

/// The fill's Cell for dtw_kernel_lanes: StandardCell, but on AArch64 its min is
/// std::fmin, one fminnm, where std::min is a compare and a select (fcmgt + bif)
/// on the chain from one row's `left` to the next. fminnm differs from std::min
/// only for NaN and for the sign of zero, and finite query/panel values meet neither:
/// input validation refuses NaN and ±inf before any lane runs, a cost on finite
/// values is |a - b| or d * d (never -0, never NaN), and a cost that overflows is
/// +inf, which both mins order alike. A DDTW derivative or an interpolated sample
/// that overflows to ±inf gives a NaN cost (|+inf - (+inf)|): fmin drops a NaN that
/// std::min keeps on the diagonal, so the AArch64 lanes can return +inf or max()
/// where the per-pair kernel returns NaN, for a pair with no finite distance.
/// Elsewhere this is StandardCell: x86's fmin is three instructions
/// (minpd, cmpunordpd, blendvpd) where its min is one.
#if defined(__aarch64__) || defined(_M_ARM64)
struct LanesCell : StandardCell
{
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return std::fmin(std::fmin(diag, up), left) + cost;
  }
  template <typename T>
  T combine_interior(T diag, T up, T left, T cost,
                     std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return combine(diag, up, left, cost, 0, 0);
  }
};
#else
struct LanesCell : StandardCell
{
  template <typename T>
  T combine(T diag, T up, T left, T cost,
            std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    // Value-returning ternary (identical to std::min's `(b < a) ? b : a` ordering)
    // avoids const-reference stack temporaries that block MSVC/GCC SIMD vectorisation.
    const T m1 = (up < diag) ? up : diag;
    const T m2 = (left < m1) ? left : m1;
    return m2 + cost;
  }
  template <typename T>
  T combine_interior(T diag, T up, T left, T cost,
                     std::size_t /*short_idx*/, std::size_t /*long_idx*/) const noexcept
  {
    return combine(diag, up, left, cost, 0, 0);
  }
};
#endif

/// Pairs per dtw_kernel_lanes call: enough chains that the FP pipes, not one
/// chain's min-then-add latency, set the pace. 128 bytes of T (16 doubles, 32
/// floats) on AArch64 and on x86-64 under Clang and GCC; 64 bytes of T (8 doubles,
/// 16 floats) under MSVC x64 and WebAssembly SIMD128 so two-column blocking fits
/// cleanly in architectural vector registers without stack spills.
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__x86_64__)
template <typename T>
inline constexpr std::size_t dtw_lanes = 128 / sizeof(T);
inline constexpr bool dtw_lanes_packed = true;
#elif defined(_M_X64) || defined(__wasm_simd128__)
template <typename T>
inline constexpr std::size_t dtw_lanes = 64 / sizeof(T);
inline constexpr bool dtw_lanes_packed = true;
#else
template <typename T>
inline constexpr std::size_t dtw_lanes = 64 / sizeof(T);
inline constexpr bool dtw_lanes_packed = false;
#endif

namespace detail {

  /// A row of the lanes' buffers: one value a lane.
  template <typename T>
  struct alignas(64) LanesRow
  {
    T v[dtw_lanes<T>];
  };

  /// The lanes kernel's two thread_local buffers, one pair a T (both of its distances share them): reached here so
  /// that core::release_thread_scratch() can empty them (detail::lanes_scratch, core/dtw_lanes.cpp). Two thread_locals,
  /// as the kernel declared them, returned as references (`auto [y_buf, s_buf]` binds the buffers): one pair object
  /// moved a column loop of the lanes, and a cell loop of TWE's, under GCC.
  template <typename T>
  std::pair<std::vector<LanesRow<T>> &, std::vector<LanesRow<T>> &> lanes_bufs()
  {
    thread_local std::vector<LanesRow<T>> y_buf, s_buf;
    return { y_buf, s_buf };
  }

  /// Ragged-tail end-row snapshot buffer for dtw_kernel_lanes_packed, kept separate from y_buf
  /// (which holds PackedLanesView::rows) so panel.rows and end_rows never alias under DTWC_LANES_RESTRICT.
  template <typename T>
  std::vector<LanesRow<T>> &lanes_end_buf()
  {
    thread_local std::vector<LanesRow<T>> end_buf;
    return end_buf;
  }

} // namespace detail

/// Non-owning view of W = dtw_lanes<T> interleaved series [ny_max][W], a lane past
/// its own length repeating its last sample.
template <typename T>
struct PackedLanesView
{
  const detail::LanesRow<T> *rows = nullptr;
  std::array<std::size_t, dtw_lanes<T>> ny{};
  std::size_t ny_min = 0;
  std::size_t ny_max = 0;
};

/// Owning pre-packed panel of W = dtw_lanes<T> series, reusable across many queries x.
template <typename T>
struct PackedLanesPanel
{
  std::vector<detail::LanesRow<T>> rows;
  std::array<std::size_t, dtw_lanes<T>> ny{};
  std::size_t ny_min = 0;
  std::size_t ny_max = 0;

  PackedLanesView<T> view() const noexcept
  {
    return { rows.data(), ny, ny_min, ny_max };
  }
};

namespace detail {

  template <typename T, typename Transform>
  PackedLanesView<T> pack_lanes_with(const std::span<const T> *ys, std::vector<LanesRow<T>> &buf,
                                     Transform &&transform)
  {
    constexpr std::size_t W = dtw_lanes<T>;
    std::array<std::size_t, W> ny{};
    std::size_t ny_min = std::numeric_limits<std::size_t>::max(), ny_max = 0;
    for (std::size_t w = 0; w < W; ++w) {
      ny[w] = ys[w].size();
      ny_min = std::min(ny_min, ny[w]);
      ny_max = std::max(ny_max, ny[w]);
    }
    if (ny_min == 0) return { nullptr, ny, 0, ny_max };
    if (buf.size() < ny_max) buf.resize(ny_max);
    LanesRow<T> *DTWC_LANES_RESTRICT Y = buf.data();
    for (std::size_t w = 0; w < W; ++w) {
      const std::span<const T> src = transform(ys[w]);
      const T *DTWC_LANES_RESTRICT yw = src.data();
      const std::size_t n = src.size();
      std::size_t t = 0;
      for (; t < n; ++t) Y[t].v[w] = yw[t];
      const T last = yw[n - 1];
      for (; t < ny_max; ++t) Y[t].v[w] = last;
    }
    return { Y, ny, ny_min, ny_max };
  }

  template <typename T>
  PackedLanesView<T> pack_lanes_into(const std::span<const T> *ys, std::vector<LanesRow<T>> &buf)
  {
    return pack_lanes_with<T>(ys, buf, [](std::span<const T> s) noexcept { return s; });
  }

} // namespace detail

/// DTW between xs and a pre-packed panel of W = dtw_lanes<T> series, with pointwise distance
/// dist(xs[i], panel.rows[j].v[w]). Supports both unbanded (band < 0) and Sakoe-Chiba banded
/// (band >= 0) execution across equal-length and ragged series.
/// Register-blocks kCols columns per sweep over s[0..nx) when unbanded (4 on AVX-512, 2 on
/// AVX2/NEON/WASM), cutting L1D load/store traffic on s[] by 2x-4x with zero register spills.
template <typename T, typename Dist, typename Cell>
std::array<T, dtw_lanes<T>> dtw_kernel_lanes_packed(std::span<const T> xs, const PackedLanesView<T> &panel,
                                                    int band, Dist dist_in, Cell cell_in)
{
  const Dist dist = dist_in;
  const Cell cell = cell_in;
  constexpr std::size_t W = dtw_lanes<T>;
  constexpr T maxValue = std::numeric_limits<T>::max();
  const T *DTWC_LANES_RESTRICT x = xs.data();
  const std::size_t nx = xs.size();
  const std::size_t ny_min = panel.ny_min;
  const std::size_t ny_max = panel.ny_max;
  std::array<T, W> out;
  if (nx == 0 || ny_min == 0) {
    out.fill(maxValue);
    return out;
  }

  using Row = detail::LanesRow<T>;
  // maxValue in every lane. A row, or the lanes of `left`, is set by copying it whole:
  // Apple clang turns a loop that stores a constant row (or reads this row lane by
  // lane, which it folds to that constant) into a call of memset_pattern16.
  static constexpr Row kUnreachable = [] {
    Row r{};
    for (T &e : r.v) e = maxValue;
    return r;
  }();

  auto &s_buf = detail::lanes_bufs<T>().second;
  if (s_buf.size() < nx) s_buf.resize(nx);
  const bool ragged_tail = (ny_min < ny_max);
  Row *DTWC_LANES_RESTRICT end_rows = nullptr;
  if (ragged_tail) {
    auto &end_buf = detail::lanes_end_buf<T>();
    const std::size_t tail_count = ny_max - (ny_min - 1);
    if (end_buf.size() < tail_count) end_buf.resize(tail_count);
    end_rows = end_buf.data();
  }
  const Row *DTWC_LANES_RESTRICT Y = panel.rows;
  Row *DTWC_LANES_RESTRICT s = s_buf.data();

  // Column 0 (y[0]). The rows above its band hold maxValue: the band only moves
  // up, so a later column first meets such a row as an unreachable `up`.
  constexpr bool has_row_weight = requires(const Cell &c) { c.row_weight(std::size_t{ 0 }, std::size_t{ 0 }); };
  const std::size_t hi0 = dtw_band_bounds(band, 0, nx).second;
  {
    T y0[W];
    for (std::size_t w = 0; w < W; ++w) y0[w] = Y[0].v[w];
    T *DTWC_LANES_RESTRICT s0 = s[0].v;
    const T x0 = x[0];
    if constexpr (has_row_weight) {
      const T w00 = cell.row_weight(0, 0);
      for (std::size_t w = 0; w < W; ++w) s0[w] = w00 * dist(x0, y0[w]);
    } else {
      for (std::size_t w = 0; w < W; ++w) s0[w] = cell.seed(dist(x0, y0[w]), 0, 0);
    }
    for (std::size_t i = 1; i < hi0; ++i) {
      const T xi = x[i];
      T *DTWC_LANES_RESTRICT si = s[i].v;
      const T *DTWC_LANES_RESTRICT sim1 = s[i - 1].v;
      if constexpr (has_row_weight) {
        const T wi0 = cell.row_weight(i, 0);
        for (std::size_t w = 0; w < W; ++w)
          si[w] = cell.boundary_w(sim1[w], dist(xi, y0[w]), wi0);
      } else {
        for (std::size_t w = 0; w < W; ++w)
          si[w] = cell.boundary(sim1[w], dist(xi, y0[w]), i, 0);
      }
    }
  }
  for (std::size_t i = hi0; i < nx; ++i) s[i] = kUnreachable;
  if (ragged_tail && ny_min == 1) end_rows[0] = s[nx - 1];

  // Multi-column register-blocked sweep for unconstrained DTW (nx > 1 keeps 1-row matrices
  // on the 1-column loop so -fassociative-math never regroups across columns without a min).
  // Stops before ny_min - 1 when ragged_tail is true so every lane-ending column runs in the
  // 1-column loop below with zero branch inside the multi-column loop.
  const std::size_t multi_limit = ragged_tail ? (ny_min - 1) : ny_min;
  std::size_t j = 1;
#if defined(__AVX512F__)
  constexpr std::size_t kCols = 4;
  for (; band < 0 && nx > 1 && j + kCols <= multi_limit; j += kCols) {
    T y0[W], y1[W], y2[W], y3[W], diag0[W], l0[W], l1[W], l2[W], l3[W];
    for (std::size_t w = 0; w < W; ++w) {
      y0[w] = Y[j].v[w];
      y1[w] = Y[j + 1].v[w];
      y2[w] = Y[j + 2].v[w];
      y3[w] = Y[j + 3].v[w];
    }
    const T x0 = x[0];
    T *DTWC_LANES_RESTRICT s0 = s[0].v;
    if constexpr (has_row_weight) {
      const T w00 = cell.row_weight(0, j), w01 = cell.row_weight(0, j + 1);
      const T w02 = cell.row_weight(0, j + 2), w03 = cell.row_weight(0, j + 3);
      for (std::size_t w = 0; w < W; ++w) {
        const T old_s0 = s0[w];
        diag0[w] = old_s0;
        l0[w] = cell.boundary_w(old_s0, dist(x0, y0[w]), w00);
        l1[w] = cell.boundary_w(l0[w], dist(x0, y1[w]), w01);
        l2[w] = cell.boundary_w(l1[w], dist(x0, y2[w]), w02);
        l3[w] = cell.boundary_w(l2[w], dist(x0, y3[w]), w03);
        s0[w] = l3[w];
      }
      for (std::size_t i = 1; i < nx; ++i) {
        const T xi = x[i];
        const T wi0 = cell.row_weight(i, j), wi1 = cell.row_weight(i, j + 1);
        const T wi2 = cell.row_weight(i, j + 2), wi3 = cell.row_weight(i, j + 3);
        T *DTWC_LANES_RESTRICT si = s[i].v;
        for (std::size_t w = 0; w < W; ++w) {
          const T up0 = si[w];
          const T n0 = cell.combine_w(diag0[w], up0, l0[w], dist(xi, y0[w]), wi0);
          const T n1 = cell.combine_w(l0[w], n0, l1[w], dist(xi, y1[w]), wi1);
          const T n2 = cell.combine_w(l1[w], n1, l2[w], dist(xi, y2[w]), wi2);
          const T n3 = cell.combine_w(l2[w], n2, l3[w], dist(xi, y3[w]), wi3);
          diag0[w] = up0;
          l0[w] = n0;
          l1[w] = n1;
          l2[w] = n2;
          l3[w] = n3;
          si[w] = n3;
        }
      }
    } else {
      for (std::size_t w = 0; w < W; ++w) {
        const T old_s0 = s0[w];
        diag0[w] = old_s0;
        l0[w] = cell.boundary(old_s0, dist(x0, y0[w]), 0, j);
        l1[w] = cell.boundary(l0[w], dist(x0, y1[w]), 0, j + 1);
        l2[w] = cell.boundary(l1[w], dist(x0, y2[w]), 0, j + 2);
        l3[w] = cell.boundary(l2[w], dist(x0, y3[w]), 0, j + 3);
        s0[w] = l3[w];
      }
      for (std::size_t i = 1; i < nx; ++i) {
        const T xi = x[i];
        T *DTWC_LANES_RESTRICT si = s[i].v;
        for (std::size_t w = 0; w < W; ++w) {
          const T up0 = si[w];
          const T n0 = cell.combine_interior(diag0[w], up0, l0[w], dist(xi, y0[w]), i, j);
          const T n1 = cell.combine_interior(l0[w], n0, l1[w], dist(xi, y1[w]), i, j + 1);
          const T n2 = cell.combine_interior(l1[w], n1, l2[w], dist(xi, y2[w]), i, j + 2);
          const T n3 = cell.combine_interior(l2[w], n2, l3[w], dist(xi, y3[w]), i, j + 3);
          diag0[w] = up0;
          l0[w] = n0;
          l1[w] = n1;
          l2[w] = n2;
          l3[w] = n3;
          si[w] = n3;
        }
      }
    }
  }
#else
  constexpr std::size_t kCols = 2;
  for (; band < 0 && nx > 1 && j + kCols <= multi_limit; j += kCols) {
    T y0[W], y1[W], diag0[W], l0[W], l1[W];
    for (std::size_t w = 0; w < W; ++w) {
      y0[w] = Y[j].v[w];
      y1[w] = Y[j + 1].v[w];
    }
    const T x0 = x[0];
    T *DTWC_LANES_RESTRICT s0 = s[0].v;
    if constexpr (has_row_weight) {
      const T w00 = cell.row_weight(0, j), w01 = cell.row_weight(0, j + 1);
      for (std::size_t w = 0; w < W; ++w) {
        const T old_s0 = s0[w];
        diag0[w] = old_s0;
        l0[w] = cell.boundary_w(old_s0, dist(x0, y0[w]), w00);
        l1[w] = cell.boundary_w(l0[w], dist(x0, y1[w]), w01);
        s0[w] = l1[w];
      }
      for (std::size_t i = 1; i < nx; ++i) {
        const T xi = x[i];
        const T wi0 = cell.row_weight(i, j), wi1 = cell.row_weight(i, j + 1);
        T *DTWC_LANES_RESTRICT si = s[i].v;
        for (std::size_t w = 0; w < W; ++w) {
          const T up0 = si[w];
          const T n0 = cell.combine_w(diag0[w], up0, l0[w], dist(xi, y0[w]), wi0);
          const T n1 = cell.combine_w(l0[w], n0, l1[w], dist(xi, y1[w]), wi1);
          diag0[w] = up0;
          l0[w] = n0;
          l1[w] = n1;
          si[w] = n1;
        }
      }
    } else {
      for (std::size_t w = 0; w < W; ++w) {
        const T old_s0 = s0[w];
        diag0[w] = old_s0;
        l0[w] = cell.boundary(old_s0, dist(x0, y0[w]), 0, j);
        l1[w] = cell.boundary(l0[w], dist(x0, y1[w]), 0, j + 1);
        s0[w] = l1[w];
      }
      for (std::size_t i = 1; i < nx; ++i) {
        const T xi = x[i];
        T *DTWC_LANES_RESTRICT si = s[i].v;
        for (std::size_t w = 0; w < W; ++w) {
          const T up0 = si[w];
          const T n0 = cell.combine_interior(diag0[w], up0, l0[w], dist(xi, y0[w]), i, j);
          const T n1 = cell.combine_interior(l0[w], n0, l1[w], dist(xi, y1[w]), i, j + 1);
          diag0[w] = up0;
          l0[w] = n0;
          l1[w] = n1;
          si[w] = n1;
        }
      }
    }
  }
#endif

  // Remaining columns (1-column tail when unbanded, or every column when banded).
  for (; j < ny_max; ++j) {
    const auto [lo, hi] = dtw_band_bounds(band, j, nx);
    T y[W], diag[W], left[W];
    for (std::size_t w = 0; w < W; ++w) y[w] = Y[j].v[w];
    std::size_t i = lo;
    if (lo == 0) {
      const T x0 = x[0];
      T *DTWC_LANES_RESTRICT s0 = s[0].v;
      if constexpr (has_row_weight) {
        const T w0j = cell.row_weight(0, j);
        for (std::size_t w = 0; w < W; ++w) {
          const T old_s0 = s0[w];
          diag[w] = old_s0;
          const T l = cell.boundary_w(old_s0, dist(x0, y[w]), w0j);
          left[w] = l;
          s0[w] = l;
        }
      } else {
        for (std::size_t w = 0; w < W; ++w) {
          const T old_s0 = s0[w];
          diag[w] = old_s0;
          const T l = cell.boundary(old_s0, dist(x0, y[w]), 0, j);
          left[w] = l;
          s0[w] = l;
        }
      }
      i = 1;
    } else if (lo < nx) {
      const T *DTWC_LANES_RESTRICT slo = s[lo - 1].v;
      for (std::size_t w = 0; w < W; ++w)
        diag[w] = slo[w];                  // dp[lo-1, j-1], inside the previous column's band
      std::copy_n(kUnreachable.v, W, left); // dp[lo-1, j], outside this column's band
    } else {
      s[nx - 1] = kUnreachable; // row nx - 1 has left the Sakoe-Chiba band
    }
    if constexpr (has_row_weight) {
      for (; i < hi; ++i) {
        const T xi = x[i];
        const T wij = cell.row_weight(i, j);
        T *DTWC_LANES_RESTRICT si = s[i].v;
        for (std::size_t w = 0; w < W; ++w) {
          const T up = si[w]; // dp[i, j-1]
          left[w] = cell.combine_w(diag[w], up, left[w], dist(xi, y[w]), wij);
          diag[w] = up;
          si[w] = left[w];
        }
      }
    } else {
      for (; i < hi; ++i) {
        const T xi = x[i];
        T *DTWC_LANES_RESTRICT si = s[i].v;
        for (std::size_t w = 0; w < W; ++w) {
          const T up = si[w]; // dp[i, j-1]
          left[w] = cell.combine(diag[w], up, left[w], dist(xi, y[w]), i, j);
          diag[w] = up;
          si[w] = left[w];
        }
      }
    }
    if (ragged_tail && j >= ny_min - 1) end_rows[j - (ny_min - 1)] = s[nx - 1];
  }

  if (!ragged_tail) {
    for (std::size_t w = 0; w < W; ++w) out[w] = s[nx - 1].v[w];
  } else {
    for (std::size_t w = 0; w < W; ++w) out[w] = end_rows[panel.ny[w] - ny_min].v[w];
  }
  return out;
}

/// DTW between xs and each of ys[0 .. dtw_lanes<T>), with the pointwise distance
/// dist(xs[i], ys[w][j]). band < 0 is unconstrained and band >= 0 is the Sakoe-Chiba
/// band |i - j| <= band; in both modes each ys[w] may have its own length.
template <typename T, typename Dist, typename Cell>
std::array<T, dtw_lanes<T>> dtw_kernel_lanes(std::span<const T> xs, const std::span<const T> *ys,
                                             int band, Dist dist, Cell cell)
{
  if (xs.empty()) {
    std::array<T, dtw_lanes<T>> out;
    out.fill(std::numeric_limits<T>::max());
    return out;
  }
  auto &y_buf = detail::lanes_bufs<T>().first;
  const auto panel = detail::pack_lanes_into<T>(ys, y_buf);
  return dtw_kernel_lanes_packed<T>(xs, panel, band, dist, cell);
}

#undef DTWC_LANES_RESTRICT
#undef DTWC_KERNEL_FORCE_INLINE

} // namespace dtwc::core
