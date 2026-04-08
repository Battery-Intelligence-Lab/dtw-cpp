/**
 * @file warping_missing_arow.hpp
 * @brief DTW-AROW: diagonal-only alignment for missing values (NaN-aware).
 *
 * @details Implements the DTW-AROW recurrence from Yurtman et al. (ECML-PKDD 2023).
 * When x[i] or y[j] is NaN, the warping path is restricted to the DIAGONAL direction
 * only (one-to-one alignment), preventing many-to-one "free stretching" through
 * missing regions.
 *
 * Standard DTW recurrence:
 *   C(i,j) = min(C(i-1,j-1), C(i-1,j), C(i,j-1)) + d(x[i], y[j])
 *
 * AROW recurrence:
 *   if is_missing(x[i]) or is_missing(y[j]):
 *       C(i,j) = C(i-1,j-1)    // diagonal only, zero local cost
 *   else:
 *       C(i,j) = min(C(i-1,j-1), C(i-1,j), C(i,j-1)) + d(x[i], y[j])
 *
 * Boundary treatment: at boundaries (first row/column), only one direction of
 * movement is available (no many-to-one cheating possible), so missing values
 * at boundaries propagate from the single available predecessor with zero cost
 * (NOT set to +inf, which would cascade and make the matrix unreachable).
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

#include "settings.hpp"
#include "warping.hpp"           // for detail::L1Dist, detail::SquaredL2Dist
#include "core/scratch_matrix.hpp"
#include "core/dtw_options.hpp"  // for core::MetricType
#include "missing_utils.hpp"     // for is_missing() — bitwise NaN, safe under -ffast-math

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <span>      // for span

namespace dtwc {

// =========================================================================
//  AROW implementation helpers
// =========================================================================

namespace detail {

/**
 * @brief Linear-space DTW-AROW implementation.
 *
 * @details O(min(m,n)) space. The rolling buffer stores the current column (short side).
 * The diag variable tracks C(i-1,j-1) before it is overwritten, exactly as in
 * dtwFull_L_impl, but the update rule differs: when a value is missing, we use
 * only the diagonal predecessor with zero additional cost.
 *
 * Boundary conditions:
 *   - C(0,0): if either is missing -> 0, else distance(x[0], y[0])
 *   - C(i,0) (first column): if either is missing -> C(i-1,0) (propagate), else C(i-1,0)+dist
 *   - Interior C(i,j): AROW recurrence
 *
 * The outer loop iterates over the "long" series, the inner loop over the "short" side.
 * This maps to the C matrix where short_ptr indexes rows (i) and long_ptr indexes columns (j).
 */
template <typename data_t, typename DistFn>
data_t dtwAROW_L_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                      DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  if (nx == 0 || ny == 0) return maxValue;
  if (x == y && nx == ny) return 0;

  // Orient so short_ptr has fewer elements (rolling buffer size = m_short)
  const data_t* short_ptr;
  const data_t* long_ptr;
  size_t m_short, m_long;
  if (nx <= ny) {
    short_ptr = x; m_short = nx;
    long_ptr  = y; m_long  = ny;
  } else {
    short_ptr = y; m_short = ny;
    long_ptr  = x; m_long  = nx;
  }

  // Thread-local rolling buffer (short side = rows in C matrix)
  thread_local static std::vector<data_t> col;
  col.resize(m_short);

  // --- First column (j=0): only vertical movement possible ---
  // C(0,0)
  if (is_missing(short_ptr[0]) || is_missing(long_ptr[0]))
    col[0] = data_t(0);
  else
    col[0] = distance(short_ptr[0], long_ptr[0]);

  // C(i,0) for i=1..m_short-1: propagate from C(i-1,0)
  for (size_t i = 1; i < m_short; ++i) {
    if (is_missing(short_ptr[i]) || is_missing(long_ptr[0]))
      col[i] = col[i - 1];  // zero additional cost, use vertical predecessor
    else
      col[i] = col[i - 1] + distance(short_ptr[i], long_ptr[0]);
  }

  // --- Remaining columns (j=1..m_long-1) ---
  for (size_t j = 1; j < m_long; ++j) {
    const bool long_missing = is_missing(long_ptr[j]);

    // C(0,j): first row — only horizontal movement possible
    // diag for this cell is col[0] from the previous iteration (= C(0, j-1))
    data_t diag = col[0];  // save C(0, j-1) before overwrite
    if (is_missing(short_ptr[0]) || long_missing)
      col[0] = diag;  // propagate: C(0,j) = C(0,j-1), zero cost
    else
      col[0] = diag + distance(short_ptr[0], long_ptr[j]);

    // C(i,j) for i=1..m_short-1: full AROW recurrence
    for (size_t i = 1; i < m_short; ++i) {
      const data_t old_col_i = col[i];  // = C(i, j-1) (vertical predecessor)

      if (is_missing(short_ptr[i]) || long_missing) {
        // AROW: diagonal only, zero local cost
        // diag here holds C(i-1, j-1)
        col[i] = diag;
      } else {
        // Standard recurrence: min(diag, col[i-1], old_col_i) + dist
        // diag     = C(i-1, j-1)
        // col[i-1] = C(i-1, j)   (already updated this iteration)
        // old_col_i = C(i, j-1)
        const data_t minimum = std::min(diag, std::min(col[i - 1], old_col_i));
        col[i] = minimum + distance(short_ptr[i], long_ptr[j]);
      }

      diag = old_col_i;  // advance diagonal: next cell needs C(i-1, j-1) = current C(i, j-1)
    }
  }

  return col[m_short - 1];
}

/// Linear-space AROW — span overload.
template <typename data_t, typename DistFn>
data_t dtwAROW_L_impl(std::span<const data_t> x, std::span<const data_t> y,
                      DistFn distance)
{
  return dtwAROW_L_impl(x.data(), x.size(), y.data(), y.size(), distance);
}

/**
 * @brief Full-matrix DTW-AROW implementation.
 *
 * @details O(m*n) space. Stores the entire cost matrix — useful for debugging and
 * verifying the linear-space implementation. Uses the same AROW recurrence.
 */
template <typename data_t, typename DistFn>
data_t dtwAROW_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                    DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const int mx = static_cast<int>(nx);
  const int my = static_cast<int>(ny);

  if (mx == 0 || my == 0) return maxValue;
  if (x == y && nx == ny) return 0;

  thread_local core::ScratchMatrix<data_t> C;
  C.resize(mx, my);

  // C(0,0)
  if (is_missing(x[0]) || is_missing(y[0]))
    C(0, 0) = data_t(0);
  else
    C(0, 0) = distance(x[0], y[0]);

  // First column (j=0): only vertical movement
  for (int i = 1; i < mx; ++i) {
    if (is_missing(x[i]) || is_missing(y[0]))
      C(i, 0) = C(i - 1, 0);
    else
      C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]);
  }

  // First row (i=0): only horizontal movement
  for (int j = 1; j < my; ++j) {
    if (is_missing(x[0]) || is_missing(y[j]))
      C(0, j) = C(0, j - 1);
    else
      C(0, j) = C(0, j - 1) + distance(x[0], y[j]);
  }

  // Interior cells: full AROW recurrence
  for (int j = 1; j < my; ++j) {
    for (int i = 1; i < mx; ++i) {
      if (is_missing(x[i]) || is_missing(y[j])) {
        C(i, j) = C(i - 1, j - 1);
      } else {
        const data_t minimum = std::min(C(i - 1, j - 1), std::min(C(i - 1, j), C(i, j - 1)));
        C(i, j) = minimum + distance(x[i], y[j]);
      }
    }
  }

  return C(mx - 1, my - 1);
}

/// Full-matrix AROW — span overload.
template <typename data_t, typename DistFn>
data_t dtwAROW_impl(std::span<const data_t> x, std::span<const data_t> y,
                    DistFn distance)
{
  return dtwAROW_impl(x.data(), x.size(), y.data(), y.size(), distance);
}

/**
 * @brief Banded DTW-AROW implementation (full-matrix with Sakoe-Chiba band).
 *
 * @details Uses the full-matrix approach restricted to the Sakoe-Chiba band.
 * Cells outside the band are initialised to +inf and never updated.
 * When band < 0, delegates to the unbanded linear-space version.
 */
template <typename data_t, typename DistFn>
data_t dtwAROW_banded_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                           int band, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const int mx = static_cast<int>(nx);
  const int my = static_cast<int>(ny);

  if (mx == 0 || my == 0) return maxValue;
  if (x == y && nx == ny) return 0;

  thread_local core::ScratchMatrix<data_t> C;
  C.resize(mx, my);

  // Initialise entire matrix to +inf
  for (int i = 0; i < mx; ++i)
    for (int j = 0; j < my; ++j)
      C(i, j) = maxValue;

  // Compute band boundaries for each row i.
  // NOTE: AROW uses the simpler std::ceil/floor(center +/- window) without the
  // round-100 trick used by detail::band_bounds(). This is intentional: AROW
  // operates on a full matrix (not a rolling buffer), so off-by-one at band
  // edges is harmless -- out-of-band cells are +inf and never selected. The
  // other banded variants use rolling buffers where the round-100 trick is
  // needed to avoid floating-point boundary drift.
  const double slope = (my > 1 && mx > 1)
    ? static_cast<double>(my - 1) / (mx - 1)
    : 1.0;
  const double window = std::max(static_cast<double>(band), slope / 2.0);

  // C(0,0)
  if (is_missing(x[0]) || is_missing(y[0]))
    C(0, 0) = data_t(0);
  else
    C(0, 0) = distance(x[0], y[0]);

  // First column (j=0), within band
  for (int i = 1; i < mx; ++i) {
    const double center = slope * i;
    const int lo = static_cast<int>(std::ceil(center - window));
    const int hi = static_cast<int>(std::floor(center + window));
    if (lo > 0) break;  // j=0 is outside band for all subsequent i
    if (is_missing(x[i]) || is_missing(y[0]))
      C(i, 0) = C(i - 1, 0);
    else
      C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]);
    (void)hi;
  }

  // First row (i=0), within band
  for (int j = 1; j < my; ++j) {
    const double center = j / slope;  // corresponding i-position for this j
    const int lo = static_cast<int>(std::ceil(center - window / slope));
    if (lo > 0) break;  // i=0 is outside band for all subsequent j
    if (is_missing(x[0]) || is_missing(y[j]))
      C(0, j) = C(0, j - 1);
    else
      C(0, j) = C(0, j - 1) + distance(x[0], y[j]);
  }

  // Interior cells: AROW within band
  for (int i = 1; i < mx; ++i) {
    const double center = slope * i;
    const int lo = std::max(1, static_cast<int>(std::ceil(center - window)));
    const int hi = std::min(my - 1, static_cast<int>(std::floor(center + window)));

    for (int j = lo; j <= hi; ++j) {
      const data_t diag = C(i - 1, j - 1);
      if (is_missing(x[i]) || is_missing(y[j])) {
        C(i, j) = (diag == maxValue) ? maxValue : diag;
      } else {
        const data_t up   = C(i - 1, j);
        const data_t left = C(i, j - 1);
        const data_t minimum = std::min(diag, std::min(up, left));
        C(i, j) = (minimum == maxValue) ? maxValue : minimum + distance(x[i], y[j]);
      }
    }
  }

  return C(mx - 1, my - 1);
}

/// Banded AROW — span overload.
template <typename data_t, typename DistFn>
data_t dtwAROW_banded_impl(std::span<const data_t> x, std::span<const data_t> y,
                           int band, DistFn distance)
{
  return dtwAROW_banded_impl(x.data(), x.size(), y.data(), y.size(), band, distance);
}

} // namespace detail

// =========================================================================
//  Public API — DTW-AROW
// =========================================================================

/**
 * @brief Computes DTW-AROW distance (linear space, O(min(m,n)) memory).
 *
 * @details When x[i] or y[j] is NaN, the warping path is restricted to the
 * diagonal direction only (zero local cost), preventing free stretching through
 * missing regions. Uses O(min(m,n)) space — no backtracking.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW-AROW distance.
 */
template <typename data_t = double>
data_t dtwAROW_L(std::span<const data_t> x, std::span<const data_t> y,
                 core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwAROW_L_impl(x, y, dist);
  });
}

/**
 * @brief Computes DTW-AROW distance (full matrix, O(m*n) memory).
 *
 * @details Same recurrence as dtwAROW_L but stores the full cost matrix for
 * debugging and verification. The result is identical to dtwAROW_L.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW-AROW distance.
 */
template <typename data_t = double>
data_t dtwAROW(std::span<const data_t> x, std::span<const data_t> y,
               core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwAROW_impl(x, y, dist);
  });
}

/**
 * @brief Computes DTW-AROW distance with Sakoe-Chiba band constraint.
 *
 * @details Restricts the warping path to a band of width @p band around the
 * diagonal, in addition to the AROW missing-value constraint. When band < 0,
 * falls back to dtwAROW_L (unbanded linear-space).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param band Sakoe-Chiba bandwidth. Negative means unconstrained.
 * @param metric Pointwise distance metric (default: L1).
 * @return The banded DTW-AROW distance.
 */
template <typename data_t = double>
data_t dtwAROW_banded(std::span<const data_t> x, std::span<const data_t> y,
                      int band = settings::DEFAULT_BAND_LENGTH,
                      core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwAROW_L<data_t>(x, y, metric);
  const auto m_short = static_cast<int>(std::min(x.size(), y.size()));
  const auto m_long  = static_cast<int>(std::max(x.size(), y.size()));
  if (m_short <= 1 || m_long <= band + 1) return dtwAROW_L<data_t>(x, y, metric);
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwAROW_banded_impl(x, y, band, dist);
  });
}

// Vector convenience overloads (vector -> span implicit conversion is non-deduced).
template <typename data_t = double>
data_t dtwAROW_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_L<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, metric);
}

template <typename data_t = double>
data_t dtwAROW(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, metric);
}

template <typename data_t = double>
data_t dtwAROW_banded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      int band = settings::DEFAULT_BAND_LENGTH,
                      core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_banded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, band, metric);
}

// =========================================================================
//  Pointer + length overloads (zero-copy for bindings)
// =========================================================================

/// DTW-AROW, linear space (pointer + length).
template <typename data_t = double>
data_t dtwAROW_L(const data_t* x, size_t nx, const data_t* y, size_t ny,
                 core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwAROW_L_impl(x, nx, y, ny, dist);
  });
}

/// DTW-AROW, full matrix (pointer + length).
template <typename data_t = double>
data_t dtwAROW(const data_t* x, size_t nx, const data_t* y, size_t ny,
               core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwAROW_impl(x, nx, y, ny, dist);
  });
}

/// DTW-AROW, banded (pointer + length).
template <typename data_t = double>
data_t dtwAROW_banded(const data_t* x, size_t nx, const data_t* y, size_t ny,
                      int band = settings::DEFAULT_BAND_LENGTH,
                      core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwAROW_L<data_t>(x, nx, y, ny, metric);
  const auto m_short = static_cast<int>(std::min(nx, ny));
  const auto m_long  = static_cast<int>(std::max(nx, ny));
  if (m_short <= 1 || m_long <= band + 1) return dtwAROW_L<data_t>(x, nx, y, ny, metric);
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwAROW_banded_impl(x, nx, y, ny, band, dist);
  });
}

} // namespace dtwc
