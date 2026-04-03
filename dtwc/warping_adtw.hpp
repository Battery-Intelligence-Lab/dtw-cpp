/**
 * @file warping_adtw.hpp
 * @brief Amerced Dynamic Time Warping (ADTW) functions.
 *
 * @details ADTW adds a penalty for non-diagonal (horizontal/vertical) steps in
 * the warping path, discouraging time stretching/compression:
 *
 *   C(i,j) = d(x[i], y[j]) + min(C(i-1,j-1),
 *                                  C(i-1,j) + penalty,
 *                                  C(i,j-1) + penalty)
 *
 * Reference: Herrmann, M. & Shifaz, A. (2023). "Amercing: An intuitive and
 *            effective constraint for dynamic time warping."
 *            Pattern Recognition, 137, 109301.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "settings.hpp"
#include "warping.hpp"
#include "core/scratch_matrix.hpp"

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for ceil, floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <utility>   // for pair, tie

namespace dtwc {

/**
 * @brief Computes the full ADTW distance using a rolling-buffer (O(min(m,n)) memory).
 *
 * @tparam data_t Data type of sequence elements.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param penalty Penalty added to non-diagonal (horizontal/vertical) warping steps.
 * @return The ADTW distance.
 */
template <typename data_t>
data_t adtwFull_L(const data_t *x, size_t nx, const data_t *y, size_t ny, data_t penalty)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  if (nx == 0 || ny == 0) return maxValue;
  if (x == y && nx == ny) return 0; // Same object => distance 0.
  thread_local static std::vector<data_t> short_side(10000);

  const data_t *short_ptr;
  const data_t *long_ptr;
  size_t m_short;
  size_t m_long;
  if (nx < ny) {
    short_ptr = x; m_short = nx;
    long_ptr = y; m_long = ny;
  } else {
    short_ptr = y; m_short = ny;
    long_ptr = x; m_long = nx;
  }

  short_side.resize(m_short);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  // Base: C(0,0) = d(short[0], long[0])
  short_side[0] = distance(short_ptr[0], long_ptr[0]);

  // First column: only vertical steps (non-diagonal) => each incurs penalty
  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + penalty + distance(short_ptr[i], long_ptr[0]);

  // Fill remaining columns
  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    // First row of this column: horizontal step (non-diagonal) => incurs penalty
    short_side[0] += penalty + distance(short_ptr[0], long_ptr[j]);

    for (size_t i = 1; i < m_short; i++) {
      // short_side[i-1] = C(i-1, j)  — already updated = left
      // short_side[i]   = C(i, j-1)  — not yet updated = below
      // diag            = C(i-1, j-1)
      const data_t min1 = std::min(short_side[i - 1], short_side[i]) + penalty;
      const data_t dist = distance(short_ptr[i], long_ptr[j]);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
    }
  }

  return short_side.back();
}

template <typename data_t>
data_t adtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y, data_t penalty)
{
  return adtwFull_L(x.data(), x.size(), y.data(), y.size(), penalty);
}


/**
 * @brief Computes the banded ADTW distance using a Sakoe-Chiba band.
 *
 * @tparam data_t Data type of sequence elements.
 * @param x First sequence.
 * @param y Second sequence.
 * @param band Sakoe-Chiba bandwidth. Negative means unbanded (falls back to adtwFull_L).
 * @param penalty Penalty added to non-diagonal warping steps.
 * @return The ADTW distance.
 */
template <typename data_t = double>
data_t adtwBanded(const data_t *x, size_t nx, const data_t *y, size_t ny,
                  int band, data_t penalty)
{
  if (band < 0) return adtwFull_L<data_t>(x, nx, y, ny, penalty);

  thread_local core::ScratchMatrix<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const data_t *short_ptr;
  const data_t *long_ptr;
  int m_short;
  int m_long;
  if (nx < ny) {
    short_ptr = x; m_short = static_cast<int>(nx);
    long_ptr = y; m_long = static_cast<int>(ny);
  } else {
    short_ptr = y; m_short = static_cast<int>(ny);
    long_ptr = x; m_long = static_cast<int>(nx);
  }

  C.resize(m_long, m_short);
  C.fill(maxValue);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  if ((m_short == 0) || (m_long == 0)) return maxValue;
  if ((m_short == 1) || (m_long == 1)) return adtwFull_L<data_t>(x, nx, y, ny, penalty);
  if (m_long <= (band + 1)) return adtwFull_L<data_t>(x, nx, y, ny, penalty);

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max((double)band, slope / 2);

  auto get_bounds = [slope, window](int idx) {
    const auto yval = slope * idx;
    const int low = std::ceil(std::round(100 * (yval - window)) / 100.0);
    const int high = std::floor(std::round(100 * (yval + window)) / 100.0) + 1;
    return std::pair(low, high);
  };

  C(0, 0) = distance(long_ptr[0], short_ptr[0]);

  {
    const auto [lo, hi] = get_bounds(0);
    for (int i = 1; i < hi; ++i)
      C(i, 0) = C(i - 1, 0) + penalty + distance(long_ptr[i], short_ptr[0]);
  }

  for (int j = 1; j < m_short; j++) {
    const auto [lo, hi] = get_bounds(j);
    if (lo <= 0)
      C(0, j) = C(0, j - 1) + penalty + distance(long_ptr[0], short_ptr[j]);

    const auto high = std::min(hi, m_long);
    for (int i = std::max(lo, 1); i < high; ++i) {
      const auto diag = C(i - 1, j - 1);
      const auto left = C(i, j - 1);
      const auto below = C(i - 1, j);
      // Diagonal is free; horizontal and vertical incur penalty
      const auto minimum = std::min(diag, std::min(left + penalty, below + penalty));
      C(i, j) = minimum + distance(long_ptr[i], short_ptr[j]);
    }
  }

  return C(m_long - 1, m_short - 1);
}

template <typename data_t = double>
data_t adtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t penalty)
{
  return adtwBanded(x.data(), x.size(), y.data(), y.size(), band, penalty);
}

// -------------------------------------------------------------------------
// Multivariate ADTW variants (interleaved layout: x[t * ndim + d]).
// -------------------------------------------------------------------------

/**
 * @brief Multivariate ADTW (full, linear-space).
 *
 * @details Replaces scalar abs(a-b) with L1 sum over ndim features.
 *          When ndim == 1, delegates to the existing scalar adtwFull_L().
 *
 * @tparam data_t Data type.
 * @param x        First series (flat interleaved: nx_steps * ndim elements).
 * @param nx_steps Number of timesteps in x.
 * @param y        Second series (flat interleaved: ny_steps * ndim elements).
 * @param ny_steps Number of timesteps in y.
 * @param ndim     Features per timestep.
 * @param penalty  Penalty for non-diagonal warping steps.
 */
template <typename data_t = double>
data_t adtwFull_L_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, data_t penalty = 1.0)
{
  if (ndim == 1) return adtwFull_L(x, nx_steps, y, ny_steps, penalty);

  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  if (nx_steps == 0 || ny_steps == 0) return maxValue;
  if (x == y && nx_steps == ny_steps) return 0;

  const data_t *short_ptr, *long_ptr;
  size_t m_short, m_long;
  if (nx_steps < ny_steps) {
    short_ptr = x; m_short = nx_steps; long_ptr = y; m_long = ny_steps;
  } else {
    short_ptr = y; m_short = ny_steps; long_ptr = x; m_long = nx_steps;
  }

  auto mv_dist = [ndim](const data_t *a, const data_t *b) {
    data_t sum = 0;
    for (size_t d = 0; d < ndim; ++d) sum += std::abs(a[d] - b[d]);
    return sum;
  };

  thread_local std::vector<data_t> short_side;
  short_side.resize(m_short);

  // Base: C(0,0) = d(short[0], long[0])
  short_side[0] = mv_dist(short_ptr, long_ptr);

  // First column: only vertical steps (non-diagonal) => each incurs penalty
  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + penalty + mv_dist(short_ptr + i * ndim, long_ptr);

  // Fill remaining columns
  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    // First row of this column: horizontal step (non-diagonal) => incurs penalty
    short_side[0] += penalty + mv_dist(short_ptr, long_ptr + j * ndim);

    for (size_t i = 1; i < m_short; i++) {
      // short_side[i-1] = C(i-1, j)  — already updated
      // short_side[i]   = C(i, j-1)  — not yet updated
      // diag            = C(i-1, j-1)
      const data_t min1 = std::min(short_side[i - 1], short_side[i]) + penalty;
      const data_t dist = mv_dist(short_ptr + i * ndim, long_ptr + j * ndim);
      const data_t next = std::min(diag, min1) + dist;
      diag = short_side[i];
      short_side[i] = next;
    }
  }

  return short_side.back();
}

/**
 * @brief Multivariate ADTW banded.
 *
 * @details When band < 0, delegates to adtwFull_L_mv.
 *          When ndim == 1, delegates to the scalar adtwBanded().
 *          For multivariate banded, falls back to full MV for now.
 *
 * @tparam data_t Data type.
 * @param x, nx_steps  First series (flat interleaved).
 * @param y, ny_steps  Second series (flat interleaved).
 * @param ndim         Features per timestep.
 * @param band         Sakoe-Chiba band. Negative means unbanded.
 * @param penalty      Penalty for non-diagonal warping steps.
 */
template <typename data_t = double>
data_t adtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, int band = settings::DEFAULT_BAND_LENGTH, data_t penalty = 1.0)
{
  if (band < 0) return adtwFull_L_mv(x, nx_steps, y, ny_steps, ndim, penalty);
  if (ndim == 1) return adtwBanded(x, nx_steps, y, ny_steps, band, penalty);
  // For banded MV ADTW, delegate to full MV (optimize later if needed)
  return adtwFull_L_mv(x, nx_steps, y, ny_steps, ndim, penalty);
}

} // namespace dtwc
