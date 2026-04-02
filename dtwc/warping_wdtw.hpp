/**
 * @file warping_wdtw.hpp
 * @brief Weighted Dynamic Time Warping (WDTW) distance functions.
 *
 * @details WDTW multiplies each pointwise distance by a weight that depends on
 *          the absolute index difference |i - j| between the matched points.
 *          Larger index deviations get heavier penalties. The weight vector is
 *          typically a logistic function: w(d) = w_max / (1 + exp(-g*(d - m/2)))
 *          where m is the maximum possible deviation and g controls steepness.
 *
 *          Reference: Y.-S. Jeong, M.-K. Jeong, O.-A. Omitaomu, "Weighted dynamic
 *          time warping for time series classification", Pattern Recognition, 44(9),
 *          2231-2240 (2011).
 *
 * @author Claude (generated)
 * @date 28 Mar 2026
 */

#pragma once

#include "settings.hpp"

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for exp, ceil, floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <utility>   // for pair
#include <numeric>   // for iota

namespace dtwc {

/**
 * @brief Generate WDTW weight vector using the logistic function.
 *
 * @details w(d) = w_max / (1 + exp(-g * (d - m/2)))
 *          where d is the index deviation (0..max_dev), g controls steepness,
 *          and w_max is the maximum weight.
 *
 * @tparam data_t Data type.
 * @param max_dev Maximum index deviation (typically max(len_x, len_y) - 1).
 * @param g Steepness parameter for the logistic weight function.
 * @param w_max Maximum weight value.
 * @return Weight vector of size max_dev + 1.
 */
template <typename data_t>
std::vector<data_t> wdtw_weights(int max_dev, data_t g = 0.05, data_t w_max = 1.0)
{
  std::vector<data_t> weights(max_dev + 1);
  const data_t half_dev = static_cast<data_t>(max_dev) / 2.0;
  for (int d = 0; d <= max_dev; ++d) {
    weights[d] = w_max / (1.0 + std::exp(-g * (d - half_dev)));
  }
  return weights;
}


/**
 * @brief Computes the full Weighted DTW distance between two sequences.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @param weights Weight vector indexed by |i - j|.
 * @return The WDTW distance.
 */
template <typename data_t>
data_t wdtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
                const std::vector<data_t> &weights)
{
  if (&x == &y) return 0;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short = static_cast<int>(short_vec.size());
  const int m_long = static_cast<int>(long_vec.size());

  if (m_short == 0 || m_long == 0) return maxValue;

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  // Use rolling column buffer (col stores one column at a time, indexed by long_vec rows)
  thread_local std::vector<data_t> col;
  col.resize(m_long);

  // We iterate columns j = 0..m_short-1 (short side).
  // col[i] stores C(i, j_prev). After processing column j, col[i] = C(i, j).

  // Determine weight index: in the cost matrix, row i = long_vec index, col j = short_vec index.
  // If x.size() < y.size(), then short_vec = x, long_vec = y, and the cost matrix
  // has long_vec on rows and short_vec on columns. The original weight deviation
  // between x[ix] and y[iy] is |ix - iy|. When short_vec = x and long_vec = y,
  // ix = j (short index), iy = i (long index), so dev = |i - j|.
  // When short_vec = y and long_vec = x, ix = i, iy = j, so dev = |i - j| as well.

  // First column (j = 0)
  col[0] = weights[0] * distance(long_vec[0], short_vec[0]);
  for (int i = 1; i < m_long; ++i) {
    col[i] = col[i - 1] + weights[i] * distance(long_vec[i], short_vec[0]);
  }

  // Remaining columns j = 1..m_short-1
  for (int j = 1; j < m_short; ++j) {
    data_t diag = col[0]; // C(0, j-1)
    col[0] = col[0] + weights[j] * distance(long_vec[0], short_vec[j]); // C(0,j) = C(0,j-1) + w*d

    for (int i = 1; i < m_long; ++i) {
      const int dev = std::abs(i - j);
      const auto dist = weights[dev] * distance(long_vec[i], short_vec[j]);
      const data_t old_col_i = col[i]; // C(i, j-1) — vertical predecessor
      const data_t minimum = std::min(diag, std::min(col[i - 1], old_col_i));
      col[i] = minimum + dist;
      diag = old_col_i; // for next i: diag = C(i, j-1) = C((i+1)-1, j-1)
    }
  }

  return col[m_long - 1];
}


/**
 * @brief Computes the banded Weighted DTW distance between two sequences.
 *
 * @details Uses a Sakoe-Chiba band and a rolling column buffer (one column at a time).
 *          Memory usage is O(m_long) instead of O(m_long * m_short).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @param weights Weight vector indexed by |i - j|.
 * @param band The bandwidth parameter controlling the Sakoe-Chiba band.
 * @return The WDTW distance.
 */
template <typename data_t>
data_t wdtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  const std::vector<data_t> &weights, int band = settings::DEFAULT_BAND_LENGTH)
{
  if (band < 0) return wdtwFull<data_t>(x, y, weights);

  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short = static_cast<int>(short_vec.size());
  const int m_long = static_cast<int>(long_vec.size());

  if (m_short == 0 || m_long == 0) return maxValue;
  if (m_short == 1 || m_long == 1) return wdtwFull<data_t>(x, y, weights);
  if (m_long <= (band + 1)) return wdtwFull<data_t>(x, y, weights);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  auto get_bounds = [slope, window](int j) {
    const auto center = slope * j;
    const int low = static_cast<int>(std::ceil(std::round(100 * (center - window)) / 100.0));
    const int high = static_cast<int>(std::floor(std::round(100 * (center + window)) / 100.0)) + 1;
    return std::pair(low, high);
  };

  // Rolling column buffer: col[i] stores C(i, j) after processing column j.
  thread_local std::vector<data_t> col;
  col.assign(m_long, maxValue);

  // First column (j = 0): initialize col[0..hi) with cumulative weighted sums
  {
    const auto [lo, hi] = get_bounds(0);
    const int high = std::min(hi, m_long);
    col[0] = weights[0] * distance(long_vec[0], short_vec[0]);
    for (int i = 1; i < high; ++i) {
      col[i] = col[i - 1] + weights[i] * distance(long_vec[i], short_vec[0]);
    }
  }

  // Columns j = 1..m_short-1
  for (int j = 1; j < m_short; ++j) {
    const auto [lo, hi] = get_bounds(j);
    const int low = std::max(lo, 0);
    const int high = std::min(hi, m_long);

    // We need diag = C(i-1, j-1) at the start of each row sweep.
    // Before overwriting col[low], grab it if it will serve as diag for low+1.
    // For i = low: diag = col[low - 1] from previous column (if low > 0).
    data_t diag = (low > 0) ? col[low - 1] : maxValue;

    // Handle row i = 0 specially (only possible if low == 0)
    if (low == 0) {
      const data_t old_col_0 = col[0]; // C(0, j-1)
      const int dev = j; // |0 - j| = j
      col[0] = old_col_0 + weights[dev] * distance(long_vec[0], short_vec[j]);
      diag = old_col_0;

      // Process rows 1..high-1
      for (int i = 1; i < high; ++i) {
        const int dev_i = std::abs(i - j);
        const auto dist = weights[dev_i] * distance(long_vec[i], short_vec[j]);
        const data_t old_col_i = col[i]; // C(i, j-1)
        const data_t minimum = std::min(diag, std::min(col[i - 1], old_col_i));
        col[i] = minimum + dist;
        diag = old_col_i;
      }
    } else {
      // Invalidate cells below the band from the previous column that are
      // now out-of-band. The cell at col[low-1] is outside band for column j,
      // but we already captured it as diag above. We must NOT invalidate it
      // before capturing — we already have it in diag.

      // For i = low (first row in band): there is no col[i-1] in-band for this column
      // if low-1 is outside the band. col[low-1] from the previous column is the
      // diag value, and there is no horizontal predecessor.
      {
        const int i = low;
        const int dev_i = std::abs(i - j);
        const auto dist = weights[dev_i] * distance(long_vec[i], short_vec[j]);
        const data_t old_col_i = col[i]; // C(i, j-1) = vertical predecessor

        // Horizontal predecessor col[i-1] was just set in this column?
        // No, i = low and we haven't processed low-1 for this column.
        // col[i-1] might be from previous column but outside current band.
        // We mark it as maxValue for the horizontal predecessor.
        const data_t horiz = maxValue; // col[low-1] is outside band for column j
        const data_t minimum = std::min(diag, std::min(horiz, old_col_i));
        col[i] = minimum + dist;
        diag = old_col_i;
      }

      for (int i = low + 1; i < high; ++i) {
        const int dev_i = std::abs(i - j);
        const auto dist = weights[dev_i] * distance(long_vec[i], short_vec[j]);
        const data_t old_col_i = col[i]; // C(i, j-1)
        const data_t minimum = std::min(diag, std::min(col[i - 1], old_col_i));
        col[i] = minimum + dist;
        diag = old_col_i;
      }
    }

    // Invalidate cells that have fallen outside the band
    // Cells below low or at/above high are invalid for future use as vertical predecessors
    if (lo > 0) {
      // col[lo-1] is out of band; set to maxValue so it's not used as a vertical predecessor
      col[lo - 1] = maxValue;
    }
    // Cells at high..m_long-1 already have stale values from previous columns.
    // They'll be maxValue from initialization or from invalidation. But if a cell
    // was valid in a previous column and is now above the band, we must invalidate it.
    // However, with the Sakoe-Chiba band, the high bound increases monotonically,
    // so cells above high were either never set or already correctly maxValue.
    // The low bound also increases, so we only need to invalidate the cell just below low.
  }

  return col[m_long - 1];
}

// -------------------------------------------------------------------------
// Convenience overloads: accept g parameter instead of precomputed weights.
// -------------------------------------------------------------------------

/// WDTW with g parameter (computes weights internally).
template <typename data_t = double>
data_t wdtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t g)
{
  const int max_dev = static_cast<int>(std::max(x.size(), y.size()));
  auto w = wdtw_weights<data_t>(max_dev, g);
  return wdtwBanded(x, y, w, band);
}

/// Full WDTW with g parameter.
template <typename data_t = double>
data_t wdtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
                data_t g)
{
  const int max_dev = static_cast<int>(std::max(x.size(), y.size()));
  auto w = wdtw_weights<data_t>(max_dev, g);
  return wdtwFull(x, y, w);
}

// -------------------------------------------------------------------------
// Multivariate WDTW variants (interleaved layout: x[t * ndim + d]).
// -------------------------------------------------------------------------

/**
 * @brief Multivariate WDTW (full, linear-space).
 *
 * @details Replaces scalar abs(a-b) with L1 sum over ndim features.
 *          When ndim == 1, delegates to the existing scalar wdtwFull().
 *
 * @tparam data_t Data type.
 * @param x        First series (flat interleaved: nx_steps * ndim elements).
 * @param nx_steps Number of timesteps in x.
 * @param y        Second series (flat interleaved: ny_steps * ndim elements).
 * @param ny_steps Number of timesteps in y.
 * @param ndim     Features per timestep.
 * @param g        Logistic weight steepness.
 */
template <typename data_t = double>
data_t wdtwFull_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                   size_t ndim, data_t g = 0.05)
{
  if (ndim == 1) {
    std::vector<data_t> vx(x, x + nx_steps), vy(y, y + ny_steps);
    const int max_dev = static_cast<int>(std::max(nx_steps, ny_steps)) - 1;
    auto w = wdtw_weights<data_t>(max_dev, g);
    return wdtwFull(vx, vy, w);
  }

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

  const int max_dev = static_cast<int>(m_long) - 1;
  auto weights = wdtw_weights<data_t>(max_dev, g);

  auto mv_dist = [ndim](const data_t *a, const data_t *b) {
    data_t sum = 0;
    for (size_t d = 0; d < ndim; ++d) sum += std::abs(a[d] - b[d]);
    return sum;
  };

  thread_local std::vector<data_t> col;
  col.resize(m_long);

  // First column (j = 0)
  col[0] = weights[0] * mv_dist(long_ptr, short_ptr);
  for (size_t i = 1; i < m_long; ++i)
    col[i] = col[i - 1] + weights[i] * mv_dist(long_ptr + i * ndim, short_ptr);

  // Remaining columns j = 1..m_short-1
  for (size_t j = 1; j < m_short; ++j) {
    data_t diag = col[0];
    col[0] = col[0] + weights[j] * mv_dist(long_ptr, short_ptr + j * ndim);

    for (size_t i = 1; i < m_long; ++i) {
      const int dev = std::abs(static_cast<int>(i) - static_cast<int>(j));
      const auto dist = weights[dev] * mv_dist(long_ptr + i * ndim, short_ptr + j * ndim);
      const data_t old_col_i = col[i];
      col[i] = std::min(diag, std::min(col[i - 1], old_col_i)) + dist;
      diag = old_col_i;
    }
  }

  return col[m_long - 1];
}

/**
 * @brief Multivariate WDTW banded.
 *
 * @details When band < 0, delegates to wdtwFull_mv.
 *          When ndim == 1, delegates to the scalar wdtwBanded().
 *          For multivariate banded, falls back to full MV for now.
 *
 * @tparam data_t Data type.
 * @param x, nx_steps  First series (flat interleaved).
 * @param y, ny_steps  Second series (flat interleaved).
 * @param ndim         Features per timestep.
 * @param band         Sakoe-Chiba band. Negative means unbanded.
 * @param g            Logistic weight steepness.
 */
template <typename data_t = double>
data_t wdtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, int band = settings::DEFAULT_BAND_LENGTH, data_t g = 0.05)
{
  if (band < 0) return wdtwFull_mv(x, nx_steps, y, ny_steps, ndim, g);
  if (ndim == 1) {
    std::vector<data_t> vx(x, x + nx_steps), vy(y, y + ny_steps);
    return wdtwBanded(vx, vy, band, g);
  }
  // For banded MV WDTW, delegate to full MV (optimize later if needed)
  return wdtwFull_mv(x, nx_steps, y, ny_steps, ndim, g);
}

} // namespace dtwc
