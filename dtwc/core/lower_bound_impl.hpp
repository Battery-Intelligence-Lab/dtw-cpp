/**
 * @file lower_bound_impl.hpp
 * @brief Lower bound functions for DTW pruning (LB_Keogh, LB_Kim).
 *
 * @details Header-only implementations of O(n) and O(1) lower bounds on DTW
 *          distance. These enable early-abandon pruning when building distance
 *          matrices: if LB > current best, the full DTW can be skipped.
 *
 *          References:
 *          - E. Keogh, C. A. Ratanamahatana, "Exact indexing of dynamic time
 *            warping," Knowledge and Information Systems 7.3 (2005): 358-386.
 *          - S.-W. Kim, S. Park, W. W. Chu, "An index-based approach for
 *            similarity search supporting time warping in large sequence
 *            databases," Proc. 17th ICDE (2001): 607-614.
 *
 * @author Claude (AI assistant)
 * @date 28 Mar 2026
 */

#pragma once

#include <algorithm> // for min, max, min_element, max_element
#include <cmath>     // for abs
#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace dtwc::core {

/**
 * @brief Compute Sakoe-Chiba upper/lower envelopes for a time series.
 *
 * @details The envelopes are precomputed ONCE per series and reused O(N) times
 *          when computing LB_Keogh against multiple query series. For each
 *          position i, the upper envelope is the max and the lower envelope is
 *          the min of the series values within the Sakoe-Chiba band [i-band, i+band].
 *
 * @tparam T Numeric data type (float, double).
 * @param series Input time series pointer.
 * @param n Length of the series.
 * @param band Sakoe-Chiba band width (half-window radius).
 * @param upper_out Output: upper envelope (must be pre-allocated to n elements).
 * @param lower_out Output: lower envelope (must be pre-allocated to n elements).
 */
template <typename T>
void compute_envelopes(const T *series, std::size_t n, int band,
                       T *upper_out, T *lower_out)
{
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = (i > static_cast<std::size_t>(band)) ? i - static_cast<std::size_t>(band) : 0;
    const std::size_t hi = std::min(i + static_cast<std::size_t>(band) + 1, n);
    T max_val = series[lo];
    T min_val = series[lo];
    for (std::size_t j = lo + 1; j < hi; ++j) {
      if (series[j] > max_val) max_val = series[j];
      if (series[j] < min_val) min_val = series[j];
    }
    upper_out[i] = max_val;
    lower_out[i] = min_val;
  }
}

/**
 * @brief Convenience overload: compute envelopes from std::vector.
 *
 * @tparam T Numeric data type.
 * @param series Input time series.
 * @param band Sakoe-Chiba band width.
 * @param upper_out Output upper envelope vector (resized to match series).
 * @param lower_out Output lower envelope vector (resized to match series).
 */
template <typename T>
void compute_envelopes(const std::vector<T> &series, int band,
                       std::vector<T> &upper_out, std::vector<T> &lower_out)
{
  upper_out.resize(series.size());
  lower_out.resize(series.size());
  if (series.empty()) return;
  compute_envelopes(series.data(), series.size(), band, upper_out.data(), lower_out.data());
}

/**
 * @brief LB_Keogh: O(n) lower bound on DTW distance using envelopes.
 *
 * @details Computes a lower bound on DTW(query, candidate) using the L1
 *          (absolute difference) metric. The envelopes must be precomputed from
 *          the CANDIDATE series via compute_envelopes(). The bound is tight when
 *          the Sakoe-Chiba band constraint is used in the DTW computation.
 *
 *          If query[i] lies within [lower[i], upper[i]], it contributes 0 to the
 *          lower bound. Otherwise, the contribution is the distance to the
 *          nearest envelope boundary.
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer.
 * @param n Length of the query (must match envelope length).
 * @param upper Upper envelope of the candidate series.
 * @param lower Lower envelope of the candidate series.
 * @return Lower bound value (always <= true DTW distance under the band constraint).
 */
template <typename T>
T lb_keogh(const T *query, std::size_t n,
           const T *upper, const T *lower)
{
  T sum = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    if (query[i] > upper[i])
      sum += std::abs(query[i] - upper[i]);
    else if (query[i] < lower[i])
      sum += std::abs(query[i] - lower[i]);
    // else: query[i] within envelope, contributes 0
  }
  return sum;
}

/**
 * @brief Convenience overload: LB_Keogh from std::vectors.
 *
 * @tparam T Numeric data type.
 * @param query Query series.
 * @param upper Upper envelope of the candidate.
 * @param lower Lower envelope of the candidate.
 * @return Lower bound value.
 */
template <typename T>
T lb_keogh(const std::vector<T> &query,
           const std::vector<T> &upper, const std::vector<T> &lower)
{
  const std::size_t n = query.size();
  return lb_keogh(query.data(), n, upper.data(), lower.data());
}

/**
 * @brief LB_Kim: cheap lower bound using first, last, min, max features.
 *
 * @details A very cheap lower bound that compares extremal features of the two
 *          series. Because DTW must align the first elements together and the
 *          last elements together, the absolute differences of those pairs are
 *          lower bounds. The min-vs-min and max-vs-max comparisons add further
 *          tightness.
 *
 *          Complexity: O(1) if min/max features are precomputed per series.
 *          This convenience overload computes them on the fly in O(n).
 *
 * @tparam T Numeric data type.
 * @param x First series pointer.
 * @param nx Length of first series.
 * @param y Second series pointer.
 * @param ny Length of second series.
 * @return Lower bound value (always <= true DTW distance).
 */
template <typename T>
T lb_kim(const T *x, std::size_t nx, const T *y, std::size_t ny)
{
  if (nx == 0 || ny == 0) return T(0);

  // Feature 1: first elements must align
  T lb = std::abs(x[0] - y[0]);

  // Feature 2: last elements must align
  lb = std::max(lb, std::abs(x[nx - 1] - y[ny - 1]));

  // Features 3-4: min and max comparisons (optional, requires series length >= 2)
  if (nx >= 2 && ny >= 2) {
    const auto [x_min_it, x_max_it] = std::minmax_element(x, x + nx);
    const auto [y_min_it, y_max_it] = std::minmax_element(y, y + ny);

    // The minimum of one series can at best align with the minimum of the other
    lb = std::max(lb, std::abs(*x_min_it - *y_min_it));
    lb = std::max(lb, std::abs(*x_max_it - *y_max_it));
  }

  return lb;
}

/**
 * @brief Convenience overload: LB_Kim from std::vectors.
 *
 * @tparam T Numeric data type.
 * @param x First series.
 * @param y Second series.
 * @return Lower bound value.
 */
template <typename T>
T lb_kim(const std::vector<T> &x, const std::vector<T> &y)
{
  return lb_kim(x.data(), x.size(), y.data(), y.size());
}

// ======================================================================
//  Wrapper types for precomputed LB data
// ======================================================================

/// Precomputed summary statistics for O(1) LB_Kim.
struct SeriesSummary {
  double first = 0, last = 0, min_val = 0, max_val = 0;
};

/// Compute summary from a time series vector.
inline SeriesSummary compute_summary(const std::vector<double> &series)
{
  if (series.empty()) return {};
  SeriesSummary s;
  s.first = series.front();
  s.last = series.back();
  s.min_val = *std::min_element(series.begin(), series.end());
  s.max_val = *std::max_element(series.begin(), series.end());
  return s;
}

/// LB_Kim using precomputed summaries -- O(1).
inline double lb_kim(const SeriesSummary &a, const SeriesSummary &b)
{
  double d = 0;
  d = std::max(d, std::abs(a.first - b.first));
  d = std::max(d, std::abs(a.last - b.last));
  d = std::max(d, std::abs(a.min_val - b.min_val));
  d = std::max(d, std::abs(a.max_val - b.max_val));
  return d;
}

/// Precomputed upper/lower envelopes for LB_Keogh.
struct Envelope {
  std::vector<double> upper, lower;
};

/// Compute envelope from a time series vector with given band width.
inline Envelope compute_envelope(const std::vector<double> &series, int band)
{
  Envelope env;
  env.upper.resize(series.size());
  env.lower.resize(series.size());
  if (!series.empty())
    compute_envelopes(series.data(), series.size(), band, env.upper.data(), env.lower.data());
  return env;
}

/// Convenience overload: LB_Keogh from vector + precomputed Envelope.
inline double lb_keogh(const std::vector<double> &query, const Envelope &env)
{
  const std::size_t n = std::min(query.size(), env.upper.size());
  if (n == 0) return 0.0;
  return lb_keogh(query.data(), n, env.upper.data(), env.lower.data());
}

/// Symmetric LB_Keogh: max of both directions.
inline double lb_keogh_symmetric(
  const std::vector<double> &x, const Envelope &env_x,
  const std::vector<double> &y, const Envelope &env_y)
{
  double lb1 = lb_keogh(x, env_y);
  double lb2 = lb_keogh(y, env_x);
  return std::max(lb1, lb2);
}

// ======================================================================
//  Runtime metric enum + LB compatibility
// ======================================================================

/// Runtime-selectable distance metric enum.
enum class DistanceMetric { L1, L2, SquaredL2 };

/// Check whether lower-bound pruning is valid for a given metric.
inline bool lb_pruning_compatible(DistanceMetric m)
{
  return m == DistanceMetric::L1;
}

} // namespace dtwc::core
