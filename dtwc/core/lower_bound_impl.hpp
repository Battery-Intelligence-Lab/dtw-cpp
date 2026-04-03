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
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 28 Mar 2026
 */

#pragma once

#include <algorithm> // for min, max, minmax_element
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
  if (n == 0) return;
  const std::size_t w = static_cast<std::size_t>(std::max(band, 0));

  // Sliding window min/max using direct scan over the band window.
  // O(n * min(band, n)) total. For the typical DTW use case (band << n),
  // this is cache-friendly (contiguous reads within the band) and allocation-free.
  // Envelopes are computed once per series and reused O(N) times for LB pruning.
  for (std::size_t p = 0; p < n; ++p) {
    const std::size_t lo = (p >= w) ? p - w : 0;
    const std::size_t hi = std::min(p + w + 1, n);
    T max_val = series[lo];
    T min_val = series[lo];
    for (std::size_t j = lo + 1; j < hi; ++j) {
      if (series[j] > max_val) max_val = series[j];
      if (series[j] < min_val) min_val = series[j];
    }
    upper_out[p] = max_val;
    lower_out[p] = min_val;
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
  #pragma omp simd reduction(+:sum)
  for (std::size_t i = 0; i < n; ++i) {
    T excess_upper = query[i] - upper[i];
    T excess_lower = lower[i] - query[i];
    sum += std::max(T(0), std::max(excess_upper, excess_lower));
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
  auto [it_min, it_max] = std::minmax_element(series.begin(), series.end());
  s.min_val = *it_min;
  s.max_val = *it_max;
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

// ======================================================================
//  Multivariate LB_Keogh (per-channel envelopes, interleaved layout)
// ======================================================================

/**
 * @brief Compute per-channel Sakoe-Chiba envelopes for a multivariate series.
 *
 * @details Each channel is enveloped independently with stride @p ndim.
 *          Input and output arrays use the same interleaved layout:
 *          element at timestep t, channel d is at index t*ndim+d.
 *          For ndim==1 this delegates to compute_envelopes() (zero overhead).
 *
 *          The per-channel lower bound computed from these envelopes is a
 *          valid lower bound on dependent multivariate DTW (Keogh 2005).
 *
 * @tparam T Numeric data type (float, double).
 * @param series Input multivariate series, interleaved layout, n_steps * ndim elements.
 * @param n_steps Number of timesteps.
 * @param ndim Number of channels (dimensions).
 * @param band Sakoe-Chiba band width (half-window radius).
 * @param upper_out Output: upper envelope, same interleaved layout (pre-allocated).
 * @param lower_out Output: lower envelope, same interleaved layout (pre-allocated).
 */
template <typename T>
void compute_envelopes_mv(const T *series, std::size_t n_steps, std::size_t ndim,
                          int band, T *upper_out, T *lower_out)
{
  if (n_steps == 0 || ndim == 0) return;
  if (ndim == 1) {
    compute_envelopes(series, n_steps, band, upper_out, lower_out);
    return;
  }

  const std::size_t w = static_cast<std::size_t>(std::max(band, 0));

  for (std::size_t d = 0; d < ndim; ++d) {
    for (std::size_t p = 0; p < n_steps; ++p) {
      const std::size_t lo = (p >= w) ? p - w : 0;
      const std::size_t hi = std::min(p + w + 1, n_steps);
      T max_val = series[lo * ndim + d];
      T min_val = series[lo * ndim + d];
      for (std::size_t j = lo + 1; j < hi; ++j) {
        T val = series[j * ndim + d];
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
      }
      upper_out[p * ndim + d] = max_val;
      lower_out[p * ndim + d] = min_val;
    }
  }
}

/**
 * @brief LB_Keogh for multivariate interleaved series using L1 metric.
 *
 * @details Sums per-channel LB_Keogh contributions across all channels.
 *          This is a valid lower bound on dependent multivariate DTW under the
 *          Sakoe-Chiba band constraint (proven in Keogh & Ratanamahatana 2005).
 *          For ndim==1 this delegates to the scalar lb_keogh() (zero overhead).
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer, interleaved layout, n_steps * ndim elements.
 * @param n_steps Number of timesteps.
 * @param ndim Number of channels.
 * @param upper Upper envelope of the candidate (from compute_envelopes_mv).
 * @param lower Lower envelope of the candidate (from compute_envelopes_mv).
 * @return Lower bound value (always <= true DTW distance under the band constraint).
 */
template <typename T>
T lb_keogh_mv(const T *query, std::size_t n_steps, std::size_t ndim,
              const T *upper, const T *lower)
{
  if (ndim == 1) return lb_keogh(query, n_steps, upper, lower);

  T sum = T(0);
  for (std::size_t i = 0; i < n_steps; ++i) {
    for (std::size_t d = 0; d < ndim; ++d) {
      const std::size_t idx = i * ndim + d;
      T excess_upper = query[idx] - upper[idx];
      T excess_lower = lower[idx] - query[idx];
      sum += std::max(T(0), std::max(excess_upper, excess_lower));
    }
  }
  return sum;
}

// ======================================================================
//  SquaredL2 LB_Keogh variants
// ======================================================================

/**
 * @brief LB_Keogh with SquaredL2 metric: sum of squared distances to envelope boundary.
 *
 * @details A valid lower bound on DTW computed with the SquaredL2 metric.
 *          If the query point lies within [lower[i], upper[i]], the contribution
 *          is zero. Otherwise it is the square of the distance to the nearest boundary.
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer.
 * @param n Length of the series (must match envelope length).
 * @param upper Upper envelope of the candidate.
 * @param lower Lower envelope of the candidate.
 * @return SquaredL2 lower bound value.
 */
template <typename T>
T lb_keogh_squared(const T *query, std::size_t n,
                   const T *upper, const T *lower)
{
  T sum = T(0);
  for (std::size_t i = 0; i < n; ++i) {
    T excess = T(0);
    if (query[i] > upper[i]) excess = query[i] - upper[i];
    else if (query[i] < lower[i]) excess = lower[i] - query[i];
    sum += excess * excess;
  }
  return sum;
}

/**
 * @brief LB_Keogh SquaredL2 for multivariate interleaved series.
 *
 * @details Sums squared per-channel LB contributions across all channels.
 *          Valid lower bound on dependent multivariate DTW using SquaredL2 metric.
 *          For ndim==1 this delegates to lb_keogh_squared() (zero overhead).
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer, interleaved layout, n_steps * ndim elements.
 * @param n_steps Number of timesteps.
 * @param ndim Number of channels.
 * @param upper Upper envelope of the candidate (from compute_envelopes_mv).
 * @param lower Lower envelope of the candidate (from compute_envelopes_mv).
 * @return SquaredL2 lower bound value.
 */
template <typename T>
T lb_keogh_mv_squared(const T *query, std::size_t n_steps, std::size_t ndim,
                      const T *upper, const T *lower)
{
  if (ndim == 1) return lb_keogh_squared(query, n_steps, upper, lower);

  T sum = T(0);
  for (std::size_t i = 0; i < n_steps; ++i) {
    for (std::size_t d = 0; d < ndim; ++d) {
      const std::size_t idx = i * ndim + d;
      T excess = T(0);
      if (query[idx] > upper[idx]) excess = query[idx] - upper[idx];
      else if (query[idx] < lower[idx]) excess = lower[idx] - query[idx];
      sum += excess * excess;
    }
  }
  return sum;
}

} // namespace dtwc::core
