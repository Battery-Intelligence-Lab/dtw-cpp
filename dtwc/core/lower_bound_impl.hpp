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

#include <algorithm> // for min, max, fill, max_element, min_element
#include <cmath>     // for abs
#include <cstddef>   // for size_t
#include <type_traits> // for is_same_v
#include <vector>    // for vector

#ifdef DTWC_HAS_HIGHWAY
namespace dtwc::simd {
// Declared in dtwc/simd/lb_keogh_simd.cpp; signature matches the SIMD kernel.
double lb_keogh_highway(const double *query, const double *upper,
                        const double *lower, std::size_t n);
}  // namespace dtwc::simd
#endif

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

  // Fast path: band covers entire series → envelopes are global min/max.
  if (w >= n) {
    const T *end = series + n;
    const T gmax = *std::max_element(series, end);
    const T gmin = *std::min_element(series, end);
    std::fill(upper_out, upper_out + n, gmax);
    std::fill(lower_out, lower_out + n, gmin);
    return;
  }

  // O(n) Lemire sliding-window min/max for centered window [p-w, p+w].
  //
  // Decomposes into two one-sided trailing/leading windows:
  //   lmax[p] = max over [max(0, p-w), p]       (trailing, forward pass)
  //   rmax[p] = max over [p, min(n-1, p+w)]     (leading, backward pass)
  //   upper[p] = max(lmax[p], rmax[p])
  // Same for min.
  //
  // Ring buffers (contiguous std::vector, NOT std::deque) keep index deques
  // cache-friendly. Capacity w+2 > window width w+1, so the buffer never
  // overflows. Each element is pushed and popped at most once → O(n) total.
  //
  // Reference: D. Lemire, "Streaming Maximum-Minimum Filter Using No More
  // Than Three Comparisons per Element," 2006.

  const std::size_t cap = w + 2;
  std::vector<std::size_t> dmax(cap), dmin(cap);
  std::size_t mx_h = 0, mx_t = 0, mn_h = 0, mn_t = 0;

  // Temporary storage for trailing-window (left-side) results.
  std::vector<T> lmax(n), lmin(n);

  // --- Forward pass: lmax[p] = max([max(0,p-w)..p]), lmin symmetric ---
  for (std::size_t p = 0; p < n; ++p) {
    // Evict indices that have left the trailing window [p-w, p]
    if (mx_h != mx_t && dmax[mx_h % cap] + w < p) ++mx_h;
    if (mn_h != mn_t && dmin[mn_h % cap] + w < p) ++mn_h;
    // Maintain monotone decreasing (max) and increasing (min) deques
    while (mx_h != mx_t && series[dmax[(mx_t - 1) % cap]] <= series[p]) --mx_t;
    while (mn_h != mn_t && series[dmin[(mn_t - 1) % cap]] >= series[p]) --mn_t;
    dmax[mx_t++ % cap] = p;
    dmin[mn_t++ % cap] = p;
    lmax[p] = series[dmax[mx_h % cap]];
    lmin[p] = series[dmin[mn_h % cap]];
  }

  // --- Backward pass: leading window [p, min(n-1,p+w)], merged with lmax/lmin ---
  mx_h = mx_t = mn_h = mn_t = 0;
  for (std::ptrdiff_t q = static_cast<std::ptrdiff_t>(n) - 1; q >= 0; --q) {
    const std::size_t p = static_cast<std::size_t>(q);
    // Evict indices past the leading window [p, p+w]
    if (mx_h != mx_t && dmax[mx_h % cap] > p + w) ++mx_h;
    if (mn_h != mn_t && dmin[mn_h % cap] > p + w) ++mn_h;
    // Maintain monotone deques
    while (mx_h != mx_t && series[dmax[(mx_t - 1) % cap]] <= series[p]) --mx_t;
    while (mn_h != mn_t && series[dmin[(mn_t - 1) % cap]] >= series[p]) --mn_t;
    dmax[mx_t++ % cap] = p;
    dmin[mn_t++ % cap] = p;
    // Centered window = union of trailing + leading; take max/min of both sides
    upper_out[p] = std::max(lmax[p], series[dmax[mx_h % cap]]);
    lower_out[p] = std::min(lmin[p], series[dmin[mn_h % cap]]);
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
#ifdef DTWC_HAS_HIGHWAY
  if constexpr (std::is_same_v<T, double>) {
    return dtwc::simd::lb_keogh_highway(query, upper, lower, n);
  }
#endif
  T sum = T(0);
#if defined(_MSC_VER)
  // MSVC does not support OpenMP reduction clauses on simd directives.
  // The branchless ternaries below map to vmaxpd — MSVC can auto-vectorize them.
#else
  #pragma omp simd reduction(+:sum)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    const T eu = query[i] - upper[i];  // positive when query is above the upper envelope
    const T el = lower[i] - query[i];  // positive when query is below the lower envelope
    // Decompose max(0, max(eu,el)) → max(0,eu) + max(0,el).
    // For a valid envelope L<=U: eu+el = (q-U)+(L-q) = L-U <= 0, so at most one term
    // is positive. Each ternary maps to a single vmaxpd with zero — two independent
    // SIMD ops instead of a nested std::max call chain with a data dependency.
    const T cu = eu > T(0) ? eu : T(0);
    const T cl = el > T(0) ? el : T(0);
    sum += cu + cl;
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

  // O(n) Lemire sliding-window per channel. Interleaved layout: series[t*ndim+d].
  // Deque indices store timesteps (not flat offsets). Same ring-buffer approach as
  // the scalar compute_envelopes — see that function for algorithm commentary.

  if (w >= n_steps) {
    // Fast path: band spans all timesteps → per-channel global min/max.
    for (std::size_t d = 0; d < ndim; ++d) {
      T gmax = series[d], gmin = series[d];
      for (std::size_t t = 1; t < n_steps; ++t) {
        T v = series[t * ndim + d];
        if (v > gmax) gmax = v;
        if (v < gmin) gmin = v;
      }
      for (std::size_t t = 0; t < n_steps; ++t) {
        upper_out[t * ndim + d] = gmax;
        lower_out[t * ndim + d] = gmin;
      }
    }
    return;
  }

  const std::size_t cap = w + 2;
  std::vector<std::size_t> dmax(cap), dmin(cap);
  std::vector<T> lmax(n_steps), lmin(n_steps);

  for (std::size_t d = 0; d < ndim; ++d) {
    std::size_t mx_h = 0, mx_t = 0, mn_h = 0, mn_t = 0;

    // Forward pass: lmax[p] = max([max(0,p-w)..p]) for channel d
    for (std::size_t p = 0; p < n_steps; ++p) {
      const T v = series[p * ndim + d];
      if (mx_h != mx_t && dmax[mx_h % cap] + w < p) ++mx_h;
      if (mn_h != mn_t && dmin[mn_h % cap] + w < p) ++mn_h;
      while (mx_h != mx_t && series[dmax[(mx_t - 1) % cap] * ndim + d] <= v) --mx_t;
      while (mn_h != mn_t && series[dmin[(mn_t - 1) % cap] * ndim + d] >= v) --mn_t;
      dmax[mx_t++ % cap] = p;
      dmin[mn_t++ % cap] = p;
      lmax[p] = series[dmax[mx_h % cap] * ndim + d];
      lmin[p] = series[dmin[mn_h % cap] * ndim + d];
    }

    // Backward pass: leading window merged with lmax/lmin
    mx_h = mx_t = mn_h = mn_t = 0;
    for (std::ptrdiff_t q = static_cast<std::ptrdiff_t>(n_steps) - 1; q >= 0; --q) {
      const std::size_t p = static_cast<std::size_t>(q);
      const T v = series[p * ndim + d];
      if (mx_h != mx_t && dmax[mx_h % cap] > p + w) ++mx_h;
      if (mn_h != mn_t && dmin[mn_h % cap] > p + w) ++mn_h;
      while (mx_h != mx_t && series[dmax[(mx_t - 1) % cap] * ndim + d] <= v) --mx_t;
      while (mn_h != mn_t && series[dmin[(mn_t - 1) % cap] * ndim + d] >= v) --mn_t;
      dmax[mx_t++ % cap] = p;
      dmin[mn_t++ % cap] = p;
      upper_out[p * ndim + d] = std::max(lmax[p], series[dmax[mx_h % cap] * ndim + d]);
      lower_out[p * ndim + d] = std::min(lmin[p], series[dmin[mn_h % cap] * ndim + d]);
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
      const T eu = query[idx] - upper[idx];
      const T el = lower[idx] - query[idx];
      const T cu = eu > T(0) ? eu : T(0);
      const T cl = el > T(0) ? el : T(0);
      sum += cu + cl;
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
#if defined(_MSC_VER)
  // MSVC does not support OpenMP reduction clauses on simd directives.
#else
  #pragma omp simd reduction(+:sum)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    const T eu = query[i] - upper[i];  // positive when query is above the upper envelope
    const T el = lower[i] - query[i];  // positive when query is below the lower envelope
    // Branchless: at most one of eu, el is positive (L<=U for valid envelopes).
    const T cu = eu > T(0) ? eu : T(0);
    const T cl = el > T(0) ? el : T(0);
    const T excess = cu + cl;
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
      const T eu = query[idx] - upper[idx];
      const T el = lower[idx] - query[idx];
      const T cu = eu > T(0) ? eu : T(0);
      const T cl = el > T(0) ? el : T(0);
      const T excess = cu + cl;
      sum += excess * excess;
    }
  }
  return sum;
}

} // namespace dtwc::core
