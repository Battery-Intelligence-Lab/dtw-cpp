/**
 * @file lower_bound_impl.hpp
 * @brief Lower bound functions for DTW pruning (LB_Keogh, LB_Kim).
 *
 * @details Header-only implementations of O(n) and O(1) lower bounds on DTW
 *          distance. A consumer that does not need every exact distance may
 *          skip a pair when a bound clears its threshold. The legacy exact
 *          distance-matrix route can only use a bound to select a cutoff
 *          attempt; an abandoned pair is recomputed without a cutoff.
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

#include <algorithm>   // for min, max, fill, max_element, min_element
#include <cmath>       // for abs
#include <cstddef>     // for size_t
#include <limits>      // for numeric_limits
#include <span>        // for span
#include <type_traits> // for is_same_v
#include <vector>      // for vector

#include "distance_metric.hpp" // for L1Metric, SquaredL2Metric (delta functors)

namespace dtwc::core {

/**
 * @brief Compute Sakoe-Chiba upper/lower envelopes for a time series.
 *
 * @details The envelopes are precomputed ONCE per series and reused O(N) times
 *          when computing LB_Keogh against multiple query series. For each
 *          position i, the upper envelope is the max and the lower envelope is
 *          the min of the series values within the Sakoe-Chiba band [i-band, i+band].
 *
 *          A negative band is coerced to radius zero; it does not request a
 *          full-DTW envelope. For full DTW, an admissible Keogh construction
 *          needs the global envelope (radius at least n-1). The input and two
 *          output ranges must not overlap; this unchecked pointer routine does
 *          not detect destructive aliasing (F46).
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
 * @param band Sakoe-Chiba band width. Negative values are coerced to radius
 *             zero, not interpreted as full DTW.
 * @param upper_out Output upper envelope vector (resized to match series).
 * @param lower_out Output lower envelope vector (resized to match series).
 *
 * @warning series, upper_out, and lower_out must be three distinct vectors;
 *          aliasing is not checked (F46).
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
 *          the CANDIDATE series. Against a fixed DTW radius w, admissibility
 *          requires an envelope radius r >= w that covers every candidate
 *          index reachable from each included query row. Full DTW therefore
 *          requires a global envelope. Inputs must be finite, lower[i] <=
 *          upper[i], and all three arrays must contain at least n elements.
 *          This unchecked kernel cannot validate shape, radius, or source
 *          provenance (F46). The result has the same units as an L1 DTW sum.
 *          For unequal lengths, D2 separately derives the admissibility of
 *          including the first min(query length, candidate length) rows when
 *          the fixed window is feasible and covered; that prefix result is a
 *          repository theorem, not part of the source paper's equal-length
 *          proposition.
 *
 *          If query[i] lies within [lower[i], upper[i]], it contributes 0 to the
 *          lower bound. Otherwise, the contribution is the distance to the
 *          nearest envelope boundary.
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer.
 * @param n Number of query rows included in the bound.
 * @param upper Upper envelope of the candidate series.
 * @param lower Lower envelope of the candidate series.
 * @return Lower-bound value under the stated coverage assumptions.
 */
template <typename T>
T lb_keogh(const T *query, std::size_t n,
           const T *upper, const T *lower)
{
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
 *
 * @warning The current overload does not validate either envelope length. Both
 *          vectors must contain at least query.size() elements (F46).
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
 *          The current implementation is in L1 units: every feature gap is an
 *          unsquared absolute difference. It is not generally admissible for a
 *          squared-L2 DTW even though lb_kim_valid<SquaredL2Metric> currently
 *          advertises that combination; F47 owns that runtime/trait mismatch.
 *
 *          Complexity: O(1) if min/max features are precomputed per series.
 *          This convenience overload computes them on the fly in O(n).
 *
 * @tparam T Numeric data type.
 * @param x First series pointer.
 * @param nx Length of first series.
 * @param y Second series pointer.
 * @param ny Length of second series.
 * @return L1-valued lower bound for finite, nonempty scalar L1/scalar-L2 DTW.
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

/// Compute summary from a time series span.
inline SeriesSummary compute_summary(std::span<const double> series)
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

/// Compute summary from a time series vector (convenience overload).
inline SeriesSummary compute_summary(const std::vector<double> &series)
{
  return compute_summary(std::span<const double>(series));
}

/// LB_Kim using precomputed summaries -- O(1), in L1 units (F47).
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
///
/// Envelope carries no source-length or radius provenance, and its mutable
/// arrays may have different lengths. Public validation is deferred to F46.
struct Envelope {
  std::vector<double> upper, lower;
};

/// Compute envelope from a span; a negative band remains radius zero (F46).
/// The returned Envelope does not record the source length or resolved radius.
inline Envelope compute_envelope(std::span<const double> series, int band)
{
  Envelope env;
  env.upper.resize(series.size());
  env.lower.resize(series.size());
  if (!series.empty())
    compute_envelopes(series.data(), series.size(), band, env.upper.data(), env.lower.data());
  return env;
}

/// Compute envelope from a time series vector (convenience overload).
inline Envelope compute_envelope(const std::vector<double> &series, int band)
{
  return compute_envelope(std::span<const double>(series), band);
}

/// LB_Keogh from span + precomputed Envelope.
///
/// Current F46 behavior truncates to min(query.size(), env.upper.size()),
/// ignores env.lower.size(), and cannot verify source or radius provenance.
inline double lb_keogh(std::span<const double> query, const Envelope &env)
{
  const auto n = std::min(query.size(), env.upper.size());
  return lb_keogh(query.data(), n, env.upper.data(), env.lower.data());
}

/// Convenience overload with the same unchecked F46 truncation behavior.
inline double lb_keogh(const std::vector<double> &query, const Envelope &env)
{
  const std::size_t n = std::min(query.size(), env.upper.size());
  if (n == 0) return 0.0;
  return lb_keogh(query.data(), n, env.upper.data(), env.lower.data());
}

/// Symmetric LB_Keogh: max of both directions.
inline double lb_keogh_symmetric(
  std::span<const double> x, const Envelope &env_x,
  std::span<const double> y, const Envelope &env_y)
{
  double lb1 = lb_keogh(x, env_y);
  double lb2 = lb_keogh(y, env_x);
  return std::max(lb1, lb2);
}

/// Convenience overload for vectors.
inline double lb_keogh_symmetric(
  const std::vector<double> &x, const Envelope &env_x,
  const std::vector<double> &y, const Envelope &env_y)
{
  return lb_keogh_symmetric(
    std::span<const double>(x), env_x,
    std::span<const double>(y), env_y);
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
 *          The bounds formed from these per-channel envelopes are
 *          repository-derived extensions, not claims from the scalar Keogh
 *          proposition. Coordinatewise projection proves the dependent
 *          additive-cost form; summing separately minimized scalar bounds
 *          proves the independent additive-cost form. Both inherit the scalar
 *          coverage, finiteness, negative-band, and non-aliasing preconditions.
 *
 * @tparam T Numeric data type (float, double).
 * @param series Input multivariate series, interleaved layout, n_steps * ndim elements.
 * @param n_steps Number of timesteps.
 * @param ndim Number of channels (dimensions).
 * @param band Sakoe-Chiba band width (half-window radius). Negative values
 *             are coerced to radius zero, not interpreted as full DTW.
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
 *          This repository-derived extension is admissible for dependent
 *          multivariate DTW with an additive L1 channel cost and one shared
 *          path. It is also admissible for independent DTW whose objective
 *          sums the separately minimized per-channel L1 DTWs. Both require the
 *          scalar envelope coverage and finiteness assumptions.
 *          Channels must share a commensurate unit or be scaled/nondimensionalized
 *          before the unweighted sum has a physical-unit interpretation.
 *          For ndim==1 this delegates to the scalar lb_keogh() (zero overhead).
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer, interleaved layout, n_steps * ndim elements.
 * @param n_steps Number of timesteps.
 * @param ndim Number of channels.
 * @param upper Upper envelope of the candidate (from compute_envelopes_mv).
 * @param lower Lower envelope of the candidate (from compute_envelopes_mv).
 * @return Lower-bound value under the stated additive-cost and coverage assumptions.
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
 * @details A lower bound on DTW computed with the SquaredL2 metric when the
 *          scalar envelope coverage, shape, and finiteness assumptions stated
 *          for lb_keogh() hold. The result is in squared-data units.
 *          If the query point lies within [lower[i], upper[i]], the contribution
 *          is zero. Otherwise it is the square of the distance to the nearest boundary.
 *
 * @tparam T Numeric data type.
 * @param query Query series pointer.
 * @param n Number of query rows included in the bound; both envelope arrays
 *          must contain at least n elements.
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
 *          This repository-derived extension is admissible for dependent
 *          multivariate DTW with additive squared channel costs and one shared
 *          path, and for independent DTW that sums separately minimized
 *          per-channel squared-L2 DTWs. It requires the scalar coverage and
 *          finiteness assumptions and returns squared-data units; channels
 *          must be commensurate or pre-scaled for that unit ledger.
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

// ======================================================================
//  LB_Enhanced (Tan, Petitjean & Webb, SDM 2019) — elastic-band bound
// ======================================================================
//
//  Reference: C. W. Tan, F. Petitjean, G. I. Webb, "Elastic bands across the
//  path: A new framework and method to lower bound DTW," SDM 2019
//  (arXiv:1808.09617), Eq. 3.7 & Theorem 3.2. Cross-checked against the
//  authors' MATLAB (lbEnhanced.m) and Java (LbEnhanced.java).
//
//  Idea. Near the two ends the boundary conditions pin the warping path
//  (A[0]<->B[0], A[n-1]<->B[n-1]), so the first/last V columns admit only a
//  narrow set of alignments; we take the exact minimum over each such "band".
//  The middle uses the ordinary LB_Keogh envelope term. Theorem 3.2 proves the
//  band sets L_1..L_V, the middle envelope sets, and the mirror sets R_1..R_V
//  are MUTUALLY DISJOINT and each is crossed by every warping path, so the sum
//  of per-set minima is <= DTW_w for any V <= n/2 and any nonnegative metric.
//
//  The near-side clip (j < i only) is what keeps the sets disjoint: extending a
//  band arm to j > i would place a cell in two adjacent bands and can push the
//  sum above DTW. NOTE: LB_Enhanced is NOT provably >= LB_Keogh pointwise — the
//  column arm delta(A[j],B[i]) may undercut the Keogh term at a position; its
//  advantage over Keogh is empirical/on-average (SDM 2019 §5).
//
//  Validity requires: |A|==|B|==n, the SAME window w in the bound and the DTW
//  it bounds, the SAME metric, and V <= n/2 (enforced by nBands=min(V,n/2)).

/**
 * @brief LB_Enhanced core: elastic-band lower bound on banded DTW_w.
 *
 * @tparam T      Numeric type.
 * @tparam Metric Pointwise cost functor, metric(a,b) (L1Metric or SquaredL2Metric).
 * @param A       Query series pointer (length n).
 * @param B       Candidate series pointer (length n).
 * @param n       Common series length.
 * @param upper_B Upper envelope of B (radius = band).
 * @param lower_B Lower envelope of B.
 * @param band    Sakoe-Chiba window radius w; negative values become zero and
 *                values above n-1 become the equivalent global radius n-1.
 * @param V       Bands per end (>= 1); clamped to n/2 for validity.
 * @param metric  Pointwise cost (default L1).
 * @return Lower bound (summed cost, same metric as the bounded DTW).
 */
template <typename T, typename Metric = L1Metric>
T lb_enhanced(const T *A, const T *B, std::size_t n,
              const T *upper_B, const T *lower_B,
              int band, int V = 5, Metric metric = Metric{})
{
  if (n == 0) return T(0);
  if (n == 1) return metric(A[0], B[0]);

  const int ni = static_cast<int>(n);
  const int w = std::min(std::max(band, 0), ni - 1);
  int nBands = std::min(V, ni / 2);
  if (nBands < 1) nBands = 1;

  // (1) Forced corners (boundary conditions).
  T d = metric(A[0], B[0]) + metric(A[ni - 1], B[ni - 1]);

  // (2) V-1 elastic bands at each end (band 0 == the corner, already counted).
  for (int i = 1; i < nBands; ++i) {
    const int ir = ni - 1 - i;                     // mirror index from the right
    T minL = metric(A[i], B[i]);                   // diagonal cell (i,i)
    T minR = metric(A[ir], B[ir]);                 // diagonal cell (ir,ir)
    const int jlo = std::max(0, i - w);
    for (int j = jlo; j < i; ++j) {                // near side only: j < i
      const int jr = ni - 1 - j;
      minL = std::min(minL, std::min(metric(A[i], B[j]), metric(A[j], B[i])));
      minR = std::min(minR, std::min(metric(A[ir], B[jr]), metric(A[jr], B[ir])));
    }
    d += minL + minR;
  }

  // (3) LB_Keogh middle over [nBands, n-nBands) (A vs envelope of B).
  for (int i = nBands; i < ni - nBands; ++i) {
    if (A[i] > upper_B[i]) d += metric(A[i], upper_B[i]);
    else if (A[i] < lower_B[i]) d += metric(A[i], lower_B[i]);
  }
  return d;
}

/// LB_Enhanced from span query + precomputed Envelope of the candidate (L1).
template <typename Metric = L1Metric>
double lb_enhanced(std::span<const double> query, std::span<const double> candidate,
                   const Envelope &env_candidate, int band, int V = 5,
                   Metric metric = Metric{})
{
  const std::size_t n = query.size();
  if (n == 0 || candidate.size() != n || env_candidate.upper.size() != n) return 0.0;
  return lb_enhanced<double, Metric>(query.data(), candidate.data(), n,
                                     env_candidate.upper.data(),
                                     env_candidate.lower.data(), band, V, metric);
}

/// Symmetric LB_Enhanced: max over both query/candidate roles (>= either direction).
template <typename Metric = L1Metric>
double lb_enhanced_symmetric(std::span<const double> x, const Envelope &env_x,
                             std::span<const double> y, const Envelope &env_y,
                             int band, int V = 5, Metric metric = Metric{})
{
  const double lb_xy = lb_enhanced<Metric>(x, y, env_y, band, V, metric);
  const double lb_yx = lb_enhanced<Metric>(y, x, env_x, band, V, metric);
  return std::max(lb_xy, lb_yx);
}

// ======================================================================
//  LB_Webb (Webb & Petitjean, Pattern Recognition 2021) — envelope bound
// ======================================================================
//
//  Reference: G. I. Webb & F. Petitjean, "Tight lower bounds for dynamic time
//  warping," Pattern Recognition 115 (2021) 107895 (arXiv:2102.07076), Alg. 2
//  & Thm 2. Implemented clean-room from the algorithm (the authors' Java is
//  GPL-3.0 and is NOT reproduced here).
//
//  Two passes over equal-length A, B with window w:
//    Pass 1 (bridge, A vs envelope of B): the ordinary one-directional LB_Keogh
//      contribution, while tracking how many CONSECUTIVE positions A has stayed
//      "free" (inside, or outside in a way covered by the secondary envelope).
//    Pass 2 (B vs envelope of A): add a correction term for each B position
//      outside A's envelope. Thm 2 guarantees these terms do not double-count
//      the pass-1 contribution: when the surrounding window is fully free the
//      full term delta(B_i,U^A_i) is safe; otherwise the secondary envelope
//      UL^B/LU^B supplies the exact overlap to subtract off.
//
//  Because pass 1 IS one-directional LB_Keogh and every pass-2 term is
//  nonnegative, LB_Webb(A,B) >= LB_Keogh(A, env B) ALWAYS; the symmetric
//  version therefore dominates symmetric LB_Keogh. (LB_Webb is NOT ordered
//  against LB_Improved — tighter on most UCR sets, looser on some.)
//
//  This implementation OMITS the paper's MinLRPaths corner DP (an O(1) exact
//  tightening of the first/last 3 alignments): running the plain bridge at the
//  corners only LOOSENS the bound, never breaks validity. The validity contract
//  LB_Webb <= DTW_w is the hard gate (adversarial test).
//
//  Free-flag alignment: pass 1 sets free[k] once the trailing window [k-2w, k]
//  is entirely free; a B column j is "free" iff its centred window [j-w, j+w]
//  is free, i.e. free[j+w] (capped at n-1 in the tail — a conservative, still
//  valid, subset test). Counters init to w to model the pinned pre/post-series
//  boundary as free.
//
//  Validity requires: |A|==|B|==n, the SAME window w in the bound, envelopes,
//  and the DTW it bounds, and a metric satisfying Thm 2 (L1 with equality;
//  SquaredL2 satisfied).

/// Precomputed envelopes for LB_Webb: primary U,L plus secondary LU=L(U), UL=U(L).
struct WebbEnvelope {
  std::vector<double> upper;  ///< U(S)
  std::vector<double> lower;  ///< L(S)
  std::vector<double> lu;     ///< L(U(S)) — lower envelope of the upper envelope
  std::vector<double> ul;     ///< U(L(S)) — upper envelope of the lower envelope
};

/// Compute the four LB_Webb envelope arrays for a series (window radius = band).
inline WebbEnvelope compute_webb_envelope(std::span<const double> series, int band)
{
  WebbEnvelope we;
  const std::size_t n = series.size();
  we.upper.resize(n); we.lower.resize(n); we.lu.resize(n); we.ul.resize(n);
  if (n == 0) return we;
  compute_envelopes(series.data(), n, band, we.upper.data(), we.lower.data());
  std::vector<double> scratch(n);
  // LU = lower envelope of the upper envelope (discard the upper-of-upper).
  compute_envelopes(we.upper.data(), n, band, scratch.data(), we.lu.data());
  // UL = upper envelope of the lower envelope (discard the lower-of-lower).
  compute_envelopes(we.lower.data(), n, band, we.ul.data(), scratch.data());
  return we;
}

/// Convenience overload from a std::vector.
inline WebbEnvelope compute_webb_envelope(const std::vector<double> &series, int band)
{
  return compute_webb_envelope(std::span<const double>(series), band);
}

/**
 * @brief LB_Webb (one-directional): lower bound on banded DTW_w, A query, B candidate.
 *
 * @tparam Metric Pointwise cost functor (L1Metric or SquaredL2Metric).
 * @param A     Query series (length n).
 * @param ea    Webb envelopes of A (window = band).
 * @param B     Candidate series (length n).
 * @param eb    Webb envelopes of B (window = band).
 * @param band  Window radius w; negative values become zero and values above
 *              n-1 become the equivalent global radius n-1.
 * @param free_scratch  Scratch of at least n chars, reused across calls (may be nullptr).
 * @param metric Pointwise cost.
 * @return Lower bound (summed cost). >= LB_Keogh(A, env B).
 */
template <typename Metric = L1Metric>
double lb_webb(std::span<const double> A, const WebbEnvelope &ea,
               std::span<const double> B, const WebbEnvelope &eb,
               int band, std::vector<char> *free_scratch = nullptr,
               Metric metric = Metric{})
{
  const std::size_t n = A.size();
  if (n == 0 || B.size() != n || ea.upper.size() != n || eb.upper.size() != n)
    return 0.0;
  const std::size_t w = std::min(
    static_cast<std::size_t>(std::max(band, 0)), n - 1);
  constexpr std::size_t max_size = std::numeric_limits<std::size_t>::max();
  const std::size_t two_w = w > max_size - w ? max_size : 2 * w;
  const auto increment_saturated = [](std::size_t value) noexcept {
    constexpr std::size_t limit = std::numeric_limits<std::size_t>::max();
    return value == limit ? value : value + 1;
  };

  std::vector<char> local;
  std::vector<char> &freeAbove = free_scratch ? *free_scratch : local;
  // Layout: first n entries = free-above flags, next n = free-below flags.
  freeAbove.assign(2 * n, 0);
  char *Fa = freeAbove.data();
  char *Fb = freeAbove.data() + n;

  const auto &UB = eb.upper; const auto &LB = eb.lower;   // envelope of B
  const auto &ULA = ea.ul;   const auto &LUA = ea.lu;     // secondary of A

  double b = 0.0;
  std::size_t cUp = w, cLo = w;                            // pre-series counted free
  for (std::size_t i = 0; i < n; ++i) {
    const double ai = A[i];
    if (ai > UB[i]) {
      b += metric(ai, UB[i]);
      cUp = 0;
      cLo = (UB[i] >= ULA[i]) ? increment_saturated(cLo) : 0;
    } else if (ai < LB[i]) {
      b += metric(ai, LB[i]);
      cLo = 0;
      cUp = (LB[i] <= LUA[i]) ? increment_saturated(cUp) : 0;
    } else {
      cUp = increment_saturated(cUp);
      cLo = increment_saturated(cLo);
    }
    Fa[i] = (cUp > two_w) ? 1 : 0;
    Fb[i] = (cLo > two_w) ? 1 : 0;
  }

  const auto &UA = ea.upper; const auto &LA = ea.lower;   // envelope of A
  const auto &ULB = eb.ul;   const auto &LUB = eb.lu;     // secondary of B
  for (std::size_t j = 0; j < n; ++j) {
    const std::size_t idx = j + std::min(w, n - 1 - j);
    const double bj = B[j];
    if (Fa[idx] && bj > UA[j]) {
      b += metric(bj, UA[j]);
    } else if (Fb[idx] && bj < LA[j]) {
      b += metric(bj, LA[j]);
    } else if (bj > ULB[j] && ULB[j] >= UA[j]) {
      b += metric(bj, UA[j]) - metric(ULB[j], UA[j]);
    } else if (bj < LUB[j] && LUB[j] <= LA[j]) {
      b += metric(bj, LA[j]) - metric(LUB[j], LA[j]);
    }
  }
  return b;
}

/// Symmetric LB_Webb: max over both roles. Dominates symmetric LB_Keogh.
template <typename Metric = L1Metric>
double lb_webb_symmetric(std::span<const double> x, const WebbEnvelope &ex,
                         std::span<const double> y, const WebbEnvelope &ey,
                         int band, std::vector<char> *free_scratch = nullptr,
                         Metric metric = Metric{})
{
  const double lb_xy = lb_webb<Metric>(x, ex, y, ey, band, free_scratch, metric);
  const double lb_yx = lb_webb<Metric>(y, ey, x, ex, band, free_scratch, metric);
  return std::max(lb_xy, lb_yx);
}

} // namespace dtwc::core
