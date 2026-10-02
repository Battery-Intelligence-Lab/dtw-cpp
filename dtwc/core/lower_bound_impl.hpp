/**
 * @file lower_bound_impl.hpp
 * @brief Envelopes and LB_Keogh, the lower bound TADPole prunes pairs with.
 *
 * @details Header-only O(n) lower bound on DTW distance. A consumer that does
 *          not need every exact distance may skip a pair when the bound clears
 *          its threshold; an exact all-pairs matrix cannot use it.
 *
 *          References:
 *          - E. Keogh, C. A. Ratanamahatana, "Exact indexing of dynamic time
 *            warping," Knowledge and Information Systems 7.3 (2005): 358-386.
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 28 Mar 2026
 */

#pragma once

#include <algorithm>   // for min, max, fill, max_element, min_element
#include <cstddef>     // for size_t
#include <span>        // for span
#include <vector>      // for vector

namespace dtwc::core {

/**
 * @brief Compute Sakoe-Chiba upper/lower envelopes for a time series.
 *
 * @details The envelopes are precomputed ONCE per series and reused O(N) times
 *          when computing LB_Keogh against multiple query series. For each
 *          position i, the upper envelope is the max and the lower envelope is
 *          the min of the series values within the Sakoe-Chiba band [i-band, i+band].
 *
 *          A negative band requests the full-DTW envelope: full DTW may align
 *          any two indices, so every position holds the global min and max,
 *          the only admissible Keogh envelope for it. The input and two
 *          output ranges must not overlap; this unchecked pointer routine does
 *          not detect destructive aliasing.
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
  const std::size_t w = (band < 0) ? n : static_cast<std::size_t>(band);

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
 * @param band Sakoe-Chiba band width; negative means full DTW (the global
 *             envelope).
 * @param upper_out Output upper envelope vector (resized to match series).
 * @param lower_out Output lower envelope vector (resized to match series).
 *
 * @warning series, upper_out, and lower_out must be three distinct vectors;
 *          aliasing is not checked.
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
 *          provenance. The result has the same units as an L1 DTW sum.
 *          For unequal lengths, this repository separately derives the
 *          admissibility of including the first min(query length, candidate
 *          length) rows when the fixed window is feasible and covered
 *          (tests/unit/core/test_lb_keogh_derivation.cpp); that prefix result is
 *          a repository theorem, not part of the source paper's equal-length
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
 *          vectors must contain at least query.size() elements.
 */
template <typename T>
T lb_keogh(const std::vector<T> &query,
           const std::vector<T> &upper, const std::vector<T> &lower)
{
  const std::size_t n = query.size();
  return lb_keogh(query.data(), n, upper.data(), lower.data());
}

/// Precomputed upper/lower envelopes for LB_Keogh.
///
/// Envelope carries no source-length or radius provenance, and its mutable
/// arrays may be resized independently, so every entry point checks EVERY
/// array it will index before touching one.
struct Envelope {
  std::vector<double> upper, lower;
};

/// True when EVERY envelope array can be indexed over [0, n). LB_Keogh admits a
/// shorter prefix, so it needs coverage, not equality.
inline bool envelope_covers(const Envelope &env, std::size_t n) noexcept
{
  return env.upper.size() >= n && env.lower.size() >= n;
}

/// Compute envelope from a span; a negative band builds the global envelope.
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
/// Includes the first min(query.size(), env.upper.size()) rows: by the
/// unequal-length prefix theorem the dropped rows contribute only nonnegative
/// terms, so the prefix bound stays admissible. A ragged envelope (a `lower`
/// shorter than that prefix) would be read out of bounds, so 0 -- itself a
/// valid bound -- is returned instead. Radius provenance is still unrecorded.
inline double lb_keogh(std::span<const double> query, const Envelope &env)
{
  const auto n = std::min(query.size(), env.upper.size());
  if (n == 0 || !envelope_covers(env, n)) return 0.0;
  return lb_keogh(query.data(), n, env.upper.data(), env.lower.data());
}

/// Convenience overload with the same prefix and coverage contract.
inline double lb_keogh(const std::vector<double> &query, const Envelope &env)
{
  return lb_keogh(std::span<const double>(query), env);
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

} // namespace dtwc::core
