/**
 * @file z_normalize.hpp
 * @brief Z-normalization utility for time series.
 *
 * @details Provides in-place and copying z-normalization (zero mean, unit
 * standard deviation). Series with near-zero standard deviation are set to
 * all zeros to avoid numerical instability.
 *
 * Algorithm: three passes for numerical stability.
 *   Pass 1: compute mean = sum(x) / n
 *   Pass 2: compute stddev = sqrt(sum((x-mean)²) / n)
 *   Pass 3: normalize in place: (x - mean) * inv_stddev
 *
 * The SIMD variant (z_normalize_simd.cpp) fuses passes 1+2 via the
 * König-Huygens identity (var = E[x²] - mean²), halving memory passes.
 * The scalar version keeps the three-pass approach for numerical stability:
 * - König-Huygens suffers catastrophic cancellation when mean >> stddev
 *   (e.g. a series at offset 1e12 with small variation loses all variance bits).
 * - The three-pass approach computes (x - mean) exactly for constant series,
 *   giving exact zero output rather than a tiny FP residual.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace dtwc::core {

/// Z-normalize a series in place: subtract mean, divide by standard deviation.
/// If the standard deviation is below 1e-10, all values are set to zero.
template <typename T>
void z_normalize(T *series, size_t n)
{
  if (n == 0) return;
  if (n == 1) { series[0] = static_cast<T>(0); return; }

  // Pass 1: sum(x) → mean
  T sum = 0;
#if defined(_MSC_VER)
  // MSVC does not support OpenMP reduction clauses on simd directives.
  // The compiler can still auto-vectorize the simple reduction loop.
#else
  #pragma omp simd reduction(+:sum)
#endif
  for (size_t i = 0; i < n; ++i)
    sum += series[i];
  const T mean = sum / static_cast<T>(n);

  // Pass 2: sum((x - mean)²) → stddev.
  // Computing deviations from the known mean is numerically stable even when
  // mean >> stddev (the case König-Huygens handles poorly).
  T sq_sum = 0;
#if defined(_MSC_VER)
  // MSVC does not support OpenMP reduction clauses on simd directives.
#else
  #pragma omp simd reduction(+:sq_sum)
#endif
  for (size_t i = 0; i < n; ++i) {
    const T dev = series[i] - mean;
    sq_sum += dev * dev;
  }
  const T stddev = std::sqrt(sq_sum / static_cast<T>(n));

  // Pass 3: normalize in place.
  if (stddev > static_cast<T>(1e-10)) {
    const T inv_stddev = T(1) / stddev;
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
      series[i] = (series[i] - mean) * inv_stddev;
  } else {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
      series[i] = 0;
  }
}

/// Return a z-normalized copy of the input series.
template <typename T>
std::vector<T> z_normalized(const T *series, size_t n)
{
  std::vector<T> result(series, series + n);
  z_normalize(result.data(), result.size());
  return result;
}

} // namespace dtwc::core
