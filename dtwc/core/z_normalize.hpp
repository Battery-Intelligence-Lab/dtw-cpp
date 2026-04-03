/**
 * @file z_normalize.hpp
 * @brief Z-normalization utility for time series.
 *
 * @details Provides in-place and copying z-normalization (zero mean, unit
 * standard deviation). Series with near-zero standard deviation are set to
 * all zeros to avoid numerical instability.
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

  T sum = 0;
#if defined(_MSC_VER)
  // MSVC does not support OpenMP reduction clauses on simd directives.
#else
  #pragma omp simd reduction(+:sum)
#endif
  for (size_t i = 0; i < n; ++i)
    sum += series[i];
  T mean = sum / static_cast<T>(n);

  T sq_sum = 0;
#if defined(_MSC_VER)
  // MSVC does not support OpenMP reduction clauses on simd directives.
#else
  #pragma omp simd reduction(+:sq_sum)
#endif
  for (size_t i = 0; i < n; ++i) {
    T d = series[i] - mean;
    sq_sum += d * d;
  }
  T stddev = std::sqrt(sq_sum / static_cast<T>(n));

  if (stddev > static_cast<T>(1e-10)) {
    T inv_stddev = T(1) / stddev;
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
