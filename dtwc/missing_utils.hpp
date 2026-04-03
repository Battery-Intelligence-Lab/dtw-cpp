/**
 * @file missing_utils.hpp
 * @brief Utilities for handling missing data (NaN) in time series.
 *
 * @details NaN handling design:
 *   - Build flags use explicit fast-math sub-flags (-fno-math-errno, -fassociative-math,
 *     etc.) but NOT -ffinite-math-only, so std::isnan() is correct and available.
 *   - is_missing() is a thin semantic wrapper over std::isnan() — prefer this name
 *     in domain code for clarity ("missing value" not "not-a-number").
 *   - DenseDistanceMatrix tracks computed entries via std::vector<bool> (not NaN
 *     sentinels), which is safe regardless of floating-point mode.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dtwc {

/// NaN check — semantic alias for std::isnan for use in missing-data code paths.
/// Build flags omit -ffinite-math-only so std::isnan() is reliable at all opt levels.
template <typename T>
inline bool is_missing(T val) noexcept
{
  static_assert(std::is_floating_point_v<T>, "is_missing only supports floating-point types");
  return std::isnan(val);
}

/// Returns true if any element in the vector is NaN.
template <typename T>
bool has_missing(const std::vector<T> &v)
{
  for (const auto &x : v)
    if (is_missing(x)) return true;
  return false;
}

/// Returns the fraction of NaN values in the vector (0.0 if empty).
template <typename T>
double missing_rate(const std::vector<T> &v)
{
  if (v.empty()) return 0.0;
  size_t count = 0;
  for (const auto &x : v)
    if (is_missing(x)) ++count;
  return static_cast<double>(count) / static_cast<double>(v.size());
}

/// Linear interpolation of NaN gaps.
/// Interior NaN: linearly interpolated between nearest observed neighbors.
/// Leading NaN: filled with first observed value (NOCB).
/// Trailing NaN: filled with last observed value (LOCF).
/// Throws std::runtime_error if ALL values are NaN.
template <typename T>
std::vector<T> interpolate_linear(const std::vector<T> &v)
{
  if (v.empty()) return {};

  size_t first_valid = v.size();
  size_t last_valid = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (!is_missing(v[i])) {
      if (first_valid == v.size()) first_valid = i;
      last_valid = i;
    }
  }
  if (first_valid == v.size())
    throw std::runtime_error("interpolate_linear: all values are NaN");

  std::vector<T> result(v.size());

  // NOCB: fill leading NaN with first observed value
  for (size_t i = 0; i < first_valid; ++i)
    result[i] = v[first_valid];

  // Interior: linear interpolation
  size_t prev_valid = first_valid;
  result[first_valid] = v[first_valid];
  for (size_t i = first_valid + 1; i <= last_valid; ++i) {
    if (!is_missing(v[i])) {
      result[i] = v[i];
      if (i - prev_valid > 1) {
        T start = v[prev_valid];
        T end = v[i];
        T span = static_cast<T>(i - prev_valid);
        for (size_t j = prev_valid + 1; j < i; ++j) {
          T frac = static_cast<T>(j - prev_valid) / span;
          result[j] = start + frac * (end - start);
        }
      }
      prev_valid = i;
    }
  }

  // LOCF: fill trailing NaN with last observed value
  for (size_t i = last_valid + 1; i < v.size(); ++i)
    result[i] = v[last_valid];

  return result;
}

} // namespace dtwc
