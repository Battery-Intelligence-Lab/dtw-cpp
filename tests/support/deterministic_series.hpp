/**
 * @file deterministic_series.hpp
 * @brief Deterministic data and dense-reference assembly shared by tests and
 *        benchmarks.
 *
 * @details These helpers preserve the historical STL distribution types and
 *          draw schedules. They are reproducible within a supported compiler
 *          floating-point profile, but are not a portable random-data format.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtwc::test_support {

/// One benchmark series: a fresh engine and one [-1, 1] draw per element.
inline std::vector<double> benchmark_series(
    std::size_t length, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> series(length);
  for (auto &value : series)
    value = dist(rng);
  return series;
}

/// Benchmark rows: a fresh engine for each base_seed + row.
inline std::vector<std::vector<double>> benchmark_series_set(
    std::size_t count, std::size_t length, unsigned base_seed)
{
  std::vector<std::vector<double>> series;
  series.reserve(count);
  for (std::size_t row = 0; row < count; ++row) {
    series.push_back(
      benchmark_series(length, base_seed + static_cast<unsigned>(row)));
  }
  return series;
}

/// Accelerator rows: one continuous engine across the row-major N*L draws.
inline std::vector<std::vector<double>> accelerator_series_set(
    std::size_t count, std::size_t length, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  std::vector<std::vector<double>> series(count);
  for (auto &row : series) {
    row.resize(length);
    for (auto &value : row)
      value = dist(rng);
  }
  return series;
}

/// Dense zero-diagonal symmetric matrix from upper-triangle distance calls.
template <class SeriesCollection, class Distance>
auto symmetric_zero_diagonal_matrix(
    const SeriesCollection &series, Distance distance)
{
  using Series = decltype(series[std::size_t{0}]);
  using Result = std::remove_cvref_t<
    std::invoke_result_t<Distance &, Series, Series>>;

  const std::size_t count = series.size();
  std::vector<Result> matrix(count * count, Result{});
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = i + 1; j < count; ++j) {
      const Result value = std::invoke(distance, series[i], series[j]);
      matrix[i * count + j] = value;
      matrix[j * count + i] = value;
    }
  }
  return matrix;
}

} // namespace dtwc::test_support
