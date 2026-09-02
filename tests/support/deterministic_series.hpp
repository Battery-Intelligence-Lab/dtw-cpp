/**
 * @file deterministic_series.hpp
 * @brief Deterministic data and dense-reference assembly shared by tests and
 *        benchmarks.
 *
 * @details The generators are a portable random-data format: for a given seed
 *          and length they produce byte-identical doubles on every conforming
 *          C++20 implementation, independent of standard library and, under
 *          IEEE round-to-nearest, of the project's floating-point flags.
 *          `std::mt19937` is bit-exact by the standard; the integer-to-double
 *          conversion below replaces the standard uniform real distribution,
 *          whose mapping is implementation-defined.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtwc::test_support {

namespace detail {

/// Signed 53-bit draw, genrand_res53 schedule: two engine outputs per value.
inline std::int64_t next_signed53(std::mt19937 &rng)
{
  const std::uint64_t hi = rng() >> 5; // 27 bits
  const std::uint64_t lo = rng() >> 6; // 26 bits
  // hi*2^26 + lo < 2^53, so the sum is an exact integer in [0, 2^53).
  const std::uint64_t bits = (hi << 26) + lo;
  // Recentre to [-2^52, 2^52); the subtraction is exact in 64-bit integers.
  return static_cast<std::int64_t>(bits) - (INT64_C(1) << 52);
}

// Range scales are hexadecimal literals, not products: nothing for
// -fassociative-math to reassociate and no addition for /fp:contract or
// -ffp-contract to fold into an FMA.
inline constexpr double kUnitScale = 0x1p-52;   // 2^-52  -> [-1, 1)
inline constexpr double kDecaScale = 0x1.4p-49; // 10*2^-52 -> [-10, 10)

/// Scale a signed 53-bit draw. Exactly one IEEE-754 rounded multiply.
inline double scaled(std::int64_t k, double scale)
{
  // |k| <= 2^52, so the int64 -> double conversion is exact; a single
  // multiply is correctly rounded and therefore identical under every
  // permitted FP relaxation, under IEEE round-to-nearest. For kUnitScale the
  // multiply is a power-of-two scaling and thus exact, with no rounding at all.
  // x87 excess precision (FLT_EVAL_METHOD == 2) is safe too: 10*k needs at most
  // 56 significant bits, well inside the 64-bit extended significand, so the
  // intermediate is exact and the single rounding to double cannot be a double
  // rounding.
  return static_cast<double>(k) * scale;
}

} // namespace detail

/// One benchmark series: a fresh engine and one [-1, 1) draw per element.
inline std::vector<double> benchmark_series(
    std::size_t length, unsigned seed)
{
  std::mt19937 rng(seed);
  std::vector<double> series(length);
  for (auto &value : series)
    value = detail::scaled(detail::next_signed53(rng), detail::kUnitScale);
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

/// Accelerator rows: one continuous engine across the row-major N*L draws,
/// each in [-10, 10).
inline std::vector<std::vector<double>> accelerator_series_set(
    std::size_t count, std::size_t length, unsigned seed)
{
  std::mt19937 rng(seed);
  std::vector<std::vector<double>> series(count);
  for (auto &row : series) {
    row.resize(length);
    for (auto &value : row)
      value = detail::scaled(detail::next_signed53(rng), detail::kDecaScale);
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
