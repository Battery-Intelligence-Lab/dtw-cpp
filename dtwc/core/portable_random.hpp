/**
 * @file portable_random.hpp
 * @brief Implementation-independent sampling from std::mt19937_64.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dtwc::core {

namespace detail {

  /** Return the high half of the exact 64-by-64-bit unsigned product. */
  [[nodiscard]] constexpr std::uint64_t multiply_high(
    std::uint64_t lhs, std::uint64_t rhs) noexcept
  {
    constexpr std::uint64_t half_mask = 0xffff'ffffULL;
    const std::uint64_t lhs_low = lhs & half_mask;
    const std::uint64_t lhs_high = lhs >> 32;
    const std::uint64_t rhs_low = rhs & half_mask;
    const std::uint64_t rhs_high = rhs >> 32;

    const std::uint64_t low_product = lhs_low * rhs_low;
    const std::uint64_t middle = lhs_high * rhs_low + (low_product >> 32);
    const std::uint64_t middle_low = middle & half_mask;
    const std::uint64_t middle_high = middle >> 32;
    const std::uint64_t upper_middle = middle_low + lhs_low * rhs_high;
    return lhs_high * rhs_high + middle_high + (upper_middle >> 32);
  }

} // namespace detail

/** Return an unbiased integer in [0, upper_exclusive). */
[[nodiscard]] inline std::uint64_t portable_bounded(
  std::mt19937_64 &engine, std::uint64_t upper_exclusive)
{
  if (upper_exclusive == 0)
    throw std::invalid_argument("portable_bounded: upper bound must be positive.");
  // A singleton domain contains no entropy.  Besides avoiding useless work,
  // this makes selection scans consume exactly one draw per genuine choice.
  if (upper_exclusive == 1) return 0;

  // Lemire's multiply-high map is unbiased after rejecting the short low
  // interval.  For ordinary collection sizes, `low < upper_exclusive` is
  // vanishingly rare, so the integer division needed for the threshold stays
  // off the Fisher-Yates hot path.  Unsigned negation is defined modulo 2^64.
  for (;;) {
    const std::uint64_t value = engine();
    const std::uint64_t low = value * upper_exclusive;
    if (low < upper_exclusive) {
      const std::uint64_t threshold = -upper_exclusive % upper_exclusive;
      if (low < threshold) continue;
    }
    return detail::multiply_high(value, upper_exclusive);
  }
}

/** Return a reproducible binary64 value in [0, 1) using exactly 53 random bits. */
[[nodiscard]] inline double portable_unit_interval(std::mt19937_64 &engine) noexcept
{
  constexpr double scale = 0x1.0p-53;
  return static_cast<double>(engine() >> 11) * scale;
}

/** Return a reproducible binary64 value in [0, upper_exclusive). */
[[nodiscard]] inline double portable_real_below(
  std::mt19937_64 &engine, double upper_exclusive)
{
  if (!std::isfinite(upper_exclusive) || upper_exclusive <= 0.0)
    throw std::invalid_argument(
      "portable_real_below: upper bound must be finite and positive.");
  const double value = portable_unit_interval(engine) * upper_exclusive;
  // Correct the possible round-to-upper-endpoint case without another draw.
  return value < upper_exclusive
           ? value
           : std::nextafter(upper_exclusive, 0.0);
}

/** Select one positive weight using a fixed cumulative mapping. */
template <typename ForwardIterator>
[[nodiscard]] std::size_t portable_weighted_index(
  ForwardIterator first, ForwardIterator last, double total,
  std::mt19937_64 &engine)
{
  if (!std::isfinite(total) || total <= 0.0)
    throw std::invalid_argument(
      "portable_weighted_index: total must be finite and positive.");

  const double threshold = portable_real_below(engine, total);
  double cumulative = 0.0;
  std::size_t index = 0;
  std::size_t last_positive = 0;
  bool has_positive_weight = false;
  std::size_t selected_index = 0;
  bool selected = false;
  for (; first != last; ++first, ++index) {
    const double weight = static_cast<double>(*first);
    if (!std::isfinite(weight) || weight < 0.0)
      throw std::invalid_argument(
        "portable_weighted_index: weights must be finite and non-negative.");
    if (weight > 0.0) {
      last_positive = index;
      has_positive_weight = true;
    }
    cumulative += weight;
    if (!selected && threshold < cumulative) {
      selected_index = index;
      selected = true;
    }
  }
  if (!has_positive_weight)
    throw std::invalid_argument(
      "portable_weighted_index: at least one weight must be positive.");
  // Summation can finish one ulp below the independently accumulated `total`.
  return selected ? selected_index : last_positive;
}

/** Stable selection sample of sorted indices from [0, population_size). */
template <typename Index>
[[nodiscard]] std::vector<Index> portable_sample_indices(
  Index population_size, Index sample_size, std::mt19937_64 &engine)
{
  static_assert(std::is_integral_v<Index>);
  if constexpr (std::is_signed_v<Index>) {
    if (population_size < 0 || sample_size < 0)
      throw std::invalid_argument(
        "portable_sample_indices: sizes must be non-negative.");
  }
  if (sample_size > population_size)
    throw std::invalid_argument(
      "portable_sample_indices: sample exceeds population.");

  using unsigned_index = std::make_unsigned_t<Index>;
  const auto population = static_cast<unsigned_index>(population_size);
  auto needed = static_cast<unsigned_index>(sample_size);
  if constexpr (sizeof(unsigned_index) > sizeof(std::size_t)) {
    if (needed > static_cast<unsigned_index>(
          std::numeric_limits<std::size_t>::max()))
      throw std::length_error(
        "portable_sample_indices: sample does not fit address space.");
  }
  std::vector<Index> sample;
  sample.reserve(static_cast<std::size_t>(needed));
  if (needed == 0) return sample;

  unsigned_index candidate = 0;
  for (auto remaining = population; remaining > 0;
       --remaining, ++candidate) {
    // Continue after the sample is complete.  This fixed full-population
    // bounded-call schedule makes consecutive samples independent of where
    // the last choice occurred. The final singleton consumes no engine word;
    // rejection in an earlier call may consume more than one.
    if (portable_bounded(engine, static_cast<std::uint64_t>(remaining))
        < static_cast<std::uint64_t>(needed)) {
      sample.push_back(static_cast<Index>(candidate));
      --needed;
    }
  }
  return sample;
}

/** Fisher-Yates shuffle with a library-independent draw-to-index mapping. */
template <typename RandomAccessIterator>
void portable_shuffle(
  RandomAccessIterator first, RandomAccessIterator last, std::mt19937_64 &engine)
{
  using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
  const difference_type count = last - first;
  for (difference_type target = 1; target < count; ++target) {
    const auto offset = static_cast<difference_type>(
      portable_bounded(engine, static_cast<std::uint64_t>(target) + 1));
    if (offset != target) std::iter_swap(first + target, first + offset);
  }
}

} // namespace dtwc::core
