/**
 * @file medoid_assignment_policy.hpp
 * @brief Shared finite-state and ordered-sum policy for medoid assignment.
 *
 * Full assignment scans are owned by their algorithms; this header centralizes
 * only the numerical contract they must all enforce. A matrix is checked once
 * where it enters a Problem (DistanceMatrix::all_computed), so a loop that only
 * reads a filled matrix checks nothing; a loop that calls the DTW function
 * checks each distance it computes; every published objective is the
 * point-ordered sum, checked once.
 *
 * The scans are NOT candidates for a single shared loop: they differ in
 * parallel-vs-serial execution, index space, distance signature and per-element
 * side effects, so folding them together needs runtime policy switches in the
 * library's hottest loops.
 */

#pragma once

#include "../base/error.hpp"
#include "../base/settings.hpp" // index_t

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace dtwc::core::detail {

[[noreturn]] inline void throw_nonfinite_medoid_distance(
  std::string_view caller, std::size_t point, index_t medoid_slot,
  index_t medoid_index)
{
  throw InvalidInput(
    std::string(caller)
    + ": non-finite nearest-medoid distance at point "
    + std::to_string(point) + ", medoid slot "
    + std::to_string(medoid_slot) + " (index "
    + std::to_string(medoid_index) + ").");
}

inline double require_finite_medoid_distance(
  double value, std::string_view caller, std::size_t point,
  index_t medoid_slot, index_t medoid_index)
{
  if (!std::isfinite(value))
    throw_nonfinite_medoid_distance(
      caller, point, medoid_slot, medoid_index);
  return value;
}

/**
 * Point-ordered binary64 accumulator for a published medoid objective.
 *
 * The volatile store after every addition is narrow and intentional: the
 * project enables floating-point reassociation, but assignment objectives are
 * a cross-route byte contract. Exact finite DBL_MAX and negative finite
 * values remain valid. A zero result is canonicalized to positive zero.
 */
class OrderedMedoidObjective
{
public:
  void add(double value) noexcept { total_ = total_ + value; }

  void add(std::span<const double> values) noexcept
  {
    for (const double value : values) add(value);
  }

  [[nodiscard]] double value() const noexcept
  {
    const double result = total_;
    return result == 0.0 ? 0.0 : result;
  }

private:
  volatile double total_ = 0.0;
};

/// The objective `caller` publishes. A sum is finite exactly when every
/// distance it adds is finite and no partial sum overflows, so this one check
/// stands for one per distance.
inline double finite_objective(double total, std::string_view caller)
{
  if (!std::isfinite(total))
    throw InvalidInput(std::string(caller)
                       + ": the nearest-medoid objective is not finite (a distance or their sum overflowed).");
  return total;
}

inline double ordered_medoid_objective(
  std::span<const double> values, std::string_view caller)
{
  OrderedMedoidObjective total;
  total.add(values);
  return finite_objective(total.value(), caller);
}

} // namespace dtwc::core::detail
