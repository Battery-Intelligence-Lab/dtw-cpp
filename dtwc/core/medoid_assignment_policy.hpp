/**
 * @file medoid_assignment_policy.hpp
 * @brief Shared finite-state and ordered-sum policy for medoid assignment.
 *
 * Full assignment scans are owned by their algorithms; this header centralizes
 * only the numerical contract they must all enforce (finiteness and the
 * point-ordered published objective).
 *
 * The scans are NOT candidates for a single shared loop: they differ in
 * parallel-vs-serial execution, index space, distance signature and per-element
 * side effects, so folding them together needs runtime policy switches in the
 * library's hottest loops. See medoid_utils.hpp (D1).
 */

#pragma once

#include "../error.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace dtwc::core::detail {

[[noreturn]] inline void throw_nonfinite_medoid_distance(
  std::string_view caller, std::size_t point, int medoid_slot,
  int medoid_index)
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
  int medoid_slot, int medoid_index)
{
  if (!std::isfinite(value))
    throw_nonfinite_medoid_distance(
      caller, point, medoid_slot, medoid_index);
  return value;
}

[[noreturn]] inline void throw_nonfinite_candidate_distance(
  std::string_view caller, std::size_t point, int candidate_index)
{
  throw InvalidInput(
    std::string(caller)
    + ": non-finite candidate distance at point "
    + std::to_string(point) + ", candidate index "
    + std::to_string(candidate_index) + ".");
}

inline double require_finite_candidate_distance(
  double value, std::string_view caller, std::size_t point,
  int candidate_index)
{
  if (!std::isfinite(value))
    throw_nonfinite_candidate_distance(
      caller, point, candidate_index);
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
  explicit OrderedMedoidObjective(
    std::string_view caller, std::size_t first_point = 0) noexcept
    : caller_(caller), next_point_(first_point)
  {
  }

  void add(double value)
  {
    if (!std::isfinite(value)) {
      throw InvalidInput(
        std::string(caller_)
        + ": non-finite nearest-medoid distance at point "
        + std::to_string(next_point_) + ".");
    }

    const double current = total_;
    const double next = current + value;
    if (!std::isfinite(next)) {
      throw InvalidInput(
        std::string(caller_)
        + ": nearest-medoid objective became non-finite after point "
        + std::to_string(next_point_) + ".");
    }
    total_ = next;
    ++next_point_;
  }

  void add(std::span<const double> values)
  {
    for (const double value : values) add(value);
  }

  [[nodiscard]] double value() const noexcept
  {
    const double result = total_;
    return result == 0.0 ? 0.0 : result;
  }

private:
  std::string_view caller_;
  std::size_t next_point_;
  volatile double total_ = 0.0;
};

inline double ordered_medoid_objective(
  std::span<const double> values, std::string_view caller)
{
  OrderedMedoidObjective total(caller);
  total.add(values);
  return total.value();
}

} // namespace dtwc::core::detail
