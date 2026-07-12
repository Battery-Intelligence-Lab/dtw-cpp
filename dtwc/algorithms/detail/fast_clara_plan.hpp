/**
 * @file fast_clara_plan.hpp
 * @brief Checked, allocation-free FastCLARA dimension planning.
 */

#pragma once

#include "../fast_clara.hpp"

#include <cstdint>
#include <string_view>

namespace dtwc::algorithms::detail {

struct ClaraPlan
{
  int n_points;
  int sample_size;
};

/** Validate controls that do not depend on the dataset size. */
void validate_clara_controls(
  const CLARAOptions &options, std::string_view caller);

/** Resolve and validate dimensions before any sample or result allocation. */
[[nodiscard]] ClaraPlan resolve_clara_plan(
  std::int64_t n_points, const CLARAOptions &options,
  std::string_view caller);

/** Reject a full-data sample on the RAM-limited streaming route. */
void validate_streaming_clara_plan(
  const ClaraPlan &plan, std::string_view caller);

} // namespace dtwc::algorithms::detail
