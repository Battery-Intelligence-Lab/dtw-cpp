/**
 * @file fast_pam_plan.hpp
 * @brief Checked FastPAM dimensions at the int-indexed clustering boundary.
 */

#pragma once

#include <cstddef>
#include <string_view>

namespace dtwc::algorithms::detail {

struct FastPamPlan
{
  int n_points;
  int n_clusters;
};

/** Resolve the point count before any distance-matrix or result allocation. */
[[nodiscard]] int checked_fast_pam_point_count(
  std::size_t n_points, std::string_view caller);

/** Resolve the point count and validate the requested cluster count. */
[[nodiscard]] FastPamPlan resolve_fast_pam_plan(
  std::size_t n_points, int n_clusters, std::string_view caller);

} // namespace dtwc::algorithms::detail
