/**
 * @file nearest_medoid.hpp
 * @brief Shared nearest-open-medoid scan for the MIP backends.
 *
 * @details The Benders assignment subproblem and every Lagrangian primal repair
 * answer the same question: which of the k open medoids serves point j, and at
 * what cost? This is the single definition, and it fixes the tie-break: the
 * FIRST minimum wins, so ties resolve to the lowest position in the open-medoid
 * array.
 *
 * Performance contract: this sits inside the LR subgradient primal repair, which
 * runs every iteration. It is a header-inline function template over the
 * distance accessor — never `std::function`, never virtual, and it allocates
 * nothing, so the caller's accessor inlines into the scan.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#pragma once

#include <limits>

namespace dtwc::mip {

/// Position of the serving medoid within the open set, and its distance.
struct NearestMedoid
{
  int position{ 0 };                                              ///< index into the open-medoid array, in [0, k).
  double distance{ std::numeric_limits<double>::infinity() };     ///< D(open[position], point).
};

/**
 * @brief Nearest open medoid of one point.
 * @param k    Number of open medoids; must be >= 1.
 * @param dist Callable `double(int position)` returning D(open[position], point).
 * @return The winning position and its distance; position 0 when k distances tie
 *         or are all non-finite.
 */
template <typename Dist>
[[nodiscard]] inline NearestMedoid nearest_medoid(int k, Dist dist)
{
  NearestMedoid best{};
  for (int t = 0; t < k; ++t) {
    const double d = dist(t);
    if (d < best.distance) {
      best.position = t;
      best.distance = d;
    }
  }
  return best;
}

} // namespace dtwc::mip
