/**
 * @file decode_assignment.hpp
 * @brief Decode an optimal p-median assignment matrix into medoids and labels.
 */

#pragma once

#include "../base/error.hpp"
#include "../core/clustering_result.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dtwc::mip {

/// `x[facility, point]` is binary; the open medoids are the diagonal. HiGHS
/// stores facility-major (`f·n + p`), Gurobi point-major (`f + p·n`). A solution
/// that is not exactly k medoids with one open medoid per point is SolverError.
inline core::ClusteringResult decode_assignment(std::span<const double> x, std::size_t n, int k,
                                                bool point_major, std::string_view backend)
{
  const auto chosen = [&](std::size_t f, std::size_t p) { return x[point_major ? f + p * n : f * n + p] > 0.5; };
  const auto fail = [&](const std::string &what) {
    throw SolverError(std::string(backend) + " returned an invalid p-median solution: " + what);
  };
  core::ClusteringResult result;
  std::vector<int> slot_of(n, -1);
  for (std::size_t f = 0; f < n; ++f)
    if (chosen(f, f)) {
      slot_of[f] = static_cast<int>(result.medoid_indices.size());
      result.medoid_indices.push_back(static_cast<int>(f));
    }
  if (result.medoid_indices.size() != static_cast<std::size_t>(k))
    fail(std::to_string(result.medoid_indices.size()) + " medoids for k = " + std::to_string(k) + ".");
  result.labels.assign(n, -1);
  for (std::size_t p = 0; p < n; ++p) {
    int assignments = 0;
    for (std::size_t f = 0; f < n; ++f)
      if (chosen(f, p)) {
        result.labels[p] = slot_of[f];
        ++assignments;
      }
    if (assignments != 1 || result.labels[p] < 0)
      fail("point " + std::to_string(p) + " is not assigned to exactly one open medoid.");
  }
  return result;
}

} // namespace dtwc::mip
