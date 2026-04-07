/**
 * @file clustering_result.hpp
 * @brief Pure data struct for clustering output.
 *
 * @details Holds cluster assignments, medoid indices, cost, and convergence
 * information returned by clustering algorithms.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <cstddef>
#include <vector>

namespace dtwc::core {

/// Result of a clustering algorithm run.
struct ClusteringResult {
  std::vector<int> labels;          ///< Cluster assignment per point [0, k).
  std::vector<int> medoid_indices;  ///< Index of medoid for each cluster [0, N).
  double total_cost = 0.0;          ///< Sum of distances to nearest medoid.
  int iterations = 0;               ///< Number of iterations until convergence.
  bool converged = false;           ///< Whether the algorithm converged.

  /// Returns the number of clusters.
  int n_clusters() const { return static_cast<int>(medoid_indices.size()); }

  /// Returns the number of data points.
  size_t n_points() const { return labels.size(); }
};

} // namespace dtwc::core
