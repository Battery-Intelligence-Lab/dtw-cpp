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

#include "../base/settings.hpp" // index_t

#include <vector>

namespace dtwc::core {

/// Result of a clustering algorithm run.
struct ClusteringResult {
  std::vector<index_t> labels;         ///< Cluster assignment per point [0, k).
  std::vector<index_t> medoid_indices; ///< Index of medoid for each cluster [0, N).
  double total_cost = 0.0;             ///< Sum of distances to nearest medoid.
  int iterations = 0;                  ///< Number of iterations until convergence.
  bool converged = false;              ///< Whether the algorithm converged.

  /// Returns the number of clusters.
  index_t n_clusters() const { return static_cast<index_t>(medoid_indices.size()); }

  /// Returns the number of data points.
  index_t n_points() const { return static_cast<index_t>(labels.size()); }
};

} // namespace dtwc::core
