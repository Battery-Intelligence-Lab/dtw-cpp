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

#include <optional>
#include <vector>

namespace dtwc::core {

/// How FastCLARA's assignment pass was split under cpu+gpu: the GPU took the leading `gpu_share` of the points (the
/// effective share, its points / N: forced again, it repeats the split exactly), the CPU's team the rest. The rates,
/// in points a second, are those that set a calibrated share: the CPU's team less the GPU's driving thread, and the
/// GPU through its pipeline; 0 where the share was forced or the design's recorded ratio stood in.
struct HybridSplit {
  double gpu_share = 0.0;
  double cpu_rate = 0.0;
  double gpu_rate = 0.0;
  const char *source = "forced"; ///< "forced", "calibrated" or "recorded"
};

/// Result of a clustering algorithm run.
struct ClusteringResult {
  std::vector<index_t> labels;         ///< Cluster assignment per point [0, k).
  std::vector<index_t> medoid_indices; ///< Index of medoid for each cluster [0, N).
  double total_cost = 0.0;             ///< Sum of distances to nearest medoid.
  int iterations = 0;                  ///< Number of iterations until convergence.
  bool converged = false;              ///< Whether the algorithm converged.
  std::optional<HybridSplit> hybrid;   ///< FastCLARA on cpu+gpu: how its pass was split (empty elsewhere).

  /// Returns the number of clusters.
  index_t n_clusters() const { return static_cast<index_t>(medoid_indices.size()); }

  /// Returns the number of data points.
  index_t n_points() const { return static_cast<index_t>(labels.size()); }
};

} // namespace dtwc::core
