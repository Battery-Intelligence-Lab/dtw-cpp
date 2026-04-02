/**
 * @file hierarchical.hpp
 * @brief Agglomerative hierarchical clustering (single/complete/average linkage).
 *
 * @details Small-N feature with a hard max_points guard. Ward's linkage is
 * intentionally excluded as it is mathematically invalid for DTW distances
 * (DTW does not satisfy the squared Euclidean distance identity required by
 * Ward's formula).
 *
 * @date 02 Apr 2026
 */

#pragma once

#include "../core/clustering_result.hpp"

#include <vector>

namespace dtwc {

class Problem;

namespace algorithms {

/// Linkage criterion for agglomerative hierarchical clustering.
enum class Linkage {
  Single,   ///< d(A∪B, C) = min(d(A,C), d(B,C))
  Complete, ///< d(A∪B, C) = max(d(A,C), d(B,C))
  Average   ///< d(A∪B, C) = (|A|*d(A,C) + |B|*d(B,C)) / (|A|+|B|)  [UPGMA]
};

/// A single merge step recorded in the dendrogram.
struct DendrogramStep {
  int cluster_a;  ///< First merged cluster (always < cluster_b for determinism)
  int cluster_b;  ///< Second merged cluster
  double distance; ///< Merge distance
  int new_size;   ///< Size of the merged cluster
};

/// Full dendrogram produced by build_dendrogram().
struct Dendrogram {
  std::vector<DendrogramStep> merges; ///< N-1 merge steps in merge order
  int n_points = 0;
};

/// Options for build_dendrogram().
struct HierarchicalOptions {
  Linkage linkage = Linkage::Average;
  int max_points = 2000; ///< Hard guard — throws std::runtime_error if N exceeds this
};

/**
 * @brief Build a dendrogram from a Problem with a fully computed distance matrix.
 *
 * @param prob  Problem with distance matrix already filled (fillDistanceMatrix() called).
 * @param opts  Linkage criterion and max_points guard.
 * @return Dendrogram containing N-1 merge steps.
 *
 * @throws std::runtime_error if N > opts.max_points.
 * @throws std::runtime_error if prob.isDistanceMatrixFilled() is false.
 */
Dendrogram build_dendrogram(Problem &prob, const HierarchicalOptions &opts = {});

/**
 * @brief Cut a dendrogram to produce k flat clusters with medoids.
 *
 * Replays the last N-k merges using union-find, then assigns medoids
 * by minimising each cluster member's sum of distances to cluster peers.
 * Tie-breaking: smallest original index wins.
 *
 * @param dend  Dendrogram produced by build_dendrogram().
 * @param prob  Problem whose distance matrix is used for medoid computation.
 * @param k     Number of clusters (1 <= k <= dend.n_points).
 * @return core::ClusteringResult with labels, medoid_indices, and total_cost.
 *
 * @note Does NOT mutate Problem::clusters_ind or Problem::centroids_ind.
 */
core::ClusteringResult cut_dendrogram(const Dendrogram &dend, Problem &prob, int k);

} // namespace algorithms
} // namespace dtwc
