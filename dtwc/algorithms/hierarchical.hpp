/**
 * @file hierarchical.hpp
 * @brief Agglomerative hierarchical clustering (single/complete/average linkage).
 *
 * @details Small-N feature with a hard max_points guard. Ward's linkage is
 * intentionally excluded as it is mathematically invalid for DTW distances
 * (DTW does not satisfy the squared Euclidean distance identity required by
 * Ward's formula).
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

#include "../core/clustering_result.hpp"
#include "../base/error.hpp"
#include "../base/names.hpp"

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

inline constexpr Name<Linkage> linkage_names[]{
  { "single", Linkage::Single },
  { "complete", Linkage::Complete },
  { "average", Linkage::Average },
};

/// A single merge step recorded in the dendrogram.
struct DendrogramStep {
  index_t cluster_a; ///< First merged cluster (always < cluster_b for determinism)
  index_t cluster_b; ///< Second merged cluster
  double distance;   ///< Merge distance
  index_t new_size;  ///< Size of the merged cluster
};

/// Full dendrogram produced by build_dendrogram().
struct Dendrogram {
  std::vector<DendrogramStep> merges; ///< N-1 merge steps in merge order
  index_t n_points = 0;
};

/// Options for build_dendrogram().
struct HierarchicalOptions {
  Linkage linkage = Linkage::Average;
  index_t max_points = 2000; ///< Hard guard — throws InvalidInput if N exceeds this
};

/**
 * @brief Build a dendrogram from a Problem, filling its distance matrix first.
 *
 * @param prob  Problem with data; its distance matrix is filled if it is not.
 * @param opts  Linkage criterion and max_points guard.
 * @return Dendrogram containing N-1 merge steps.
 *
 * @throws InvalidInput if N > opts.max_points (checked before the fill).
 */
Dendrogram build_dendrogram(Problem &prob, const HierarchicalOptions &opts = {});

/**
 * @brief Cut a dendrogram to produce k flat clusters with medoids.
 *
 * Replays the FIRST N-k merges using union-find (the remaining k-1 merges are
 * the ones that would collapse the k surviving clusters), then assigns medoids
 * by minimising each cluster member's sum of distances to cluster peers.
 * Tie-breaking: smallest original index wins.
 *
 * @param dend  Dendrogram produced by build_dendrogram().
 * @param prob  Problem whose distance matrix is used for medoid computation.
 * @param k     Number of clusters (1 <= k <= dend.n_points).
 * @return core::ClusteringResult with labels, medoid_indices, and total_cost.
 *
 * @throws InvalidInput if `dend` is not a well-formed dendrogram over
 *         `prob` — n_points != prob.size(), merges.size() != n_points - 1, a
 *         cluster id outside [0, n_points), or a merge list that does not
 *         reduce N points to k components.
 *
 * @note 2.0: writes the result back into `prob` (clusters_ind,
 *       centroids_ind, n_clusters) so scores work with no manual wiring. 1.x
 *       left prob untouched (and the bindings did not wire cut_dendrogram).
 */
core::ClusteringResult cut_dendrogram(const Dendrogram &dend, Problem &prob, index_t k);

} // namespace algorithms
} // namespace dtwc
