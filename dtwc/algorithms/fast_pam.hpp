/**
 * @file fast_pam.hpp
 * @brief FasterPAM k-medoids clustering algorithm.
 *
 * @details Implements the eager FasterPAM SWAP from:
 *   Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager k-medoids clustering:
 *   O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms."
 *   Information Systems 101:101804.
 *
 * FasterPAM is a true PAM SWAP that considers swapping any medoid with any
 * non-medoid globally, unlike Lloyd iteration which only updates medoids
 * within their own cluster. It maintains nearest and second-nearest medoid
 * information per point, so evaluating one candidate against every medoid costs
 * O(N), and it performs each improving swap as soon as it is found.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "../core/clustering_result.hpp"

#include <cstdint>

namespace dtwc {

class Problem; // Forward declaration

/**
 * @brief Run FasterPAM k-medoids clustering (BUILD via K-means++).
 *
 * @param prob      Problem instance with data loaded. fill_distance_matrix() will
 *                  be called if the distance matrix is not yet filled.
 * @param n_clusters Number of clusters (k).
 * @param max_iter  Maximum number of SWAP iterations (default: 100).
 * @return core::ClusteringResult containing labels, medoid indices, total cost, etc.
 *
 * @note 2.0 (Task 1.6): on return this WRITES the result back into `prob`
 *       (prob.centroids_ind = medoids, prob.clusters_ind = labels,
 *       prob.n_clusters() = n_clusters), so scores::silhouette(prob) etc. work
 *       with no manual wiring. In 1.x it left prob untouched and the bindings
 *       wired the result in; that binding auto-wire moves into core here.
 * @note Requires prob to have data loaded (prob.size() > 0).
 * @throws InvalidInput if the problem is empty or `n_clusters` is outside `[1, N]`.
 */
core::ClusteringResult fast_pam(Problem& prob, int n_clusters, int max_iter = 100);

/**
 * Deterministic FastPAM entry point with an invocation-local BUILD seed.
 *
 * BUILD uses k-median++ D-sampling because PAM minimizes the sum of DTW
 * distances. This intentionally differs from squared-objective barycenter
 * k-means initialization, whose weights are already squared local costs.
 * @throws InvalidInput if the problem is empty or `n_clusters` is outside `[1, N]`.
 */
core::ClusteringResult fast_pam_seeded(Problem& prob, int n_clusters,
                                       std::uint64_t random_seed, int max_iter = 100);

} // namespace dtwc
