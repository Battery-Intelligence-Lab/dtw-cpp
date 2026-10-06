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

#include "../base/settings.hpp" // DEFAULT_RANDOM_SEED
#include "../core/clustering_result.hpp"

#include <cstdint>

namespace dtwc {

class Problem; // Forward declaration

/**
 * @brief Run FasterPAM k-medoids clustering (BUILD via k-medoids++).
 *
 * BUILD is k-median++ D-sampling (core::kmedoids_pp): PAM minimizes a sum of DTW
 * distances, so a point's sampling weight is its distance to the nearest medoid.
 * Barycenter k-means samples D² instead, because its objective is squared.
 *
 * @param prob      Problem instance with data loaded. fill_distance_matrix() will
 *                  be called if the distance matrix is not yet filled.
 * @param n_clusters Number of clusters (k).
 * @param max_iter  Maximum number of SWAP iterations (default: 100). 0 returns the
 *                  BUILD medoids without a SWAP (`converged` false), which is how the
 *                  BUILD phase is observed alone; k = 1 needs no SWAP and always
 *                  returns the exact 1-median.
 * @param seed      BUILD seed (default settings::DEFAULT_RANDOM_SEED): one seed gives
 *                  one result on every platform; dtwc::randGenerator is not read.
 * @return core::ClusteringResult containing labels, medoid indices, total cost, etc.
 *
 * @note 2.0: on return this WRITES the result back into `prob`
 *       (prob.centroids_ind = medoids, prob.clusters_ind = labels,
 *       prob.n_clusters() = n_clusters), so scores::silhouette(prob) etc. work
 *       with no manual wiring. In 1.x it left prob untouched and the bindings
 *       wired the result in; that binding auto-wire moves into core here.
 * @note Requires prob to have data loaded (prob.size() > 0).
 * @throws InvalidInput if the problem is empty, `n_clusters` is outside `[1, N]` or
 *         `max_iter` is negative.
 */
core::ClusteringResult fast_pam(Problem& prob, index_t n_clusters, int max_iter = 100,
                                std::uint64_t seed = settings::DEFAULT_RANDOM_SEED);

} // namespace dtwc
