/**
 * @file fast_pam.hpp
 * @brief FastPAM1 k-medoids clustering algorithm.
 *
 * @details Implements the FastPAM1 algorithm from:
 *   Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager k-medoids clustering:
 *   O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms."
 *   JMLR, 22(1), 4653-4688.
 *
 * FastPAM1 is a true PAM SWAP that considers swapping any medoid with any
 * non-medoid globally, unlike Lloyd iteration which only updates medoids
 * within their own cluster. It maintains nearest and second-nearest medoid
 * information per point for O(N*k) swap evaluation instead of O(N*k*N).
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "../core/clustering_result.hpp"

#include <vector>

namespace dtwc {

class Problem; // Forward declaration

/// Backward-compatible alias: FastPAMResult is now core::ClusteringResult.
using FastPAMResult = core::ClusteringResult;

/**
 * @brief Run FastPAM1 k-medoids clustering.
 *
 * @param prob      Problem instance with data loaded. fillDistanceMatrix() will
 *                  be called if the distance matrix is not yet filled.
 * @param n_clusters Number of clusters (k).
 * @param max_iter  Maximum number of SWAP iterations (default: 100).
 * @return core::ClusteringResult containing labels, medoid indices, total cost, etc.
 *
 * @note This function does NOT modify prob.centroids_ind or prob.clusters_ind.
 *       It returns a standalone ClusteringResult.
 * @note Requires prob to have data loaded (prob.size() > 0).
 */
FastPAMResult fast_pam(Problem& prob, int n_clusters, int max_iter = 100);

} // namespace dtwc
