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
#include <cstdint>

namespace dtwc {

class Problem; // Forward declaration

/// Backward-compatible alias: FastPAMResult is now core::ClusteringResult.
using FastPAMResult = core::ClusteringResult;

/**
 * @brief Which SWAP algorithm to run (all from Schubert & Rousseeuw 2021).
 *
 * - `FastPAM1Naive` — the pre-5.1 baseline: single best swap per pass, but the
 *   per-candidate gain is the naive nested O(N·k) loop ⇒ O(N²·k) per iteration.
 *   Kept only as the bench baseline and the digit-identity oracle.
 * - `FastPAM1` — the paper's FastPAM1: same single-best-swap result, but each
 *   candidate's ΔTD for ALL medoids is found in ONE O(N) pass (Eq. 11
 *   decomposition), so it is O(N²) per iteration and parallel over candidates.
 *   Digit-identical to `FastPAM1Naive`, k× faster. The default.
 * - `FasterPAM` — eager: performs swaps as they are found and shares removal
 *   loss across medoids, converging in far fewer (often 1) sweeps. O(N²) per
 *   sweep, but sequential (each swap changes state for later candidates in the
 *   sweep). Never worse in objective than FastPAM1; wins wall-time at large k.
 *
 * All three reach a local optimum of the same swap neighbourhood. The enum lets
 * the bench A/B them on an identical BUILD (isolating swap cost).
 */
enum class PAMVariant { FastPAM1Naive, FastPAM1, FasterPAM };

/**
 * @brief Run FastPAM/FasterPAM k-medoids clustering (BUILD via K-means++).
 *
 * @param prob      Problem instance with data loaded. fillDistanceMatrix() will
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
 */
FastPAMResult fast_pam(Problem& prob, int n_clusters, int max_iter = 100);

/**
 * Deterministic FastPAM entry point with an invocation-local BUILD seed.
 *
 * BUILD uses k-median++ D-sampling because PAM minimizes the sum of DTW
 * distances. This intentionally differs from squared-objective barycenter
 * k-means initialization, whose weights are already squared local costs.
 */
FastPAMResult fast_pam_seeded(Problem& prob, int n_clusters,
                              std::uint64_t random_seed, int max_iter = 100);

/**
 * @brief Run the SWAP phase only, from a caller-supplied initial medoid set.
 *
 * Factored out of fast_pam so a benchmark can run BOTH variants from the SAME
 * BUILD result — comparing swap wall-time and final objective apples-to-apples
 * (the O(N²) distance-matrix fill and the K-means++ BUILD are shared, not timed).
 *
 * @param prob            Problem with data loaded; fillDistanceMatrix() ensured.
 * @param initial_medoids k distinct medoid indices in [0, N) (the BUILD result).
 * @param max_iter        Maximum SWAP iterations.
 * @param variant         FastPAM1 or FasterPAM.
 * @return ClusteringResult; also written back into prob (see fast_pam note).
 * @throws std::runtime_error on invalid initial_medoids (empty/dup/out-of-range).
 */
FastPAMResult fast_pam_swap(Problem& prob, const std::vector<int>& initial_medoids,
                            int max_iter = 100, PAMVariant variant = PAMVariant::FasterPAM);

} // namespace dtwc
