/**
 * @file fast_clara.hpp
 * @brief FastCLARA: scalable k-medoids via subsampling + FastPAM.
 *
 * @details Implements CLARA (Clustering Large Applications) using FastPAM1
 *   on random subsamples. Reference:
 *   - Kaufman, L. & Rousseeuw, P.J. (1990). "Finding Groups in Data."
 *     Wiley Series in Probability and Statistics.
 *   - Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager k-medoids
 *     clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
 *     algorithms." JMLR, 22(1), 4653-4688.
 *
 * CLARA avoids O(N^2) memory by running PAM on subsamples of size s << N,
 * then assigning all N points to the best medoids found.
 *
 * @date 29 Mar 2026
 */

#pragma once

#include "../core/clustering_result.hpp"

namespace dtwc {

class Problem; // Forward declaration

namespace algorithms {

/// Options for the FastCLARA algorithm.
struct CLARAOptions {
  int n_clusters = 3;       ///< Number of clusters (k).
  int sample_size = -1;     ///< Subsample size. -1 = auto (40 + 2*k).
  int n_samples = 5;        ///< Number of subsamples to try.
  int max_iter = 100;       ///< Max PAM iterations per subsample.
  unsigned random_seed = 42; ///< RNG seed for reproducibility.
};

/**
 * @brief Run FastCLARA: scalable k-medoids via subsampling + FastPAM.
 *
 * @param prob      Problem instance with data loaded. The full distance matrix
 *                  is NOT computed (that's the whole point of CLARA).
 * @param opts      CLARAOptions controlling subsample size, repetitions, etc.
 * @return core::ClusteringResult with labels, medoid_indices, total_cost.
 *
 * @note When sample_size >= N, falls back to a single FastPAM run on all data.
 */
core::ClusteringResult fast_clara(Problem& prob, const CLARAOptions& opts);

} // namespace algorithms
} // namespace dtwc
