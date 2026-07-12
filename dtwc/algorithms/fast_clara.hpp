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
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

#include "../core/clustering_result.hpp"
#include "../settings.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

namespace dtwc {

class Problem; // Forward declaration

namespace algorithms {

  /// Options for the FastCLARA algorithm.
  struct CLARAOptions
  {
    int n_clusters = 3;                                   ///< Number of clusters (k).
    int sample_size = -1;                                 ///< Subsample size. -1 = overflow-safe auto policy.
    int n_samples = 5;                                    ///< Number of subsamples to try.
    int max_iter = 100;                                   ///< Max PAM iterations per subsample.
    unsigned random_seed = settings::DEFAULT_RANDOM_SEED; ///< Reproducible RNG seed.

    // RAM-aware chunked processing
    size_t ram_limit_bytes = 0;         ///< 0 = no limit (all data in RAM).
    std::filesystem::path parquet_path; ///< Parquet file for streaming (empty = data in RAM).
    std::string parquet_column;         ///< Column name for Parquet reader.
    bool use_float32 = false;           ///< Load chunks as float32 (2x memory saving).
  };

  /**
   * @brief Run FastCLARA: scalable k-medoids via subsampling + FastPAM.
   *
   * @param prob      Problem instance with data loaded. The full distance matrix
   *                  is NOT computed (that's the whole point of CLARA).
   * @param opts      CLARAOptions controlling subsample size, repetitions, etc.
   * @return core::ClusteringResult with labels, medoid_indices, total_cost.
   *
   * @note When sample_size resolves to N, in-memory data falls back to one
 * FastPAM run. A streaming Parquet dataset that exceeds the RAM limit rejects
 * that request rather than loading all rows or repeating identical full runs.
 * @throws InvalidInput for invalid dimensions/options, including N > INT_MAX.
 */
  core::ClusteringResult fast_clara(Problem &prob, const CLARAOptions &opts);

} // namespace algorithms
} // namespace dtwc
