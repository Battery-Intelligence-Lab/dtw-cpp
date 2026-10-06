/**
 * @file fast_clara.hpp
 * @brief FastCLARA: scalable k-medoids via subsampling + FastPAM.
 *
 * @details Implements CLARA (Clustering Large Applications) using FasterPAM
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
#include "../base/settings.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace dtwc {

class Problem; // Forward declaration

namespace algorithms {

  /// Options for the FastCLARA algorithm.
  struct CLARAOptions
  {
    index_t n_clusters = 3;                                    ///< Number of clusters (k).
    index_t sample_size = -1;                                  ///< Subsample size. -1 = auto policy.
    int n_samples = 5;                                         ///< Number of subsamples to try.
    int max_iter = 100;                                        ///< Max PAM iterations per subsample.
    std::uint64_t random_seed = settings::DEFAULT_RANDOM_SEED; ///< Reproducible RNG seed.

    // Streaming from Parquet: fast_clara_parquet's (dtwc_io); fast_clara refuses it.
    size_t ram_limit_bytes = 0;         ///< Series-data cap; 0 = no limit.
    std::filesystem::path parquet_path; ///< Parquet file to stream (empty = the Problem's series).
    std::string parquet_column;         ///< Column name for the Parquet reader.
    bool use_float32 = false;           ///< Read chunks as float32 (half the memory).
    bool force_parquet_streaming = false; ///< Stream even if the series fit (dtwc_cl's planner decided).
  };

  /**
   * @brief Run FastCLARA: scalable k-medoids via subsampling + FastPAM.
   *
   * @param prob      Problem instance with data loaded. The full distance matrix
   *                  is NOT computed (that's the whole point of CLARA).
   * @param opts      CLARAOptions controlling subsample size, repetitions, etc.
   * @return core::ClusteringResult with labels, medoid_indices, total_cost.
   *
   * @note When sample_size resolves to N, the data falls back to one FastPAM run.
   * Non-full assignment evaluates the configured bound DTW function directly;
   * an existing parent distance cache is ignored and left unchanged.
   * Each sample Problem takes `prob`'s distance settings, metric and device
   * (GPU index, precision). An in-memory sample is a view of `prob`'s series on
   * the CPU and a copy on a GPU, which uploads owned series. On a GPU device the
   * sample matrices fill on the GPU, and so does the assignment where the GPU
   * is CUDA's; Metal has no kernel for the assignment, which then runs on the
   * CPU (a verbose line says so).
 * @throws InvalidInput for invalid dimensions/options, and for a Parquet
 *         stream (parquet_path with ram_limit_bytes), which is fast_clara_parquet's;
 *         IOError for force_parquet_streaming: the core's FastCLARA reads no
 *         Parquet (fast_clara_parquet streams it, in a build with Parquet);
 *         DeviceError for a request the GPU cannot honour (Float32 series, a
 *         variant or missing-data strategy its kernels lack).
 */
  core::ClusteringResult fast_clara(Problem &prob, const CLARAOptions &opts);

#ifdef DTWC_HAS_PARQUET
  /**
   * @brief fast_clara, with the series streamed from opts.parquet_path when
   *        opts.ram_limit_bytes asks for it: forced, or because the file's
   *        resident estimate exceeds the limit (the RAM-limited route of
   *        dtwc_cl --ram-limit). Otherwise it is fast_clara itself.
   *
   * A streamed run needs a settings-only `prob` (no hidden resident-plus-chunk
   * memory peak) and list-per-row Parquet; a stream whose sample_size resolves
   * to N is refused rather than loading all rows or repeating identical full
   * runs. It is dtwc_io's (algorithms/fast_clara_parquet.cpp), which a build
   * with Parquet links: the core reads no Parquet.
   */
  core::ClusteringResult fast_clara_parquet(Problem &prob, const CLARAOptions &opts);
#endif

} // namespace algorithms
} // namespace dtwc
