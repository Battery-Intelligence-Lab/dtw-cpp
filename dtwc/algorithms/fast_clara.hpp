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
    /// CUDA's assignment: its host buffers within this many bytes, a block down to one series (0: blocks of a
    /// launch's bound); dtwc_cl's memory plan sets it where an explicit --ram-limit leaves the default blocks no room.
    size_t gpu_staging_bytes = 0;
    /// cpu+gpu: the share of the points, the leading ones, whose assignment runs on the GPU, the rest on the CPU's
    /// team. -1 measures it at the start of the pass (a calibration block timed on each device; the share and both
    /// rates are the result's `hybrid`); a value in [0, 1] forces it, so a run repeats exactly. Another device refuses
    /// any value but -1.
    double gpu_share = -1.0;

    // Streaming from Parquet: fast_clara_parquet's (dtwc_io); fast_clara refuses it.
    size_t ram_limit_bytes = 0;         ///< Series-data cap; 0 = no limit.
    std::filesystem::path parquet_path; ///< Parquet file to stream (empty = the Problem's series).
    std::string parquet_column;         ///< Column name for the Parquet reader.
    index_t parquet_skip_cols = 0;      ///< Leading Parquet columns dropped (--skip-cols).
    index_t parquet_skip_rows = 0;      ///< Leading Parquet rows dropped (--skip-rows).
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
   * CPU (a verbose line says so). On cpu+gpu (CUDA) the samples fill on the GPU
   * and the assignment pass is split: the GPU takes the leading share of the
   * points (opts.gpu_share), the CPU's team the rest, in one parallel region;
   * under FP64 L1 the results equal both single-device runs whatever the split.
 * @throws InvalidInput for invalid dimensions/options, and for a Parquet
 *         stream (parquet_path with ram_limit_bytes), which is fast_clara_parquet's;
 *         IOError for force_parquet_streaming: the core's FastCLARA reads no
 *         Parquet (fast_clara_parquet streams it, in a build with Parquet);
 *         DeviceError for a request the GPU cannot honour (Float32 series, a
 *         variant or missing-data strategy its kernels lack).
 */
  core::ClusteringResult fast_clara(Problem &prob, const CLARAOptions &opts);

  /// cpu+gpu's split of a pass over `n_points` series in words: the GPU's points and share (17 digits: forced again,
  /// it repeats the split), how the share was set and, where measured, both rates. dtwc_cl's -v and run report print it.
  std::string describe(const core::HybridSplit &split, std::size_t n_points);

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
