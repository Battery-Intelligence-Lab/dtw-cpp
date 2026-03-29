/**
 * @file mpi_distance_matrix.hpp
 * @brief MPI-distributed pairwise DTW distance matrix computation.
 *
 * @details Distributes N(N-1)/2 pair computations across P MPI ranks.
 *          Each rank computes a contiguous block of the upper triangle,
 *          then results are gathered via MPI_Allreduce (SUM over
 *          non-overlapping positions).
 *          All ranks must hold all N series (replicated data).
 *
 * @date 29 Mar 2026
 */

#pragma once

#ifdef DTWC_HAS_MPI

#include <vector>
#include <cstddef>
#include <utility>

namespace dtwc::mpi {

/**
 * @brief Options for MPI-distributed distance matrix computation.
 */
struct MPIDistMatOptions {
  /// Sakoe-Chiba band width. Negative means full DTW (no band).
  int band = -1;

  /// Print progress info on rank 0.
  bool verbose = false;
};

/**
 * @brief Result of MPI-distributed distance matrix computation.
 */
struct MPIDistMatResult {
  /// Full NxN distance matrix in row-major order. Symmetric, zero diagonal.
  std::vector<double> matrix;

  /// Number of series.
  size_t n = 0;

  /// This process's MPI rank.
  int rank = 0;

  /// Total number of MPI processes.
  int world_size = 0;

  /// Number of pairs computed locally by this rank.
  size_t local_pairs = 0;

  /// Wall-clock time for local computation (seconds).
  double local_time_sec = 0.0;
};

/**
 * @brief Compute the full NxN DTW distance matrix distributed across MPI ranks.
 *
 * All ranks must call this with the same series data.
 * Returns the full symmetric matrix on ALL ranks (via MPI_Allreduce).
 *
 * @param series Vector of N time series (each a vector<double>).
 * @param opts   Options controlling band width and verbosity.
 * @return MPIDistMatResult with the full distance matrix.
 */
MPIDistMatResult compute_distance_matrix_mpi(
    const std::vector<std::vector<double>> &series,
    const MPIDistMatOptions &opts = {});

/**
 * @brief Initialize MPI if not already initialized. Safe to call multiple times.
 */
void ensure_mpi_initialized();

/**
 * @brief Get current MPI rank and world size.
 * @return Pair of (rank, world_size).
 */
std::pair<int, int> mpi_rank_and_size();

} // namespace dtwc::mpi

#endif // DTWC_HAS_MPI
