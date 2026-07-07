/**
 * @file mpi_distance_matrix.cpp
 * @brief Implementation of MPI-distributed DTW distance matrix.
 *
 * @details Each MPI rank computes a contiguous block of the N(N-1)/2
 *          upper-triangle pairs. Results are combined via MPI_Allreduce
 *          with MPI_SUM (non-overlapping positions are zero, so sum
 *          reconstructs the full matrix).
 *
 *          Within each rank, OpenMP is used for thread-level parallelism
 *          when available.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include "mpi_distance_matrix.hpp"
#include "../parallelisation.hpp"

#ifdef DTWC_HAS_MPI

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>
#include <utility>

#include "../warping.hpp"
#include "../detail/decode_pair.hpp"

namespace dtwc::mpi {

void ensure_mpi_initialized()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }
}

std::pair<int, int> mpi_rank_and_size()
{
  ensure_mpi_initialized();
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return { rank, size };
}

namespace {

/**
 * @brief Decode a linear upper-triangle index to (i, j) pair indices.
 *
 * Thin adapter over the SSOT dtwc::detail::decode_pair (see
 * dtwc/detail/decode_pair.hpp). Previously this file held one of three
 * divergent copies; the shared FP64 + int64 implementation is the audited-
 * correct one, so all three backends now share it.
 */
std::pair<size_t, size_t> decode_pair(size_t k, size_t N)
{
  std::int64_t i = 0, j = 0;
  dtwc::detail::decode_pair(static_cast<std::int64_t>(k),
                            static_cast<std::int64_t>(N), i, j);
  return { static_cast<size_t>(i), static_cast<size_t>(j) };
}

} // anonymous namespace

MPIDistMatResult compute_distance_matrix_mpi(
    const std::vector<std::vector<double>> &series,
    const MPIDistMatOptions &opts)
{
  ensure_mpi_initialized();

  auto [rank, world_size] = mpi_rank_and_size();
  const size_t N = series.size();
  const size_t total_pairs = N * (N - 1) / 2;

  MPIDistMatResult result;
  result.n = N;
  result.rank = rank;
  result.world_size = world_size;
  result.matrix.resize(N * N, 0.0);

  if (N <= 1) {
    return result;
  }

  // Divide pairs among ranks: contiguous blocks with remainder distributed
  const size_t pairs_per_rank = total_pairs / static_cast<size_t>(world_size);
  const size_t remainder = total_pairs % static_cast<size_t>(world_size);

  const auto urank = static_cast<size_t>(rank);
  const size_t start_k = urank * pairs_per_rank + std::min(urank, remainder);
  const size_t local_count = pairs_per_rank + (urank < remainder ? 1 : 0);

  result.local_pairs = local_count;

  if (opts.verbose && rank == 0) {
    std::cout << "[MPI] distance matrix: N=" << N
              << ", total_pairs=" << total_pairs
              << ", world_size=" << world_size
              << ", pairs_per_rank~=" << pairs_per_rank << "\n";
  }

  const auto start_time = std::chrono::high_resolution_clock::now();

  // Local buffer for computed distances and their (i,j) coordinates
  std::vector<double> local_distances(local_count);
  std::vector<std::pair<size_t, size_t>> local_ij(local_count);

  // Compute this rank's share of pairs
  // Lock-free by design: each thread writes only to local_distances[idx] and
  // local_ij[idx] at its own index — no two threads access the same element.
  // Use OpenMP within each rank for thread-level parallelism
#ifdef _OPENMP
  const int mpi_chunk = dtwc::omp_chunk_size(static_cast<int>(local_count));
#pragma omp parallel for schedule(dynamic, mpi_chunk)
#endif
  for (int idx = 0; idx < static_cast<int>(local_count); ++idx) {
    const size_t k = start_k + static_cast<size_t>(idx);
    auto [i, j] = decode_pair(k, N);
    local_ij[idx] = { i, j };

    double d;
    if (opts.band >= 0) {
      d = dtwBanded<double>(series[i], series[j], opts.band);
    } else {
      d = dtwFull_L<double>(series[i], series[j]);
    }
    local_distances[idx] = d;
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  result.local_time_sec = std::chrono::duration<double>(end_time - start_time).count();

  // Fill local results into matrix (both upper and lower triangle)
  for (size_t idx = 0; idx < local_count; ++idx) {
    auto [i, j] = local_ij[idx];
    result.matrix[i * N + j] = local_distances[idx];
    result.matrix[j * N + i] = local_distances[idx];
  }

  // Combine across ranks: MPI_Allreduce with SUM.
  // Each rank wrote to non-overlapping (i,j) positions; uncomputed
  // positions are 0.0. Summing yields the full matrix on every rank.
  std::vector<double> global_matrix(N * N, 0.0);
  MPI_Allreduce(
      result.matrix.data(),
      global_matrix.data(),
      static_cast<int>(N * N),
      MPI_DOUBLE,
      MPI_SUM,
      MPI_COMM_WORLD);
  result.matrix = std::move(global_matrix);

  if (opts.verbose && rank == 0) {
    std::cout << "[MPI] distance matrix complete: "
              << result.local_time_sec << "s local compute\n";
  }

  return result;
}

} // namespace dtwc::mpi

#endif // DTWC_HAS_MPI
