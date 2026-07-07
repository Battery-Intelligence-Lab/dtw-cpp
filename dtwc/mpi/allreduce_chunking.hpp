/**
 * @file allreduce_chunking.hpp
 * @brief Overflow-safe chunking for MPI_Allreduce element counts.
 *
 * @details The classic MPI_Allreduce takes its element count as a 32-bit
 *          `int`. For a full NxN distance matrix, N*N exceeds INT_MAX once
 *          N > ~46340, so a single `static_cast<int>(N*N)` silently truncates
 *          (in fact wraps to a negative value) and corrupts the reduction.
 *          This header exposes the pure integer arithmetic that splits an
 *          arbitrarily large element count into consecutive chunks, each of at
 *          most INT_MAX elements, so the reduction can be issued as a sequence
 *          of standard MPI_Allreduce calls that work with every MPI version
 *          (no MPI-4 large-count API required).
 *
 *          Intentionally header-only and free of any <mpi.h> / DTWC_HAS_MPI
 *          dependency, so the index math is unit-testable on hosts without an
 *          MPI runtime — this project's default build has DTWC_ENABLE_MPI=OFF.
 *          The live consumer is dtwc::mpi::compute_distance_matrix_mpi() in
 *          mpi_distance_matrix.cpp (compiled only when DTWC_HAS_MPI is set),
 *          which calls the identical logic exercised by the unit test.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#pragma once

#include <cstddef>
#include <limits>

namespace dtwc::mpi {

/// Largest element count that fits MPI_Allreduce's 32-bit `int` count argument.
inline constexpr std::size_t max_allreduce_chunk =
  static_cast<std::size_t>(std::numeric_limits<int>::max());

/**
 * @brief Element count for the reduction chunk starting at @p offset.
 *
 * Splits a buffer of @p total elements into consecutive chunks, each at most
 * ::max_allreduce_chunk (INT_MAX) elements, so every returned value fits the
 * `int` count parameter of MPI_Allreduce without truncation. The caller
 * advances @p offset by ::max_allreduce_chunk each iteration; the buffer
 * offset itself stays 64-bit (std::size_t), only the per-call count is int.
 *
 * @param offset Start index of the chunk within the buffer.
 * @param total  Total number of elements in the buffer.
 * @return Number of elements in this chunk; 0 once @p offset >= @p total.
 */
inline int allreduce_chunk_count(std::size_t offset, std::size_t total) noexcept
{
  if (offset >= total) return 0;
  const std::size_t remaining = total - offset;
  return static_cast<int>(remaining < max_allreduce_chunk ? remaining
                                                          : max_allreduce_chunk);
}

} // namespace dtwc::mpi
