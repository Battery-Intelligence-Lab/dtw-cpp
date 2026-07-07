/**
 * @file unit_test_mpi_allreduce_chunking.cpp
 * @brief Host-side tests for the MPI_Allreduce overflow-safe chunking math.
 *
 * @details Exercises the LIVE public logic in <mpi/allreduce_chunking.hpp>
 *          (dtwc::mpi::allreduce_chunk_count and dtwc::mpi::max_allreduce_chunk).
 *          That header is the exact code dtwc::mpi::compute_distance_matrix_mpi()
 *          (dtwc/mpi/mpi_distance_matrix.cpp) uses to bound every MPI_Allreduce
 *          call's element count to INT_MAX. The .cpp is compiled only when
 *          DTWC_ENABLE_MPI=ON (OFF by default on this machine), so the MPI
 *          reduction itself needs an MPI CI runner; but the chunking arithmetic
 *          is deliberately MPI-free so this bug (int overflow of N*N) is pinned
 *          on a live-in-the-MPI-build header here, not on dead code.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <mpi/allreduce_chunking.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>

using dtwc::mpi::allreduce_chunk_count;
using dtwc::mpi::max_allreduce_chunk;

namespace {

// Replays exactly the loop in compute_distance_matrix_mpi(): walk the buffer in
// max_allreduce_chunk strides, summing per-call counts. Returns the total
// covered and, via out-params, the min/max per-call count observed.
std::size_t replay_reduction_loop(std::size_t total_elems,
                                  int &min_count,
                                  long long &max_count,
                                  std::size_t &num_calls)
{
  std::size_t covered = 0;
  num_calls = 0;
  min_count = std::numeric_limits<int>::max();
  max_count = 0;
  for (std::size_t off = 0; off < total_elems; off += max_allreduce_chunk) {
    const int count = allreduce_chunk_count(off, total_elems);
    if (count < min_count) min_count = count;
    if (count > max_count) max_count = count;
    covered += static_cast<std::size_t>(count);
    ++num_calls;
  }
  return covered;
}

} // namespace

TEST_CASE("max_allreduce_chunk equals INT_MAX", "[mpi][chunking]")
{
  // Registered expectation: the cap is exactly the largest signed 32-bit int.
  REQUIRE(max_allreduce_chunk
          == static_cast<std::size_t>(std::numeric_limits<int>::max()));
}

TEST_CASE("Small buffer reduces in a single chunk", "[mpi][chunking]")
{
  // Registered band: total < INT_MAX -> one chunk == total, then 0.
  const std::size_t total = 100;
  REQUIRE(allreduce_chunk_count(0, total) == 100);
  REQUIRE(allreduce_chunk_count(total, total) == 0); // offset == total
  REQUIRE(allreduce_chunk_count(total + 5, total) == 0); // offset > total
}

TEST_CASE("Empty buffer yields no chunks", "[mpi][chunking]")
{
  // Registered band: total == 0 -> every count is 0 (loop body never runs).
  REQUIRE(allreduce_chunk_count(0, 0) == 0);
}

TEST_CASE("Exactly INT_MAX elements is one full chunk", "[mpi][chunking]")
{
  // Registered band: total == INT_MAX -> single chunk of INT_MAX, then 0.
  const std::size_t total = max_allreduce_chunk;
  REQUIRE(allreduce_chunk_count(0, total)
          == std::numeric_limits<int>::max());
  REQUIRE(allreduce_chunk_count(max_allreduce_chunk, total) == 0);

  int min_c = 0;
  long long max_c = 0;
  std::size_t calls = 0;
  REQUIRE(replay_reduction_loop(total, min_c, max_c, calls) == total);
  REQUIRE(calls == 1);
}

TEST_CASE("INT_MAX + 1 elements splits into two chunks (the overflow case)",
          "[mpi][chunking][overflow]")
{
  // This is the exact defect: the old static_cast<int>(N*N) wrapped a value
  // > INT_MAX to a NEGATIVE int count and corrupted the reduction.
  // Registered band: total == INT_MAX+1 -> chunk 0 == INT_MAX, chunk 1 == 1;
  // both counts strictly positive and <= INT_MAX; loop covers total exactly.
  const std::size_t total = max_allreduce_chunk + 1;

  REQUIRE(allreduce_chunk_count(0, total)
          == std::numeric_limits<int>::max());
  REQUIRE(allreduce_chunk_count(max_allreduce_chunk, total) == 1);

  int min_c = 0;
  long long max_c = 0;
  std::size_t calls = 0;
  const std::size_t covered = replay_reduction_loop(total, min_c, max_c, calls);
  REQUIRE(covered == total);
  REQUIRE(calls == 2);
  REQUIRE(min_c > 0);                                       // never non-positive
  REQUIRE(max_c <= std::numeric_limits<int>::max());        // never overflows int
}

TEST_CASE("N*N reduction for N=46341 stays within int per chunk",
          "[mpi][chunking][overflow]")
{
  // N=46341 is the first N where N*N (2,147,488,281) exceeds INT_MAX
  // (2,147,483,647). The old code truncated this to a negative int.
  // Registered band: loop covers all N*N elements; first chunk == INT_MAX,
  // remainder == N*N - INT_MAX == 4634; every per-call count in (0, INT_MAX].
  const std::size_t N = 46341;
  const std::size_t total = N * N; // computed in size_t: no overflow
  REQUIRE(total > static_cast<std::size_t>(std::numeric_limits<int>::max()));

  REQUIRE(allreduce_chunk_count(0, total)
          == std::numeric_limits<int>::max());
  REQUIRE(allreduce_chunk_count(max_allreduce_chunk, total)
          == static_cast<int>(total - max_allreduce_chunk));
  REQUIRE(total - max_allreduce_chunk == 4634);

  int min_c = 0;
  long long max_c = 0;
  std::size_t calls = 0;
  const std::size_t covered = replay_reduction_loop(total, min_c, max_c, calls);
  REQUIRE(covered == total);
  REQUIRE(calls == 2);
  REQUIRE(min_c > 0);
  REQUIRE(max_c <= std::numeric_limits<int>::max());
}
