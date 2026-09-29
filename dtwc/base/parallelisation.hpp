/**
 * @file parallelisation.hpp
 * @brief Header for parallelisation functions.
 *
 * @details This header file provides functionalities for parallelising tasks using standard parallelisation.
 * It includes functions for running individual tasks in parallel and adjusting
 * the level of parallelism. Functions are templated to support various task types.
 *
 * @date 15 Dec 2021
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "settings.hpp" // index_t

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc {

/// @brief Emit ONE process-wide single-thread warning. Defined in env.cpp;
///        forward-declared here — not via `#include "env.hpp"` — to keep this
///        hot-path header light and free of an include cycle.
void warn_if_single_threaded();

/// @brief Returns the number of available OpenMP threads (1 if OpenMP is absent).
///        On first call, emits a loud stderr warning if running single-threaded —
///        this is the chokepoint every compute path funnels through (via
///        omp_chunk_size / run), so no compute path is silently serial. The
///        warning text lives in env.cpp (warn_if_single_threaded).
inline int get_max_threads()
{
#ifdef _OPENMP
  warn_if_single_threaded();
  return omp_get_max_threads();
#else
  warn_if_single_threaded();
  return 1;
#endif
}

/// @brief Computes an optimal OpenMP dynamic chunk size at runtime.
/// @param n_iterations Total loop iterations.
/// @param chunks_per_thread Target number of chunks per thread for load balancing (default: 4).
/// @return Chunk size >= 1, scaled to the machine's thread count.
///
/// Heuristic: each thread gets ~chunks_per_thread work units. This balances
/// dispatch overhead (fewer, larger chunks) against load imbalance (more, smaller chunks).
/// For 168 threads with N=8926: chunk=13. For 16 threads with N=28: chunk=1.
inline index_t omp_chunk_size_for(index_t n_iterations, int chunks_per_thread, int nthreads)
{
  assert(chunks_per_thread > 0 && nthreads > 0); // a literal and an OpenMP thread count
  return std::max<index_t>(1, n_iterations / (index_t{ nthreads } * chunks_per_thread));
}

inline int omp_chunk_size(int n_iterations, int chunks_per_thread = 4)
{
  // At most n_iterations, so it fits in int.
  return static_cast<int>(omp_chunk_size_for(n_iterations, chunks_per_thread, get_max_threads()));
}

/**
 * @brief Runs task_indv(i) for every i in [0, i_end), on OpenMP threads with
 *        dynamic scheduling (tasks of uneven cost balance), else serially.
 *
 * An exception may not leave an OpenMP region, so each thread keeps its first
 * failure in its own slot and skips its remaining iterations; after the join the
 * caller rethrows one of the stored failures (which one, when several threads
 * fail, may vary between runs). A serial run lets a failure propagate directly.
 *
 * @tparam Tfun The type of the task function.
 * @param task_indv The task; in a parallel run it is called from several threads.
 * @param i_end Number of iterations.
 * @param isParallel Run on OpenMP threads (default true).
 * @param chunks_per_thread Dynamic-scheduling granularity (default is 4).
 * @param max_workers Upper bound on workers for THIS region only (0 = the
 *        OpenMP default), applied through a num_threads(...) clause. Nothing
 *        process-wide is mutated, so one constrained call cannot pin later ones.
 */
template <typename Tfun>
void run_openmp(Tfun &task_indv, size_t i_end,
                [[maybe_unused]] bool isParallel = true,
                [[maybe_unused]] int chunks_per_thread = 4,
                [[maybe_unused]] int max_workers = 0)
{
  const auto end = static_cast<index_t>(i_end);
#ifdef _OPENMP
  if (isParallel) {
    const int available = get_max_threads();
    const int nthreads =
      (max_workers > 0) ? std::min(max_workers, available) : available;
    const index_t chunk = omp_chunk_size_for(end, chunks_per_thread, nthreads);
    // One slot per thread. `failed` is what the loop tests: the MSVC STL's
    // exception_ptr::operator bool is an out-of-line call.
    struct Slot {
      std::exception_ptr failure;
      bool failed = false;
    };
    std::vector<Slot> slots(static_cast<size_t>(nthreads));
#pragma omp parallel num_threads(nthreads)
    {
      Slot &slot = slots[static_cast<size_t>(omp_get_thread_num())];
#pragma omp for schedule(dynamic, chunk) nowait
      for (index_t i = 0; i < end; i++) {
        if (slot.failed) continue;
        try {
          task_indv(static_cast<size_t>(i));
        } catch (...) {
          slot.failure = std::current_exception();
          slot.failed = true;
        }
      }
    }
    for (const auto &slot : slots)
      if (slot.failed) std::rethrow_exception(slot.failure);
    return;
  }
#endif
  for (index_t i = 0; i < end; i++)
    task_indv(static_cast<size_t>(i));
}

/**
 * @brief A wrapper function to control the degree of parallelism in task execution.
 *
 * @details This function provides a higher level of control for parallel task execution.
 * It decides whether to use parallelism based on the provided maximum number of parallel workers.
 * When numMaxParallelWorkers > 1, it sets the OpenMP thread count accordingly.
 *
 * @tparam Tfun The type of the task function.
 * @param task_indv Reference to the task function to be executed.
 * @param i_end The upper bound of the loop index.
 * @param numMaxParallelWorkers The maximum number of parallel workers (default is 32).
 *        Set to 1 for serial execution.
 */
template <typename Tfun>
void run(Tfun &task_indv, size_t i_end, size_t numMaxParallelWorkers = 32)
{
  const bool useParallel = (numMaxParallelWorkers != 1);

  // Region-local limit only, so a later unconstrained fill still sees the
  // machine's full thread count.
  int requestedThreads = 0;
  if (useParallel && numMaxParallelWorkers > 0) {
    const int maxThreads = get_max_threads();
    requestedThreads = static_cast<int>(std::min(
      numMaxParallelWorkers, static_cast<size_t>(maxThreads)));
  }

  run_openmp(task_indv, i_end, useParallel, 4, requestedThreads);
}

} // namespace dtwc
