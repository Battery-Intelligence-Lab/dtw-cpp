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

#include "settings.hpp"
#include "types/Range.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc {

/// @brief Emit ONE process-wide single-thread warning (Task 3.6). Defined in
///        env.cpp; forward-declared here — not via `#include "env.hpp"` — to keep
///        this hot-path header light and free of an include cycle. Shares the
///        single process-once guard with the dtwc::Env constructor, so a
///        front-end that both constructs env() and computes warns at most once.
void warn_if_single_threaded();

/// @brief Returns the number of available OpenMP threads (1 if OpenMP is absent).
///        On first call, emits a loud stderr warning if running single-threaded —
///        this is the chokepoint every compute path funnels through (via
///        omp_chunk_size / run), so paths that never construct dtwc::env() (Python
///        compute bindings, direct C++ Problem use) are not silently serial
///        (Task 3.6, review finding H1). The warning string is the SSOT in
///        env.cpp (detail::sequential_warning_text) — no duplication here.
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
inline int omp_chunk_size(int n_iterations, int chunks_per_thread = 4)
{
  const int nthreads = get_max_threads();
  return std::max(1, n_iterations / (nthreads * chunks_per_thread));
}

/**
 * @brief Runs a given task in parallel using OpenMP if available.
 *
 * This function executes the provided task in parallel, leveraging OpenMP's dynamic scheduling.
 * The dynamic scheduling is advantageous for tasks with varying completion times. It allows for
 * better load balancing across threads. Falls back to serial execution if OpenMP is not available.
 *
 * @tparam Tfun The type of the task function.
 * @param task_indv Reference to the task function to be executed.
 * @param i_end The upper bound of the loop index.
 * @param isParallel Flag to enable/disable parallel execution (default is true).
 */
template <typename Tfun>
void run_openmp(Tfun &task_indv, size_t i_end, [[maybe_unused]] bool isParallel = true)
{
  // OpenMP requires signed loop variables for compatibility with older compilers
  if (i_end > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Loop bound exceeds maximum int value for OpenMP loop");
  }
  const int end = static_cast<int>(i_end);

#ifdef _OPENMP
  if (isParallel) {
    const int chunk = omp_chunk_size(end);
    // An exception may not leave an OpenMP structured block. Give each loop
    // index its own preallocated slot, then rethrow the lowest-index failure on
    // the caller thread after the implicit join. The atomic cutoff prevents
    // useful work above the best known failing index while still allowing all
    // lower indices to run, so scheduling cannot change which error wins.
    std::vector<std::exception_ptr> failures(static_cast<size_t>(end));
    std::atomic<int> earliest_failure{end};
#pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < end; i++) {
      if (i > earliest_failure.load(std::memory_order_acquire)) continue;
      try {
        task_indv(static_cast<size_t>(i));
      } catch (...) {
        failures[static_cast<size_t>(i)] = std::current_exception();
        int observed = earliest_failure.load(std::memory_order_relaxed);
        while (i < observed
               && !earliest_failure.compare_exchange_weak(
                 observed, i, std::memory_order_release,
                 std::memory_order_relaxed)) {}
      }
    }
    for (const auto &failure : failures)
      if (failure) std::rethrow_exception(failure);
  } else {
    for (int i = 0; i < end; i++)
      task_indv(static_cast<size_t>(i));
  }
#else
  // Serial fallback when OpenMP is not available
  for (int i = 0; i < end; i++)
    task_indv(static_cast<size_t>(i));
#endif
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

  if (useParallel && numMaxParallelWorkers > 0) {
    const int maxThreads = get_max_threads();
#ifdef _OPENMP
    // Respect the requested thread limit, but don't exceed system maximum
    const int requestedThreads = static_cast<int>(std::min(
      numMaxParallelWorkers,
      static_cast<size_t>(maxThreads)));
    omp_set_num_threads(requestedThreads);
#else
    (void)maxThreads; // warning already emitted by get_max_threads()
#endif
  }

  run_openmp(task_indv, i_end, useParallel);
}

} // namespace dtwc
