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
#include <cstddef>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dtwc {

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
#pragma omp parallel for schedule(dynamic) // Dynamic scheduling for varying task times
    for (int i = 0; i < end; i++)
      task_indv(static_cast<size_t>(i));
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

#ifdef _OPENMP
  if (useParallel && numMaxParallelWorkers > 0) {
    // Respect the requested thread limit, but don't exceed system maximum
    const int requestedThreads = static_cast<int>(std::min(
      numMaxParallelWorkers,
      static_cast<size_t>(omp_get_max_threads())));
    omp_set_num_threads(requestedThreads);
  }
#endif

  run_openmp(task_indv, i_end, useParallel);
}

} // namespace dtwc