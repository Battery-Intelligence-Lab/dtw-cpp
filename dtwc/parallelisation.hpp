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

#include <cstddef>
#include <omp.h>

namespace dtwc {

/**
 * @brief Runs a given task in parallel using OpenMP.
 *
 * This function executes the provided task in parallel, leveraging OpenMP's dynamic scheduling.
 * The dynamic scheduling is advantageous for tasks with varying completion times. It allows for
 * better load balancing across threads.
 *
 * @tparam Tfun The type of the task function.
 * @param task_indv Reference to the task function to be executed.
 * @param i_end The upper bound of the loop index.
 * @param isParallel Flag to enable/disable parallel execution (default is true).
 */
template <typename Tfun>
void run_openmp(Tfun &task_indv, size_t i_end, bool isParallel = true)
{
  if (isParallel) {
#pragma omp parallel for schedule(dynamic) // As some take less time static scheduling is 2x slower.
    for (int i = 0; i < i_end; i++)
      task_indv(i);
  } else
    for (int i = 0; i < i_end; i++)
      task_indv(i);
}

/**
 * @brief A wrapper function to control the degree of parallelism in task execution.
 *
 * @details This function provides a higher level of control for parallel task execution.
 * It decides whether to use parallelism based on the provided maximum number of parallel workers.
 * It delegates the task execution to 'run_openmp'.
 *
 * @tparam Tfun The type of the task function.
 * @param task_indv Reference to the task function to be executed.
 * @param i_end The upper bound of the loop index.
 * @param numMaxParallelWorkers The maximum number of parallel workers (default is 32).
 */
template <typename Tfun>
void run(Tfun &task_indv, size_t i_end, size_t numMaxParallelWorkers = 32)
{
  run_openmp(task_indv, i_end, numMaxParallelWorkers != 1);
}
} // namespace dtwc