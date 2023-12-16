/*
 * parallelisation.hpp
 *
 * Parallelisation functions

 * Created on: 15 Dec 2021
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"
#include "types/Range.hpp"

#include <cstddef>
#include <omp.h>


namespace dtwc {

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


template <typename Tfun>
void run(Tfun &task_indv, size_t i_end, size_t numMaxParallelWorkers = 32)
{
  run_openmp(task_indv, i_end, numMaxParallelWorkers != 1);
}
} // namespace dtwc