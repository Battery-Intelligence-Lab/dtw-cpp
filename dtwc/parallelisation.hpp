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

// #include <execution>
// #include <algorithm>
#include <cstddef>
#include "omp.h"


namespace dtwc {

// namespace ex = std::execution;

// template <typename Tfun>
// void run_std(Tfun &task_indv, size_t i_end, bool isParallel = settings::isParallel)
// {
//   auto range = Range(i_end);
//   if (isParallel)
//     std::for_each(ex::par_unseq, range.begin(), range.end(), task_indv);
//   else
//     std::for_each(ex::seq, range.begin(), range.end(), task_indv);
// }

template <typename Tfun>
void run_openmp(Tfun &task_indv, size_t i_end, bool isParallel = settings::isParallel)
{
  if (isParallel) {
#pragma omp parallel for schedule(dynamic) // As some take less time static scheduling is 2x slower.
    for (int i = 0; i < i_end; i++) {
      task_indv(i);
    }
  } else
    for (int i = 0; i < i_end; i++)
      task_indv(i);
}


template <typename Tfun>
void run(Tfun &task_indv, size_t i_end, size_t numMaxParallelWorkers = settings::numMaxParallelWorkers)
{
  std::cout << "Standard algorithms parallelisation is being used." << std::endl;
  run_openmp(task_indv, i_end, numMaxParallelWorkers > 1);
}
} // namespace dtwc