/*
 * parallelisation.hpp
 *
 * Parallelisation functions

 * Created on: 15 Dec 2021
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"

#include <execution>
#include <algorithm>

namespace dtwc {

namespace ex = std::execution;

template <typename Tfun>
void run_std(Tfun &task_indv, size_t i_end, bool isParallel = settings::isParallel)
{
  auto range = Range(i_end);
  if (isParallel)
    std::for_each(ex::par_unseq, range.begin(), range.end(), task_indv);
  else
    std::for_each(ex::seq, range.begin(), range.end(), task_indv);
}


template <typename Tfun>
void run(Tfun &task_indv, size_t i_end, size_t numMaxParallelWorkers = settings::numMaxParallelWorkers)
{
  std::cout << "Standard algorithms parallelisation is being used." << std::endl;
  run_std(task_indv, i_end);
}
} // namespace dtwc