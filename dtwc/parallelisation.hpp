/*
 * parallelisation.hpp
 *
 * Parallelisation functions

 * Created on: 15 Dec 2021
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"

#include <vector>
#include <thread>
#include <execution>
#include <algorithm>


namespace dtwc {
template <typename Tfun> // #TODO change with parallel algorithms.
void run_legacy(Tfun task_indv, int i_end, ind_t numMaxParallelWorkers = settings::numMaxParallelWorkers)
{

  auto task_par = [&](int i_begin, int i_end, int Nth) {
    while (i_begin < i_end) {
      task_indv(i_begin);
      i_begin += Nth;
    }
  };

  if constexpr (settings::isParallel) {
    if (numMaxParallelWorkers == 1)
      task_par(0, i_end, 1);
    else {
      if (numMaxParallelWorkers < 1)
        numMaxParallelWorkers = std::thread::hardware_concurrency();

      const ind_t N_th_max = std::min(numMaxParallelWorkers, std::thread::hardware_concurrency());

      std::vector<std::thread> threads;
      threads.reserve(N_th_max);

      for (ind_t i_begin = 0; i_begin < N_th_max; i_begin++) //!< indices for the threads
      {
        //!< Multi threaded simul:

        threads.emplace_back(task_par, i_begin, i_end, N_th_max);
      }

      for (auto &th : threads) {
        if (th.joinable())
          th.join();
      }
    }
  } else {
    task_par(0, i_end, 1);
  }
}

#if USE_STD_PAR_ALGORITMHS
namespace ex = std::execution;
#endif

template <typename Tfun>
void run_std(Tfun &task_indv, size_t i_end, ind_t numMaxParallelWorkers = settings::numMaxParallelWorkers)
{
#if USE_STD_PAR_ALGORITMHS

  auto range = Range(i_end);

  if constexpr (settings::isParallel)
    std::for_each(ex::par_unseq, range.begin(), range.end(), task_indv);
  else
    std::for_each(ex::seq, range.begin(), range.end(), task_indv);
#endif
}


template <typename Tfun>
void run(Tfun &task_indv, size_t i_end, ind_t numMaxParallelWorkers = settings::numMaxParallelWorkers)
{
#if USE_STD_PAR_ALGORITMHS
  // std::cout << "Standard algorithms parallelisation is being used." << std::endl;
  run_std(task_indv, i_end, numMaxParallelWorkers);
#else
  // std::cout << "Thread-based parallelisation is being used." << std::endl;
  run_legacy(task_indv, i_end, numMaxParallelWorkers);
#endif
}
} // namespace dtwc