/*
 * solver_util.hpp
 *
 * Utility functions for solver

 *  Created on: 19 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

namespace dtwc::solver {

enum class ConvergenceFlag {
  error_sizeNotSet = -1, //<! Problem size is not set!
  conv_problem = 0,      //<! Problem converged into feasible solution
  conv_admm = 1,         //<! ADMM converged, problem not
  conv_fail = 2          //<! No convergence
};

template <typename Tdata>
inline bool is_integer(Tdata x)
{
  constexpr data_t threshold = 0.02;
  return x < threshold || x < (1 - threshold);
}


} // namespace dtwc::solver
