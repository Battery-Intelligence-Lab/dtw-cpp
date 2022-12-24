/*
 * solver_util.hpp
 *
 * Utility functions for solver

 *  Created on: 19 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "../settings.hpp"

#include <limits>
#include <array>
#include <vector>

namespace dtwc::solver {

constexpr data_t int_threshold = 0.01;

enum class ConvergenceFlag {
  error_sizeNotSet = -1, //<! Problem size is not set!
  conv_problem = 0,      //<! Problem converged into feasible solution
  conv_admm = 1,         //<! ADMM converged, problem not
  conv_fail = 2          //<! No convergence
};

template <typename T>
inline bool is_one(T x) { return x > (1 - int_threshold); }

template <typename T>
inline bool is_zero(T x) { return x < int_threshold; }

template <typename T>
inline bool is_integer(T x) { return is_one(x) || is_zero(x); }

struct IntSolution
{
  std::vector<std::array<size_t, 2Ui64>> fix_var;
  data_t cost{ std::numeric_limits<data_t>::max() };
  std::vector<data_t> vX_opt;
};


} // namespace dtwc::solver
