/**
 * @file Solver.hpp
 * @brief Solver enum for MIP solver selection.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

#include "../error.hpp"

namespace dtwc {

enum class Solver {
  Gurobi, //<! Gurobi solver for MIP solution
  HiGHS   //<! HiGHS solver for MIP solution.
};

inline void validate_solver(Solver value)
{
  switch (value) {
  case Solver::Gurobi:
  case Solver::HiGHS:
    return;
  }
  throw InvalidInput("Invalid Solver value.");
}
}
