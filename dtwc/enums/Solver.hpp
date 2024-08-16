/**
 * @file Solver.hpp
 * @brief Solver enum for MIP solver selection.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

namespace dtwc {

enum class Solver {
  Gurobi, //<! Gurobi solver for MIP solution
  HiGHS   //<! HiGHS solver for MIP solution.
};
}