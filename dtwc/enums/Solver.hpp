/**
 * @file Solver.hpp
 * @brief Solver enum for MIP solver selection.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

#include "../base/names.hpp"

namespace dtwc {

/// Mixed-integer programming solver backend.
enum class Solver {
  Gurobi, ///< Gurobi solver for MIP solution
  HiGHS   ///< HiGHS solver for MIP solution
};

/// Table of solver names recognized by CLI and config parsers.
inline constexpr Name<Solver> solver_names[]{
  { "highs", Solver::HiGHS },
  { "gurobi", Solver::Gurobi },
};
}
