/*
 * Solver.hpp
 *
 * Solver enum for MIP solver selection

 *  Created on: 11 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

namespace dtwc {

enum class Solver {
  Gurobi, //<! Gurobi solver for MIP solution
  HiGHS   //<! HiGHS solver for MIP solution.

}