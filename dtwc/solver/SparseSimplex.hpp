/*
 * SparseSimplex.hpp
 *
 * LP solution

 *  Created on: 26 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"
#include "SimplexTable.hpp"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>


#include <tuple>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>
#include <set>
#include <limits>


namespace dtwc {
class Problem;
}

namespace dtwc::solver {

class SparseSimplex
{
  using VectorType = std::vector<double>;
  // -1 means None.
  int nGomory{ -1 };

  SimplexTable table;
  EqualityConstraints eq;
  VectorType c; // c*x = cost.

public:
  SparseSimplex() = default;
  SparseSimplex(Problem &prob);
  SparseSimplex(EqualityConstraints &eq_, const VectorType &c_) : eq(eq_), c(c_) {}

  void gomory();
  void gomoryAlgorithm()
  {
    while (nGomory != 0) gomory();
  }

  std::pair<std::vector<double>, double> getResults() const;
  std::tuple<bool, bool> simplex();
};

} // namespace dtwc::solver
