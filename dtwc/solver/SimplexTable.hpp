/*
 * SimplexSolver.hpp
 *
 * Sparse implementation of a Simplex table.

 *  Created on: 22 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"


#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <tuple>
#include <stdexcept>
#include <set>
#include <limits>
#include <map>


namespace dtwc::solver {

class SimplexTable
{
  // Table is   mtab x ntab
  // Inner table is m x n
  std::vector<std::map<int, double>> innerTable; // Each is a column
  std::vector<double> reducedCosts, rhs;
  double negativeObjective{};

  int mtab{}, ntab{};

public:
  SimplexTable() = default;
  SimplexTable(int mtab_, int ntab_) : mtab{ mtab_ }, ntab{ ntab_ }, innerTable(ntab_ - 1), reducedCosts(ntab_ - 1), rhs(mtab_ - 1) {}
};


} // namespace dtwc::solver
