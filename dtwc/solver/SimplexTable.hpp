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

  void createTableau(int Nb, int Nc)
  {
    const auto Neq = Nb + 1;
    const auto Nineq = Nb * (Nb - 1);
    const auto Nconstraints = Neq + Nineq;

    const auto Nvar_original = Nb * Nb;
    const auto N_slack = Nineq;
    const auto Nvar = Nvar_original + N_slack; // x1--xN^2  + s_slack


    *this = SimplexTable(Nconstraints + 1, Nvar + Nconstraints + 1);

    negativeObjective = -(Nb + Nc);
    rhs[0] = Nc;
    for (int i = 0; i < Nb; ++i)
      rhs[i + 1] = 1;
  }
};


} // namespace dtwc::solver
