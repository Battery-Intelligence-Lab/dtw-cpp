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
#include "EqualityConstraints.hpp"


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

  int rows() const { return mtab; }
  int cols() const { return ntab; }


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

  int getRow(int col) const
  {
    if (col < 0 || col >= (ntab - 1)) // Check if index is in the valid range
      throw std::runtime_error(fmt::format("The index of the variable ({}) must be between 0 and {}", col, ntab - 2));

    int rowIndex = -1; // Using -1 to represent None
    // So this is checking there is one and only one 1.0 element! -> #TODO change with something keeping book of basic variables in future.
    for (auto [key, value] : innerTable[col])
      if (!isAround(value, 0)) // The entry is non zero
      {
        if (rowIndex == -1 && isAround(value, 1.0)) // The entry is one, and the index has not been found yet.
          rowIndex = key;
        else
          return -1;
      }

    return rowIndex;
  }

  double getObjective() const { return -negativeObjective; }

  double getValue(int k) const
  {
    const auto basicRowNo = getRow(k);
    return (basicRowNo != -1) ? rhs[basicRowNo] : 0.0;
  }

  double getRHS(int k) const { return rhs[k]; }

  void pivoting(int p, int q)
  {
    // Make p,q element one and eliminate all other nonzero elements in that column by basic row operations.
    const double thepivot = table(q, p);
    if (isAround(thepivot, 0.0))
      throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));

    table.row(q) /= thepivot; // Make (p,q) one.

    for (int i = 0; i < table.rows(); ++i)
      if (i != q)
        table.row(i) -= table(i, p) * table.row(q);
  }
};


} // namespace dtwc::solver
