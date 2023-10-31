/*
 * SimplexRowTable.hpp
 *
 * Sparse implementation of a Simplex table but this time each row.

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
#include <map>


namespace dtwc::solver {

class SimplexRowTable
{
  // Table is   mtab x ntab
  // Inner table is m x n
  int mtab{}, ntab{};
  std::vector<std::map<int, double>> innerTable; // Each is a row
  std::vector<double> reducedCosts, rhs;
  std::vector<int> rowIndices;
  double negativeObjective{};

public:
  SimplexRowTable() = default;
  SimplexRowTable(int mtab_, int ntab_) : mtab{ mtab_ }, ntab{ ntab_ }, innerTable(mtab - 1),
                                          reducedCosts(ntab_ - 1), rhs(mtab_ - 1), rowIndices(ntab_ - 1) {}

  int rows() const { return mtab; }
  int cols() const { return ntab; }

  void clear()
  {
    rowIndices.clear();
    rhs.clear();
    reducedCosts.clear();

    for (auto &map : innerTable) // Should not remove allocated memory for maps.
      map.clear();
  }

  void removeColumns(int a, int b)
  {
    // Removes columns [a, b)
    for (auto &row : innerTable) {
      auto itBegin = row.lower_bound(a);
      auto itEnd = row.upper_bound(b);
      row.erase(itBegin, itEnd);
    }

    reducedCosts.erase(reducedCosts.begin() + a, reducedCosts.begin() + b);
    ntab -= (b - a);
  }


  void createPhaseOneTableau(const EqualityConstraints &eq)
  {
    // Table size (m + 1, m + n + 1)
    // table.block(0, 0, m, n) = A;
    // table.block(0, n, m, m) = SimplexRowTable::Identity(m, m);
    // table.block(0, n + m, m, 1) = b;

    // // Set the first n columns of the last row
    // for (int k = 0; k < n; ++k)
    //   table.row(m)(k) = -table.block(0, k, m, 1).sum();

    // // Set columns n through n+m of the last row to 0.0
    // table.block(m, n, 1, m).setZero();

    // // Set the last element of the last row
    // table(m, n + m) = -table.block(0, n + m, m, 1).sum();

    const int m = eq.A.rows(), n = eq.A.cols();
    mtab = m + 1;
    ntab = m + n + 1;
    clear();              // Clear the things.
    innerTable.resize(m); // Adding m auxillary variables.
    rhs = eq.b;
    reducedCosts.resize(n + m, 0.0); // Set columns n through n+m of the last row to 0.0

    for (const auto [key, val] : eq.A.data) {
      const auto [i, j] = key;
      innerTable[i][j] = val;
      reducedCosts[j] -= val; // Set the first n columns of the last row
    }

    for (int i = 0; i < m; i++)
      innerTable[i][n + i] = 1.0;

    negativeObjective = -std::reduce(rhs.begin(), rhs.end());
  }

  int getRow(int col) const;

  double getObjective() const { return -negativeObjective; }

  double getValue(int k) const
  {
    const auto basicRowNo = getRow(k);
    return (basicRowNo != -1) ? rhs[basicRowNo] : 0.0;
  }

  double getRHS(int k) const { return rhs[k]; }

  double getReducedCost(int k) const { return reducedCosts[k]; }

  double inner(int i, int j) { return innerTable[i][j]; }


  double &setReducedCost(int k) { return reducedCosts[k]; }
  void setNegativeObjective(double val) { negativeObjective = val; }

  int findNegativeCost();

  int findMinStep(int p);

  std::tuple<int, int, bool, bool> simplexTableau();

  void pivoting(int p, int q);

  std::pair<bool, bool> simplexAlgorithmTableau();
};


} // namespace dtwc::solver
