/*
 * SimplexFlatTable.hpp
 *
 * Sparse implementation of a Simplex table but this time each row.
 * Using vectors for a flat map instead of std::map to avoid allocations.
 *  Created on: 30 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"
#include "sparse_util.hpp"
#include "EqualityConstraints.hpp"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <tuple>
#include <stdexcept>
#include <map>
#include <execution>


namespace dtwc::solver {


class SimplexFlatTable
{
  // Table is   mtab x ntab
  // Inner table is m x n
  std::vector<std::vector<Element>> innerTable; // Each is a column
  std::vector<double> reducedCosts, rhs;
  double negativeObjective{};

  int mtab{}, ntab{};

public:
  SimplexFlatTable() = default;
  SimplexFlatTable(int mtab_, int ntab_) : mtab{ mtab_ }, ntab{ ntab_ }, innerTable(ntab_ - 1),
                                           reducedCosts(ntab_ - 1), rhs(mtab_ - 1) {}

  int rows() const { return mtab; }
  int cols() const { return ntab; }

  void clear()
  {
    rhs.clear();
    reducedCosts.clear();

    for (auto &map : innerTable) // Should not remove allocated memory for maps.
      map.clear();
  }

  void removeColumns(int a, int b)
  {
    // Removes columns [a, b)
    innerTable.erase(innerTable.begin() + a, innerTable.begin() + b);
    reducedCosts.erase(reducedCosts.begin() + a, reducedCosts.begin() + b);
    ntab -= (b - a);
  }


  void createPhaseOneTableau(const EqualityConstraints &eq)
  {
    const int m = eq.A.rows(), n = eq.A.cols();
    mtab = m + 1;
    ntab = m + n + 1;
    clear();                  // Clear the things.
    innerTable.resize(m + n); // Adding m auxillary variables.
    rhs = eq.b;
    reducedCosts.resize(n + m, 0.0); // Set columns n through n+m of the last row to 0.0

    for (const auto [key, val] : eq.A.data) {
      const auto [i, j] = key;
      innerTable[j].emplace_back(i, val);
      reducedCosts[j] -= val; // Set the first n columns of the last row
    }

    for (int i = 0; i < m; i++)
      innerTable[n + i].emplace_back(i, 1.0);

    negativeObjective = -std::reduce(rhs.begin(), rhs.end());

    std::for_each(innerTable.begin(), innerTable.end(), [](auto &elemVec) {
      std::sort(elemVec.begin(), elemVec.end(), CompElementIndices{});
    }

    );
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

  double inner(int j, int i)
  {
    const auto &vec = innerTable[i];
    auto it = std::lower_bound(vec.begin(), vec.end(), j, [](const Element &elem, int idx) {
      return elem.index < idx;
    });

    if (it != vec.end() && it->index == j)
      return it->value;

    return 0.0;
  }


  double &setReducedCost(int k) { return reducedCosts[k]; }
  void setNegativeObjective(double val) { negativeObjective = val; }

  int findNegativeCost();
  int findMinStep(int p);

  void pivoting(int p, int q);
  std::tuple<int, int, bool, bool> simplexTableau();
  std::pair<bool, bool> simplexAlgorithmTableau();
};

} // namespace dtwc::solver
