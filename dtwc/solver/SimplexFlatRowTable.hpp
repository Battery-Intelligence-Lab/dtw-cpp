/*
 * SimplexFlatRowTable.hpp
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


class SimplexFlatRowTable
{
  // Table is   mtab x ntab
  // Inner table is m x n
  std::vector<std::vector<Element>> innerTable; // Each is a row
  std::vector<double> reducedCosts, rhs;
  double negativeObjective{};

  int mtab{}, ntab{};

public:
  SimplexFlatRowTable() = default;
  SimplexFlatRowTable(int mtab_, int ntab_) : mtab{ mtab_ }, ntab{ ntab_ }, innerTable(mtab - 1),
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
    for (auto &row : innerTable) {
      auto itBegin = std::lower_bound(row.begin(), row.end(), a, [](const Element &elem, int idx) {
        return elem.index < idx;
      });
      auto itEnd = std::upper_bound(itBegin, row.end(), b, [](int idx, const Element &elem) {
        return idx < elem.index;
      });

      row.erase(itBegin, itEnd);
    }

    reducedCosts.erase(reducedCosts.begin() + a, reducedCosts.begin() + b);
    ntab -= (b - a);
  }


  void createPhaseOneTableau(const EqualityConstraints &eq)
  {
    // Table size (m + 1, m + n + 1)
    // table.block(0, 0, m, n) = A;
    // table.block(0, n, m, m) = SimplexFlatRowTable::Identity(m, m);
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
      innerTable[i].emplace_back(j, val);
      reducedCosts[j] -= val; // Set the first n columns of the last row
    }

    for (int i = 0; i < m; i++)
      innerTable[i].emplace_back(n + i, 1.0);

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

  double inner(int i, int j)
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

  std::tuple<int, int, bool, bool> simplexTableau();

  void pivoting(int p, int q);

  std::pair<bool, bool> simplexAlgorithmTableau();
};

} // namespace dtwc::solver