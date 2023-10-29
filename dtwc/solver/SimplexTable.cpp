/*
 * SimplexTable.cpp
 *
 * Sparse implementation of a Simplex table.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#include "SimplexTable.hpp"
#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"
#include "EqualityConstraints.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>

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


int SimplexTable::getRow(int col) const
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


int SimplexTable::findNegativeCost()
{
  // Returns -1 if table is optimal otherwise first negative reduced cost.
  // Find the first negative cost, if there are none, then table is optimal.
  int p = -1;
  for (int i = 0; i < reducedCosts.size(); i++) // Reduced cost.
    if (reducedCosts[i] < -epsilon) {
      p = i;
      break;
    }

  return p;
}

int SimplexTable::findMinStep(int p)
{
  constexpr double tol = 1e-10;
  // Calculate the maximum step that can be done along the basic direction d[p]
  double minStep = std::numeric_limits<double>::infinity();
  int q{ -1 }; // row of min step. -1 means unbounded.

  for (auto [k, minus_d] : innerTable[p])
    if (minus_d > tol) {
      const auto step = rhs[k] / minus_d;
      if (step < minStep) {
        minStep = step;
        q = k;
      }
    }

  return q;
}

std::tuple<int, int, bool, bool> SimplexTable::simplexTableau()
{
  // Find the first negative cost, if there are none, then table is optimal.
  const int p = findNegativeCost();

  if (p == -1) // The table is optimal
    return std::make_tuple(-1, -1, true, true);

  const int q = findMinStep(p);

  if (q == -1) // The table is unbounded
    return std::make_tuple(-1, -1, false, false);

  return std::make_tuple(p, q, false, true);
}


void SimplexTable::pivoting(int p, int q)
{
  // p column, q = row.
  // Make p,q element one and eliminate all other nonzero elements in that column by basic row operations.
  const double thepivot = innerTable[p][q];
  if (isAround(thepivot, 0.0))
    throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));

  auto &pivotCol = innerTable[p]; //

  for (int i = 0; i < innerTable.size(); i++) {
    if (i == p) continue; // Dont delete the pivot column yet.

    auto currentCol_q = innerTable[i].find(q);
    if (currentCol_q != innerTable[i].end()) // We have a row like that!
    {
      (currentCol_q->second) /= thepivot; // Normalise by pivot;

      reducedCosts[i] -= reducedCosts[p] * (currentCol_q->second); // Remove from last row.

      for (auto itr = innerTable[p].begin(); itr != innerTable[p].end(); ++itr)
        if (itr->first != q) {
          auto it_now = innerTable[i].find(itr->first);
          if (it_now != innerTable[i].end()) // If that row exists only then subtract otherwise equate.
            (it_now->second) -= (itr->second) * (currentCol_q->second);
          else
            innerTable[i][itr->first] = -(itr->second) * (currentCol_q->second);
        }

      std::erase_if(innerTable[i], [](const auto &item) {
        auto const &[key, value] = item;
        return isAround(value, 0.0); // Remove zero elements to make it compact.
      });
    }
  }

  rhs[q] /= thepivot;
  // We always have RHS.
  for (auto [key, val] : innerTable[p])
    if (key != q)
      rhs[key] -= val * rhs[q];

  negativeObjective -= rhs[q] * reducedCosts[p];

  // Deal with the pivot column now.
  innerTable[p].clear();
  innerTable[p][q] = 1;
  reducedCosts[p] = 0;
}


std::pair<bool, bool> SimplexTable::simplexAlgorithmTableau()
{
  while (true) {
    auto [colPivot, rowPivot, optimal, bounded] = simplexTableau();

    if (optimal) return { true, false };
    if (!bounded) return { false, true };

    pivoting(colPivot, rowPivot);
  }
}

} // namespace dtwc::solver
