/*
 * SimplexFlatTable.cpp
 *
 * Sparse implementation of a Simplex table.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#include "SimplexFlatTable.hpp"
#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"
#include "EqualityConstraints.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include "../timing.hpp"

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
#include <execution>


namespace dtwc::solver {


int SimplexFlatTable::getRow(int col) const
{
  if (col < 0 || col >= (ntab - 1)) // Check if index is in the valid range
    throw std::runtime_error(fmt::format("The index of the variable ({}) must be between 0 and {}", col, ntab - 2));

  int rowIndex = -1; // Using -1 to represent None
  // So this is checking there is one and only one 1.0 element! -> #TODO change with something keeping book of basic variables in future.

  if (!isAround(reducedCosts[col], 0.0)) return rowIndex;

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

int SimplexFlatTable::findNegativeCost()
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

int SimplexFlatTable::findMinStep(int p)
{
  // Calculate the maximum step that can be done along the basic direction d[p]
  constexpr double tol = 1e-10;
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

std::tuple<int, int, bool, bool> SimplexFlatTable::simplexTableau()
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

void SimplexFlatTable::pivoting(int p, int q)
{
  // p column, q = row.
  // Make p,q element one and eliminate all other nonzero elements in that column by basic row operations.
  const double thepivot = inner(q, p);
  if (isAround(thepivot, 0.0))
    throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));


  auto oneTask = [this, thepivot, p, q](int i) {
    if (i == p) return; // Dont delete the pivot column yet.


    auto &colNow = innerTable[i];
    auto &pivotCol = innerTable[p];

    auto it_q = std::lower_bound(colNow.begin(), colNow.end(), q, [](const Element &elem, int idx) {
      return elem.index < idx;
    });

    if (it_q == colNow.end() || it_q->index != q) return; // We don't have that index.

    (it_q->value) /= thepivot; // Normalise by pivot;

    const auto q_val = (it_q->value);

    reducedCosts[i] -= reducedCosts[p] * q_val; // Remove from last row.

    auto N_now = colNow.size();
    auto N_piv = pivotCol.size();
    size_t i_now{}, i_piv{};


    while (i_now != N_now && i_piv != N_piv) {
      const auto [key_piv, val_piv] = pivotCol[i_piv];
      auto &[key_now, val_now] = colNow[i_now];

      if (key_now == q)
        ++i_piv;
      else if (key_now < key_piv)
        ++i_now;
      else if (key_now == key_piv) {
        val_now -= q_val * val_piv;
        ++i_now;
        ++i_piv;
      } else {
        colNow.emplace_back(key_piv, -q_val * val_piv);
        ++i_piv;
      }
    }

    while (i_piv != N_piv) {
      const auto [key_piv, val_piv] = pivotCol[i_piv];
      colNow.emplace_back(key_piv, -q_val * val_piv);
      ++i_piv;
    }

    std::sort(colNow.begin(), colNow.end(), CompElementValuesAndIndices{}); // Places zero values to end.

    while (!colNow.empty() && isAround(colNow.back().value)) // Remove zeroes from end.
      colNow.pop_back();
  };

  dtwc::run(oneTask, innerTable.size(), 1);

  rhs[q] /= thepivot;
  // We always have RHS.
  for (auto [key, val] : innerTable[p])
    if (key != q)
      rhs[key] -= val * rhs[q];

  negativeObjective -= rhs[q] * reducedCosts[p];

  // Deal with the pivot column now.
  innerTable[p].clear();
  innerTable[p].emplace_back(q, 1);
  reducedCosts[p] = 0;
}


std::pair<bool, bool> SimplexFlatTable::simplexAlgorithmTableau()
{
  size_t iter{};
  double duration_table{}, duration_pivoting{};
  while (true) {
    dtwc::Clock clk;
    auto [colPivot, rowPivot, optimal, bounded] = simplexTableau();
    duration_table += clk.duration();

    if (optimal) return { true, false };
    if (!bounded) return { false, true };

    clk = dtwc::Clock{};

    pivoting(colPivot, rowPivot);
    duration_pivoting += clk.duration();


    if (iter % 100 == 0) {
      std::cout << "Iteration " << iter << " is finished!" << std::endl;
      std::cout << "Duration table: ";
      dtwc::Clock::print_duration(std::cout, duration_table);
      std::cout << "Duration pivoting: ";
      dtwc::Clock::print_duration(std::cout, duration_pivoting);

      duration_table = 0;
      duration_pivoting = 0;

      size_t innerSize = 0;
      for (auto &map : innerTable) {
        innerSize += map.size();
      }

      std::cout << "Inner size: " << innerSize << '\n';
    }

    iter++;
  }
}

} // namespace dtwc::solver
