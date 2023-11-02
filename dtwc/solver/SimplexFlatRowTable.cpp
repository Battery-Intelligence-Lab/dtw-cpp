/*
 * SimplexFlatRowTable.cpp
 *
 * Sparse implementation of a Simplex table.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#include "SimplexFlatRowTable.hpp"
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


int SimplexFlatRowTable::getRow(int col) const
{
  if (col < 0 || col >= (ntab - 1)) // Check if index is in the valid range
    throw std::runtime_error(fmt::format("The index of the variable ({}) must be between 0 and {}", col, ntab - 2));

  int rowIndex = -1; // Using -1 to represent None
  // So this is checking there is one and only one 1.0 element! -> #TODO change with something keeping book of basic variables in future.

  if (!isAround(reducedCosts[col], 0.0)) return rowIndex;

  for (int row = 0; row < innerTable.size(); row++) {
    for (const auto [key, val] : innerTable[row])
      if (key == col) {                           // The entry is non-zero
        if (rowIndex == -1 && isAround(val, 1.0)) // The entry is one, and the index has not been found yet.
          rowIndex = row;
        else
          return -1;
      } else if (key > col)
        break;
  }

  return rowIndex;
}

int SimplexFlatRowTable::findMostNegativeCost() const
{
  // Returns -1 if table is optimal, otherwise the index of the most negative reduced cost.
  int p = -1;
  double mostNegativeValue = -epsilon; // start with a small negative threshold value

  for (int i = 0; i < reducedCosts.size(); i++) {
    if (reducedCosts[i] < mostNegativeValue) {
      mostNegativeValue = reducedCosts[i];
      p = i;
    }
  }

  return p;
}

int SimplexFlatRowTable::findNegativeCost() const
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

int SimplexFlatRowTable::findMinStep(int p) const
{

  // Calculate the maximum step that can be done along the basic direction d[p]
  using StepIndexPair = std::pair<double, int>;
  StepIndexPair initial = { std::numeric_limits<double>::infinity(), -1 }; // row of min step. -1 means unbounded.

  auto result = std::transform_reduce(
    std::execution::par_unseq, // Use parallel execution policy
    innerTable.begin(),
    innerTable.end(),
    initial,
    [](const StepIndexPair &a, const StepIndexPair &b) -> StepIndexPair {
      return (a.first < b.first) ? a : b;
    },
    [p, this](const auto &row) -> StepIndexPair {
      double minus_d = -1;
      for (const auto [key, val] : row)
        if (key == p) {
          minus_d = val;
          break;
        } else if (key > p)
          break;

      if (minus_d > epsilon) {
        int rowIndex = (&row - &innerTable[0]);
        return { rhs[rowIndex] / minus_d, rowIndex };
      }

      return { std::numeric_limits<double>::infinity(), -1 };
    });

  return result.second;
}

std::tuple<int, int, bool, bool> SimplexFlatRowTable::simplexTableau()
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

void SimplexFlatRowTable::pivoting(int p, int q)
{
  // p column, q = row.
  // Make p,q element one and eliminate all other nonzero elements in that column by basic row operations.
  const double thepivot = inner(q, p);
  if (isAround(thepivot, 0.0))
    throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));

  auto &pivotRow = innerTable[q];

  const auto reducedcost_p = reducedCosts[p];
  rhs[q] /= thepivot;
  negativeObjective -= rhs[q] * reducedcost_p;

  for (auto &[key, val] : pivotRow) // Make the pivot row normalised.
  {
    val /= thepivot;
    reducedCosts[key] -= reducedcost_p * val; // Remove from last row.
  }

  auto oneTask = [this, thepivot, p, q, &pivotRow](auto &rowNow) {
    if (&rowNow == &pivotRow) return; // Do not process pivot row.

    double p_val;
    for (const auto [key, val] : rowNow)
      if (key == p) {
        p_val = val;
        break;
      } else if (key > p)
        return;

    const int rowIndex = &rowNow - &innerTable[0];

    rhs[rowIndex] -= p_val * rhs[q];

    auto N_now = rowNow.size();
    auto N_piv = pivotRow.size();
    size_t i_now{}, i_piv{};

    while (i_now != N_now && i_piv != N_piv) {
      const auto [key_piv, val_piv] = pivotRow[i_piv];
      auto &[key_now, val_now] = rowNow[i_now];

      if (key_now < key_piv)
        ++i_now;
      else if (key_now == key_piv) {
        val_now -= p_val * val_piv;
        ++i_now;
        ++i_piv;
      } else {
        rowNow.emplace_back(key_piv, -p_val * val_piv);
        ++i_piv;
      }
    }

    while (i_piv != N_piv) {
      const auto [key_piv, val_piv] = pivotRow[i_piv];
      rowNow.emplace_back(key_piv, -p_val * val_piv);
      ++i_piv;
    }

    std::sort(rowNow.begin(), rowNow.end(), CompElementValuesAndIndices{}); // Places zero values to end.

    while (!rowNow.empty() && isAround(rowNow.back().value)) // Remove zeroes from end.
      rowNow.pop_back();
  };

  std::for_each(std::execution::par_unseq, innerTable.begin(), innerTable.end(), oneTask);
}


std::pair<bool, bool> SimplexFlatRowTable::simplexAlgorithmTableau()
{
  size_t iter{};
  double duration_table{}, duration_pivoting{};
 // static std::vector<int> colNumbers;
   // colNumbers.reserve(rhs.size());
    

  while (true) {
    dtwc::Clock clk;
    //colNumbers.clear();
    auto [colPivot, rowPivot, optimal, bounded] = simplexTableau();
    duration_table += clk.duration();

    if (optimal) return { true, false };
    if (!bounded) return { false, true };

    clk = dtwc::Clock{};

    pivoting(colPivot, rowPivot);
    duration_pivoting += clk.duration();


    if (iter % 100 == 0) {
      std::cout << "Iteration " << iter << " is finished!\n";
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

      std::cout << "Inner size per row: " << (double)innerSize / innerTable.size() 
                << " per " << innerTable.size() << '\n';
    }

    iter++;
  }
}

} // namespace dtwc::solver
