/*
 * SimplexRowTable.cpp
 *
 * Sparse implementation of a Simplex table.

 *  Created on: 29 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#include "SimplexRowTable.hpp"
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


int SimplexRowTable::getRow(int col) const
{
  if (col < 0 || col >= (ntab - 1)) // Check if index is in the valid range
    throw std::runtime_error(fmt::format("The index of the variable ({}) must be between 0 and {}", col, ntab - 2));

  int rowIndex = -1; // Using -1 to represent None
  // So this is checking there is one and only one 1.0 element! -> #TODO change with something keeping book of basic variables in future.

  if (!isAround(reducedCosts[col], 0.0)) return -1;

  for (int row = 0; row < innerTable.size(); row++) {
    auto it = innerTable[row].find(col);
    if (it != innerTable[row].end()) {
      if (!isAround(it->second, 0)) // The entry is non zero
      {
        if (rowIndex == -1 && isAround(it->second, 1.0)) // The entry is one, and the index has not been found yet.
          rowIndex = row;
        else
          return -1;
      }
    }
  }

  return rowIndex;
}

int SimplexRowTable::findNegativeCost()
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

int SimplexRowTable::findMinStep(int p)
{

  // Calculate the maximum step that can be done along the basic direction d[p]

  constexpr double tol = 1e-10;
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
    [p, tol, this](const auto &row) -> StepIndexPair {
      auto it = row.find(p);
      if (it != row.end()) {
        auto [k, minus_d] = *it;
        if (minus_d > tol) {
          int rowIndex = std::distance(innerTable.begin(), std::find(innerTable.begin(), innerTable.end(), row));
          return { rhs[rowIndex] / minus_d, rowIndex };
        }
      }
      return { std::numeric_limits<double>::infinity(), -1 };
    });

  return result.second;
}

std::tuple<int, int, bool, bool> SimplexRowTable::simplexTableau()
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

void SimplexRowTable::pivoting(int p, int q)
{
  // p column, q = row.
  // Make p,q element one and eliminate all other nonzero elements in that column by basic row operations.
  const double thepivot = innerTable[q][p];
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

  auto oneTask = [this, thepivot, p, q, pivotRow](int i) {
    if (q == i) return; // Do not process pivot row.

    auto &rowNow = innerTable[i];

    auto it_p = rowNow.find(p);
    if (it_p == rowNow.end()) return;

    const auto p_val = it_p->second;
    auto it_now = rowNow.begin();
    auto it_pivot = pivotRow.begin();

    rhs[i] -= p_val * rhs[q];

    while (it_now != rowNow.end() && it_pivot != pivotRow.end()) {
      const auto [key_piv, val_piv] = (*it_pivot);
      auto &[key_now, val_now] = (*it_now);

      if (key_now < key_piv)
        ++it_now;
      else if (key_now == key_piv) {
        val_now -= p_val * val_piv;
        ++it_now;
        ++it_pivot;
      } else {
        rowNow.insert(it_now, { key_piv, -(p_val * val_piv) });
        ++it_pivot;
      }
    }

    while (it_pivot != pivotRow.end()) {
      rowNow[it_pivot->first] = -p_val * (it_pivot->second);
      ++it_pivot;
    }


    std::erase_if(rowNow, [](const auto &item) {
      auto const &[key, value] = item;
      return isAround(value, 0.0); // Remove zero elements to make it compact.
    });
  };

  dtwc::run(oneTask, innerTable.size());
}


std::pair<bool, bool> SimplexRowTable::simplexAlgorithmTableau()
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
