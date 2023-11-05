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

  // So this is checking there is one and only one 1.0 element! -> #TODO change with something keeping book of basic variables in future.
  const bool isBasic = isAround(reducedCosts[col], 0.0) && innerTable[col].size() == 1 && isAround(innerTable[col][0].value, 1);
  return isBasic ? innerTable[col][0].index : -1; // Using -1 to represent None
}

int SimplexFlatTable::findNegativeCost() const
{
  // Returns -1 if table is optimal otherwise first negative reduced cost.
  // Find the first negative cost, if there are none, then table is optimal.
  int p = -1;
  for (int i = 0; i < reducedCosts.size(); i++) // Reduced cost.
    if (reducedCosts[i] < -epsilon / 2) {
      p = i;
      break;
    }

  return p;
}

int SimplexFlatTable::findMostNegativeCost() const
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

int SimplexFlatTable::findMinStep(int p) const
{
  // Calculate the maximum step that can be done along the basic direction d[p]
  double minStep = std::numeric_limits<double>::infinity();
  int q{ -1 }; // row of min step. -1 means unbounded.

  for (auto [k, minus_d] : innerTable[p])
    if (minus_d > epsilon) {
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
  if (isAround(thepivot, 0.0)) {

    std::cout << "The pivot is too close to zero for (" << q << "," << p << ')' << std::endl;
    for (auto elem : innerTable[p])
      std::cout << "(" << elem.index << ',' << elem.value << ")\n";

    std::cout << std::endl;

    throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));
  }

  auto &pivotCol = innerTable[p];

  auto oneTask = [this, thepivot, p, q, &pivotCol](auto &colNow) {
    // if (!std::is_sorted(colNow.begin(), colNow.end(), CompElementIndices{})) {
    //   std::cout << "Series is not sorted" << std::endl;
    //   throw "Given column is not sorted!";
    // }

    if (&colNow == &pivotCol) return; // Do not process pivot row.

    const int colIndex = &colNow - &innerTable[0];

    auto it_q = std::lower_bound(colNow.begin(), colNow.end(), q, [](const Element &elem, int idx) {
      return elem.index < idx;
    });

    if (it_q == colNow.end() || it_q->index != q) return; // We don't have that index.

    (it_q->value) /= thepivot; // Normalise by pivot;

    const auto q_val = (it_q->value);

    reducedCosts[colIndex] -= reducedCosts[p] * q_val; // Remove from last row.

    if (reducedCosts[colIndex] < -1e20) {
      std::cout << "Reduced cost is too low!  " << reducedCosts[colIndex] << std::endl;
      throw "Reduced cost is too low!!!!\n";
    }

    auto N_now = colNow.size();
    auto N_piv = pivotCol.size();

    thread_local std::vector<Element> temporary;
    temporary.clear();
    size_t i_now{}, i_piv{};

    while (i_now != N_now && i_piv != N_piv) {
      const auto &eNow = colNow[i_now];
      const auto &ePiv = pivotCol[i_piv];

      if (ePiv.index == q)
        ++i_piv;
      else if (eNow.index < ePiv.index) {
        if (!isAround(eNow.value))
          temporary.push_back(eNow);
        ++i_now;
      } else if (eNow.index == ePiv.index) {
        const auto new_value = eNow.value - q_val * ePiv.value;
        if (!isAround(new_value))
          temporary.emplace_back(Element{ eNow.index, new_value });
        ++i_now;
        ++i_piv;
      } else {
        const auto new_value = -q_val * ePiv.value;
        if (!isAround(new_value))
          temporary.emplace_back(ePiv.index, new_value);
        ++i_piv;
      }
    }

    while (i_now != N_now)
      if (!isAround(colNow[i_now].value))
        temporary.push_back(colNow[i_now++]);

    while (i_piv != N_piv) {
      const auto &ePiv = pivotCol[i_piv++];
      const auto new_value = -q_val * ePiv.value;
      if (!isAround(new_value))
        temporary.emplace_back(ePiv.index, new_value);
    }

    std::swap(temporary, colNow);
  };

  std::for_each(std::execution::par_unseq, innerTable.begin(), innerTable.end(), oneTask);

  rhs[q] /= thepivot;
  // We always have RHS.
  for (auto [key, val] : innerTable[p])
    if (key != q) {
      const auto new_value = rhs[key] - val * rhs[q];
      rhs[key] = new_value; // isAround(new_value) ? 0.0 : new_value;
    }

  negativeObjective -= rhs[q] * reducedCosts[p];

  // Deal with the pivot column now.
  pivotCol.clear();
  pivotCol.emplace_back(q, 1);
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


    if (iter % 500 == 0) {
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

      std::cout << "Inner size per row: " << (double)innerSize / innerTable.size()
                << " per " << innerTable.size() << '\n';


      std::cout << "Minimum reduced cost: " << *std::min_element(reducedCosts.begin(), reducedCosts.end()) << '\n';
    }

    iter++;
  }
}

} // namespace dtwc::solver
