/*
 * SparseSimplex.cpp
 *
 * LP solution

 *  Created on: 26 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#include "SparseSimplex.hpp"
#include "SimplexTable.hpp"

#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"
#include "../Problem.hpp"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <tuple>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>
#include <set>
#include <limits>


namespace dtwc::solver {

void SparseSimplex::warmStartPhaseOne()
{
  if (Nb == 0 || Nc == 0) return; // it is not defined by clustering problem so we dont know what to do.
  dtwc::Clock clk{};
  std::cout << "Warmstart starting!" << std::endl;
  // Just activate first Nc variables as clusters and fill all to the first cluster.
  for (int i = 0; i < Nb; i++) {
    const int p = (i < Nc) ? (i * (Nb + 1)) : i;
    const int q = table.findMinStep(p);
    if (q != -1)
      table.pivoting(p, q);
  }

  // const int slack_begin = (Nb * Nb) + (Nb - 1);
  // const int slack_end = (Nb * Nb) + (Nb - 1) * Nb;

  // for (int p = slack_begin; p < slack_end; p++) { // Make slack variables one.
  //   const int q = table.findMinStep(p);
  //   if (q != -1)
  //     table.pivoting(p, q);

  //   std::cout << "Pivoting variable " << p << " of " << slack_end << std::endl;
  // }


  std::cout << "Warmstart ended in " << clk << std::endl;
};


std::tuple<bool, bool> SparseSimplex::simplex()
{
  // bool unbounded{}, optimal{};
  const int m = eq.A.rows(), n = eq.A.cols();

  // Ensure all elements of b are non-negative
  for (int i = 0; i < m; ++i)
    if (eq.b[i] < 0) {
      eq.b[i] = -eq.b[i];
      for (auto it = eq.A.row_begin(i), it_end = eq.A.row_end(i); it != it_end; ++it)
        it->second = -it->second;
    }
  std::cout << "Creating Phase-I table." << std::endl;
  table.createPhaseOneTableau(eq);
  warmStartPhaseOne();

  std::cout << "Running algorithm with Phase-I table." << std::endl;
  auto [optimal, unbounded] = table.simplexAlgorithmTableau();

  if (unbounded) return { true, false };
  if (table.getObjective() > epsilon) return { false, true }; // Infeasible problem

  std::vector<int> basicRows(m + n);
  std::set<int> basicIndices;
  std::set<int> tobeCleaned;

  for (int k = 0; k < m + n; ++k) {
    basicRows[k] = table.getRow(k);
    if (basicRows[k] != -1) {
      basicIndices.insert(k);
      if (k >= n) tobeCleaned.insert(k); // If k>= n are basic indices then clean them.
    }
  }

  if (!tobeCleaned.empty())
    throw "There are things to be cleaned!\n";

  // while (!tobeCleaned.empty()) {
  //   int auxiliaryColumn = *tobeCleaned.begin();
  //   tobeCleaned.erase(tobeCleaned.begin());

  //   int rowpivotIndex = basicRows[auxiliaryColumn];
  //   VectorXd rowpivot = phaseOneTableau.row(rowpivotIndex);

  //   std::vector<int> originalNonbasic;
  //   for (int i = 0; i < n; ++i)
  //     if (basicIndices.find(i) == basicIndices.end())
  //       originalNonbasic.push_back(i);

  //   int colpivot = -1;
  //   double maxVal = std::numeric_limits<double>::epsilon();

  //   for (int col : originalNonbasic)
  //     if (std::abs(rowpivot(col)) > maxVal) {
  //       maxVal = std::abs(rowpivot(col));
  //       colpivot = col;
  //     }

  //   if (colpivot != -1) {
  //     pivoting(phaseOneTableau, colpivot, rowpivotIndex);
  //     basicRows[colpivot] = rowpivotIndex;
  //     basicRows[auxiliaryColumn] = -1;
  //   } else {
  //     phaseOneTableau.conservativeResize(phaseOneTableau.rows() - 1, phaseOneTableau.cols());
  //     for (int k = 0; k < m + n; ++k)
  //       basicRows[k] = phaseOneTableau.getRow(k);
  //   }
  // }

  std::cout << "Creating Phase-II table." << std::endl;

  table.removeColumns(n, n + m);

  basicRows.resize(n);
  for (int k = 0; k < n; ++k)
    basicRows[k] = table.getRow(k);

  // Calculate last row
  std::vector<int> basicIndicesList, nonbasicIndicesList;
  for (int i = 0; i < n; ++i)
    if (basicRows[i] != -1)
      basicIndicesList.push_back(i);
    else
      nonbasicIndicesList.push_back(i);

  for (int k : nonbasicIndicesList) {
    double sumVal = 0.0;
    for (int j : basicIndicesList)
      sumVal += c[j] * table.inner(basicRows[j], k);

    table.setReducedCost(k) = c[k] - sumVal;
  }

  double lastRowSum = 0.0;
  for (int j : basicIndicesList)
    lastRowSum += c[j] * table.getRHS(basicRows[j]);

  table.setNegativeObjective(-lastRowSum);

  // Phase II
  std::cout << "Running algorithm with Phase-II table." << std::endl;
  auto [optimal2, unbounded2] = table.simplexAlgorithmTableau(); // it was startPhaseTwo
  return { unbounded2, false };
}

std::pair<std::vector<double>, double> SparseSimplex::getResults() const
{
  int n = table.cols() - 1; // finalTableau

  std::vector<double> solution(n);
  for (int k = 0; k < n; ++k)
    solution[k] = table.getValue(k);

  return { solution, table.getObjective() }; // Solution and optimal cost
}


void SparseSimplex::gomory()
{
  fmt::println("==================================");
  fmt::println("Problem with {} variables and {} constraints", eq.A.cols(), eq.A.rows());

  bool unbounded{}, infeasible{};
  std::tie(unbounded, infeasible) = simplex(); // Assuming simplex is defined

  if (unbounded) {
    fmt::println("Unbounded problem");
    nGomory = 0;
    return;
  }

  if (infeasible) {
    fmt::println("Infeasible problem");
    nGomory = 0;
    return;
  }

  auto [solution, copt] = getResults(); // Assuming getResults is defined
  if constexpr (settings::debug_Simplex)
    fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);


  int m_now = eq.A.rows();
  int n_now = eq.A.cols();

  nGomory = 0;
  for (int i = 0; i < (table.rows() - 1); i++) {
    double rhs_now = table.getRHS(i);
    if (isFractional(rhs_now)) // Scan last column
    {
      eq.A(m_now + nGomory, n_now + nGomory) = 1.0; // Slack variable for new row.

      for (auto j = 0; j < eq.A.cols(); j++) {
        auto val = std::floor(table.inner(i, j)); // gamma

        if (!isAround(val))
          eq.A(m_now + nGomory, j) = val;
      }

      eq.b.emplace_back(std::floor(rhs_now));
      nGomory++; // One more gomory cut.
    }
  }

  eq.A.expand(m_now + nGomory, n_now + nGomory);

  c.resize(n_now + nGomory); // add zeros
  fmt::println("Number of Gomory cuts: {}", nGomory);
}

SparseSimplex::SparseSimplex(Problem &prob)
{
  Nb = prob.data.size();
  Nc = prob.cluster_size();

  eq = defaultConstraints(Nb, Nc);

  c.resize(Nb * Nb);

  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      c[i + j * Nb] = prob.distByInd_scaled(i, j);
}


} // namespace dtwc::solver
