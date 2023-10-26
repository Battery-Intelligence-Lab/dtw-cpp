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

void pivoting(SimplexTable &table, int p, int q)
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

std::tuple<int, int, bool, bool> simplexTableau(const SimplexTable &table)
{
  const int mtab = table.rows(), ntab = table.cols();
  const int m = mtab - 1, n = ntab - 1;

  // Find the first negative cost, if there are none, then table is optimal.
  int p = -1;
  for (int i = 0; i < n; i++) // Reduced cost.
    if (table(Eigen::last, i) < -epsilon) {
      p = i;
      break;
    }

  if (p == -1) // The table is optimal
    return std::make_tuple(-1, -1, true, true);

  double vkTolerance = 1e-10;
  // Calculate the maximum step that can be done along the basic direction d[p]
  double minStep = std::numeric_limits<double>::infinity();
  int q{ -1 }; // row of min step. -1 means unbounded.

  for (int k = 0; k < m; k++)
    if (table(k, p) > vkTolerance) {
      const auto step = table(k, Eigen::last) / table(k, p); // table(k,p) = minus(d)
      if (step < minStep) {
        minStep = step;
        q = k;
      }
    }

  if (q == -1) // The table is unbounded
    return std::make_tuple(-1, -1, false, false);

  return std::make_tuple(p, q, false, true);
}

std::pair<bool, bool> simplexAlgorithmTableau(SimplexTable &table)
{
  while (true) {
    auto [colPivot, rowPivot, optimal, bounded] = simplexTableau(table);

    if (optimal) return { true, false };
    if (!bounded) return { false, true };

    pivoting(table, colPivot, rowPivot);
  }
}

SimplexTable createTableau(const SimplexTable &A, const VectorXd &b, VectorXd &c)
{
  const int m = A.rows(), n = A.cols();
  SimplexTable table(m + 1, n + m + 1);

  table.block(0, 0, m, n) = A;
  table.block(0, n, m, m) = SimplexTable::Identity(m, m);
  table.block(0, n + m, m, 1) = b;

  // Set the first n columns of the last row
  for (int k = 0; k < n; ++k)
    table.row(m)(k) = -table.block(0, k, m, 1).sum();

  // Set columns n through n+m of the last row to 0.0
  table.block(m, n, 1, m).setZero();

  // Set the last element of the last row
  table(m, n + m) = -table.block(0, n + m, m, 1).sum();

  return table;
}

std::tuple<SimplexTable, bool, bool> simplex(SimplexTable &A, VectorXd &b, VectorXd &c)
{
  // bool unbounded{}, optimal{};
  const int m = A.rows(), n = A.cols();

  if (b.rows() != m)
    throw std::runtime_error(fmt::format("Incompatible sizes: A is {}x{}, b is of length {}, and should be {}", m, n, b.rows(), m));

  if (c.size() != n)
    throw std::runtime_error(fmt::format("Incompatible sizes: A is {}x{}, c is of length {}, and should be {}", m, n, c.size(), n));

  // Ensure all elements of b are non-negative
  for (int i = 0; i < m; ++i)
    if (b(i) < 0) {
      A.row(i) = -A.row(i);
      b(i) = -b(i);
    }

  SimplexTable phaseOneTableau = createTableau(A, b, c);

  auto [optimal, unbounded] = simplexAlgorithmTableau(phaseOneTableau);

  if (unbounded) return { phaseOneTableau, true, false };

  const auto isInfeasible = phaseOneTableau(Eigen::last, Eigen::last) < -epsilon;
  if (isInfeasible) return { phaseOneTableau, false, true }; // Infeasible problem

  std::vector<int> basicRows(m + n);
  std::set<int> basicIndices;
  std::set<int> tobeCleaned;

  for (int k = 0; k < m + n; ++k) {
    basicRows[k] = phaseOneTableau.getRow(k);
    if (basicRows[k] != -1) {
      basicIndices.insert(k);
      if (k >= n) tobeCleaned.insert(k); // If k>= n are basic indices then clean them.
    }
  }

  int ZeroCount = phaseOneTableau.unaryExpr([](double elem) { return isAround(elem, 0.0); }).count();

  std::cout << "Table:\n"
            << phaseOneTableau.rows() << 'x' << phaseOneTableau.cols() << " : " << ZeroCount << '\n';

  while (!tobeCleaned.empty()) {
    int auxiliaryColumn = *tobeCleaned.begin();
    tobeCleaned.erase(tobeCleaned.begin());

    int rowpivotIndex = basicRows[auxiliaryColumn];
    VectorXd rowpivot = phaseOneTableau.row(rowpivotIndex);

    std::vector<int> originalNonbasic;
    for (int i = 0; i < n; ++i)
      if (basicIndices.find(i) == basicIndices.end())
        originalNonbasic.push_back(i);

    int colpivot = -1;
    double maxVal = std::numeric_limits<double>::epsilon();

    for (int col : originalNonbasic)
      if (std::abs(rowpivot(col)) > maxVal) {
        maxVal = std::abs(rowpivot(col));
        colpivot = col;
      }

    if (colpivot != -1) {
      pivoting(phaseOneTableau, colpivot, rowpivotIndex);
      basicRows[colpivot] = rowpivotIndex;
      basicRows[auxiliaryColumn] = -1;
    } else {
      phaseOneTableau.conservativeResize(phaseOneTableau.rows() - 1, phaseOneTableau.cols());
      for (int k = 0; k < m + n; ++k)
        basicRows[k] = phaseOneTableau.getRow(k);
    }
  }

  SimplexTable leftPart = phaseOneTableau.leftCols(n);
  SimplexTable rightPart = phaseOneTableau.rightCols(phaseOneTableau.cols() - n - m);

  SimplexTable phaseTwoTableau(leftPart.rows(), leftPart.cols() + rightPart.cols());
  phaseTwoTableau << leftPart, rightPart;

  // Reset basicRows for startPhaseTwo -> phaseTwoTableau
  basicRows.resize(n);
  for (int k = 0; k < n; ++k)
    basicRows[k] = phaseTwoTableau.getRow(k);

  // Calculate last row
  std::vector<int> basicIndicesList, nonbasicIndicesList;
  for (int i = 0; i < n; ++i) {
    if (basicRows[i] != -1)
      basicIndicesList.push_back(i);
    else
      nonbasicIndicesList.push_back(i);
  }

  for (int k : nonbasicIndicesList) {
    double sumVal = 0.0;
    for (int j : basicIndicesList) {
      //  std::cout << "Value: " << phaseTwoTableau(basicRows[j], k) << '\n';
      sumVal += c(j) * phaseTwoTableau(basicRows[j], k);
    }

    phaseTwoTableau(phaseTwoTableau.rows() - 1, k) = c(k) - sumVal;
  }

  double lastRowSum = 0.0;
  for (int j : basicIndicesList)
    lastRowSum += c(j) * phaseTwoTableau(basicRows[j], phaseTwoTableau.cols() - 1);

  phaseTwoTableau(phaseTwoTableau.rows() - 1, phaseTwoTableau.cols() - 1) = -lastRowSum;

  // Phase II
  auto [optimal2, unbounded2] = simplexAlgorithmTableau(phaseTwoTableau); // it was startPhaseTwo
  return { phaseTwoTableau, unbounded2, false };
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
  fmt::println("Problem with {} variables and {} constraints", eq.cols(), eq.rows());

  bool unbounded{}, infeasible{};
  std::tie(table, unbounded, infeasible) = simplex(A, b, c); // Assuming simplex is defined

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

  std::vector<int> fractionalMask;

  for (int i = 0; i < (table.rows() - 1); i++)
    if (isFractional(table.getRHS(i))) // Scan last column
      fractionalMask.push_back(i);

  nGomory = fractionalMask.size();
  fmt::println("Number of Gomory cuts: {}", nGomory);

  SimplexTable gamma = table(fractionalMask, Eigen::seqN(0, table.cols() - 1)).array().floor();
  VectorXd bplus = table(fractionalMask, table.cols() - 1).array().floor();

  SimplexTable newA(A.rows() + nGomory, A.cols() + nGomory);
  VectorXd newb(A.rows() + nGomory);

  newA << A, SimplexTable::Zero(A.rows(), nGomory),
    gamma, SimplexTable::Identity(nGomory, nGomory);

  newb << b, bplus;


  A = newA;
  b = newb;
  c.resize(A.cols() + nGomory); // add zeros
}

SparseSimplex::SparseSimplex(Problem &prob)
{
  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();

  eq = defaultConstraints(Nb, Nc);

  c.resize(Nb * Nb);

  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      c[i + j * Nb] = prob.distByInd_scaled(i, j);
}

} // namespace dtwc::solver
