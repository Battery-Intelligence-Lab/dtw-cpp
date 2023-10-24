/*
 * SimplexSolver.hpp
 *
 * LP solution

 *  Created on: 22 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "Simplex.hpp"

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
#include <Eigen/Dense>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>
#include <set>
#include <limits>


namespace dtwc::solver {

using Eigen::MatrixXd, Eigen::VectorXd;

int getRow(const MatrixXd &tableau, int index)
{
  if (index < 0 || index >= (tableau.cols() - 1)) // Check if index is in the valid range
    throw std::runtime_error(fmt::format("The index of the variable ({}) must be between 0 and {}", index, tableau.cols() - 2));

  int rowIndex = -1; // Using -1 to represent None
  // So this is checking there is one and only one 1.0 element! -> #TODO change with something keeping book of basic variables in future.
  for (int j = 0; j < tableau.rows(); ++j)
    if (!isAround(tableau(j, index), 0)) // The entry is non zero
    {
      if (rowIndex == -1 && isAround(tableau(j, index), 1.0)) // The entry is one, and the index has not been found yet.
        rowIndex = j;
      else
        return -1;
    }

  return rowIndex;
}

void pivoting(MatrixXd &tableau, int p, int q)
{
  // Make p,q element one and eliminate all other nonzero elements in that column by basic row operations.
  const double thepivot = tableau(q, p);
  if (isAround(thepivot, 0.0))
    throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));

  tableau.row(q) /= thepivot; // Make (p,q) one.

  for (int i = 0; i < tableau.rows(); ++i)
    if (i != q)
      tableau.row(i) -= tableau(i, p) * tableau.row(q);
}

std::tuple<int, int, bool, bool> simplexTableau(const MatrixXd &tableau)
{
  int mtab = tableau.rows(), ntab = tableau.cols();
  int m = mtab - 1, n = ntab - 1;

  VectorXd reducedCost = tableau.row(mtab - 1).head(ntab - 1);

  // Find the first negative cost, if there are none, then table is optimal.
  int p = -1;
  for (int i = 0; i < reducedCost.size(); i++)
    if (reducedCost(i) < -epsilon) {
      p = i;
      break;
    }

  if (p == -1) // The tableau is optimal
    return std::make_tuple(-1, -1, true, true);

  double vkTolerance = 1e-10;
  // Calculate the maximum step that can be done along the basic direction d[p]
  double minStep = std::numeric_limits<double>::infinity();
  int q{ -1 }; // row of min step. -1 means unbounded.

  for (int k = 0; k < m; k++)
    if (tableau(k, p) > vkTolerance) {
      const auto step = tableau(k, Eigen::last) / tableau(k, p); // tableau(k,p) = minus(d)
      if (step < minStep) {
        minStep = step;
        q = k;
      }
    }

  if (q == -1) // The tableau is unbounded
    return std::make_tuple(-1, -1, false, false);

  return std::make_tuple(p, q, false, true);
}

std::tuple<MatrixXd, bool, bool> simplexAlgorithmTableau(const MatrixXd &input_tableau)
{
  MatrixXd tableau = input_tableau;

  while (true) {
    auto [colPivot, rowPivot, optimal, bounded] = simplexTableau(tableau);

    if (optimal) return { tableau, true, false };
    if (!bounded) return { tableau, false, true };

    pivoting(tableau, colPivot, rowPivot);
  }
}

MatrixXd createTableau(const MatrixXd &A, const VectorXd &b, VectorXd &c)
{
  const int m = A.rows(), n = A.cols();
  MatrixXd tableau(m + 1, n + m + 1);

  tableau.block(0, 0, m, n) = A;
  tableau.block(0, n, m, m) = MatrixXd::Identity(m, m);
  tableau.block(0, n + m, m, 1) = b;

  // Set the first n columns of the last row
  for (int k = 0; k < n; ++k)
    tableau.row(m)(k) = -tableau.block(0, k, m, 1).sum();

  // Set columns n through n+m of the last row to 0.0
  tableau.block(m, n, 1, m).setZero();

  // Set the last element of the last row
  tableau(m, n + m) = -tableau.block(0, n + m, m, 1).sum();

  return tableau;
}

std::tuple<MatrixXd, bool, bool> simplex(MatrixXd &A, VectorXd &b, VectorXd &c)
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

  MatrixXd tableau = createTableau(A, b, c);

  auto [phaseOneTableau, optimal, unbounded] = simplexAlgorithmTableau(tableau);

  if (unbounded) return { tableau, true, false };

  const auto isInfeasible = phaseOneTableau(Eigen::last, Eigen::last) < -epsilon;
  if (isInfeasible) return { phaseOneTableau, false, true }; // Infeasible problem

  std::vector<int> basicRows(m + n);
  for (int k = 0; k < m + n; ++k)
    basicRows[k] = getRow(phaseOneTableau, k);

  std::set<int> basicIndices;
  for (int i = 0; i < basicRows.size(); ++i)
    if (basicRows[i] != -1)
      basicIndices.insert(i);

  std::set<int> tobeCleaned;
  for (int i = n; i < n + m; ++i)
    if (basicIndices.find(i) != basicIndices.end())
      tobeCleaned.insert(i);


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
      for (int k = 0; k < m + n; ++k) {
        basicRows[k] = getRow(phaseOneTableau, k);
      }
    }
  }

  MatrixXd leftPart = phaseOneTableau.leftCols(n);
  MatrixXd rightPart = phaseOneTableau.rightCols(phaseOneTableau.cols() - n - m);

  MatrixXd startPhaseTwo(leftPart.rows(), leftPart.cols() + rightPart.cols());
  startPhaseTwo << leftPart, rightPart;

  // Reset basicRows for startPhaseTwo
  basicRows.resize(n);
  for (int k = 0; k < n; ++k)
    basicRows[k] = getRow(startPhaseTwo, k);

  // Calculate last row
  std::vector<int> basicIndicesList, nonbasicIndicesList;
  for (int i = 0; i < basicRows.size(); ++i) {
    if (basicRows[i] != -1)
      basicIndicesList.push_back(i);
    else
      nonbasicIndicesList.push_back(i);
  }

  for (int k : nonbasicIndicesList) {
    double sumVal = 0.0;
    for (int j : basicIndicesList)
      sumVal += c(j) * startPhaseTwo(basicRows[j], k);

    startPhaseTwo(startPhaseTwo.rows() - 1, k) = c(k) - sumVal;
  }

  double lastRowSum = 0.0;
  for (int j : basicIndicesList)
    lastRowSum += c(j) * startPhaseTwo(basicRows[j], startPhaseTwo.cols() - 1);

  startPhaseTwo(startPhaseTwo.rows() - 1, startPhaseTwo.cols() - 1) = -lastRowSum;

  // Phase II
  auto [phaseTwoTableau, optimal2, unbounded2] = simplexAlgorithmTableau(startPhaseTwo);
  return { phaseTwoTableau, unbounded2, false };
}

std::pair<std::vector<double>, double> Simplex::getResults() const
{
  int n = tableau.cols() - 1; // finalTableau

  std::vector<double> solution(n);
  for (int k = 0; k < n; ++k) {
    const auto basicRowNo = getRow(tableau, k);
    solution[k] = (basicRowNo != -1) ? tableau(basicRowNo, Eigen::last) : 0.0;
  }

  return { solution, -tableau(Eigen::last, Eigen::last) }; // Solution and optimal cost
}


void Simplex::gomory()
{
  fmt::println("==================================");
  fmt::println("Problem with {} variables and {} constraints", A.cols(), A.rows());

  bool unbounded{}, infeasible{};
  std::tie(tableau, unbounded, infeasible) = simplex(A, b, c); // Assuming simplex is defined

  auto [solution, copt] = getResults(); // Assuming getResults is defined

  if constexpr (settings::debug_Simplex)
    fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);

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

  std::vector<int> fractionalMask;

  for (int i = 0; i < (tableau.rows() - 1); i++)
    if (isFractional(tableau(i, Eigen::last))) // Scan last column
      fractionalMask.push_back(i);

  nGomory = fractionalMask.size();
  fmt::println("Number of Gomory cuts: {}", nGomory);

  MatrixXd gamma = tableau(fractionalMask, Eigen::seqN(0, tableau.cols() - 1)).array().floor();
  VectorXd bplus = tableau(fractionalMask, tableau.cols() - 1).array().floor();

  MatrixXd newA(A.rows() + nGomory, A.cols() + nGomory);
  VectorXd newb(A.rows() + nGomory);
  VectorXd newc(A.cols() + nGomory);

  newA << A, MatrixXd::Zero(A.rows(), nGomory),
    gamma, MatrixXd::Identity(nGomory, nGomory);

  newb << b, bplus;

  newc << c, VectorXd::Zero(nGomory);

  A = newA;
  b = newb;
  c = newc;
}


Simplex::Simplex(int Nb, int Nc)
{
  // Create Simplex type problem with Nb = number of batteries and Nc = number of clusters.
}


Simplex::Simplex(Problem &prob)
{
  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();


  const auto Neq = Nb + 1;
  const auto Nineq = Nb * (Nb - 1);
  const auto Nconstraints = Neq + Nineq;

  const auto Nvar_original = Nb * Nb;
  const auto N_slack = Nineq;
  const auto Nvar = Nvar_original + N_slack; // x1--xN^2  + s_slack

  A = MatrixXd::Zero(Nconstraints, Nvar);
  b = VectorXd::Zero(Nconstraints);
  c = VectorXd::Zero(Nvar);

  // Create A matrix:
  A.bottomRightCorner(N_slack, N_slack) = MatrixXd::Identity(N_slack, N_slack);

  for (int i = 0; i < Nb; ++i) {
    A(0, i * (Nb + 1)) = 1.0;                                // Sum of diagonals is Nc
    A.block(1, Nb * i, Nb, Nb) = MatrixXd::Identity(Nb, Nb); // Every element belongs to one cluster.

    // ---------------
    int shift = 0;
    for (int j = 0; j < Nb; j++) {
      const int block_begin_row = Nb + 1 + (Nb - 1) * i;
      const int block_begin_col = Nb * i;
      if (i == j) {
        A.block(block_begin_row, block_begin_col + j, Nb - 1, 1) = -1 * MatrixXd::Ones(Nb - 1, 1);
        shift = 1;
      } else
        A(block_begin_row + j - shift, block_begin_col + j) = 1;
    }
  }

  // Create b matrix:
  b(0) = Nc;
  for (int i = 0; i < Nb; ++i)
    b(i + 1) = 1;

  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      c[i + j * Nb] = prob.distByInd_scaled(i, j);
}

} // namespace dtwc::solver
