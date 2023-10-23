/*
 * SimplexSolver.hpp
 *
 * LP solution

 *  Created on: 22 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "cg.hpp"
#include "ConstraintOperator.hpp"
#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"


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
constexpr double epsilon = 1e-8;

bool isAround(double x, double y = 0.0) { return std::abs(x - y) <= epsilon; }
bool isFractional(double x) { return std::abs(x - std::round(x)) > epsilon; }

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

MatrixXd pivoting(const MatrixXd &tableau, int p, int q)
{
  int m = tableau.rows(), n = tableau.cols();

  // Check the provided pivot indices
  if (q >= m)
    throw std::runtime_error(fmt::format("The row of the pivot ({}) must be between 0 and {}", q, m - 1));

  if (p >= n)
    throw std::runtime_error(fmt::format("The column of the pivot ({}) must be between 0 and {}", p, n - 1));

  const double thepivot = tableau(q, p);
  if (isAround(thepivot, 0.0))
    throw std::runtime_error(fmt::format("The pivot is too close to zero: {}", thepivot));

  MatrixXd newtableau = MatrixXd::Zero(m, n);
  newtableau.row(q) = tableau.row(q) / thepivot;

  for (int i = 0; i < m; ++i)
    if (i != q)
      newtableau.row(i) = tableau.row(i) - tableau(i, p) * newtableau.row(q);

  return newtableau;
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

  VectorXd xb = tableau.col(ntab - 1).head(m);
  VectorXd minusd = tableau.col(p).head(m);
  VectorXd steps(m);

  for (int k = 0; k < m; k++) {
    if (minusd(k) > vkTolerance)
      steps(k) = xb(k) / minusd(k);
    else
      steps(k) = std::numeric_limits<double>::infinity();
  }

  VectorXd::Index q;
  double minStep = steps.minCoeff(&q);

  if (std::isinf(minStep)) // The tableau is unbounded
    return std::make_tuple(-1, -1, false, false);
  else
    return std::make_tuple(p, q, false, true);
}

std::tuple<MatrixXd, bool, bool> simplexAlgorithmTableau(const MatrixXd &input_tableau)
{
  MatrixXd tableau = input_tableau;

  while (true) {
    // We assume simplexTableau returns a tuple with four elements: colPivot, rowPivot, optimal, bounded
    auto [colPivot, rowPivot, optimal, bounded] = simplexTableau(tableau);

    if (optimal) return { tableau, true, false };
    if (!bounded) return { tableau, false, true };

    tableau = pivoting(tableau, colPivot, rowPivot);
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
      phaseOneTableau = pivoting(phaseOneTableau, colpivot, rowpivotIndex); // Assuming pivoting is defined
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
  auto [phaseTwoTableau, optimal2, unbounded2] = simplexAlgorithmTableau(startPhaseTwo); // Assuming this function is defined
  return { phaseTwoTableau, unbounded2, false };
}

std::pair<std::vector<double>, double> getResults(const MatrixXd &finalTableau)
{
  int n = finalTableau.cols() - 1;

  std::vector<double> solution(n);
  for (int k = 0; k < n; ++k) {
    const auto basicRowNo = getRow(finalTableau, k);
    solution[k] = (basicRowNo != -1) ? finalTableau(basicRowNo, Eigen::last) : 0.0;
  }

  return { solution, -finalTableau(Eigen::last, Eigen::last) }; // Solution and optimal cost
}


std::tuple<int, MatrixXd, VectorXd, VectorXd, MatrixXd>
  gomory(MatrixXd &A, VectorXd &b, VectorXd &c)
{
  int m = A.rows(), n = A.cols();

  fmt::println("==================================");
  fmt::println("Problem with {} variables and {} constraints", n, m);

  MatrixXd tableau;
  bool unbounded, infeasible;
  std::tie(tableau, unbounded, infeasible) = simplex(A, b, c); // Assuming simplex is defined

  auto [solution, copt] = getResults(tableau); // Assuming getResults is defined

  fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);

  if (unbounded) {
    fmt::println("Unbounded problem");
    return { 0, MatrixXd(), VectorXd(), VectorXd(), MatrixXd() };
  }

  if (infeasible) {
    fmt::println("Infeasible problem");
    return { 0, {}, {}, {}, {} };
  }

  std::vector<int> fractionalMask;

  for (int i = 0; i < (tableau.rows() - 1); i++)
    if (isFractional(tableau(i, Eigen::last))) // Scan last column
      fractionalMask.push_back(i);

  int nGomory = fractionalMask.size();
  MatrixXd gamma = tableau(fractionalMask, Eigen::seqN(0, tableau.cols() - 1)).array().floor();
  VectorXd bplus = tableau(fractionalMask, tableau.cols() - 1).array().floor();

  MatrixXd newA(m + nGomory, n + nGomory);
  VectorXd newb(m + nGomory);
  VectorXd newc(n + nGomory);

  newA << A, MatrixXd::Zero(m, nGomory),
    gamma, MatrixXd::Identity(nGomory, nGomory);

  newb << b, bplus;

  newc << c, VectorXd::Zero(nGomory);

  fmt::println("Number of Gomory cuts: {}", nGomory);

  return { nGomory, newA, newb, newc, tableau };
}

MatrixXd gomoryAlgorithm(MatrixXd A, VectorXd b, VectorXd c)
{
  int nGomory = -1;
  MatrixXd tableu;
  while (nGomory != 0)
    std::tie(nGomory, A, b, c, tableu) = gomory(A, b, c);

  return tableu; // This returns the matrix A. Adjust as needed.
}


class Simplex
{
  double EPS_ADMM_FACTOR{ 1e-2 };
  size_t N{ 0 }, Nc{ 0 };
  int mtab{}, ntab{};
  MatrixXd tableu;

public:
  size_t maxIterations{ 15000 }; // Default 4000
  size_t numItrConv{ 50 };       // Check convergence every 200 iteration.

  // data_t cost() { return std::inner_product(q.begin(), q.end(), vX.begin(), 0.0); }

  // inline bool isSolutionInteger() { return std::all_of(vX.cbegin(), vX.cend(), is_integer<data_t>); }

  ConvergenceFlag solve()
  {
  }
};


} // namespace dtwc::solver
