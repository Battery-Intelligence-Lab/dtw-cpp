/*
 * SimplexSolver.hpp
 *
 * LP solution

 *  Created on: 22 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

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


namespace dtwc {
class Problem;
}

namespace dtwc::solver {

using Eigen::MatrixXd, Eigen::VectorXd;


class Simplex
{
  // -1 means None.
  size_t N{ 0 }, Nc{ 0 };
  int mtab{}, ntab{};
  MatrixXd tableau;
  MatrixXd A;    // A*x = b
  VectorXd b, c; // c*x = cost.
  int nGomory{ -1 };

public:
  size_t maxIterations{ 15000 }; // Default 4000
  size_t numItrConv{ 50 };       // Check convergence every 200 iteration.

  Simplex(MatrixXd A_, VectorXd b_, VectorXd c_) : A(A_), b(b_), c(c_) {}
  Simplex() = default;
  Simplex(int Nb, int Nc);
  Simplex(Problem &prob);

  void gomory();

  void gomoryAlgorithm()
  {
    while (nGomory != 0) gomory();
  }

  std::pair<std::vector<double>, double> getResults() const;

  // ConvergenceFlag solve()
  // {
  // }
};


int getRow(const MatrixXd &tableau, int index);

void inline pivoting(MatrixXd &tableau, int p, int q);

std::tuple<int, int, bool, bool> inline simplexTableau(const MatrixXd &tableau);

std::tuple<MatrixXd, bool, bool> inline simplexAlgorithmTableau(const MatrixXd &input_tableau);

MatrixXd inline createTableau(const MatrixXd &A, const VectorXd &b, VectorXd &c);

std::tuple<MatrixXd, bool, bool> inline simplex(MatrixXd &A, VectorXd &b, VectorXd &c);

} // namespace dtwc::solver
