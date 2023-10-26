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
#include "SimplexTable.hpp"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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
using MatrixTypeRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixType = MatrixTypeRowMajor; // Faster than column-major


class Simplex
{
  // -1 means None.
  size_t N{ 0 }, Nc{ 0 };
  int mtab{}, ntab{};
  MatrixType tableau;
  MatrixType A;  // A*x = b
  VectorXd b, c; // c*x = cost.
  int nGomory{ -1 };

  SimplexTable table;

public:
  Simplex(MatrixType A_, VectorXd b_, VectorXd c_) : A(A_), b(b_), c(c_) {}
  Simplex() = default;
  Simplex(Problem &prob);

  void gomory();
  void gomoryAlgorithm()
  {
    while (nGomory != 0) gomory();
  }

  std::pair<std::vector<double>, double> getResults() const;
};


int getRow(const MatrixType &tableau, int index);
void inline pivoting(MatrixType &tableau, int p, int q);
std::tuple<int, int, bool, bool> inline simplexTableau(const MatrixType &tableau);
std::pair<bool, bool> inline simplexAlgorithmTableau(MatrixType &input_tableau);
MatrixType inline createTableau(const MatrixType &A, const VectorXd &b, VectorXd &c);
std::tuple<MatrixType, bool, bool> inline simplex(MatrixType &A, VectorXd &b, VectorXd &c);

} // namespace dtwc::solver
