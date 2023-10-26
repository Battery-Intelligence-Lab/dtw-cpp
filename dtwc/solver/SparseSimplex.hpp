/*
 * SparseSimplex.hpp
 *
 * LP solution

 *  Created on: 26 Oct 2023
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
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>
#include <set>
#include <limits>


namespace dtwc {
class Problem;
}

namespace dtwc::solver {

class SparseSimplex
{
  using VectorType = std::vector<double>;
  // -1 means None.
  VectorType c; // c*x = cost.
  int nGomory{ -1 };

  SimplexTable table;
  EqualityConstraints eq;

public:
  // SparseSimplex(MatrixType A_, VectorXd b_, VectorXd c_) : A(A_), b(b_), c(c_) {}
  SparseSimplex() = default;
  SparseSimplex(Problem &prob);

  void gomory();
  void gomoryAlgorithm()
  {
    while (nGomory != 0) gomory();
  }

  std::pair<std::vector<double>, double> getResults() const;
};


void inline pivoting(SimplexTable &tableau, int p, int q);
std::tuple<int, int, bool, bool> inline simplexTableau(const SimplexTable &tableau);
std::pair<bool, bool> inline simplexAlgorithmTableau(SimplexTable &input_tableau);
SimplexTable inline createTableau(const SimplexTable &A, const VectorType &b, VectorType &c);
std::tuple<SimplexTable, bool, bool> inline simplex(EqualityConstraints &eq, VectorType &c);

} // namespace dtwc::solver
