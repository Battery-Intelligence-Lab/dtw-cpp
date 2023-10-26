/*
 * SimplexSolver.hpp
 *
 * Helper class for sparse equality constraint.

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
#include <stdexcept>
#include <set>
#include <limits>
#include <map>
#include <iomanip>


namespace dtwc::solver {

struct Coordinate
{
  int row{}, col{}; // Row and column of the value
};

bool operator<(const Coordinate &c1, const Coordinate &c2)
{
  return (c1.row < c2.row) || (c1.row == c2.row && c1.col < c2.col);
}


struct SparseMatrix
{
  std::map<Coordinate, double> data;
  int m{}, n{}; // rows and columns

  SparseMatrix() = default;
  SparseMatrix(int m_, int n_) : m{ m_ }, n{ n_ } {}

  double operator()(int x, int y) const
  {
    auto it = data.find(Coordinate{ x, y });
    return (it != data.end()) ? (it->second) : 0.0;
  }

  double &operator()(int x, int y) { return data[Coordinate{ x, y }]; }

  void compress()
  {
    std::erase_if(data, [](const auto &item) {
      auto const &[key, value] = item;
      return isAround(value, 0.0);
    });
  }

  void print()
  {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (j != 0) std::cout << ',';
        std::cout << std::setw(3) << (*this)(i, j);
      }
      std::cout << '\n';
    }
  }

  int rows() { return m; }
  int cols() { return n; }
};

class EqualityConstraints
{
  SparseMatrix A;
  std::vector<double> b;
  int m{}, n{}; // rows and columns
public:
  EqualityConstraints() = default;
  EqualityConstraints(int m_, int n_) : A(m_, n_), b(m_, 0.0) {}

  double &coeff_A(int x, int y) { return A(x, y); }
  double &coeff_b(int x) { return b[x]; }

  void print_A() { A.print(); }
};

EqualityConstraints inline defaultConstraints(int Nb, int Nc)
{
  const auto Neq = Nb + 1;
  const auto Nineq = Nb * (Nb - 1);
  const auto Nconstraints = Neq + Nineq;

  const auto Nvar_original = Nb * Nb;
  const auto N_slack = Nineq;
  const auto Nvar = Nvar_original + N_slack; // x1--xN^2  + s_slack

  auto eq = EqualityConstraints(Nconstraints, Nvar);
  // Create b matrix:
  eq.coeff_b(0) = Nc;
  for (int i = 0; i < Nb; ++i)
    eq.coeff_b(i + 1) = 1;

  // Create A matrix:
  for (int i = 0; i < Nineq; i++)
    eq.coeff_A(Neq + i, Nvar_original + i) = 1;

  for (int i = 0; i < Nb; ++i) {
    eq.coeff_A(0, i * (Nb + 1)) = 1.0; // Sum of diagonals is Nc

    for (int j = 0; j < Nb; j++)
      eq.coeff_A(1 + j, Nb * i + j) = 1; // Every element belongs to one cluster.

    // ---------------
    int shift = 0;
    for (int j = 0; j < Nb; j++) {
      const int block_begin_row = Nb + 1 + (Nb - 1) * i;
      const int block_begin_col = Nb * i;
      if (i == j) {
        for (int k = 0; k < (Nb - 1); k++)
          eq.coeff_A(block_begin_row + k, block_begin_col + j) = -1;
        shift = 1;
      } else
        eq.coeff_A(block_begin_row + j - shift, block_begin_col + j) = 1;
    }
  }

  return eq;
};

} // namespace dtwc::solver