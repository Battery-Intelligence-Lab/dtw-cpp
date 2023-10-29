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
#include "SparseMatrix.hpp"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <span>

#include <tuple>
#include <stdexcept>
#include <set>
#include <limits>
#include <map>

namespace dtwc::solver {

struct EqualityConstraints
{
  SparseMatrix<ColumnMajor> A;
  std::vector<double> b;

  EqualityConstraints() = default;
  EqualityConstraints(int m_, int n_) : A(m_, n_), b(m_, 0.0) {}
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
  eq.b[0] = Nc;
  for (int i = 0; i < Nb; ++i)
    eq.b[i + 1] = 1;

  // Create A matrix:
  for (int i = 0; i < Nineq; i++)
    eq.A(Neq + i, Nvar_original + i) = 1;

  for (int i = 0; i < Nb; ++i) {
    eq.A(0, i * (Nb + 1)) = 1.0; // Sum of diagonals is Nc

    for (int j = 0; j < Nb; j++)
      eq.A(1 + j, Nb * i + j) = 1; // Every element belongs to one cluster.

    // ---------------
    int shift = 0;
    for (int j = 0; j < Nb; j++) {
      const int block_begin_row = Nb + 1 + (Nb - 1) * i;
      const int block_begin_col = Nb * i;
      if (i == j) {
        for (int k = 0; k < (Nb - 1); k++)
          eq.A(block_begin_row + k, block_begin_col + j) = -1;
        shift = 1;
      } else
        eq.A(block_begin_row + j - shift, block_begin_col + j) = 1;
    }
  }

  return eq;
};

} // namespace dtwc::solver