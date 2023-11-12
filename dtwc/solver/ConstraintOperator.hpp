/*
 * op_A.hpp
 *
 * operators for A matrix

 *  Created on: 10 Dec 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "../settings.hpp"
#include "../utility.hpp"
#include "EqualityConstraints.hpp"
#include <Eigen/Sparse>

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

namespace dtwc::solver {
using Eigen::VectorXd;

struct ConstraintOperator
{
  using SpVectorType = Eigen::SparseVector<double>;
  using SpMatrixType = Eigen::SparseMatrix<double>;

  size_t Nb{}, Nc{}; // Number of time-series data.

  SpVectorType b;
  SpMatrixType A;

  ConstraintOperator() = default;
  explicit ConstraintOperator(size_t Nb_, size_t Nc_) : Nb(Nb_), Nc(Nc_)
  {
    const auto Neq = Nb + 1;
    const auto Nineq = Nb * (Nb - 1);
    const auto Nconstraints = Neq + Nineq;

    const auto Nvar_original = Nb * Nb;
    const auto N_slack = Nineq;
    const auto Nvar = Nvar_original + N_slack; // x1--xN^2  + s_slack

    std::vector<Eigen::Triplet<double>> triplets;
    b = SpVectorType(Nconstraints);
    b.reserve(Nb + 1);

    auto eq = EqualityConstraints(Nconstraints, Nvar);
    // Create b matrix:
    b.coeffRef(0) = Nc;
    for (int i = 0; i < Nb; ++i)
      b.coeffRef(i + 1) = 1;

    // Create A matrix:
    for (int i = 0; i < Nineq; i++)
      triplets.emplace_back(Neq + i, Nvar_original + i, 1.0);

    for (int i = 0; i < Nb; ++i) {
      triplets.emplace_back(0, i * (Nb + 1), 1.0); // Sum of diagonals is Nc

      for (int j = 0; j < Nb; j++)
        triplets.emplace_back(1 + j, Nb * i + j, 1.0); // Every element belongs to one cluster.

      // ---------------
      int shift = 0;
      for (int j = 0; j < Nb; j++) {
        const int block_begin_row = Nb + 1 + (Nb - 1) * i;
        const int block_begin_col = Nb * i;
        if (i == j) {
          for (int k = 0; k < (Nb - 1); k++)
            triplets.emplace_back(block_begin_row + k, block_begin_col + j, -1.0);
          shift = 1;
        } else
          triplets.emplace_back(block_begin_row + j - shift, block_begin_col + j, 1.0);
      }
    }

    A = SpMatrixType(Nconstraints, Nvar);
    A.setFromTriplets(triplets.begin(), triplets.end());
  }

  auto get_Nm() const { return A.rows(); }
  auto get_Nx() const { return A.cols(); }
};
} // namespace dtwc::solver