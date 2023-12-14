/*
 * mip.cpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "Data.hpp"        // for Data
#include "types/types.hpp" // for Triplet, RowMajor
#include "mip.hpp"
#include "Problem.hpp"
#include "settings.hpp"
#include "timing.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <vector>
#include <cassert>   // for assert
#include <cstddef>   // for size_t
#include <algorithm> // for sort
#include <iostream>  // for operator<<, basic_ostream, ost...
#include <string>    // for operator<<

namespace dtwc {

template <typename T>
void extract_solution(Problem &prob, const T &solution)
{
  prob.clear_clusters();
  const auto Nb = prob.data.size();

  for (int i{ 0 }; i < Nb; i++)
    if (solution[i * (Nb + 1)] > 0.5)
      prob.centroids_ind.push_back(i);

  prob.clusters_ind.resize(Nb);

  int i_cluster = 0;
  for (auto i : prob.centroids_ind) {
    prob.cluster_members.emplace_back();
    for (int j{ 0 }; j < Nb; j++)
      if (solution[i * Nb + j] > 0.5) {
        prob.clusters_ind[j] = i_cluster;
        prob.cluster_members.back().push_back(j);
      }

    i_cluster++;
  }
}

void MIP_clustering_byHiGHS(Problem &prob)
{
  std::cout << "HiGS is being called!" << std::endl;
  dtwc::Clock clk; // Create a clock object

  prob.clear_clusters();

  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();

  const auto Neq = Nb + 1;
  const auto Nineq = Nb * (Nb - 1);
  const auto Nconstraints = Neq + Nineq;

  const auto Nvar = Nb * Nb;

#ifdef DTWC_ENABLE_HIGHS
  HighsModel model;
  model.lp_.num_col_ = Nvar;
  model.lp_.num_row_ = Nconstraints;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.offset_ = 0;

  // Initialise q vector for cost.
  model.lp_.col_cost_.resize(Nvar);
  for (int j{ 0 }; j < Nb; j++)
    for (int i{ 0 }; i < Nb; i++)
      model.lp_.col_cost_[i + j * Nb] = prob.distByInd_scaled(i, j);

  model.lp_.col_lower_.clear();
  model.lp_.col_lower_.resize(Nvar, 0.0);

  model.lp_.col_upper_.clear();
  model.lp_.col_upper_.resize(Nvar, 1.0);

  model.lp_.row_lower_.clear();
  model.lp_.row_lower_.resize(Nconstraints, -1.0);

  model.lp_.row_upper_.clear();
  model.lp_.row_upper_.resize(Nconstraints, 0.0);

  model.lp_.row_upper_[0] = model.lp_.row_lower_[0] = Nc;

  for (int i = 0; i < Nb; ++i)
    model.lp_.row_upper_[i + 1] = model.lp_.row_lower_[i + 1] = 1;


  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise; // Here the orientation of the matrix is column-wise

  const auto numel = Nb + Nb * Nb + Nb * 2 * (Nb - 1);

  model.lp_.a_matrix_.start_.clear();
  model.lp_.a_matrix_.index_.clear();
  model.lp_.a_matrix_.value_.clear();

  model.lp_.a_matrix_.start_.reserve(numel + 1);
  model.lp_.a_matrix_.index_.reserve(numel);
  model.lp_.a_matrix_.value_.reserve(numel);

  std::vector<solver::Triplet> triplets;

  triplets.reserve(numel);

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
          triplets.emplace_back(block_begin_row + k, block_begin_col + j, -1);
        shift = 1;
      } else
        triplets.emplace_back(block_begin_row + j - shift, block_begin_col + j, 1);
    }
  }

  std::sort(triplets.begin(), triplets.end(), solver::RowMajor{});

  int current{ -1 }, i_now{};

  for (const auto triplet : triplets) {

    if (current != triplet.col) {
      model.lp_.a_matrix_.start_.push_back(i_now);
      current = triplet.col;
    }

    model.lp_.a_matrix_.index_.push_back(triplet.row);
    model.lp_.a_matrix_.value_.push_back(triplet.val);
    i_now++;
  }

  model.lp_.a_matrix_.start_.push_back(i_now);

  // Now indicate that all the variables must take integer values
  model.lp_.integrality_.clear();
  model.lp_.integrality_.resize(model.lp_.num_col_, HighsVarType::kInteger);

  // Create a Highs instance
  Highs highs;

  HighsStatus return_status = highs.passModel(model); // Pass the model to HiGHS
  if (return_status != HighsStatus::kOk) {
    std::cout << "Passing the model to HiGHS was unsuccessful!\n";
    return;
  }
  return_status = highs.run(); // Solve the model
  if (return_status != HighsStatus::kOk) {
    std::cout << "Solving the model with HiGHS was unsuccessful!\n";
    return;
  }

  // Get the model status
  const HighsModelStatus &model_status = highs.getModelStatus();
  assert(model_status == HighsModelStatus::kOptimal);
  std::cout << "Model status: " << highs.modelStatusToString(model_status) << '\n';

  // Get the solution information
  const HighsInfo &info = highs.getInfo();
  std::cout << "Simplex iteration count: " << info.simplex_iteration_count << '\n';
  std::cout << "Objective function value: " << info.objective_function_value << '\n';
  std::cout << "Primal  solution status: " << highs.solutionStatusToString(info.primal_solution_status) << '\n';
  std::cout << "Dual    solution status: " << highs.solutionStatusToString(info.dual_solution_status) << '\n';
  std::cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << '\n';

  // Get the solution values
  extract_solution(prob, highs.getSolution().col_value);
#endif
}

} // namespace dtwc
