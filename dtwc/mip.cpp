/*
 * mip.cpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "mip.hpp"
#include "Problem.hpp"
#include "settings.hpp"
#include "utility.hpp"
#include "solver/Simplex.hpp"
#include "solver/SparseSimplex.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>
#include <string_view>
#include <memory>
#include <limits>
#include <Highs.h>


namespace dtwc {

template <typename T>
void MIP_clustering_bySimplex(Problem &prob)
{
  std::cout << "Simplex is being called!" << std::endl;
  dtwc::Clock clk; // Create a clock object

  prob.clear_clusters();

  thread_local auto simplexSolver = T(prob);

  std::cout << "Problem formulation finished in " << clk << '\n';
  simplexSolver.gomoryAlgorithm();
  std::cout << "Problem solution finished in " << clk << '\n';

  auto [solution, copt] = simplexSolver.getResults();

  fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);

  const auto Nb = prob.data.size();

  for (ind_t i{ 0 }; i < Nb; i++)
    if (solution[i * (Nb + 1)] > 0.5)
      prob.centroids_ind.push_back(i);

  prob.clusters_ind = std::vector<ind_t>(Nb);

  ind_t i_cluster = 0;
  for (auto i : prob.centroids_ind) {
    prob.cluster_members.emplace_back();
    for (size_t j{ 0 }; j < Nb; j++)
      if (solution[i * Nb + j] > 0.5) {
        prob.clusters_ind[j] = i_cluster;
        prob.cluster_members.back().push_back(j);
      }

    i_cluster++;
  }
}

void MIP_clustering_bySparseSimplex(Problem &prob)
{
  MIP_clustering_bySimplex<dtwc::solver::SparseSimplex>(prob);
}

void MIP_clustering_byDenseSimplex(Problem &prob)
{
  MIP_clustering_bySimplex<dtwc::solver::Simplex>(prob);
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

  HighsModel model;
  model.lp_.num_col_ = Nvar;
  model.lp_.num_row_ = Nconstraints;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.offset_ = 0;

  // Initialise q vector for cost.
  model.lp_.col_cost_.resize(Nvar);
  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
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

  //
  // Here the orientation of the matrix is column-wise
  model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
  // a_start_ has num_col_1 entries, and the last entry is the number
  // of nonzeros in A, allowing the number of nonzeros in the last
  // column to be defined
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

  std::sort(triplets.begin(), triplets.end(), solver::ColumnMajor{});

  int current_row{ -1 }, i_row{};

  for (const auto triplet : triplets) {

    if (current_row != triplet.row) {
      model.lp_.a_matrix_.start_.push_back(i_row);
      current_row = triplet.row;
    }


    //  std::cout << "Triplet: (" << triplet.row << ", " << triplet.col << ", " << triplet.val << ")\n";
    model.lp_.a_matrix_.index_.push_back(triplet.col);
    model.lp_.a_matrix_.value_.push_back(triplet.val);
    i_row++;
  }

  model.lp_.a_matrix_.start_.push_back(i_row);

  // Test.


  // Create a Highs instance
  Highs highs;
  HighsStatus return_status;
  //
  // Pass the model to HiGHS
  return_status = highs.passModel(model);
  assert(return_status == HighsStatus::kOk);
  // If a user passes a model with entries in
  // model.lp_.a_matrix_.value_ less than (the option)
  // small_matrix_value in magnitude, they will be ignored. A logging
  // message will indicate this, and passModel will return
  // HighsStatus::kWarning
  //
  // Get a const reference to the LP data in HiGHS
  const HighsLp &lp = highs.getLp();


  // Now indicate that all the variables must take integer values
  model.lp_.integrality_.resize(lp.num_col_);
  for (int col = 0; col < lp.num_col_; col++)
    model.lp_.integrality_[col] = HighsVarType::kInteger;

  highs.passModel(model);
  // Solve the model
  return_status = highs.run();
  assert(return_status == HighsStatus::kOk);

  // Get the model status
  const HighsModelStatus &model_status = highs.getModelStatus();
  assert(model_status == HighsModelStatus::kOptimal);
  std::cout << "Model status: " << highs.modelStatusToString(model_status) << '\n';
  //
  // Get the solution information
  const HighsInfo &info = highs.getInfo();
  std::cout << "Simplex iteration count: " << info.simplex_iteration_count << '\n';
  std::cout << "Objective function value: " << info.objective_function_value << '\n';
  std::cout << "Primal  solution status: " << highs.solutionStatusToString(info.primal_solution_status) << '\n';
  std::cout << "Dual    solution status: " << highs.solutionStatusToString(info.dual_solution_status) << '\n';
  std::cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << '\n';

  // Get the solution values and basis
  const HighsSolution &solution = highs.getSolution();

  for (ind_t i{ 0 }; i < Nb; i++)
    if (solution.col_value[i * (Nb + 1)] > 0.5)
      prob.centroids_ind.push_back(i);

  prob.clusters_ind = std::vector<ind_t>(Nb);

  ind_t i_cluster = 0;
  for (auto i : prob.centroids_ind) {
    prob.cluster_members.emplace_back();
    for (size_t j{ 0 }; j < Nb; j++)
      if (solution.col_value[i * Nb + j] > 0.5) {
        prob.clusters_ind[j] = i_cluster;
        prob.cluster_members.back().push_back(j);
      }

    i_cluster++;
  }
}


} // namespace dtwc
