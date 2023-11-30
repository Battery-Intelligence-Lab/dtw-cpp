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

  using std::cout;
  using std::endl;

  // Create and populate a HighsModel instance for the LP
  //
  // Min    f  =  x_0 +  x_1 + 3
  // s.t.                x_1 <= 7
  //        5 <=  x_0 + 2x_1 <= 15
  //        6 <= 3x_0 + 2x_1
  // 0 <= x_0 <= 4; 1 <= x_1
  //
  // Although the first constraint could be expressed as an upper
  // bound on x_1, it serves to illustrate a non-trivial packed
  // column-wise matrix.
  //
  HighsModel model;
  model.lp_.num_col_ = 2;
  model.lp_.num_row_ = 3;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.offset_ = 3;
  model.lp_.col_cost_ = { 1.0, 1.0 };
  model.lp_.col_lower_ = { 0.0, 1.0 };
  model.lp_.col_upper_ = { 4.0, 1.0e30 };
  model.lp_.row_lower_ = { -1.0e30, 5.0, 6.0 };
  model.lp_.row_upper_ = { 7.0, 15.0, 1.0e30 };
  //
  // Here the orientation of the matrix is column-wise
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  // a_start_ has num_col_1 entries, and the last entry is the number
  // of nonzeros in A, allowing the number of nonzeros in the last
  // column to be defined
  model.lp_.a_matrix_.start_ = { 0, 2, 5 };
  model.lp_.a_matrix_.index_ = { 1, 2, 0, 1, 2 };
  model.lp_.a_matrix_.value_ = { 1.0, 3.0, 1.0, 2.0, 2.0 };
  //
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
  //
  // Solve the model
  return_status = highs.run();
  assert(return_status == HighsStatus::kOk);
  //
  // Get the model status
  const HighsModelStatus &model_status = highs.getModelStatus();
  assert(model_status == HighsModelStatus::kOptimal);
  cout << "Model status: " << highs.modelStatusToString(model_status) << endl;
  //
  // Get the solution information
  const HighsInfo &info = highs.getInfo();
  cout << "Simplex iteration count: " << info.simplex_iteration_count << endl;
  cout << "Objective function value: " << info.objective_function_value << endl;
  cout << "Primal  solution status: " << highs.solutionStatusToString(info.primal_solution_status) << endl;
  cout << "Dual    solution status: " << highs.solutionStatusToString(info.dual_solution_status) << endl;
  cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << endl;
  const bool has_values = info.primal_solution_status;
  const bool has_duals = info.dual_solution_status;
  const bool has_basis = info.basis_validity;
  //
  // Get the solution values and basis
  const HighsSolution &solution = highs.getSolution();
  const HighsBasis &basis = highs.getBasis();
  //
  // Report the primal and solution values and basis
  for (int col = 0; col < lp.num_col_; col++) {
    cout << "Column " << col;
    if (has_values) cout << "; value = " << solution.col_value[col];
    if (has_duals) cout << "; dual = " << solution.col_dual[col];
    if (has_basis) cout << "; status: " << highs.basisStatusToString(basis.col_status[col]);
    cout << endl;
  }
  for (int row = 0; row < lp.num_row_; row++) {
    cout << "Row    " << row;
    if (has_values) cout << "; value = " << solution.row_value[row];
    if (has_duals) cout << "; dual = " << solution.row_dual[row];
    if (has_basis) cout << "; status: " << highs.basisStatusToString(basis.row_status[row]);
    cout << endl;
  }

  // Now indicate that all the variables must take integer values
  model.lp_.integrality_.resize(lp.num_col_);
  for (int col = 0; col < lp.num_col_; col++)
    model.lp_.integrality_[col] = HighsVarType::kInteger;

  highs.passModel(model);
  // Solve the model
  return_status = highs.run();
  assert(return_status == HighsStatus::kOk);
  // Report the primal solution values
  for (int col = 0; col < lp.num_col_; col++) {
    cout << "Column " << col;
    if (info.primal_solution_status) cout << "; value = " << solution.col_value[col];
    cout << endl;
  }
  for (int row = 0; row < lp.num_row_; row++) {
    cout << "Row    " << row;
    if (info.primal_solution_status) cout << "; value = " << solution.row_value[row];
    cout << endl;
  }
}


} // namespace dtwc
