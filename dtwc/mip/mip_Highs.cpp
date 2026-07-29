/**
 * @file mip_Highs.cpp
 * @brief Encapsulating mixed-integer program functions using Highs solver.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 06 Nov 2022
 */

#include "mip.hpp"
#include "solution_transaction.hpp"
#include "warm_start.hpp"
#include "../Data.hpp"        // for Data
#include "../error.hpp"       // for SolverError
#include "../types/types.hpp" // for Triplet, RowMajor
#include "../Problem.hpp"
#include "../settings.hpp"
#include "../timing.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <vector>
#include <stdexcept> // for std::runtime_error
#include <cstddef>   // for size_t
#include <algorithm> // for sort
#include <iostream>  // for operator<<, basic_ostream, ost...
#include <string>    // for operator<<, std::to_string
#include <utility>   // for move

namespace dtwc {

bool highs_solver_available() noexcept
{
#ifdef DTWC_ENABLE_HIGHS
  return true;
#else
  return false;
#endif
}

void MIP_clustering_byHiGHS(Problem &prob)
{
  if (prob.mip_settings.verbose_solver || prob.verbose())
    std::cout << "HiGHS MIP solver starting." << '\n';
  dtwc::Clock clk; // Create a clock object

#ifdef DTWC_ENABLE_HIGHS
  const auto Nb = prob.data().size();
  const auto Nc = prob.n_clusters();
  mip::ExactClusteringTransaction result_transaction(prob);

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

  prob.fill_distance_matrix();                                           // We need full distance matrix before MIP clustering.
  const auto scaling_factor = std::max(prob.max_distance() / 2.0, 1.0); // In case no distance is set.

  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      model.lp_.col_cost_[i + j * Nb] = prob.dist_by_ind(static_cast<int>(i), static_cast<int>(j)) / scaling_factor;


  model.lp_.col_lower_.clear();
  model.lp_.col_lower_.resize(Nvar, 0.0);

  model.lp_.col_upper_.clear();
  model.lp_.col_upper_.resize(Nvar, 1.0);

  model.lp_.row_lower_.clear();
  model.lp_.row_lower_.resize(Nconstraints, -1.0);

  model.lp_.row_upper_.clear();
  model.lp_.row_upper_.resize(Nconstraints, 0.0);

  model.lp_.row_upper_[0] = model.lp_.row_lower_[0] = Nc;

  for (size_t i = 0; i < Nb; ++i)
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

  for (size_t i = 0; i < Nb; ++i) {
    triplets.emplace_back(0, static_cast<int>(i * (Nb + 1)), 1.0); // Sum of diagonals is Nc

    for (size_t j = 0; j < Nb; j++)
      triplets.emplace_back(static_cast<int>(1 + j), static_cast<int>(Nb * i + j), 1.0); // Every element belongs to one cluster.

    // ---------------
    int shift = 0;
    for (size_t j = 0; j < Nb; j++) {
      const int block_begin_row = static_cast<int>(Nb + 1 + (Nb - 1) * i);
      const int block_begin_col = static_cast<int>(Nb * i);
      if (i == j) {
        for (size_t k = 0; k < (Nb - 1); k++)
          triplets.emplace_back(block_begin_row + static_cast<int>(k), block_begin_col + static_cast<int>(j), -1);
        shift = 1;
      } else
        triplets.emplace_back(block_begin_row + static_cast<int>(j) - shift, block_begin_col + static_cast<int>(j), 1);
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

  // Solver tuning from MIPSettings
  highs.setOptionValue("mip_rel_gap", prob.mip_settings.mip_gap);
  if (prob.mip_settings.time_limit_sec > 0)
    highs.setOptionValue("time_limit", static_cast<double>(prob.mip_settings.time_limit_sec));
  if (!prob.mip_settings.verbose_solver)
    highs.setOptionValue("output_flag", false);

  HighsStatus return_status = highs.passModel(model); // Pass the model to HiGHS
  if (return_status != HighsStatus::kOk)
    throw SolverError("HiGHS rejected the MIP model (passModel returned status "
                      + std::to_string(static_cast<int>(return_status)) + ").");

  // Warm start: run FastPAM and feed solution as MIP start
  if (prob.mip_settings.warm_start) {
    auto pam_result = mip::make_warm_start(prob, prob.random_seed());

    HighsSolution sol;
    sol.col_value.resize(Nvar, 0.0);
    sol.value_valid = true;

    for (int med : pam_result.medoid_indices)
      sol.col_value[static_cast<size_t>(med) * (Nb + 1)] = 1.0;

    // HiGHS indexing: A[i,j] at flat index i * Nb + j (row-major)
    for (size_t j = 0; j < Nb; ++j) {
      int med = pam_result.medoid_indices[pam_result.labels[j]];
      sol.col_value[static_cast<size_t>(med) * Nb + j] = 1.0;
    }

    highs.setSolution(sol);
  }

  return_status = highs.run(); // Solve the model
  if (return_status != HighsStatus::kOk)
    throw SolverError("HiGHS failed to solve the MIP (run returned status "
                      + std::to_string(static_cast<int>(return_status)) + ").");

  // Get the model status. Task 0.5 / audit finding #7: this guard used to be
  // assert(model_status == kOptimal), which is a no-op under NDEBUG (release
  // builds). A non-optimal solve (infeasible, unbounded, time/iteration limit)
  // then fell through to decoding an empty/invalid solution vector and returned
  // garbage or empty centroids (UB). Fail loudly.
  const HighsModelStatus &model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal)
    throw SolverError("HiGHS MIP did not solve to optimality. Model status: "
                      + highs.modelStatusToString(model_status));

  if (prob.mip_settings.verbose_solver || prob.verbose()) {
    std::cout << "Model status: " << highs.modelStatusToString(model_status) << '\n';

    const HighsInfo &info = highs.getInfo();
    std::cout << "Simplex iteration count: " << info.simplex_iteration_count << '\n'
              << "Objective function value: " << info.objective_function_value << '\n'
              << "Primal  solution status: " << highs.solutionStatusToString(info.primal_solution_status) << '\n'
              << "Dual    solution status: " << highs.solutionStatusToString(info.dual_solution_status) << '\n'
              << "Basis: " << highs.basisValidityToString(info.basis_validity) << '\n';
  }

  // Decode and validate into private vectors before atomically publishing.
  auto exact_result = mip::extract_exact_clustering(
    highs.getSolution().col_value,
    Nb,
    Nc,
    mip::AssignmentMatrixLayout::FacilityMajor,
    "HiGHS");
  result_transaction.publish(std::move(exact_result), "HiGHS");
#else
  throw SolverError(
      "HiGHS solver is unavailable; rebuild with -DDTWC_ENABLE_HIGHS=ON");
#endif
}

} // namespace dtwc
