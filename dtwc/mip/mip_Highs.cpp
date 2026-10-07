/**
 * @file mip_Highs.cpp
 * @brief The compact p-median model as arrays, and its solve by linked HiGHS.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 06 Nov 2022
 */

#include "mip.hpp"
#include "highs_support.hpp"
#include "decode_assignment.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../base/error.hpp"       // for SolverError
#include "../Problem.hpp"
#include "../base/settings.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <algorithm> // for max, fill_n
#include <cstddef>   // for size_t
#include <iostream>  // for operator<<, basic_ostream, ost...
#include <limits>    // for numeric_limits
#include <string>    // for operator<<, std::to_string

namespace dtwc {

bool highs_solver_available() noexcept
{
#ifdef DTWC_ENABLE_HIGHS
  return true;
#else
  return false;
#endif
}

mip::PMedianModel mip::build_p_median_model(Problem &prob)
{
  const auto n = static_cast<std::size_t>(prob.size());
  const auto k = prob.n_clusters();

  // HiGHS indexes with `int`; the ~3N² nonzeros are the largest count (N ≈ 26,750),
  // and a cast past INT_MAX builds a wrong model.
  const std::size_t nnz = n + n * n + 2 * n * (n - 1);
  if (nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw SolverError("HiGHS: the compact p-median model for N = " + std::to_string(n) + " has more than INT_MAX nonzeros; use Method::LRCore.");

  prob.fill_distance_matrix();                                           // We need full distance matrix before MIP clustering.
  const auto scaling_factor = std::max(prob.max_distance() / 2.0, 1.0); // In case no distance is set.

  PMedianModel model;
  const std::size_t rows = 1 + n + n * (n - 1);
  model.num_col = static_cast<int>(n * n);
  model.num_row = static_cast<int>(rows);
  model.col_cost.resize(n * n);
  for (std::size_t j{ 0 }; j < n; j++)
    for (std::size_t i{ 0 }; i < n; i++)
      model.col_cost[i + j * n] = prob.dist_by_ind(static_cast<index_t>(i), static_cast<index_t>(j)) / scaling_factor;
  model.col_lower.assign(n * n, 0.0);
  model.col_upper.assign(n * n, 1.0);
  model.integrality.assign(n * n, 1);

  model.row_lower.assign(rows, -1.0);
  model.row_upper.assign(rows, 0.0);
  model.row_lower[0] = model.row_upper[0] = static_cast<double>(k);
  std::fill_n(model.row_lower.begin() + 1, n, 1.0);
  std::fill_n(model.row_upper.begin() + 1, n, 1.0);

  model.a_start.reserve(rows + 1);
  model.a_index.reserve(nnz);
  model.a_value.reserve(nnz);
  model.a_start.push_back(0);
  const auto add = [&model](std::size_t col, double value) {
    model.a_index.push_back(static_cast<int>(col));
    model.a_value.push_back(value);
  };
  const auto end_row = [&model] { model.a_start.push_back(static_cast<int>(model.a_index.size())); };
  for (std::size_t f = 0; f < n; ++f) add(f * (n + 1), 1.0); // k medoids are open
  end_row();
  for (std::size_t p = 0; p < n; ++p) { // every point belongs to one cluster
    for (std::size_t f = 0; f < n; ++f) add(f * n + p, 1.0);
    end_row();
  }
  for (std::size_t f = 0; f < n; ++f) // ... whose medoid is open
    for (std::size_t p = 0; p < n; ++p)
      if (p != f) {
        add(f * n + p, 1.0);
        add(f * (n + 1), -1.0);
        end_row();
      }

  if (prob.mip_settings.warm_start) { // FastPAM's clustering as the MIP start
    const auto pam = fast_pam(prob, k, settings::DEFAULT_MAX_ITER, prob.random_seed());
    model.start.assign(n * n, 0.0);
    for (const auto med : pam.medoid_indices)
      model.start[static_cast<std::size_t>(med) * (n + 1)] = 1.0;
    for (std::size_t p = 0; p < n; ++p)
      model.start[static_cast<std::size_t>(pam.medoid_indices[static_cast<std::size_t>(pam.labels[p])]) * n + p] = 1.0;
  }
  return model;
}

void MIP_clustering_byHiGHS(Problem &prob)
{
  if (prob.mip_settings.verbose_solver || prob.verbose())
    std::cout << "HiGHS MIP solver starting." << '\n';

#ifdef DTWC_ENABLE_HIGHS
  const auto model = mip::build_p_median_model(prob);

  // Create a Highs instance
  Highs highs;

  // Solver tuning from MIPSettings. A rejected option must never be ignored:
  // the solve would then silently run on HiGHS defaults.
  mip::set_highs_option(highs, "mip_rel_gap", prob.mip_settings.mip_gap, "HiGHS");
  if (prob.mip_settings.time_limit_sec > 0)
    mip::set_highs_option(highs, "time_limit",
                          static_cast<double>(prob.mip_settings.time_limit_sec), "HiGHS");
  if (!prob.mip_settings.verbose_solver)
    mip::set_highs_option(highs, "output_flag", false, "HiGHS");

  HighsStatus return_status = highs.passModel( // HiGHS stores it column-wise
    model.num_col, model.num_row, static_cast<HighsInt>(model.a_value.size()),
    static_cast<HighsInt>(MatrixFormat::kRowwise), static_cast<HighsInt>(ObjSense::kMinimize), 0.0,
    model.col_cost.data(), model.col_lower.data(), model.col_upper.data(), model.row_lower.data(),
    model.row_upper.data(), model.a_start.data(), model.a_index.data(), model.a_value.data(),
    model.integrality.data());
  if (return_status != HighsStatus::kOk)
    throw SolverError("HiGHS rejected the MIP model (passModel returned status "
                      + std::to_string(static_cast<int>(return_status)) + ").");

  if (!model.start.empty()) {
    HighsSolution sol;
    sol.col_value = model.start;
    sol.value_valid = true;
    highs.setSolution(sol);
  }

  return_status = highs.run(); // Solve the model
  if (return_status != HighsStatus::kOk)
    throw SolverError("HiGHS failed to solve the MIP (run returned status "
                      + std::to_string(static_cast<int>(return_status)) + ").");

  // Get the model status. This guard used to be
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

  prob.set_result(mip::decode_assignment(highs.getSolution().col_value, static_cast<std::size_t>(prob.size()),
                                         prob.n_clusters(), false, "HiGHS"));
#else
  throw SolverError(
      "HiGHS solver is unavailable; rebuild with -DDTWC_ENABLE_HIGHS=ON");
#endif
}

} // namespace dtwc
