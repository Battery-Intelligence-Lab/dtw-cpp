/**
 * @file mip_Highs.cpp
 * @brief The compact p-median model as arrays, and its solve by linked HiGHS.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 06 Nov 2022
 */

#include "mip.hpp"
#include "decode_assignment.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../base/error.hpp"
#include "../Problem.hpp"
#include "../base/settings.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>

namespace dtwc {

bool highs_solver_available() noexcept
{
#ifdef DTWC_ENABLE_HIGHS
  return true;
#else
  return false;
#endif
}

bool mip::detail::try_solve_trivial_pmedian(Problem &prob)
{
  const auto n = static_cast<std::size_t>(prob.size());
  const auto k = prob.n_clusters();
  if (k == 1 && n >= 1) {
    prob.set_result(fast_pam_solve(prob, 1, 0, prob.random_seed()));
    return true;
  }
  if (k == static_cast<index_t>(n) && n >= 1) {
    core::ClusteringResult result;
    result.medoid_indices.resize(n);
    result.labels.resize(n);
    std::iota(result.medoid_indices.begin(), result.medoid_indices.end(), index_t{ 0 });
    std::iota(result.labels.begin(), result.labels.end(), index_t{ 0 });
    result.converged = true;
    prob.set_result(result);
    return true;
  }
  return false;
}

mip::PMedianModel mip::detail::build_p_median_model_impl(Problem &prob, bool build_csc)
{
  const auto n = static_cast<std::size_t>(prob.size());
  const auto k = prob.n_clusters();

  // The model indexes with `int`, as HiGHS and Gurobi read it; check before filling the distance matrix.
  // HiGHS uses CSC (~3N^2 nonzeros, N ~= 26,750), while Gurobi builds rows dynamically and is bounded by N^2 columns.
  const std::size_t nnz = (n == 0) ? 0 : (n + n * n + 2 * n * (n - 1));
  if (build_csc) {
    if (nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw SolverError("MIP: the compact p-median model for N = " + std::to_string(n)
                        + " has more than INT_MAX nonzeros; use Method::LRCore.");
  } else {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) / std::max<std::size_t>(n, 1))
      throw SolverError("Gurobi: the compact p-median model for N = " + std::to_string(n)
                        + " has more than INT_MAX variables; use Method::LRCore.");
  }

  prob.fill_distance_matrix(); // We need full distance matrix before MIP clustering.
  const double max_distance = prob.max_distance();
  double scaling_factor = max_distance > 0.0 ? max_distance / 2.0 : 1.0;

  std::optional<core::ClusteringResult> pam;
  if (prob.mip_settings.warm_start) { // FastPAM's clustering as the MIP start (without mutating prob on failure)
    pam = fast_pam_solve(prob, k, settings::DEFAULT_MAX_ITER, prob.random_seed());
    if (pam->total_cost > 0.0) {
      const double candidate = max_distance > 0.0 ? std::min(max_distance / 2.0, pam->total_cost) : pam->total_cost;
      if (candidate > 0.0) scaling_factor = candidate;
    }
  }

  PMedianModel model;
  const std::size_t rows = (n == 0) ? 1 : (1 + n + n * (n - 1));
  model.num_col = static_cast<int>(n * n);
  model.num_row = static_cast<int>(rows);
  model.col_cost.resize(n * n);
  for (std::size_t f = 0; f < n; ++f)
    for (std::size_t p = 0; p < n; ++p)
      model.col_cost[f * n + p] = prob.dist_by_ind(static_cast<index_t>(p), static_cast<index_t>(f)) / scaling_factor;
  model.col_lower.assign(n * n, 0.0);
  model.col_upper.assign(n * n, 1.0);
  model.integrality.assign(n * n, 0);
  for (std::size_t f = 0; f < n; ++f)
    model.integrality[f * (n + 1)] = 1;

  model.row_lower.assign(rows, -std::numeric_limits<double>::infinity());
  model.row_upper.assign(rows, 0.0);
  model.row_lower[0] = model.row_upper[0] = static_cast<double>(k);
  if (n > 0) {
    std::fill_n(model.row_lower.begin() + 1, n, 1.0);
    std::fill_n(model.row_upper.begin() + 1, n, 1.0);
  }

  if (build_csc) {
    // Direct CSC (column-wise) matrix construction: rows sorted ascending per column.
    model.a_start.reserve(n * n + 1);
    model.a_index.reserve(nnz);
    model.a_value.reserve(nnz);
    model.a_start.push_back(0);
    const auto add = [&model](std::size_t row, double value) {
      model.a_index.push_back(static_cast<int>(row));
      model.a_value.push_back(value);
    };
    const auto end_col = [&model] { model.a_start.push_back(static_cast<int>(model.a_index.size())); };
    for (std::size_t f = 0; f < n; ++f) {
      const std::size_t link_base = 1 + n + f * (n - 1);
      for (std::size_t p = 0; p < n; ++p) {
        if (p == f) {
          add(0, 1.0);
          add(1 + f, 1.0);
          for (std::size_t r = 0; r < n - 1; ++r)
            add(link_base + r, -1.0);
        } else {
          add(1 + p, 1.0);
          add(link_base + (p < f ? p : p - 1), 1.0);
        }
        end_col();
      }
    }
  }

  if (pam) {
    model.start.assign(n * n, 0.0);
    for (const auto med : pam->medoid_indices)
      model.start[static_cast<std::size_t>(med) * (n + 1)] = 1.0;
    for (std::size_t p = 0; p < n; ++p)
      model.start[static_cast<std::size_t>(pam->medoid_indices[static_cast<std::size_t>(pam->labels[p])]) * n + p] = 1.0;
  }
  return model;
}

mip::PMedianModel mip::build_p_median_model(Problem &prob)
{
  return detail::build_p_median_model_impl(prob, true);
}

bool mip::detail::try_accept_zero_cost_start(Problem &prob, const PMedianModel &model, std::string_view backend)
{
  if (model.start.empty()) return false;
  double start_obj = 0.0;
  for (std::size_t i = 0; i < model.start.size(); ++i)
    start_obj += model.col_cost[i] * model.start[i];
  if (start_obj != 0.0) return false;
  const auto n = static_cast<std::size_t>(prob.size());
  const auto k = prob.n_clusters();
  prob.set_result(mip::decode_assignment_nearest(model.start, prob.distance_matrix(), n, k, backend));
  return true;
}

#ifdef DTWC_ENABLE_HIGHS
namespace mip {
namespace {

/// Set a HiGHS option, or throw — never let a rejected option select a default.
template <typename Value>
void set_highs_option(Highs &highs, const char *name, const Value &value, std::string_view backend)
{
  if (highs.setOptionValue(name, value) == HighsStatus::kError)
    throw SolverError(std::string(backend) + ": HiGHS rejected option '"
      + std::string(name)
      + "'. The option is unknown to this HiGHS build or the value is outside "
        "its domain; continuing would silently run a different solver "
        "configuration.");
}

} // namespace
} // namespace mip
#endif

void MIP_clustering_byHiGHS(Problem &prob)
{
  if (prob.mip_settings.verbose_solver || prob.verbose())
    std::cout << "HiGHS MIP solver starting." << '\n';

#ifdef DTWC_ENABLE_HIGHS
  const auto n = static_cast<std::size_t>(prob.size());
  const auto k = prob.n_clusters();
  if (mip::detail::try_solve_trivial_pmedian(prob)) return;

  auto model = mip::detail::build_p_median_model_impl(prob, true);
  if (mip::detail::try_accept_zero_cost_start(prob, model, "HiGHS")) return;

  Highs highs;

  mip::set_highs_option(highs, "mip_rel_gap", prob.mip_settings.mip_gap, "HiGHS");
  mip::set_highs_option(highs, "mip_abs_gap", 0.0, "HiGHS");
  // HiGHS runs serially: its parallel tree search (1.16's default, a thread per core) could pick other equal-cost
  // medoids on another machine, and its own threads slowed a root solve (1.4 to 2 times on an idle machine, far more
  // on a busy one).
  mip::set_highs_option(highs, "parallel", std::string("off"), "HiGHS");
  mip::set_highs_option(highs, "threads", HighsInt{ 1 }, "HiGHS");
  if (prob.mip_settings.time_limit_sec > 0)
    mip::set_highs_option(highs, "time_limit",
                          static_cast<double>(prob.mip_settings.time_limit_sec), "HiGHS");
  if (!prob.mip_settings.verbose_solver)
    mip::set_highs_option(highs, "output_flag", false, "HiGHS");

  HighsStatus return_status = highs.passModel(
    model.num_col, model.num_row, static_cast<HighsInt>(model.a_value.size()),
    static_cast<HighsInt>(MatrixFormat::kColwise), static_cast<HighsInt>(ObjSense::kMinimize), 0.0,
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
    if (highs.setSolution(sol) == HighsStatus::kError)
      throw SolverError("HiGHS rejected the warm-start solution (setSolution returned an error).");
  }
  model = {}; // HiGHS holds its own copy now: the model's N²-sized arrays go before the solve

  // HiGHS keeps a scheduler per calling thread and refuses a run whose thread count differs from it: the run starts
  // from none and leaves none, as in Python.
  struct ThreadScheduler {
    ThreadScheduler() { Highs::resetThreadScheduler(true); }
    ~ThreadScheduler() { Highs::resetThreadScheduler(true); }
  } const scheduler;
  return_status = highs.run();
  if (return_status != HighsStatus::kOk)
    throw SolverError("HiGHS failed to solve the MIP (run returned status "
                      + std::to_string(static_cast<int>(return_status)) + ").");

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

  prob.set_result(mip::decode_assignment_nearest(highs.getSolution().col_value, prob.distance_matrix(),
                                                 n, k, "HiGHS"));
#else
  throw SolverError(
      "HiGHS solver is unavailable; rebuild with -DDTWC_ENABLE_HIGHS=ON");
#endif
}

} // namespace dtwc
