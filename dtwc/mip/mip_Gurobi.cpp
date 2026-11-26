/**
 * @file mip_Gurobi.cpp
 * @brief Exact k-medoids p-median MIP solver backed by Gurobi.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 25 Dec 2022
 */

#include "mip.hpp"
#include "decode_assignment.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../Problem.hpp"
#include "../base/error.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef DTWC_ENABLE_GUROBI
#include "gurobi_c++.h"
#endif

namespace dtwc {
void MIP_clustering_byGurobi(Problem &prob)
{
  (void)prob;
#ifdef DTWC_ENABLE_GUROBI

  const auto Nb = prob.size();
  const auto Nc = prob.n_clusters();
  const auto n_uz = static_cast<std::size_t>(Nb);

  try {
    if (Nc == 1 && Nb >= 1) {
      prob.set_result(fast_pam_solve(prob, 1, 0, prob.random_seed()));
      return;
    }
    if (Nc == Nb && Nb >= 1) {
      core::ClusteringResult result;
      result.medoid_indices.resize(n_uz);
      result.labels.resize(n_uz);
      std::iota(result.medoid_indices.begin(), result.medoid_indices.end(), index_t{ 0 });
      std::iota(result.labels.begin(), result.labels.end(), index_t{ 0 });
      result.converged = true;
      prob.set_result(result);
      return;
    }

    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // GRBModel::addVars takes an `int` count: an N*N above INT_MAX would narrow
    // into a wrong-sized (or negative) model.
    if (n_uz > static_cast<std::size_t>(std::numeric_limits<int>::max()) / std::max<std::size_t>(n_uz, 1))
      throw SolverError("Gurobi: the compact p-median model for N = " + std::to_string(Nb) + " has more than INT_MAX variables; use Method::LRCore.");

    // The model the HiGHS backend solves: N diagonal binaries y_f = x[f, f] and N*(N-1) continuous assignment variables.
    auto pmedian = mip::build_p_median_model(prob);
    if (!pmedian.start.empty()) {
      double start_obj = 0.0;
      for (std::size_t i = 0; i < pmedian.start.size(); ++i)
        start_obj += pmedian.col_cost[i] * pmedian.start[i];
      if (start_obj == 0.0) {
        prob.set_result(mip::decode_assignment_nearest(pmedian.start, prob.distance_matrix(), n_uz, Nc, "Gurobi"));
        return;
      }
    }

    const int num_col = pmedian.num_col;
    std::vector<char> vtype(static_cast<std::size_t>(num_col), GRB_CONTINUOUS);
    for (std::size_t c = 0; c < static_cast<std::size_t>(num_col); ++c)
      if (pmedian.integrality[c] != 0) vtype[c] = GRB_BINARY;

    std::unique_ptr<GRBVar[]> w{
      model.addVars(pmedian.col_lower.data(), pmedian.col_upper.data(), pmedian.col_cost.data(),
                    vtype.data(), nullptr, num_col)
    };

    // Row 0: Σ_f x[f, f] = k.
    {
      GRBLinExpr lhs = 0;
      for (std::size_t f = 0; f < n_uz; ++f)
        lhs += w[f * (n_uz + 1)];
      model.addConstr(lhs, GRB_EQUAL, pmedian.row_upper[0]);
    }
    // Rows 1..N: Σ_f x[f, p] = 1 for each point p.
    for (std::size_t p = 0; p < n_uz; ++p) {
      GRBLinExpr lhs = 0;
      for (std::size_t f = 0; f < n_uz; ++f)
        lhs += w[f * n_uz + p];
      model.addConstr(lhs, GRB_EQUAL, 1.0);
    }
    // Linking rows: x[f, p] - x[f, f] <= 0 for p != f.
    for (std::size_t f = 0; f < n_uz; ++f) {
      const GRBVar &yf = w[f * (n_uz + 1)];
      for (std::size_t p = 0; p < n_uz; ++p)
        if (p != f) model.addConstr(w[f * n_uz + p] - yf, GRB_LESS_EQUAL, 0.0);
    }

    // Solver tuning from MIPSettings
    model.set(GRB_IntParam_NumericFocus, prob.mip_settings.numeric_focus);
    model.set(GRB_IntParam_MIPFocus, prob.mip_settings.mip_focus);
    model.set(GRB_DoubleParam_MIPGap, prob.mip_settings.mip_gap);
    if (prob.mip_settings.time_limit_sec > 0)
      model.set(GRB_DoubleParam_TimeLimit, static_cast<double>(prob.mip_settings.time_limit_sec));
    if (!prob.mip_settings.verbose_solver)
      model.set(GRB_IntParam_OutputFlag, 0);

    // Warm start: FastPAM's clustering, which the model carries when MIPSettings::warm_start is set
    if (!pmedian.start.empty())
      model.set(GRB_DoubleAttr_Start, w.get(), pmedian.start.data(), num_col);
    pmedian = {}; // Gurobi holds its own copy now: the model's N²-sized arrays go before the solve

    if (prob.mip_settings.verbose_solver || prob.verbose())
      std::cout << "Finished setting up the MILP problem." << '\n';

    model.optimize();

    // Verify optimality before reading GRB_DoubleAttr_X; non-optimal solves have no primal solution.
    const int opt_status = model.get(GRB_IntAttr_Status);
    if (opt_status != GRB_OPTIMAL)
      throw SolverError(
        "Gurobi MIP did not solve to optimality (status code "
        + std::to_string(opt_status) + "). No valid clustering produced.");

    std::unique_ptr<double[]> solution{ model.get(GRB_DoubleAttr_X, w.get(), num_col) };
    prob.set_result(mip::decode_assignment_nearest(
      std::span<const double>(solution.get(), static_cast<std::size_t>(num_col)),
      prob.distance_matrix(), n_uz, Nc, "Gurobi"));

  } catch (GRBException &e) {
    // Must precede std::runtime_error (which GRBException derives from).
    throw SolverError("Gurobi optimisation failed (error code "
                      + std::to_string(e.getErrorCode()) + "): " + e.getMessage());
  } catch (const std::runtime_error &) {
    // Every DTWC++ error (the SolverError above, fast_pam's validation) passes unchanged, never rewrapped below.
    throw;
  } catch (const std::bad_alloc &) {
    // Memory the system refused is no solver failure: it leaves as std::bad_alloc, as every refused allocation does.
    throw;
  } catch (...) {
    throw SolverError("Unknown exception during Gurobi optimisation.");
  }
#else
  throw SolverError(
    "Gurobi solver is unavailable; rebuild with -DDTWC_ENABLE_GUROBI=ON");
#endif
}

} // namespace dtwc
