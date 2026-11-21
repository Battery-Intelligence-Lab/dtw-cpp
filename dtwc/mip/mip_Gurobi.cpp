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
#include "../base/error.hpp" // for SolverError
#include "../base/settings.hpp"
#include "../types/types.hpp" // for Range


#include <algorithm> // for std::max
#include <cstddef>   // for std::size_t
#include <iostream>  // for std::cout
#include <limits>    // for std::numeric_limits
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

  try {
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // GRBModel::addVars takes an `int` count: an N*N above INT_MAX would narrow
    // into a wrong-sized (or negative) model.
    const auto n_uz = static_cast<std::size_t>(Nb);
    if (n_uz > static_cast<std::size_t>(std::numeric_limits<int>::max()) / std::max<std::size_t>(n_uz, 1))
      throw SolverError("Gurobi: the compact p-median model for N = " + std::to_string(Nb) + " has more than INT_MAX variables; use Method::LRCore.");

    // Create variables
    std::unique_ptr<GRBVar[]> w{ model.addVars(static_cast<int>(n_uz * n_uz), GRB_BINARY) };

    // Branch on medoid selection first — once A[i,i] is fixed,
    // assignment is a transportation problem (TU) and LP-integral.
    for (std::size_t i = 0; i < n_uz; ++i)
      w[i * (n_uz + 1)].set(GRB_IntAttr_BranchPriority, 100);

    for (auto i : Range(Nb)) {
      GRBLinExpr lhs = 0;
      for (auto j : Range(Nb))
        lhs += w[j + i * Nb];

      model.addConstr(lhs, '=', 1.0);
    }


    for (auto j : Range(Nb))
      for (auto i : Range(Nb))
        model.addConstr(w[i + j * Nb] <= w[i * (Nb + 1)]);

    {
      GRBLinExpr lhs = 0;
      for (auto i : Range(Nb))
        lhs += w[i * (Nb + 1)];

      model.addConstr(lhs == Nc); // There should be Nc clusters.
    }

    prob.fill_distance_matrix(); // We need full distance matrix before MIP clustering.
    const double max_distance = prob.max_distance();
    const double scaling_factor = max_distance > 0.0 ? max_distance / 2.0 : 1.0; // the largest cost is 2 at any distance scale
    // Set objective
    GRBLinExpr obj = 0;
    for (auto j : Range(Nb))
      for (auto i : Range(Nb))
        obj += w[i + j * Nb] * prob.dist_by_ind(static_cast<index_t>(i), static_cast<index_t>(j)) / scaling_factor;

    model.setObjective(obj, GRB_MINIMIZE);

    // Solver tuning from MIPSettings
    model.set(GRB_IntParam_NumericFocus, prob.mip_settings.numeric_focus);
    model.set(GRB_IntParam_MIPFocus, prob.mip_settings.mip_focus);
    model.set(GRB_DoubleParam_MIPGap, prob.mip_settings.mip_gap);
    if (prob.mip_settings.time_limit_sec > 0)
      model.set(GRB_DoubleParam_TimeLimit, static_cast<double>(prob.mip_settings.time_limit_sec));
    if (!prob.mip_settings.verbose_solver)
      model.set(GRB_IntParam_OutputFlag, 0);

    // Warm start: run FastPAM and feed solution as MIP start (without mutating prob on failure)
    if (prob.mip_settings.warm_start) {
      const auto pam_result = fast_pam_solve(prob, Nc, settings::DEFAULT_MAX_ITER, prob.random_seed());

      for (std::size_t idx = 0; idx < n_uz * n_uz; ++idx)
        w[idx].set(GRB_DoubleAttr_Start, 0.0);

      for (index_t med : pam_result.medoid_indices)
        w[static_cast<std::size_t>(med) * (n_uz + 1)].set(GRB_DoubleAttr_Start, 1.0);

      // Gurobi indexing: A[i,j] at flat index i + j * Nb (column-major)
      for (std::size_t j = 0; j < n_uz; ++j) {
        const index_t med = pam_result.medoid_indices[static_cast<std::size_t>(pam_result.labels[j])];
        w[static_cast<std::size_t>(med) + j * n_uz].set(GRB_DoubleAttr_Start, 1.0);
      }
    }

    if (prob.mip_settings.verbose_solver || prob.verbose())
      std::cout << "Finished setting up the MILP problem." << '\n';

    model.optimize();

    // Verify optimality before reading GRB_DoubleAttr_X; non-optimal solves have no primal solution.
    const int opt_status = model.get(GRB_IntAttr_Status);
    if (opt_status != GRB_OPTIMAL)
      throw SolverError(
        "Gurobi MIP did not solve to optimality (status code "
        + std::to_string(opt_status) + "). No valid clustering produced.");

    std::vector<double> solution(n_uz * n_uz);
    for (std::size_t index = 0; index < solution.size(); ++index)
      solution[index] = w[index].get(GRB_DoubleAttr_X);

    prob.set_result(mip::decode_assignment(solution, Nb, Nc, true, "Gurobi"));

  } catch (GRBException &e) {
    // Wrap solver-native failures as SolverError. GRBException derives from
    // std::runtime_error in this Gurobi version, so this handler MUST precede the
    // std::runtime_error handler below — otherwise it is shadowed and never runs.
    throw SolverError("Gurobi optimisation failed (error code "
                      + std::to_string(e.getErrorCode()) + "): " + e.getMessage());
  } catch (const std::runtime_error &) {
    // Propagate our own SolverError (see above) — and any other std::runtime_error
    // raised inside the try (e.g. fast_pam validation) — unchanged; never swallow it.
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
