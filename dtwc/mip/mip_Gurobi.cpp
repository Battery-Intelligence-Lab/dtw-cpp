/**
 * @file mip_Gurobi.cpp
 * @brief Exact k-medoids p-median MIP solver backed by Gurobi.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 25 Dec 2022
 */

#include "mip.hpp"
#include "decode_assignment.hpp"
#include "../Problem.hpp"
#include "../base/error.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
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

  try {
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // GRBModel::addVars takes an `int` count: an N*N above INT_MAX would narrow
    // into a wrong-sized (or negative) model.
    const auto n_uz = static_cast<std::size_t>(Nb);
    if (n_uz > static_cast<std::size_t>(std::numeric_limits<int>::max()) / std::max<std::size_t>(n_uz, 1))
      throw SolverError("Gurobi: the compact p-median model for N = " + std::to_string(Nb) + " has more than INT_MAX variables; use Method::LRCore.");

    // The model the HiGHS backend solves: N*N binaries, x[f, p] at f*N + p, the diagonal x[f, f] opening f.
    const auto pmedian = mip::build_p_median_model(prob);
    std::unique_ptr<GRBVar[]> w{ model.addVars(pmedian.num_col, GRB_BINARY) };
    model.set(GRB_DoubleAttr_Obj, w.get(), pmedian.col_cost.data(), pmedian.num_col);

    // Branch on medoid selection first — once A[i,i] is fixed,
    // assignment is a transportation problem (TU) and LP-integral.
    for (std::size_t i = 0; i < n_uz; ++i)
      w[i * (n_uz + 1)].set(GRB_IntAttr_BranchPriority, 100);

    // The k medoids and the one medoid of each point are equalities; the others are x[f, p] - x[f, f] <= 0 (their
    // lower bound of -1 holds for any binaries).
    for (std::size_t row = 0; row < pmedian.row_upper.size(); ++row) {
      GRBLinExpr lhs = 0;
      for (auto term = static_cast<std::size_t>(pmedian.a_start[row]);
           term < static_cast<std::size_t>(pmedian.a_start[row + 1]); ++term)
        lhs += pmedian.a_value[term] * w[static_cast<std::size_t>(pmedian.a_index[term])];

      model.addConstr(lhs, pmedian.row_lower[row] == pmedian.row_upper[row] ? GRB_EQUAL : GRB_LESS_EQUAL,
                      pmedian.row_upper[row]);
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
      model.set(GRB_DoubleAttr_Start, w.get(), pmedian.start.data(), pmedian.num_col);

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

    prob.set_result(mip::decode_assignment(solution, Nb, Nc, false, "Gurobi"));

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
