/**
 * @file benders.cpp
 * @brief Benders decomposition for p-median MIP clustering using HiGHS.
 *
 * @details Implements a disaggregated Benders cut loop for the p-median
 * (k-medoids) problem:
 *   1. Warm-start with PAM heuristic for an upper bound.
 *   2. Solve a master MIP (N binary medoid vars + N continuous cost vars).
 *   3. Solve the assignment subproblem (nearest-medoid for each point).
 *   4. If lower bound >= upper bound - epsilon, stop (optimal).
 *   5. Otherwise, add N disaggregated optimality cuts and repeat.
 *
 * The disaggregated cuts are stronger than a single aggregated cut and
 * ensure finite convergence.
 *
 * Reference:
 * - Benders (1962), "Partitioning procedures for solving mixed-variables
 *   programming problems", Numerische Mathematik 4(1), 238-252.
 * - Magnanti & Wong (1981), "Accelerating Benders decomposition:
 *   algorithmic enhancement and model selection criteria", Operations
 *   Research 29(3), 464-484.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include "benders.hpp"
#include "mip.hpp"
#include "highs_support.hpp"
#include "nearest_medoid.hpp"
#include "solution_transaction.hpp"
#include "../core/clustering_result.hpp"
#include "../Problem.hpp"
#include "../error.hpp"
#include "../settings.hpp"
#include "../timing.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

namespace dtwc {

namespace {

  template <typename Callback>
  class ScopeExit
  {
  public:
    explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}

    ScopeExit(const ScopeExit &) = delete;
    ScopeExit &operator=(const ScopeExit &) = delete;

    ~ScopeExit() noexcept { callback_(); }

  private:
    Callback callback_;
  };

  template <typename Callback>
  ScopeExit(Callback) -> ScopeExit<Callback>;

} // namespace

void MIP_clustering_byBenders(Problem &prob)
{
  (void)prob;
#ifdef DTWC_ENABLE_HIGHS
  dtwc::Clock clk;
  const int Nb = static_cast<int>(prob.size());
  const int Nc = prob.n_clusters();

  if (Nb <= 0 || Nc <= 0 || Nc > Nb) {
    std::cout << "Benders: invalid problem size (N=" << Nb << ", k=" << Nc << ")\n";
    return;
  }

  // Trivial case: k == N, every point is its own medoid
  if (Nc == Nb) {
    prob.centroids_ind.resize(Nb);
    std::iota(prob.centroids_ind.begin(), prob.centroids_ind.end(), 0);
    prob.clusters_ind.resize(Nb);
    std::iota(prob.clusters_ind.begin(), prob.clusters_ind.end(), 0);
    return;
  }

  // Trivial case: k == 1, find the 1-medoid minimizing total cost
  if (Nc == 1) {
    prob.fill_distance_matrix();
    double best_1med_cost = std::numeric_limits<double>::max();
    int best_1med = 0;
    for (int i = 0; i < Nb; ++i) {
      double cost_i = 0.0;
      for (int j = 0; j < Nb; ++j)
        cost_i += prob.dist_by_ind(i, j);
      if (cost_i < best_1med_cost) {
        best_1med_cost = cost_i;
        best_1med = i;
      }
    }
    prob.centroids_ind = { best_1med };
    prob.clusters_ind.assign(Nb, 0);
    return;
  }

  // Everything below decodes into private vectors and publishes atomically, the
  // same contract the compact HiGHS/Gurobi backends use. Constructed AFTER the
  // trivial-k branches so their direct writes are not rolled back.
  mip::ExactClusteringTransaction result_transaction(prob);

  prob.fill_distance_matrix(); // Subproblem needs the full distance matrix.

  // --- Warm start from classic PAM ---
  std::vector<int> best_medoids;
  double best_cost = std::numeric_limits<double>::max();

  if (prob.mip_settings.warm_start) {
    // NOT mip::make_warm_start: this nested Lloyd must also restore method_,
    // n_repetitions and last_iterations_, which the exact-clustering transaction
    // does not own. The scoped restore below covers all five fields.
    {
      const Method caller_method = prob.method_;
      const int caller_n_repetitions = prob.n_repetitions();
      const int caller_last_iterations = prob.last_iterations_;
      auto caller_centroids = prob.centroids_ind;
      auto caller_clusters = prob.clusters_ind;
      ScopeExit restore_caller_state(
        [&prob,
         method = caller_method,
         n_repetitions = caller_n_repetitions,
         last_iterations = caller_last_iterations,
         centroids = std::move(caller_centroids),
         clusters = std::move(caller_clusters)]() mutable noexcept {
          prob.method_ = method;
          prob.set_n_repetitions(n_repetitions);
          prob.last_iterations_ = last_iterations;
          prob.centroids_ind.swap(centroids);
          prob.clusters_ind.swap(clusters);
        });
      prob.method_ = Method::Kmedoids;
      prob.set_n_repetitions(1);
      prob.cluster_by_kmedoids_lloyd_impl(false);

      best_medoids = prob.centroids_ind;
      best_cost = prob.find_total_cost();
    }

    std::cout << "Benders warm start: PAM cost = " << best_cost << "\n";
  }

  // --- Build master problem (disaggregated formulation) ---
  //
  // Variables:
  //   y_0 .. y_{N-1}         : binary, is point i a medoid?
  //   theta_0 .. theta_{N-1} : continuous, assignment cost of point j
  //
  // Objective: minimize sum_j theta_j
  //
  // Constraints:
  //   (0) sum_i y_i = k                     [cardinality]
  //   Benders cuts added iteratively:
  //   For each point j and iteration t with medoid set S^t:
  //     theta_j >= d(j, sigma^t(j)) - sum_{i in S^t} max(0, d(j, sigma^t(j)) - d(j,i)) * y_i
  //
  // This is the standard disaggregated Benders for uncapacitated facility
  // location / p-median.

  const int Nvar = 2 * Nb;            // N binary y + N continuous theta
  const int theta_base = Nb;          // theta_j at index Nb + j

  Highs highs;
  HighsModel model;

  model.lp_.num_col_ = Nvar;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.offset_ = 0;

  // Objective: minimize sum theta_j
  model.lp_.col_cost_.assign(Nvar, 0.0);
  for (int j = 0; j < Nb; ++j)
    model.lp_.col_cost_[theta_base + j] = 1.0;

  // Variable bounds
  model.lp_.col_lower_.assign(Nvar, 0.0);
  model.lp_.col_upper_.assign(Nvar, 1.0);
  for (int j = 0; j < Nb; ++j)
    model.lp_.col_upper_[theta_base + j] = 1e20; // theta_j unbounded above

  // Integrality: y_i binary, theta_j continuous
  model.lp_.integrality_.assign(Nvar, HighsVarType::kInteger);
  for (int j = 0; j < Nb; ++j)
    model.lp_.integrality_[theta_base + j] = HighsVarType::kContinuous;

  // Initial constraint: sum(y_i) = k
  model.lp_.num_row_ = 1;
  model.lp_.row_lower_ = { static_cast<double>(Nc) };
  model.lp_.row_upper_ = { static_cast<double>(Nc) };

  // Constraint matrix (sparse, column-wise)
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.start_.resize(Nvar + 1);
  for (int i = 0; i <= Nb; ++i)
    model.lp_.a_matrix_.start_[i] = i;       // each y_i: 1 entry in row 0
  for (int j = 0; j <= Nb; ++j)
    model.lp_.a_matrix_.start_[Nb + j] = Nb; // theta_j: 0 entries initially

  model.lp_.a_matrix_.index_.assign(Nb, 0);  // all y_i in row 0
  model.lp_.a_matrix_.value_.assign(Nb, 1.0);

  // Solver settings. A rejected option must never leave HiGHS on its defaults.
  if (!prob.mip_settings.verbose_solver)
    mip::set_highs_option(highs, "output_flag", false, "Benders");

  if (prob.mip_settings.mip_gap > 0.0)
    mip::set_highs_option(highs, "mip_rel_gap", prob.mip_settings.mip_gap, "Benders");

  if (prob.mip_settings.time_limit_sec > 0)
    mip::set_highs_option(highs, "time_limit",
                          static_cast<double>(prob.mip_settings.time_limit_sec), "Benders");

  HighsStatus hs = highs.passModel(model);
  if (hs != HighsStatus::kOk)
    throw SolverError("Benders: HiGHS rejected the master model (passModel returned status "
                      + std::to_string(static_cast<int>(hs)) + ").");

  // Warm start master with the PAM solution.
  if (!best_medoids.empty()) {
    // Cardinality, not just range: a short medoid set otherwise runs the whole
    // cut loop and only fails inside publish(), naming the wrong stage.
    if (static_cast<int>(best_medoids.size()) != Nc)
      throw SolverError("Benders: the warm start produced "
                        + std::to_string(best_medoids.size())
                        + " medoids but k = " + std::to_string(Nc) + ".");
    HighsSolution sol;
    sol.col_value.assign(Nvar, 0.0);
    sol.value_valid = true;
    for (int med : best_medoids) {
      if (med < 0 || med >= Nb)
        throw SolverError("Benders: warm-start medoid index " + std::to_string(med)
                          + " is outside [0, N).");
      sol.col_value[med] = 1.0;
    }
    // Set theta_j to the PAM assignment costs.
    for (int j = 0; j < Nb; ++j)
      sol.col_value[theta_base + j] =
        mip::nearest_medoid(Nc, [&](int t) { return prob.dist_by_ind(j, best_medoids[static_cast<std::size_t>(t)]); })
          .distance;
    highs.setSolution(sol);
  }

  // --- Benders iteration loop ---
  const int max_benders_iter = prob.mip_settings.max_benders_iter;
  // Tolerance for the two COST comparisons only (convergence and cut-skip); the
  // cut coefficients use benders_cut_coefficient_threshold, which must stay tiny.
  const double abs_eps = mip::benders_abs_eps(prob.max_distance());

  bool converged = false;
  std::string failure_reason;
  int iter = 0;

  for (iter = 0; iter < max_benders_iter; ++iter) {
    highs.run();

    auto model_status = highs.getModelStatus();
    if (model_status != HighsModelStatus::kOptimal) {
      failure_reason = "the master MIP did not solve to optimality at iteration "
                     + std::to_string(iter) + " (model status: "
                     + highs.modelStatusToString(model_status) + ")";
      break;
    }

    const auto &sol = highs.getSolution().col_value;

    const double master_lb = mip::benders_master_lower_bound(highs);

    // Extract medoid set
    std::vector<int> current_medoids;
    current_medoids.reserve(Nc);
    for (int i = 0; i < Nb; ++i) {
      if (sol[i] > 0.5)
        current_medoids.push_back(i);
    }

    if (static_cast<int>(current_medoids.size()) != Nc) {
      failure_reason = "the master returned " + std::to_string(current_medoids.size())
                     + " medoids instead of " + std::to_string(Nc)
                     + " at iteration " + std::to_string(iter);
      break;
    }

    // --- Subproblem: assign each point to nearest medoid ---
    const int K = static_cast<int>(current_medoids.size());
    std::vector<double> nearest_dist(Nb);

    for (int p = 0; p < Nb; ++p)
      nearest_dist[p] =
        mip::nearest_medoid(K, [&](int t) { return prob.dist_by_ind(p, current_medoids[static_cast<std::size_t>(t)]); })
          .distance;

    double actual_cost = 0.0;
    for (int p = 0; p < Nb; ++p)
      actual_cost += nearest_dist[p];

    // Update incumbent
    if (actual_cost < best_cost) {
      best_cost = actual_cost;
      best_medoids = current_medoids;
    }

    // Convergence check: master dual bound (LB) vs incumbent (UB).
    const double bound_gap = best_cost - master_lb;
    const double rel_gap = (best_cost > abs_eps) ? (bound_gap / best_cost) : bound_gap;

    if (bound_gap <= abs_eps + prob.mip_settings.mip_gap * std::abs(best_cost)) {
      converged = true;
      std::cout << "Benders converged at iteration " << iter
                << ", cost = " << best_cost
                << ", LB = " << master_lb
                << ", gap = " << rel_gap << "\n";
      break;
    }

    // --- Generate disaggregated Benders cuts ---
    // For each point j:
    //   Let sigma(j) = nearest open medoid in current set S.
    //   Cut: theta_j >= d(j, sigma(j))
    //                    - sum_{i=0}^{N-1} max(0, d(j,sigma(j)) - d(j,i)) * y_i
    //
    // This cut involves ALL potential facility locations (all N points),
    // not just the current medoid set. It states that opening any facility
    // closer to j than sigma(j) would reduce the assignment cost of j.
    //
    // Rearranged for addRow:
    //   theta_j + sum_i max(0, d_nearest - d(j,i)) * y_i >= d_nearest
    //
    // Only add a cut for point j if theta_j underestimates the true cost.

    int cuts_added = 0;
    for (int j = 0; j < Nb; ++j) {
      double theta_j = sol[theta_base + j];
      double d_nearest = nearest_dist[j];

      // Only add cut if theta_j underestimates
      if (d_nearest <= theta_j + abs_eps)
        continue;

      // Build cut: theta_j + sum_i coeff_i * y_i >= d_nearest
      // where coeff_i = max(0, d_nearest - d(j, i))
      std::vector<HighsInt> cut_idx;
      std::vector<double> cut_val;
      cut_idx.reserve(Nb + 1);
      cut_val.reserve(Nb + 1);

      // NOT abs_eps: every coeff is >= 0, so dropping a positive one makes the
      // cut STRICTER than the valid Benders cut and can remove the optimum.
      const double coeff_floor = mip::benders_cut_coefficient_threshold(d_nearest);
      for (int i = 0; i < Nb; ++i) {
        double d_ji = prob.dist_by_ind(j, i);
        double coeff = std::max(0.0, d_nearest - d_ji);
        if (coeff > coeff_floor) {
          cut_idx.push_back(static_cast<HighsInt>(i));
          cut_val.push_back(coeff);
        }
      }
      cut_idx.push_back(static_cast<HighsInt>(theta_base + j));
      cut_val.push_back(1.0);

      highs.addRow(d_nearest, 1e20,
                   static_cast<HighsInt>(cut_idx.size()),
                   cut_idx.data(), cut_val.data());
      ++cuts_added;
    }

    if (prob.mip_settings.verbose_solver)
      std::cout << "Benders iter " << iter
                << ": LB=" << master_lb
                << " actual=" << actual_cost
                << " best=" << best_cost
                << " gap=" << rel_gap
                << " cuts=" << cuts_added << "\n";
  }

  // --- Convergence contract ---
  // Method::MIP is the EXACT route. An answer produced by an exhausted cut loop
  // (or by a master that stopped early) is a heuristic of unknown quality; the
  // compact HiGHS and Gurobi backends both throw SolverError in the same
  // situation, so this one must too rather than print "complete".
  if (!converged) {
    if (failure_reason.empty())
      failure_reason = "the cut loop hit its iteration cap ("
                     + std::to_string(max_benders_iter)
                     + ") before the bound gap closed";
    throw SolverError("Benders decomposition did not prove optimality: "
                      + failure_reason
                      + ". Raise Problem::mip_settings.max_benders_iter (currently "
                      + std::to_string(max_benders_iter)
                      + "), relax Problem::mip_settings.mip_gap (currently "
                      + std::to_string(prob.mip_settings.mip_gap)
                      + "), set Problem::mip_settings.benders = \"off\" for the compact "
                        "MIP backend, or use Method::Kmedoids for a heuristic answer.");
  }

  if (best_medoids.empty())
    throw SolverError("Benders decomposition converged without a feasible medoid set.");

  const int Kb = static_cast<int>(best_medoids.size());
  core::ClusteringResult result;
  result.medoid_indices = best_medoids;
  result.labels.resize(static_cast<std::size_t>(Nb));
  for (int j = 0; j < Nb; ++j)
    result.labels[static_cast<std::size_t>(j)] =
      mip::nearest_medoid(Kb, [&](int t) { return prob.dist_by_ind(j, best_medoids[static_cast<std::size_t>(t)]); })
        .position;

  result_transaction.publish(std::move(result), "Benders");

  std::cout << "Benders decomposition complete: cost = " << best_cost
            << " (" << clk << ")\n";

#else
  throw SolverError(
      "Benders decomposition requires HiGHS; rebuild with -DDTWC_ENABLE_HIGHS=ON");
#endif
}

} // namespace dtwc
