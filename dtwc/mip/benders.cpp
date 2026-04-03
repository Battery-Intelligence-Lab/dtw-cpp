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
#include "../Problem.hpp"
#include "../settings.hpp"
#include "../timing.hpp"
#include "../types/types.hpp" // for Range

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>

namespace dtwc {

void MIP_clustering_byBenders(Problem &prob)
{
#ifdef DTWC_ENABLE_HIGHS
  dtwc::Clock clk;
  const int Nb = prob.size();
  const int Nc = prob.cluster_size();

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
    prob.fillDistanceMatrix();
    double best_1med_cost = std::numeric_limits<double>::max();
    int best_1med = 0;
    for (int i = 0; i < Nb; ++i) {
      double cost_i = 0.0;
      for (int j = 0; j < Nb; ++j)
        cost_i += prob.distByInd(i, j);
      if (cost_i < best_1med_cost) {
        best_1med_cost = cost_i;
        best_1med = i;
      }
    }
    prob.centroids_ind = { best_1med };
    prob.clusters_ind.assign(Nb, 0);
    return;
  }

  prob.fillDistanceMatrix(); // Subproblem needs the full distance matrix.

  // --- Warm start from classic PAM ---
  std::vector<int> best_medoids;
  double best_cost = std::numeric_limits<double>::max();

  if (prob.mip_settings.warm_start) {
    auto saved_centroids = prob.centroids_ind;
    auto saved_clusters = prob.clusters_ind;
    auto saved_method = prob.method;

    prob.method = Method::Kmedoids;
    prob.N_repetition = 1;
    prob.cluster_by_kMedoidsPAM();

    best_medoids = prob.centroids_ind;
    best_cost = prob.findTotalCost();

    prob.centroids_ind = saved_centroids;
    prob.clusters_ind = saved_clusters;
    prob.method = saved_method;

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

  // Solver settings
  if (!prob.mip_settings.verbose_solver)
    highs.setOptionValue("output_flag", false);

  if (prob.mip_settings.mip_gap > 0.0)
    highs.setOptionValue("mip_rel_gap", prob.mip_settings.mip_gap);

  if (prob.mip_settings.time_limit_sec > 0)
    highs.setOptionValue("time_limit", static_cast<double>(prob.mip_settings.time_limit_sec));

  HighsStatus hs = highs.passModel(model);
  if (hs != HighsStatus::kOk) {
    std::cout << "Benders: failed to pass model to HiGHS\n";
    return;
  }

  // Warm start master with PAM solution
  if (!best_medoids.empty()) {
    HighsSolution sol;
    sol.col_value.assign(Nvar, 0.0);
    sol.value_valid = true;
    for (int med : best_medoids)
      sol.col_value[med] = 1.0;
    // Set theta_j to the PAM assignment costs
    for (int j = 0; j < Nb; ++j) {
      double min_d = std::numeric_limits<double>::max();
      for (int med : best_medoids) {
        double d = prob.distByInd(j, med);
        if (d < min_d) min_d = d;
      }
      sol.col_value[theta_base + j] = min_d;
    }
    highs.setSolution(sol);
  }

  // --- Benders iteration loop ---
  const int max_benders_iter = prob.mip_settings.max_benders_iter;
  const double abs_eps = 1e-6;

  for (int iter = 0; iter < max_benders_iter; ++iter) {
    highs.run();

    auto model_status = highs.getModelStatus();
    if (model_status != HighsModelStatus::kOptimal &&
        model_status != HighsModelStatus::kObjectiveBound &&
        model_status != HighsModelStatus::kSolutionLimit) {
      std::cout << "Benders: master not optimal at iteration " << iter
                << " (status: " << static_cast<int>(model_status) << ")\n";
      break;
    }

    const auto &sol = highs.getSolution().col_value;

    // Master objective = sum theta_j = lower bound
    double theta_sum = 0.0;
    for (int j = 0; j < Nb; ++j)
      theta_sum += sol[theta_base + j];

    // Extract medoid set
    std::vector<int> current_medoids;
    current_medoids.reserve(Nc);
    for (int i = 0; i < Nb; ++i) {
      if (sol[i] > 0.5)
        current_medoids.push_back(i);
    }

    if (static_cast<int>(current_medoids.size()) != Nc) {
      std::cout << "Benders: master returned " << current_medoids.size()
                << " medoids instead of " << Nc << " at iteration " << iter << "\n";
      break;
    }

    // --- Subproblem: assign each point to nearest medoid ---
    const int K = static_cast<int>(current_medoids.size());
    std::vector<double> nearest_dist(Nb);

    for (int p = 0; p < Nb; ++p) {
      double best_d = std::numeric_limits<double>::max();
      for (int m = 0; m < K; ++m) {
        double d = prob.distByInd(p, current_medoids[m]);
        if (d < best_d)
          best_d = d;
      }
      nearest_dist[p] = best_d;
    }

    double actual_cost = 0.0;
    for (int p = 0; p < Nb; ++p)
      actual_cost += nearest_dist[p];

    // Update incumbent
    if (actual_cost < best_cost) {
      best_cost = actual_cost;
      best_medoids = current_medoids;
    }

    // Convergence check: lower bound (theta_sum) vs upper bound (best_cost)
    const double bound_gap = best_cost - theta_sum;
    const double rel_gap = (best_cost > abs_eps) ? (bound_gap / best_cost) : bound_gap;

    if (bound_gap <= abs_eps + prob.mip_settings.mip_gap * std::abs(best_cost)) {
      std::cout << "Benders converged at iteration " << iter
                << ", cost = " << best_cost
                << ", LB = " << theta_sum
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

      for (int i = 0; i < Nb; ++i) {
        double d_ji = prob.distByInd(j, i);
        double coeff = std::max(0.0, d_nearest - d_ji);
        if (coeff > abs_eps) {
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
                << ": LB=" << theta_sum
                << " actual=" << actual_cost
                << " best=" << best_cost
                << " gap=" << rel_gap
                << " cuts=" << cuts_added << "\n";
  }

  // --- Extract final solution ---
  if (best_medoids.empty()) {
    std::cout << "Benders: no feasible solution found.\n";
    return;
  }

  prob.centroids_ind = best_medoids;
  prob.clusters_ind.resize(Nb);

  for (int j = 0; j < Nb; ++j) {
    double best_d = std::numeric_limits<double>::max();
    int best_m = 0;
    for (int mi = 0; mi < static_cast<int>(best_medoids.size()); ++mi) {
      double d = prob.distByInd(j, best_medoids[mi]);
      if (d < best_d) {
        best_d = d;
        best_m = mi;
      }
    }
    prob.clusters_ind[j] = best_m;
  }

  std::cout << "Benders decomposition complete: cost = " << best_cost
            << " (" << clk << ")\n";

#else
  std::cout << "Benders decomposition requires HiGHS. "
            << "Please rebuild with -DDTWC_ENABLE_HIGHS=ON\n";
#endif
}

} // namespace dtwc
