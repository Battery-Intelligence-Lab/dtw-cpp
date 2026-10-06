/**
 * @file mip.hpp
 * @brief Collecting mixed-integer program functions.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 06 Nov 2022
 */

#pragma once

#include <vector>

namespace dtwc {
class Problem;

void MIP_clustering_byGurobi(Problem &prob);
void MIP_clustering_byHiGHS(Problem &prob);

/// Runtime capability query used by bindings and artifact smoke tests.
[[nodiscard]] bool highs_solver_available() noexcept;

/// @brief LR-core EXACT clustering entry point (Method::LRCore).
/// Fills the distance matrix, seeds an upper bound from the k-medoids heuristic,
/// runs `mip::lagrangian_root_exact` on a dense copy of D, and writes the proven
/// optimal `centroids_ind` / `clusters_ind` back into @p prob. Needs no external
/// MIP solver (uses HiGHS only for a tighter Kelley root when available).
void LR_core_clustering(Problem &prob);

namespace mip {

/// The compact p-median MIP of a Problem as the arrays HiGHS takes: linked HiGHS
/// solves it in MIP_clustering_byHiGHS, Python hands it to highspy. Column f·N + p
/// is x[f, p], point p served by medoid f; the diagonal x[f, f] opens f. Row 0 is
/// Σ_f x[f, f] = k, rows 1..N are Σ_f x[f, p] = 1, then for each f the N − 1 rows
/// x[f, p] − x[f, f] in [−1, 0], p ≠ f. Every column is integer (integrality 1,
/// HiGHS's kInteger) in [0, 1] and costs D(p, f) / max(max_distance / 2, 1).
struct PMedianModel
{
  int num_col{}, num_row{};
  std::vector<double> col_cost, col_lower, col_upper, row_lower, row_upper;
  std::vector<int> a_start, a_index; ///< The constraint matrix row-wise (CSR).
  std::vector<double> a_value;
  std::vector<int> integrality;
  std::vector<double> start; ///< FastPAM's clustering as a MIP start; empty unless MIPSettings::warm_start.
};

/// Fill prob's distance matrix and build its model. HiGHS indexes with `int`, so a
/// model with more than INT_MAX nonzeros (3N² − N) is SolverError, before the fill.
PMedianModel build_p_median_model(Problem &prob);

} // namespace mip
} // namespace dtwc
