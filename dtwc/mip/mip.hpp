/**
 * @file mip.hpp
 * @brief Collecting mixed-integer program functions.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 06 Nov 2022
 */

#pragma once

namespace dtwc {
class Problem;

void MIP_clustering_byGurobi(Problem &prob);
void MIP_clustering_byHiGHS(Problem &prob);
void MIP_clustering_byBenders(Problem &prob);

/// @brief LR-core EXACT clustering entry point (Method::LRCore, Phase 4 Task 4.4).
/// Fills the distance matrix, seeds an upper bound from the k-medoids heuristic,
/// runs `mip::lagrangian_root_exact` on a dense copy of D, and writes the proven
/// optimal `centroids_ind` / `clusters_ind` back into @p prob. Needs no external
/// MIP solver (uses HiGHS only for a tighter Kelley root when available).
void LR_core_clustering(Problem &prob);

} // namespace dtwc
