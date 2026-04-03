/**
 * @file benders.hpp
 * @brief Benders decomposition for p-median MIP clustering.
 *
 * @details Provides an exact k-medoids solver that scales to N > 200 by
 * decomposing the compact N^2-variable MIP into a master problem with N
 * binary variables and an O(Nk) assignment subproblem.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

namespace dtwc {
class Problem;

/**
 * @brief Solve the p-median clustering problem via Benders decomposition.
 *
 * @details The master problem selects k medoids (N binary variables + 1
 * continuous theta). The subproblem assigns each point to its nearest open
 * medoid and generates optimality cuts. Warm-started with the classic
 * PAM heuristic.
 *
 * Requires HiGHS; prints a diagnostic and returns without modifying
 * @p prob if HiGHS is not compiled in.
 *
 * @param prob Problem instance with filled or fillable distance matrix.
 */
void MIP_clustering_byBenders(Problem &prob);

} // namespace dtwc
