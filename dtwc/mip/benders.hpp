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

#include <algorithm>

namespace dtwc {
class Problem;

namespace mip {

/**
 * @brief Absolute tolerance, in RAW distance units, for the two Benders tests
 *        that compare whole assignment costs: the UB/LB convergence test and
 *        the per-point cut-skip test.
 *
 * @details Both compare quantities of the order of a cost, so an absolute 1e-6
 * means "exact" on DTW distances of 1e-3 and "anything goes" on 1e7. Scale it
 * by the same conditioning factor the compact HiGHS backend divides its
 * objective by (`max(max_distance()/2, 1)`, mip_Highs.cpp) so the two backends
 * accept the same RELATIVE violation. The compact backend has no 1e-6 tolerance
 * of its own — only this factor is shared, not a tolerance.
 *
 * It must NOT be used to filter cut COEFFICIENTS; see
 * benders_cut_coefficient_threshold.
 */
[[nodiscard]] inline double benders_abs_eps(double max_distance) noexcept
{
  return 1e-6 * std::max(max_distance / 2.0, 1.0);
}

/**
 * @brief Sparsity threshold for the coefficients of ONE disaggregated Benders cut
 *        `theta_j + sum_i c_i y_i >= d_nearest`, where `c_i = max(0, d_nearest - d_ji) >= 0`.
 *
 * @details Every `c_i` is non-negative by construction, so dropping a positive
 * one SHRINKS the cut's left-hand side and makes the constraint STRICTER than
 * the valid Benders cut: the master's dual bound is then inflated and the true
 * optimum can be cut off while the loop reports a proved optimum. The threshold
 * is therefore relative to this cut's own right-hand side and small enough to
 * drop only exact zeros and rounding dust — never a scaled cost tolerance.
 */
[[nodiscard]] inline double benders_cut_coefficient_threshold(double d_nearest) noexcept
{
  return 1e-12 * std::max(1.0, d_nearest);
}

/**
 * @brief The Benders master's lower bound: the solver's DUAL bound.
 *
 * @details Not the master's incumbent objective (`sum_j theta_j`), which is only
 * an UPPER bound on the master optimum — using it declared convergence while a
 * strictly better medoid set still existed. Templated on the solver type so this
 * header pulls in no HiGHS declaration (HiGHS is a PRIVATE dependency of the
 * mip-solvers target and is not on the public include path).
 */
template <typename MasterSolver>
[[nodiscard]] inline double benders_master_lower_bound(const MasterSolver &master)
{
  return master.getInfo().mip_dual_bound;
}

} // namespace mip

/**
 * @brief Solve the p-median clustering problem via Benders decomposition.
 *
 * @details The master problem selects k medoids (N binary variables + 1
 * continuous theta). The subproblem assigns each point to its nearest open
 * medoid and generates optimality cuts. Warm-started with the classic
 * PAM heuristic.
 *
 * Requires HiGHS: throws dtwc::SolverError if HiGHS is not compiled in, and
 * also if the cut loop reaches its iteration cap without closing the bound gap
 * (Method::MIP is exact — an unconverged incumbent is never published).
 *
 * @param prob Problem instance with filled or fillable distance matrix.
 */
void MIP_clustering_byBenders(Problem &prob);

} // namespace dtwc
