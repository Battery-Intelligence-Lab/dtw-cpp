/**
 * @file reduced_cost_fixing.hpp
 * @brief Beasley reduced-cost fixing for the p-median (k-medoids) core
 *        (PLAN.md Phase 4, Task 4.2). Driven by the Task 4.1 Lagrangian dual.
 *
 * @details After the Lagrangian root produces multipliers μ*, facility scores
 * ρ_i(μ*) = Σ_j min(0, D_ij − μ_j), a valid lower bound LB = L(μ*), and a valid
 * primal upper bound UB, each candidate medoid can be tested against the
 * conditional bounds obtained by FORCING it open or closed:
 *
 *   Force facility i OPEN  (i ∉ S_k): the best dual value drops to
 *       LB + (ρ_i − ρ_(k))            (swap the k-th best for i);
 *     ρ_i ≥ ρ_(k) ⇒ this is ≥ LB. If it exceeds UB, no optimum opens i ⇒ fix i CLOSED.
 *
 *   Force facility i CLOSED (i ∈ S_k): the best dual value drops to
 *       LB + (ρ_(k+1) − ρ_i)          (drop i for the (k+1)-th best);
 *     ρ_i ≤ ρ_(k) ≤ ρ_(k+1) ⇒ this is ≥ LB. If it exceeds UB, every optimum opens i ⇒ fix i OPEN.
 *
 * where ρ_(k), ρ_(k+1) are the k-th and (k+1)-th smallest scores and S_k is the
 * set of the k smallest (the facilities the dual opens). Both bounds are exact
 * conditional Lagrangian bounds (Beasley 1993), so a fixed facility is provably
 * absent from / present in EVERY optimal solution — never a heuristic prune. A
 * facility is fixed only when its bound clears UB by a magnitude-scaled margin,
 * never on floating-point rounding (see the .cpp: a certified instance whose ρ
 * ties ρ_(k) is a legitimate alternative optimum and must survive).
 *
 * Registered prediction P2 (PLAN.md Phase 4): when the root gap ≤ 1%, this
 * fixing eliminates ≥ 80% of the candidate medoids. Test verifies every
 * fixed-out medoid is absent from the brute-force optimum (N ≤ 14).
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#pragma once

#include <vector>

namespace dtwc::mip {

/// @brief Partition of the candidate facilities {0..N-1} after reduced-cost
///        fixing. `core` are the survivors still worth branching on; the three
///        sets satisfy `core ∪ fixed_closed = {0..N-1}`, `fixed_open ⊆ core`.
struct FixingResult
{
  std::vector<int> core;         ///< survivors (facilities NOT proven closed), ascending.
  std::vector<int> fixed_closed; ///< facilities proven closed in every optimum, ascending.
  std::vector<int> fixed_open;   ///< facilities proven open in every optimum, ascending.
};

/**
 * @brief Beasley reduced-cost fixing from the Lagrangian dual state.
 *
 * @param rho          Facility scores ρ_i(μ*) at the best dual multipliers, size N.
 * @param k            Number of medoids, 1 ≤ k ≤ N.
 * @param lower_bound  LB = L(μ*), a valid lower bound (may be -inf ⇒ nothing fixed).
 * @param upper_bound  UB, a valid primal upper bound (may be +inf ⇒ nothing fixed).
 * @return core / fixed_closed / fixed_open. If LB ≥ UB is not established, or the
 *         bounds are non-finite, no facility is fixed and `core` is all of {0..N-1}.
 * @throws dtwc::InvalidInput if `rho` is empty or k ∉ [1, N].
 */
FixingResult reduced_cost_fixing(const std::vector<double> &rho, int k,
                                 double lower_bound, double upper_bound);

} // namespace dtwc::mip
