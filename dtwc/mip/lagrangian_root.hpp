/**
 * @file lagrangian_root.hpp
 * @brief Lagrangian root bound for the p-median (k-medoids) problem — the bound
 *        engine of the "LR-core" exact solver (PLAN.md Phase 4, Task 4.1).
 *
 * @details Dualizing the assignment equalities `Σ_i x_ij = 1` with multipliers
 * μ ∈ R^N decouples the p-median inner problem per candidate medoid. Define the
 * facility score
 *
 *     ρ_i(μ) = Σ_j min(0, D_ij − μ_j),
 *
 * open the k most negative scores S_k(μ), and
 *
 *     L(μ) = Σ_j μ_j + Σ_{i ∈ S_k(μ)} ρ_i(μ)
 *
 * is a valid LOWER BOUND on the optimal p-median cost for every μ. Because the
 * inner polytope's matrix (Cardinality + Linking) is totally unimodular for all
 * N (UNIMODULAR.md §8.2), Geoffrion's theorem makes max_μ L(μ) equal the full LP
 * relaxation bound — obtained matrix-free, streaming D once per iteration, never
 * forming the N²-column LP. We maximize L by subgradient ascent with Polyak
 * steps off a heuristic upper bound (FastPAM), repair a feasible primal each
 * iteration (assign to nearest open medoid), and report the gap. On clustered /
 * real data the root typically certifies the heuristic optimal (prediction P1).
 *
 * Full derivation and registered predictions: `.claude/UNIMODULAR.md` §8.3–8.5.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#pragma once

#include <vector>

namespace dtwc {
class Problem;
}

namespace dtwc::mip {

/// @brief Tuning knobs for the subgradient ascent (safe defaults).
struct LagrangianParams
{
  int max_iters = 4000;        ///< Subgradient iteration cap.
  double rel_gap_tol = 1e-6;   ///< Certify optimal when (UB − LB)/max(|UB|,ε) ≤ this.
  double lambda0 = 1.0;        ///< Initial Polyak step scale λ ∈ (0, 2] (1.0 = damped, well inside the convergent range).
  int stall_halve = 20;        ///< Halve λ after this many iters with no LB improvement.
  double lambda_min = 1e-4;    ///< Floor for λ — CLAMPED here (never frozen), so diminishing steps keep converging.
  double deflect = 1.5;        ///< CFM subgradient deflection γ ∈ [0,2) — steers the step off the previous direction to kill zig-zag (0 = plain subgradient).
  int polish_period = 16;      ///< Run the O(N²) medoid polish every this many iters (a cheap O(Nk) assignment repair still runs EVERY iter).
  int kelley_max_major = 500;  ///< Cutting-plane (Kelley) variant only: cap on major iterations (each adds one cut + re-solves the small master LP).
  long max_nodes = 2000000;    ///< Exact B&B (lagrangian_root_exact) only: cap on branch-and-bound nodes before giving up (returns best-so-far, certified_optimal=false — never silent).
};

/// @brief Result of a Lagrangian-root solve. Bounds are in RAW distance units
///        (unscaled Σ_j min_{i∈S} D_ij), so they compare directly to a
///        brute-force IP oracle and to a recomputed MIP-solution cost.
struct LagrangianResult
{
  double lower_bound = 0.0;         ///< best L(μ): a valid lower bound on the optimum.
  double upper_bound = 0.0;         ///< best primal-repair cost: a valid upper bound.
  double gap = 0.0;                 ///< (upper_bound − lower_bound) / max(|upper_bound|, ε).
  bool certified_optimal = false;   ///< gap ≤ rel_gap_tol — the primal solution is proven optimal.
  std::vector<int> medoids;         ///< k medoid point indices of the best primal (ascending).
  std::vector<int> labels;          ///< labels[j] = medoid POINT INDEX serving point j.
  std::vector<double> multipliers;  ///< μ at termination (size N).
  int iterations = 0;               ///< subgradient iterations actually run (root dual).
  int n_core = 0;                   ///< candidate medoids surviving reduced-cost fixing (≤ N) = core.size().
  std::vector<int> core;            ///< the surviving candidate facilities (Task 4.2), ascending; consumed by 4.3.
  long nodes = 0;                   ///< exact B&B nodes explored (lagrangian_root_exact); 0 for the bound-only routines.
};

/**
 * @brief Lagrangian root bound on a dense distance matrix.
 *
 * @param D          Row-major N×N distances; symmetric, `D[i*N+i] == 0`, `D ≥ 0`.
 * @param N          Number of points.
 * @param k          Number of medoids, 1 ≤ k ≤ N.
 * @param initial_ub A heuristic upper bound (e.g. FastPAM cost). If ≤ 0 the
 *                   routine bootstraps its own bound from the first primal repair.
 * @param params     Subgradient tuning.
 * @return Lower/upper bounds, gap, the best primal clustering, and n_core.
 * @throws dtwc::InvalidInput if N ≤ 0 or k ∉ [1, N].
 */
LagrangianResult lagrangian_root(const double *D, int N, int k,
                                 double initial_ub = -1.0,
                                 const LagrangianParams &params = {});

/**
 * @brief Same Lagrangian bound, solved by a Kelley CUTTING-PLANE method — the
 *        "right tool" for the concave piecewise-linear dual.
 *
 * @details Rather than stepping a subgradient (O(1/√k) convergence, stalls at
 * the non-smooth optimum), this accumulates each evaluated `(L, subgradient)`
 * as a supporting hyperplane and maximizes the polyhedral outer model exactly by
 * a small LP master over the N multipliers μ (bounded to a box) plus one scalar
 * θ. Kelley converges FINITELY on a piecewise-linear function — a handful of
 * major iterations close gaps the subgradient cannot. The master LP is tiny
 * (N+1 columns, one row per major iteration), warm-started across iterations —
 * NOT the intractable N²-column compact LP.
 *
 * Requires HiGHS for the master LP (the subgradient variant above is the
 * solver-free default). Throws dtwc::SolverError if HiGHS is not compiled in.
 *
 * @param D,N,k,initial_ub,params  As lagrangian_root; uses params.kelley_max_major.
 */
LagrangianResult lagrangian_root_kelley(const double *D, int N, int k,
                                        double initial_ub = -1.0,
                                        const LagrangianParams &params = {});

/**
 * @brief EXACT p-median solve: LR-bounded branch-and-bound on y over the core —
 *        "core Benders with y-only branching" (PLAN.md Phase 4, Task 4.3).
 *
 * @details The Lagrangian dual (Task 4.1) already equals the LP/Benders master
 * bound (Geoffrion), obtained matrix-free — so there is no N²-column master to
 * re-solve per round. This routine solves the root dual once, applies
 * reduced-cost fixing (Task 4.2) to obtain the candidate `core` and the facilities
 * proven open in every optimum, then closes any residual integrality gap by
 * branch-and-bound that branches on the open/close (y) decision of a single
 * candidate at a time. Each node's lower bound is the fixed-root-dual value
 * `Σμ* + Σ_{S}ρ*_i ≤ cost(S)` (valid for every candidate open set S), so a node
 * is pruned as soon as its bound reaches the incumbent. Because the optimum is
 * a subset of the core (reduced-cost fixing never removes an optimal medoid,
 * verified in tests), the tree is exact.
 *
 * On well-separated data the root certifies immediately and the tree is a single
 * node (the P1 regime). On the adversarial regime (uniform non-metric D, large
 * integrality gap) the tree can grow; @p params.max_nodes caps it and the routine
 * then returns the best incumbent with `certified_optimal = false` (never a silent
 * wrong answer).
 *
 * @param D,N,k,initial_ub,params  As lagrangian_root; uses params.max_nodes.
 * @return `certified_optimal = true` and `lower_bound == upper_bound == optimum`
 *         when the tree is fully explored within the node cap.
 */
LagrangianResult lagrangian_root_exact(const double *D, int N, int k,
                                       double initial_ub = -1.0,
                                       const LagrangianParams &params = {});

/**
 * @brief Lagrangian root bound for a Problem: fills the distance matrix (if
 *        needed), seeds the upper bound with the in-repo k-medoids heuristic,
 *        materializes a dense copy of D, and calls the core routine.
 * @param prob   Problem with data set; its distance matrix is filled if needed.
 * @param params Subgradient tuning.
 */
LagrangianResult lagrangian_root(Problem &prob, const LagrangianParams &params = {});

} // namespace dtwc::mip
