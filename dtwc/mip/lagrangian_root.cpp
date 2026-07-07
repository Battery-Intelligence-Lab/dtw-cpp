/**
 * @file lagrangian_root.cpp
 * @brief Implementation of the Lagrangian root bound (PLAN.md Phase 4, Task 4.1).
 *
 * @details See lagrangian_root.hpp for the derivation. The hot loop streams the
 * dense distance matrix D once per subgradient iteration:
 *   1. ρ_i(μ) = Σ_j min(0, D_ij − μ_j)            — O(N²), parallel over i;
 *   2. S_k(μ) = the k most negative ρ              — O(N) selection;
 *   3. L(μ) = Σ_j μ_j + Σ_{i∈S_k} ρ_i             — the lower bound;
 *   4. primal repair: assign each j to its nearest open medoid → upper bound;
 *   5. subgradient g_j = 1 − #{i∈S_k : D_ij < μ_j}; Polyak step μ ← μ + t·g.
 * Reduced-cost (Beasley) fixing at the end reports n_core.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include "lagrangian_root.hpp"

#include "../error.hpp"
#include "../parallelisation.hpp"
#include "../Problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace dtwc::mip {

namespace {
constexpr double kEps = 1e-12; // relative-gap denominator floor.

/// k-medoids (Lloyd) local search from a seed medoid set — the LR primal repair.
/// Assignment alone identifies the right clusters on separated data but not the
/// optimal medoid WITHIN each; alternating assign / medoid-update sweeps reach the
/// local optimum (= global optimum on well-separated clusters). O(N²) per sweep,
/// warm-started, converges in 1–2 sweeps in practice. Fills @p medoids (sorted)
/// and @p labels (medoid POINT INDEX per point); returns the raw cost.
double pmedian_local_search(const double *D, int N, int k,
                            std::vector<int> &medoids, std::vector<int> &labels,
                            int max_sweeps = 32)
{
  const std::size_t Nz = static_cast<std::size_t>(N);
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<int> cluster_of(Nz, 0); // cluster id 0..k-1 per point.
  labels.assign(Nz, medoids[0]);
  double cost = 0.0;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    // Assignment: each point to its nearest current medoid.
    cost = 0.0;
    for (int j = 0; j < N; ++j) {
      double bd = inf;
      int bc = 0;
      for (int c = 0; c < k; ++c) {
        const double d = D[static_cast<std::size_t>(medoids[static_cast<std::size_t>(c)]) * Nz
                           + static_cast<std::size_t>(j)];
        if (d < bd) { bd = d; bc = c; }
      }
      cluster_of[static_cast<std::size_t>(j)] = bc;
      labels[static_cast<std::size_t>(j)] = medoids[static_cast<std::size_t>(bc)];
      cost += bd;
    }
    // Update: each cluster's medoid = member minimizing its intra-cluster sum.
    bool changed = false;
    for (int c = 0; c < k; ++c) {
      double best_sum = inf;
      int best_m = medoids[static_cast<std::size_t>(c)];
      for (int cand = 0; cand < N; ++cand) {
        if (cluster_of[static_cast<std::size_t>(cand)] != c) continue;
        double s = 0.0;
        for (int j = 0; j < N; ++j)
          if (cluster_of[static_cast<std::size_t>(j)] == c)
            s += D[static_cast<std::size_t>(cand) * Nz + static_cast<std::size_t>(j)];
        if (s < best_sum) { best_sum = s; best_m = cand; }
      }
      if (best_m != medoids[static_cast<std::size_t>(c)]) {
        medoids[static_cast<std::size_t>(c)] = best_m;
        changed = true;
      }
    }
    if (!changed) break; // converged.
  }
  std::sort(medoids.begin(), medoids.end());
  return cost;
}
} // namespace

LagrangianResult lagrangian_root(const double *D, int N, int k,
                                 double initial_ub, const LagrangianParams &params)
{
  if (N <= 0) throw InvalidInput("lagrangian_root: N must be positive");
  if (k < 1 || k > N)
    throw InvalidInput("lagrangian_root: require 1 <= k <= N (k=" + std::to_string(k)
                       + ", N=" + std::to_string(N) + ")");

  const std::size_t Nz = static_cast<std::size_t>(N);
  const double inf = std::numeric_limits<double>::infinity();

  std::vector<double> mu(Nz, 0.0);   // Lagrange multipliers (start at 0 ⇒ L(0)=0).
  std::vector<double> rho(Nz, 0.0);   // facility scores this iteration.
  std::vector<double> g(Nz, 0.0);     // subgradient this iteration.
  std::vector<double> d(Nz, 0.0);     // deflected step direction (CFM).
  std::vector<double> d_prev(Nz, 0.0);// previous deflected direction.
  std::vector<int> idx(Nz);           // scratch for k-smallest selection.
  std::vector<int> cheap_lab(Nz, 0);  // per-iter cheap assignment labels.
  double dprev_norm2 = 0.0;
  bool have_dprev = false;

  double best_lb = -inf;
  double best_primal = inf;                              // best cost of LR's OWN primal repair.
  const double seed_ub = (initial_ub > 0.0) ? initial_ub : inf; // external heuristic UB (step target only).
  std::vector<int> best_medoids, best_labels;

  // Snapshot of (ρ, ρ_(k)) at the μ that produced best_lb — the valid triple for
  // reduced-cost fixing (LB + (ρ_i − ρ_(k)) > UB ⇒ facility i cannot be open).
  std::vector<double> rho_at_best(Nz, 0.0);
  double rhok_at_best = 0.0;

  double lambda = params.lambda0;
  int stall = 0;
  int iter = 0;
  bool certified = false;

  // Trigger the shared single-thread loudness check once (Task 3.6) and get a
  // scheduling hint; the reductions below are correct serial or parallel.
  const int chunk = omp_chunk_size(N, 8);
  (void)chunk;

  for (iter = 0; iter < params.max_iters; ++iter) {
    // (1) ρ_i(μ) = Σ_j min(0, D_ij − μ_j). Independent per i ⇒ lock-free.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
      const double *Di = D + static_cast<std::size_t>(i) * Nz;
      double s = 0.0;
      for (int j = 0; j < N; ++j) {
        const double d = Di[j] - mu[static_cast<std::size_t>(j)];
        if (d < 0.0) s += d;
      }
      rho[static_cast<std::size_t>(i)] = s;
    }

    // (2) S_k = the k facilities with the smallest (most negative) ρ.
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(),
                     [&](int a, int b) { return rho[static_cast<std::size_t>(a)]
                                              < rho[static_cast<std::size_t>(b)]; });
    const double rho_k = rho[static_cast<std::size_t>(idx[static_cast<std::size_t>(k - 1)])]; // k-th smallest.

    // (3) L(μ) = Σ_j μ_j + Σ_{i∈S_k} ρ_i — a valid lower bound for this μ.
    double sum_mu = 0.0;
    for (int j = 0; j < N; ++j) sum_mu += mu[static_cast<std::size_t>(j)];
    double sum_rho_S = 0.0;
    for (int t = 0; t < k; ++t) sum_rho_S += rho[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])];
    const double L = sum_mu + sum_rho_S;

    if (L > best_lb) {
      best_lb = L;
      rho_at_best = rho;      // snapshot for reduced-cost fixing.
      rhok_at_best = rho_k;
      stall = 0;
    } else {
      ++stall;
    }

    // S_k = idx[0..k-1] drives the bound and the subgradient (below).
    // (4a) Cheap primal EVERY iter: assign each j to its nearest facility in S_k
    // (O(Nk)). A valid upper bound that keeps the Polyak target sharp; the
    // subgradient still keys off S_k, not this repair.
    double cheap_cost = 0.0;
    for (int j = 0; j < N; ++j) {
      double bd = inf;
      int bm = idx[0];
      for (int t = 0; t < k; ++t) {
        const int m = idx[static_cast<std::size_t>(t)];
        const double dd = D[static_cast<std::size_t>(m) * Nz + static_cast<std::size_t>(j)];
        if (dd < bd) { bd = dd; bm = m; }
      }
      cheap_lab[static_cast<std::size_t>(j)] = bm;
      cheap_cost += bd;
    }
    if (cheap_cost < best_primal) {
      best_primal = cheap_cost;
      best_medoids.assign(idx.begin(), idx.begin() + k);
      std::sort(best_medoids.begin(), best_medoids.end());
      best_labels = cheap_lab;
    }

    // (4b) Full O(N²) k-medoids polish only PERIODICALLY (the assignment above
    // finds the right clusters; the polish moves each medoid to the intra-cluster
    // optimum). Throttling this is the dominant per-iter saving at large N.
    if (params.polish_period > 0 && (iter % params.polish_period) == 0) {
      std::vector<int> repair_med(idx.begin(), idx.begin() + k);
      std::vector<int> repair_lab;
      const double repair_cost = pmedian_local_search(D, N, k, repair_med, repair_lab);
      if (repair_cost < best_primal) {
        best_primal = repair_cost;
        best_medoids = std::move(repair_med);
        best_labels = std::move(repair_lab);
      }
    }

    // (6) Gap / certificate against LR's own primal (self-consistent with the
    // reported medoids). best_primal is finite from iteration 0.
    const double denom = std::max(std::abs(best_primal), kEps);
    if (best_primal - best_lb <= params.rel_gap_tol * denom) {
      certified = true;
      ++iter; // count this iteration.
      break;
    }

    // (5) Subgradient g_j = 1 − #{i∈S_k : D_ij < μ_j}.
    double gnorm2 = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : gnorm2)
#endif
    for (int j = 0; j < N; ++j) {
      int served = 0;
      const double muj = mu[static_cast<std::size_t>(j)];
      for (int t = 0; t < k; ++t) { // S_k = idx[0..k-1].
        const int m = idx[static_cast<std::size_t>(t)];
        if (D[static_cast<std::size_t>(m) * Nz + static_cast<std::size_t>(j)] < muj) ++served;
      }
      const double gj = 1.0 - static_cast<double>(served);
      g[static_cast<std::size_t>(j)] = gj;
      gnorm2 += gj * gj;
    }

    if (gnorm2 == 0.0) { // μ stationary: no ascent direction ⇒ done.
      ++iter;
      break;
    }

    if (stall >= params.stall_halve) {
      // Clamp λ at the floor — never freeze to 0. Diminishing-but-positive steps
      // keep μ moving toward the dual optimum; termination is by gap or max_iters.
      lambda = std::max(lambda * 0.5, params.lambda_min);
      stall = 0;
    }

    // (7) CFM deflection: steer the step off the previous direction to damp the
    // zig-zag that stalls a plain subgradient. β > 0 only when the new subgradient
    // conflicts with the previous direction.
    double beta = 0.0;
    if (have_dprev && params.deflect > 0.0 && dprev_norm2 > 0.0) {
      double dot = 0.0;
      for (int j = 0; j < N; ++j) dot += g[static_cast<std::size_t>(j)] * d_prev[static_cast<std::size_t>(j)];
      if (dot < 0.0) beta = -params.deflect * dot / dprev_norm2;
    }
    double dnorm2 = 0.0, dg = 0.0;
    for (int j = 0; j < N; ++j) {
      const double dj = g[static_cast<std::size_t>(j)] + beta * d_prev[static_cast<std::size_t>(j)];
      d[static_cast<std::size_t>(j)] = dj;
      dnorm2 += dj * dj;
      dg += dj * g[static_cast<std::size_t>(j)];
    }
    if (dnorm2 <= 0.0 || dg <= 0.0) { // deflection degenerate / not ascent ⇒ plain subgradient.
      d = g;
      dnorm2 = gnorm2;
    }

    // Polyak step along d toward the best available UB (≥ L ⇒ step ≥ 0).
    const double ub_step = std::min(best_primal, seed_ub);
    const double step = lambda * (ub_step - L) / dnorm2;
    for (int j = 0; j < N; ++j) mu[static_cast<std::size_t>(j)] += step * d[static_cast<std::size_t>(j)];

    d_prev = d;
    dprev_norm2 = dnorm2;
    have_dprev = true;
  }

  // Final polish: the best cheap incumbent may have appeared between polish
  // periods, so polish the best medoid set once more to guarantee a true local
  // optimum in the reported primal.
  if (!best_medoids.empty()) {
    std::vector<int> fm = best_medoids;
    std::vector<int> fl;
    const double fc = pmedian_local_search(D, N, k, fm, fl);
    if (fc < best_primal) {
      best_primal = fc;
      best_medoids = std::move(fm);
      best_labels = std::move(fl);
    }
  }

  // Reduced-cost (Beasley) fixing: facility i is fixed CLOSED when opening it
  // would push the bound past the incumbent. n_core = survivors.
  const double fix_ub = std::min(best_primal, seed_ub);
  int n_core = 0;
  for (int i = 0; i < N; ++i) {
    const double extra = rho_at_best[static_cast<std::size_t>(i)] - rhok_at_best; // ≥ 0 for i ∉ S_k.
    const bool fixed_closed = (best_lb + std::max(0.0, extra) > fix_ub);
    if (!fixed_closed) ++n_core;
  }

  LagrangianResult r;
  r.lower_bound = (best_lb == -inf) ? 0.0 : best_lb;
  r.upper_bound = best_primal;
  const double denom = std::max(std::abs(best_primal), kEps);
  r.gap = (best_primal - r.lower_bound) / denom;
  r.certified_optimal = (r.gap <= params.rel_gap_tol); // final gap between valid LB and valid UB.
  (void)certified;
  r.medoids = std::move(best_medoids);
  r.labels = std::move(best_labels);
  r.multipliers = std::move(mu);
  r.iterations = iter;
  r.n_core = n_core;
  return r;
}

LagrangianResult lagrangian_root(Problem &prob, const LagrangianParams &params)
{
  const int N = static_cast<int>(prob.size());
  const int k = static_cast<int>(prob.n_clusters());
  if (N <= 0) throw InvalidInput("lagrangian_root(Problem): no data set");

  if (!prob.is_distance_matrix_filled()) prob.fill_distance_matrix();

  // Seed the upper bound with the in-repo k-medoids heuristic (FastPAM/Lloyd).
  double ub = -1.0;
  prob.cluster_by_kmedoids_lloyd();
  if (static_cast<int>(prob.centroids_ind.size()) == k
      && static_cast<int>(prob.clusters_ind.size()) == N) {
    double c = 0.0;
    for (int j = 0; j < N; ++j) c += static_cast<double>(prob.dist_by_ind(j, prob.centroid_of(j)));
    ub = c;
  }

  // Materialize a dense row-major copy of D for the streaming solver.
  std::vector<double> D(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      D[static_cast<std::size_t>(i) * static_cast<std::size_t>(N) + static_cast<std::size_t>(j)]
        = static_cast<double>(prob.dist_by_ind(i, j));

  return lagrangian_root(D.data(), N, k, ub, params);
}

} // namespace dtwc::mip
