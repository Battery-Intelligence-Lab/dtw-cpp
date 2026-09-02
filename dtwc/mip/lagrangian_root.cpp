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

#include "nearest_medoid.hpp"
#include "reduced_cost_fixing.hpp"
#include "solution_transaction.hpp"

#include "../core/clustering_result.hpp"
#include "../error.hpp"
#include "../parallelisation.hpp"
#include "../Problem.hpp"

#ifdef DTWC_ENABLE_HIGHS
#include <Highs.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
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

  // Assignment sweep: each point to its nearest current medoid. Fills cost,
  // labels and cluster_of FROM the current @p medoids, so cost/labels always
  // describe the medoid set the caller receives.
  auto assign = [&]() {
    cost = 0.0;
    for (int j = 0; j < N; ++j) {
      const auto nearest = mip::nearest_medoid(k, [&](int c) {
        return D[static_cast<std::size_t>(medoids[static_cast<std::size_t>(c)]) * Nz
                 + static_cast<std::size_t>(j)];
      });
      cluster_of[static_cast<std::size_t>(j)] = nearest.position;
      labels[static_cast<std::size_t>(j)] = medoids[static_cast<std::size_t>(nearest.position)];
      cost += nearest.distance;
    }
  };

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    assign();
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
  // Sort FIRST, then assign once, so cost/labels/cluster_of describe the final,
  // sorted medoid set even when the loop exits by exhausting max_sweeps. The
  // extra O(N·k) sweep can only lower the cost — Lloyd is monotone.
  std::sort(medoids.begin(), medoids.end());
  assign();
  return cost;
}

/// Evaluate the Lagrangian dual at μ: fills @p rho (facility scores), @p g (a
/// subgradient of L), and @p idx (idx[0..k-1] = S_k, the k smallest ρ); sets
/// @p rho_k (the k-th smallest ρ) and returns L(μ). Shared by the subgradient
/// and cutting-plane solvers so the "dual oracle" lives in exactly one place.
double evaluate_dual(const double *D, int N, std::size_t Nz, int k,
                     const std::vector<double> &mu, std::vector<double> &rho,
                     std::vector<double> &g, std::vector<int> &idx, double &rho_k)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; ++i) {
    const double *Di = D + static_cast<std::size_t>(i) * Nz;
    double s = 0.0;
    for (int j = 0; j < N; ++j) {
      const double dd = Di[j] - mu[static_cast<std::size_t>(j)];
      if (dd < 0.0) s += dd;
    }
    rho[static_cast<std::size_t>(i)] = s;
  }

  std::iota(idx.begin(), idx.end(), 0);
  std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(),
                   [&](int a, int b) { return rho[static_cast<std::size_t>(a)]
                                            < rho[static_cast<std::size_t>(b)]; });
  rho_k = rho[static_cast<std::size_t>(idx[static_cast<std::size_t>(k - 1)])];

  double sum_mu = 0.0;
  for (int j = 0; j < N; ++j) sum_mu += mu[static_cast<std::size_t>(j)];
  double sum_rho_S = 0.0;
  for (int t = 0; t < k; ++t) sum_rho_S += rho[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)])];
  const double L = sum_mu + sum_rho_S;

  // g_j = 1 − #{i ∈ S_k : D_ij < μ_j}  (a subgradient of the concave L at μ).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int j = 0; j < N; ++j) {
    int served = 0;
    const double muj = mu[static_cast<std::size_t>(j)];
    for (int t = 0; t < k; ++t) {
      const int m = idx[static_cast<std::size_t>(t)];
      if (D[static_cast<std::size_t>(m) * Nz + static_cast<std::size_t>(j)] < muj) ++served;
    }
    g[static_cast<std::size_t>(j)] = 1.0 - static_cast<double>(served);
  }
  return L;
}

/// Update the primal incumbent from S_k (idx[0..k-1]): a cheap O(Nk) nearest
/// assignment always; a full O(N²) medoid polish when @p do_polish. Mutates
/// best_* only on strict improvement.
void try_primal(const double *D, int N, std::size_t Nz, int k,
                const std::vector<int> &idx, bool do_polish, double &best_primal,
                std::vector<int> &best_medoids, std::vector<int> &best_labels,
                std::vector<int> &cheap_lab)
{
  double cheap_cost = 0.0;
  for (int j = 0; j < N; ++j) {
    const auto nearest = mip::nearest_medoid(k, [&](int t) {
      return D[static_cast<std::size_t>(idx[static_cast<std::size_t>(t)]) * Nz
               + static_cast<std::size_t>(j)];
    });
    cheap_lab[static_cast<std::size_t>(j)] = idx[static_cast<std::size_t>(nearest.position)];
    cheap_cost += nearest.distance;
  }
  if (cheap_cost < best_primal) {
    best_primal = cheap_cost;
    best_medoids.assign(idx.begin(), idx.begin() + k);
    std::sort(best_medoids.begin(), best_medoids.end());
    best_labels = cheap_lab;
  }
  if (do_polish) {
    std::vector<int> rm(idx.begin(), idx.begin() + k), rl;
    const double rc = pmedian_local_search(D, N, k, rm, rl);
    if (rc < best_primal) {
      best_primal = rc;
      best_medoids = std::move(rm);
      best_labels = std::move(rl);
    }
  }
}

/// Shared tail for both solvers: one final polish (guarantees a true local
/// optimum), Beasley reduced-cost fixing (n_core), and result assembly.
LagrangianResult finalize(const double *D, int N, int k, double best_lb,
                          double best_primal, double seed_ub,
                          std::vector<int> best_medoids, std::vector<int> best_labels,
                          std::vector<double> mu, const std::vector<double> &rho_at_best,
                          int iterations, double rel_gap_tol)
{
  const double inf = std::numeric_limits<double>::infinity();
  if (!best_medoids.empty()) {
    std::vector<int> fm = best_medoids, fl;
    const double fc = pmedian_local_search(D, N, k, fm, fl);
    if (fc < best_primal) {
      best_primal = fc;
      best_medoids = std::move(fm);
      best_labels = std::move(fl);
    }
  }
  const double fix_ub = std::min(best_primal, seed_ub);
  // Beasley reduced-cost fixing on the dual state at best_lb (Task 4.2). Uses the
  // RAW best_lb (not the display-clamped value below): with best_lb = -inf the
  // module correctly fixes nothing. rho_at_best/k determine S_k internally.
  FixingResult fix = reduced_cost_fixing(rho_at_best, k, best_lb, fix_ub);

  LagrangianResult r;
  r.lower_bound = (best_lb == -inf) ? 0.0 : best_lb;
  r.upper_bound = best_primal;
  const double denom = std::max(std::abs(best_primal), kEps);
  r.gap = (best_primal - r.lower_bound) / denom;
  r.certified_optimal = (r.gap <= rel_gap_tol);
  r.medoids = std::move(best_medoids);
  r.labels = std::move(best_labels);
  r.multipliers = std::move(mu);
  r.iterations = iterations;
  r.core = std::move(fix.core);
  r.n_core = static_cast<int>(r.core.size());
  return r;
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

  // Snapshot of ρ at the μ that produced best_lb — the state reduced-cost fixing
  // consumes (LB + (ρ_i − ρ_(k)) > UB ⇒ facility i cannot be open; Task 4.2).
  std::vector<double> rho_at_best(Nz, 0.0);

  double lambda = params.lambda0;
  int stall = 0;
  int iter = 0;

  // Trigger the shared single-thread loudness check once (Task 3.6) and get a
  // scheduling hint; the reductions below are correct serial or parallel.
  const int chunk = omp_chunk_size(N, 8);
  (void)chunk;

  for (iter = 0; iter < params.max_iters; ++iter) {
    // (1)–(3),(5) Dual oracle: L(μ), subgradient g, and S_k (idx[0..k-1]).
    double rho_k = 0.0;
    const double L = evaluate_dual(D, N, Nz, k, mu, rho, g, idx, rho_k);

    if (L > best_lb) {
      best_lb = L;
      rho_at_best = rho;      // snapshot for reduced-cost fixing.
      stall = 0;
    } else {
      ++stall;
    }

    // (4) Primal repair from S_k: cheap assignment every iter, full polish
    // periodically (throttling the O(N²) polish is the dominant large-N saving).
    try_primal(D, N, Nz, k, idx,
               params.polish_period > 0 && (iter % params.polish_period) == 0,
               best_primal, best_medoids, best_labels, cheap_lab);

    // (6) Gap / certificate against LR's own primal (self-consistent with the
    // reported medoids). best_primal is finite from iteration 0.
    const double denom = std::max(std::abs(best_primal), kEps);
    if (best_primal - best_lb <= params.rel_gap_tol * denom) {
      ++iter; // count this iteration.
      break;
    }

    double gnorm2 = 0.0;
    for (int j = 0; j < N; ++j) gnorm2 += g[static_cast<std::size_t>(j)] * g[static_cast<std::size_t>(j)];
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

  return finalize(D, N, k, best_lb, best_primal, seed_ub, std::move(best_medoids),
                  std::move(best_labels), std::move(mu), rho_at_best,
                  iter, params.rel_gap_tol);
}

LagrangianResult lagrangian_root_kelley(const double *D, int N, int k,
                                        double initial_ub, const LagrangianParams &params)
{
#ifndef DTWC_ENABLE_HIGHS
  (void)D; (void)N; (void)k; (void)initial_ub; (void)params;
  throw SolverError(
    "lagrangian_root_kelley: the cutting-plane master requires HiGHS. Rebuild with "
    "-DDTWC_ENABLE_HIGHS=ON, or use lagrangian_root (the solver-free subgradient variant).");
#else
  if (N <= 0) throw InvalidInput("lagrangian_root_kelley: N must be positive");
  if (k < 1 || k > N)
    throw InvalidInput("lagrangian_root_kelley: require 1 <= k <= N (k=" + std::to_string(k)
                       + ", N=" + std::to_string(N) + ")");

  const std::size_t Nz = static_cast<std::size_t>(N);
  const double inf = std::numeric_limits<double>::infinity();

  // Box μ_j ∈ [0, maxD] contains the dual optimum (a point's assignment price is
  // ≤ its farthest distance ≤ maxD) and keeps the master bounded.
  double maxD = 0.0;
  for (std::size_t t = 0; t < Nz * Nz; ++t) maxD = std::max(maxD, D[t]);
  if (!(maxD > 0.0)) maxD = 1.0;

  std::vector<double> mu(Nz, 0.0), rho(Nz, 0.0), g(Nz, 0.0);
  std::vector<int> idx(Nz), cheap_lab(Nz, 0);
  double best_lb = -inf, best_primal = inf;
  const double seed_ub = (initial_ub > 0.0) ? initial_ub : inf;
  std::vector<int> best_medoids, best_labels;
  std::vector<double> rho_at_best(Nz, 0.0);

  (void)omp_chunk_size(N, 8); // Task 3.6 loudness.

  // Master LP: cols μ_0..μ_{N-1} ∈ [0,maxD] and θ (col N) ∈ [-inf, UB]; maximize θ.
  HighsModel model;
  const std::size_t ncol = Nz + 1;
  model.lp_.num_col_ = static_cast<HighsInt>(ncol);
  model.lp_.num_row_ = 0;
  model.lp_.sense_ = ObjSense::kMaximize;
  model.lp_.col_cost_.assign(ncol, 0.0);
  model.lp_.col_cost_[Nz] = 1.0; // maximize θ
  model.lp_.col_lower_.assign(ncol, 0.0);
  model.lp_.col_upper_.assign(ncol, maxD);
  model.lp_.col_lower_[Nz] = -kHighsInf;
  model.lp_.col_upper_[Nz] = (seed_ub < inf) ? seed_ub : kHighsInf; // θ ≤ UB (valid: L* ≤ opt ≤ UB)
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.num_col_ = static_cast<HighsInt>(ncol);
  model.lp_.a_matrix_.num_row_ = 0;
  model.lp_.a_matrix_.start_.assign(ncol + 1, 0); // empty matrix

  Highs highs;
  highs.setOptionValue("output_flag", false);
  if (highs.passModel(model) != HighsStatus::kOk)
    throw SolverError("lagrangian_root_kelley: HiGHS rejected the master LP model.");

  int major = 0;
  std::vector<HighsInt> ridx;
  std::vector<double> rval;

  // Add the supporting hyperplane θ ≤ L + g·(μ'−μ) for the CURRENT (μ, L, g).
  auto add_cut = [&](double L) {
    ridx.clear();
    rval.clear();
    ridx.push_back(static_cast<HighsInt>(N)); // θ column, coeff +1
    rval.push_back(1.0);
    double gdotmu = 0.0;
    for (int j = 0; j < N; ++j) {
      const double gj = g[static_cast<std::size_t>(j)];
      gdotmu += gj * mu[static_cast<std::size_t>(j)];
      if (gj != 0.0) {
        ridx.push_back(static_cast<HighsInt>(j));
        rval.push_back(-gj);
      }
    }
    highs.addRow(-kHighsInf, L - gdotmu, static_cast<HighsInt>(ridx.size()),
                 ridx.data(), rval.data());
  };

  // Initial evaluation at μ = 0: seed the LB, the primal, and the θ ≤ UB bound
  // (unstabilized Kelley throws μ to box corners where L is terrible and the LB
  // never rises — so we stabilize with a BOXSTEP trust region around the best μ).
  double rho_k = 0.0;
  best_lb = evaluate_dual(D, N, Nz, k, mu, rho, g, idx, rho_k);
  rho_at_best = rho;
  try_primal(D, N, Nz, k, idx, /*do_polish=*/true, best_primal, best_medoids, best_labels, cheap_lab);
  highs.changeColBounds(static_cast<HighsInt>(N), -kHighsInf, best_primal); // θ ≤ UB
  add_cut(best_lb);

  std::vector<double> mu_hat = mu; // stability centre (best-L point so far)
  double L_hat = best_lb;
  double delta = maxD;             // trust radius (grows on serious steps, shrinks on null)
  double prev_ub = best_primal;

  for (major = 1; major <= params.kelley_max_major; ++major) {
    // Trust region: μ_j ∈ [μ̂_j − δ, μ̂_j + δ] ∩ [0, maxD].
    for (int j = 0; j < N; ++j) {
      const double lo = std::max(0.0, mu_hat[static_cast<std::size_t>(j)] - delta);
      const double hi = std::min(maxD, mu_hat[static_cast<std::size_t>(j)] + delta);
      highs.changeColBounds(static_cast<HighsInt>(j), lo, hi);
    }
    if (highs.run() != HighsStatus::kOk) break;
    if (highs.getModelStatus() != HighsModelStatus::kOptimal) break;
    const std::vector<double> &sol = highs.getSolution().col_value;
    const double theta_master = sol[Nz]; // model max over the trust region.
    for (int j = 0; j < N; ++j) mu[static_cast<std::size_t>(j)] = sol[static_cast<std::size_t>(j)];

    // Oracle at the new point; refine the model and the incumbents.
    const double L_new = evaluate_dual(D, N, Nz, k, mu, rho, g, idx, rho_k);
    try_primal(D, N, Nz, k, idx, /*do_polish=*/true, best_primal, best_medoids, best_labels, cheap_lab);
    if (best_primal < prev_ub) { // tightened UB ⇒ tighten θ bound too.
      highs.changeColBounds(static_cast<HighsInt>(N), -kHighsInf, best_primal);
      prev_ub = best_primal;
    }
    add_cut(L_new);

    if (L_new > best_lb) { best_lb = L_new; rho_at_best = rho; }
    if (L_new > L_hat + 1e-12 * std::max(1.0, std::abs(L_hat))) {
      mu_hat = mu;               // serious step: move centre, grow the region.
      L_hat = L_new;
      delta = std::min(delta * 2.0, maxD);
    } else {
      delta *= 0.5;              // null step: contract around the centre.
    }

    const double denom = std::max(std::abs(best_primal), kEps);
    if (best_primal - best_lb <= params.rel_gap_tol * denom) break; // primal certifies.
    // Global dual certificate: only valid when the trust region spans the full box.
    if (delta >= maxD
        && theta_master - best_lb <= params.rel_gap_tol * std::max(std::abs(theta_master), kEps))
      break;
    if (delta < maxD * 1e-12) break; // trust region collapsed ⇒ converged/stalled.
  }

  return finalize(D, N, k, best_lb, best_primal, seed_ub, std::move(best_medoids),
                  std::move(best_labels), std::move(mu_hat), rho_at_best,
                  major, params.rel_gap_tol);
#endif
}

LagrangianResult lagrangian_root_exact(const double *D, int N, int k,
                                       double initial_ub, const LagrangianParams &params)
{
  // 1. Root Lagrangian dual + primal. Prefer the Kelley cutting-plane root when
  //    HiGHS is present: it certifies to machine precision where the subgradient
  //    stalls, so far more instances close at the root (0 B&B nodes). Without
  //    HiGHS the solver-free subgradient root is used and the B&B closes the gap.
  //    If it already certifies, the tree is a single node — return immediately.
#ifdef DTWC_ENABLE_HIGHS
  LagrangianResult root = lagrangian_root_kelley(D, N, k, initial_ub, params);
#else
  LagrangianResult root = lagrangian_root(D, N, k, initial_ub, params);
#endif
  if (root.certified_optimal) { root.nodes = 0; return root; }

  const std::size_t Nz = static_cast<std::size_t>(N);
  const double inf = std::numeric_limits<double>::infinity();

  // 2. Recompute ρ* and Σμ* at the root multipliers (the fixed dual we bound with).
  const std::vector<double> &mu = root.multipliers;
  double sum_mu = 0.0;
  for (int j = 0; j < N; ++j) sum_mu += mu[static_cast<std::size_t>(j)];
  std::vector<double> rho(Nz, 0.0);
  for (int i = 0; i < N; ++i) {
    const double *Di = D + static_cast<std::size_t>(i) * Nz;
    double s = 0.0;
    for (int j = 0; j < N; ++j) {
      const double dd = Di[j] - mu[static_cast<std::size_t>(j)];
      if (dd < 0.0) s += dd;
    }
    rho[static_cast<std::size_t>(i)] = s;
  }

  // 3. Reduced-cost fixing (Task 4.2): the optimum ⊆ core, and fixed_open ⊆ every
  //    optimum. Pre-open the proven-open facilities; branch only over the rest.
  //    Use the dual value CONSISTENT with this ρ*, i.e. L(μ*) = Σμ* + Σ_{k
  //    smallest ρ*}, NOT root.lower_bound: the Kelley stability centre μ* can
  //    trail best_lb by a hair (serious-step threshold), and pairing a larger LB
  //    with this ρ* would OVER-fix and could drop an optimal medoid from the core.
  //    L(μ*) ≤ best_lb ⇒ fixing is weaker but provably valid (never loses the optimum).
  double lb_star = sum_mu;
  {
    std::vector<double> rho_sorted = rho;
    std::nth_element(rho_sorted.begin(), rho_sorted.begin() + (k - 1), rho_sorted.end());
    for (int t = 0; t < k; ++t) lb_star += rho_sorted[static_cast<std::size_t>(t)];
  }
  root.lower_bound = std::max(root.lower_bound, lb_star); // keep the tighter valid LB for reporting.
  const FixingResult fx = reduced_cost_fixing(rho, k, lb_star, root.upper_bound);
  const std::vector<int> forced = fx.fixed_open;
  std::vector<char> is_forced(Nz, 0);
  for (int f : forced) is_forced[static_cast<std::size_t>(f)] = 1;
  std::vector<int> cand;
  cand.reserve(fx.core.size());
  for (int c : fx.core)
    if (!is_forced[static_cast<std::size_t>(c)]) cand.push_back(c);
  // Sort branch candidates by ρ* ascending: the k−|forced| smallest form the LP's
  // tentative open set, so the prefix is both the best incumbent guess and the
  // tightest bound term.
  std::sort(cand.begin(), cand.end(), [&](int a, int b) {
    const double ra = rho[static_cast<std::size_t>(a)], rb = rho[static_cast<std::size_t>(b)];
    return ra < rb || (ra == rb && a < b);
  });
  const int C = static_cast<int>(cand.size());
  const int need = k - static_cast<int>(forced.size());
  // Reduced-cost fixing can only prove facilities open; proving MORE than k of
  // them open contradicts the cardinality constraint, so the (LB, UB) pair it
  // was given cannot both be valid. A negative `need` never reaches the leaf
  // test `remaining_need == 0` and indexes csum past its end, so refuse here.
  if (need < 0)
    throw SolverError("lagrangian_root_exact: reduced-cost fixing proved "
      + std::to_string(forced.size()) + " facilities open for k=" + std::to_string(k)
      + "; the lower/upper bound pair driving the fixing is inconsistent.");

  double forced_rho = 0.0;
  for (int f : forced) forced_rho += rho[static_cast<std::size_t>(f)];
  std::vector<double> csum(static_cast<std::size_t>(C) + 1, 0.0); // prefix sums of sorted cand ρ*.
  for (int t = 0; t < C; ++t)
    csum[static_cast<std::size_t>(t) + 1] = csum[static_cast<std::size_t>(t)]
                                          + rho[cand[static_cast<std::size_t>(t)]];

  // Actual p-median cost of an open set S (O(N·|S|)).
  auto cost_of = [&](const std::vector<int> &S) {
    const int n_open = static_cast<int>(S.size());
    double c = 0.0;
    for (int j = 0; j < N; ++j)
      c += nearest_medoid(n_open, [&](int t) {
             return D[static_cast<std::size_t>(S[static_cast<std::size_t>(t)]) * Nz
                      + static_cast<std::size_t>(j)];
           }).distance;
    return c;
  };

  double best_cost = root.upper_bound;
  std::vector<int> best_medoids = root.medoids;
  const double tol = 1e-9 * (1.0 + std::abs(best_cost));

  // 4. DFS branch-and-bound on the open/close decision of one candidate at a time.
  struct Frame { int pos; int opened; double opened_rho; std::vector<int> S; };
  std::vector<Frame> stack;
  stack.push_back({ 0, 0, 0.0, {} });
  long nodes = 0;
  bool capped = false;

  while (!stack.empty()) {
    if (nodes >= params.max_nodes) { capped = true; break; }
    Frame fr = std::move(stack.back());
    stack.pop_back();
    ++nodes;

    const int remaining_need = need - fr.opened;
    if (remaining_need == 0) { // leaf: all k chosen ⇒ evaluate the exact cost.
      std::vector<int> S = forced;
      S.insert(S.end(), fr.S.begin(), fr.S.end());
      const double c = cost_of(S);
      if (c < best_cost - tol) { best_cost = c; best_medoids = std::move(S); }
      continue;
    }
    const int rem = C - fr.pos;
    if (rem < remaining_need) continue; // not enough candidates left ⇒ infeasible.

    // Node bound: fixed-dual value with the remaining_need SMALLEST candidate ρ*
    // (the sorted-ascending prefix). Valid: Σμ* + Σ_S ρ*_i ≤ cost(S) for every S.
    const double node_lb = sum_mu + forced_rho + fr.opened_rho
                         + (csum[static_cast<std::size_t>(fr.pos) + static_cast<std::size_t>(remaining_need)]
                            - csum[static_cast<std::size_t>(fr.pos)]);
    if (node_lb >= best_cost - tol) continue; // prune.

    // Branch on cand[pos]: push the SKIP child first so the OPEN child is explored
    // first (drives strong incumbents early).
    stack.push_back({ fr.pos + 1, fr.opened, fr.opened_rho, fr.S });
    std::vector<int> S_open = fr.S;
    S_open.push_back(cand[static_cast<std::size_t>(fr.pos)]);
    stack.push_back({ fr.pos + 1, fr.opened + 1,
                      fr.opened_rho + rho[cand[static_cast<std::size_t>(fr.pos)]],
                      std::move(S_open) });
  }

  // 5. Assemble the result. On completion the incumbent is the proven optimum.
  std::sort(best_medoids.begin(), best_medoids.end());
  LagrangianResult r = root;                 // keep μ, iterations, core, n_core.
  r.medoids = best_medoids;
  r.labels.assign(Nz, best_medoids.empty() ? 0 : best_medoids[0]);
  const int n_best = static_cast<int>(best_medoids.size());
  if (n_best > 0)
    for (int j = 0; j < N; ++j) {
      const auto nearest = nearest_medoid(n_best, [&](int t) {
        return D[static_cast<std::size_t>(best_medoids[static_cast<std::size_t>(t)]) * Nz
                 + static_cast<std::size_t>(j)];
      });
      r.labels[static_cast<std::size_t>(j)] = best_medoids[static_cast<std::size_t>(nearest.position)];
    }
  r.upper_bound = best_cost;
  r.nodes = nodes;
  if (capped) {
    // Node cap hit before the tree closed — report best-so-far, NEVER a silent
    // wrong "optimal". lower_bound stays the valid root bound.
    r.lower_bound = root.lower_bound;
    const double denom = std::max(std::abs(best_cost), kEps);
    r.gap = (best_cost - r.lower_bound) / denom;
    r.certified_optimal = false;
    std::fprintf(stderr,
                 "lagrangian_root_exact: node cap %ld reached at N=%d k=%d before the tree "
                 "closed; returning best incumbent (cost %.10g, gap %.3e) UNCERTIFIED. "
                 "Raise params.max_nodes or use an exact MIP solver for a certificate.\n",
                 params.max_nodes, N, k, best_cost, r.gap);
  } else {
    r.lower_bound = best_cost; // tree fully explored ⇒ incumbent is optimal.
    r.gap = 0.0;
    r.certified_optimal = true;
  }
  return r;
}

namespace {
/// Shared Problem→dense-D preparation for the Problem-facing entry points: fills
/// the distance matrix if needed, seeds an upper bound from the in-repo k-medoids
/// heuristic (FastPAM/Lloyd), and materializes a row-major copy of D. Sets @p N,
/// @p k, @p D; returns the seed UB (or -1 if unavailable).
double prepare_dense_D(Problem &prob, int &N, int &k, std::vector<double> &D)
{
  N = static_cast<int>(prob.size());
  k = static_cast<int>(prob.n_clusters());
  if (N <= 0) throw InvalidInput("LR-core (Problem): no data set");

  if (!prob.is_distance_matrix_filled()) prob.fill_distance_matrix();

  double ub = -1.0;
  {
    // The heuristic seed is invocation-local: an unpublished transaction restores
    // centroids_ind / clusters_ind, so the bound-only entry point
    // lagrangian_root(Problem&) has no visible side effect on the caller.
    ExactClusteringTransaction seed_transaction(prob);
    prob.cluster_by_kmedoids_lloyd();
    if (static_cast<int>(prob.centroids_ind.size()) == k
        && static_cast<int>(prob.clusters_ind.size()) == N) {
      double c = 0.0;
      for (int j = 0; j < N; ++j) c += static_cast<double>(prob.dist_by_ind(j, prob.centroid_of(j)));
      ub = c;
    }
  }

  D.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      D[static_cast<std::size_t>(i) * static_cast<std::size_t>(N) + static_cast<std::size_t>(j)]
        = static_cast<double>(prob.dist_by_ind(i, j));
  return ub;
}
} // namespace

LagrangianResult lagrangian_root(Problem &prob, const LagrangianParams &params)
{
  int N = 0, k = 0;
  std::vector<double> D;
  const double ub = prepare_dense_D(prob, N, k, D);
  return lagrangian_root(D.data(), N, k, ub, params);
}

} // namespace dtwc::mip

namespace dtwc {

void LR_core_clustering(Problem &prob)
{
  validate_mip_settings(prob.mip_settings); // lr_max_nodes is consumed below.
  int N = 0, k = 0;
  std::vector<double> D;
  mip::ExactClusteringTransaction result_transaction(prob);
  const double ub = mip::prepare_dense_D(prob, N, k, D);

  // Trivial k handled by the exact solver directly (k==N ⇒ all medoids; k==1 ⇒
  // the 1-medoid) — no special-casing needed, the B&B certifies them at the root.
  mip::LagrangianParams params;
  params.max_nodes = prob.mip_settings.lr_max_nodes;
  const mip::LagrangianResult r = mip::lagrangian_root_exact(D.data(), N, k, ub, params);

  // Method::LRCore is an EXACT entry point. lagrangian_root_exact clears
  // certified_optimal when the node cap stopped the tree, and publishing that
  // incumbent as a proven optimum is exactly the silent-wrong-answer the other
  // exact backends throw on.
  if (!r.certified_optimal)
    throw SolverError("LR-core did not prove optimality (gap "
      + std::to_string(r.gap) + " after " + std::to_string(r.nodes)
      + " branch-and-bound nodes). The incumbent is a heuristic, not a certificate; "
        "raise Problem::mip_settings.lr_max_nodes (currently "
      + std::to_string(prob.mip_settings.lr_max_nodes)
      + "), use Method::MIP for an exact MIP backend, or Method::Kmedoids for a "
        "heuristic answer.");

  // Problem's convention: centroids_ind holds the medoid POINT indices;
  // clusters_ind[j] is the 0..k-1 index of j's medoid within centroids_ind.
  const int n_medoids = static_cast<int>(r.medoids.size());
  core::ClusteringResult result;
  result.medoid_indices = r.medoids;
  result.labels.assign(static_cast<std::size_t>(N), 0);
  if (n_medoids > 0)
    for (int j = 0; j < N; ++j)
      result.labels[static_cast<std::size_t>(j)] =
        mip::nearest_medoid(n_medoids, [&](int c) {
          return D[static_cast<std::size_t>(r.medoids[static_cast<std::size_t>(c)])
                   * static_cast<std::size_t>(N) + static_cast<std::size_t>(j)];
        }).position;

  // Same validation the HiGHS/Gurobi backends run: exactly k unique medoids,
  // one in-range label per point, and every medoid in its own cluster.
  result_transaction.publish(std::move(result), "LR-core");
}

} // namespace dtwc
