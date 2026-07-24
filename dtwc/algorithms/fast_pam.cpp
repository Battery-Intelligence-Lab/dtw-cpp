/**
 * @file fast_pam.cpp
 * @brief FastPAM1 / FasterPAM k-medoids SWAP (Schubert & Rousseeuw 2021).
 *
 * @details Reference: Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager
 *   k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
 *   algorithms." *Information Systems* 101:101804 (arXiv:2008.05171). BUILD uses
 *   the existing K-means++; three SWAP variants (see PAMVariant) share the O(N)
 *   ΔTD decomposition derived below.
 *
 * ── ΔTD decomposition (Eq. 11), derived from first principles ──────────────────
 * State per point o: nearest medoid slot m₁(o), d₁(o)=dist to it, d₂(o)=dist to
 * the 2nd-nearest medoid. Total deviation TD = Σ_o d₁(o). Consider removing medoid
 * slot m and inserting candidate x_c; let a = d(x_c,o). Each point o contributes:
 *   • m₁(o) ≠ m (nearest survives):  min(0, a − d₁)          [same for every m ≠ m₁]
 *   • m₁(o) = m (nearest removed):   min(d₂, a) − d₁         [falls to 2nd or x_c]
 * Summing, with a SHARED accumulator acc = Σ_o min(0, a − d₁) over ALL points and
 * correcting the over-count for points whose nearest is the removed m:
 *   ΔTD(m, x_c) = acc + ploss[m],
 *   ploss[m] = ρ(m) + Σ_{m₁(o)=m}[ min(d₂,a) − d₁ − min(0, a − d₁) ],
 *   removal loss ρ(m) = Σ_{m₁(o)=m}(d₂ − d₁).
 * The bracket simplifies per point: a < d₁ → acc += a−d₁, ploss[m₁] += d₁−d₂;
 * d₁ ≤ a < d₂ → ploss[m₁] += a−d₂; a ≥ d₂ → 0. So ΔTD for removing EVERY medoid is
 * found in ONE O(N) pass per candidate (argmin over m, then add acc once) — O(N²)
 * per iteration, not O(N²·k). (Degenerate k=1: d₂=+inf makes ρ and the correction
 * ±inf ⇒ NaN, so k=1 is special-cased to a direct argmin in fast_pam_swap.)
 * Cross-checked against the paper (Alg. 3–4) and the Rust `kmedoids` reference.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include "fast_pam.hpp"
#include "detail/fast_pam_plan.hpp"
#include "detail/medoid_utils.hpp"
#include "../Problem.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../core/portable_random.hpp"
#include "../core/distance_sampling_weights.hpp"
#include "../initialisation.hpp"
#include "../parallelisation.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::algorithms::detail {

int checked_fast_pam_point_count(
  std::size_t n_points, std::string_view caller)
{
  const std::string prefix(caller);
  if (n_points == 0)
    throw InvalidInput(prefix + ": Problem has no data points.");
  if (n_points > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw InvalidInput(
      prefix + ": N exceeds the int-indexed clustering result limit.");
  return static_cast<int>(n_points);
}

FastPamPlan resolve_fast_pam_plan(
  std::size_t n_points, int n_clusters, std::string_view caller)
{
  const int n = checked_fast_pam_point_count(n_points, caller);
  if (n_clusters <= 0 || n_clusters > n) {
    const std::string prefix(caller);
    throw InvalidInput(
      prefix + ": n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(n_clusters) + ", N=" + std::to_string(n) + ".");
  }
  return { n, n_clusters };
}

} // namespace dtwc::algorithms::detail

namespace dtwc {

namespace {

/**
 * @brief For each point, find its nearest and second-nearest medoid.
 *
 * @param prob          Problem instance (for distance lookups).
 * @param medoids       Current medoid indices.
 * @param N             Number of data points.
 * @param nearest       [out] Index into medoids array of nearest medoid for each point.
 * @param nearest_dist  [out] Distance to nearest medoid for each point.
 * @param second_dist   [out] Distance to second-nearest medoid for each point.
 */
void compute_nearest_and_second(
  Problem& prob,
  const std::vector<int>& medoids,
  int N,
  std::vector<int>& nearest,
  std::vector<double>& nearest_dist,
  std::vector<double>& second_dist)
{
  const int k = static_cast<int>(medoids.size());
  std::exception_ptr failure;
  int failure_point = N;

  // Lock-free by design: each iteration writes only to nearest[p], nearest_dist[p],
  // and second_dist[p] at its own index p — no two threads access the same element.
#pragma omp parallel for schedule(static)
  for (int p = 0; p < N; ++p) {
    try {
      double best = std::numeric_limits<double>::max();
      double second_best = std::numeric_limits<double>::max();
      int best_idx = 0;
      bool has_best = false;
      bool has_second = false;

      for (int m = 0; m < k; ++m) {
        const double d = core::detail::require_finite_medoid_distance(
          prob.dist_by_ind(p, medoids[m]), "fast_pam", p, m, medoids[m]);
        if (!has_best || d < best) {
          if (has_best) {
            second_best = best;
            has_second = true;
          }
          best = d;
          best_idx = m;
          has_best = true;
        } else if (!has_second || d < second_best) {
          second_best = d;
          has_second = true;
        }
      }

      nearest[p] = best_idx;
      nearest_dist[p] = best;
      second_dist[p] = second_best;
    } catch (...) {
#pragma omp critical(dtwc_medoid_assignment_failure)
      {
        if (p < failure_point) {
          failure_point = p;
          failure = std::current_exception();
        }
      }
    }
  }

  if (failure) std::rethrow_exception(failure);
}

/**
 * @brief Compute the total cost (sum of nearest-medoid distances).
 */
double compute_total_cost(const std::vector<double>& nearest_dist)
{
  return core::detail::ordered_medoid_objective(
    nearest_dist, "fast_pam");
}

// Deterministic best-swap selection order, shared by the naive and decomposition
// FastPAM1 so their results are DIGIT-IDENTICAL: prefer the most-negative ΔTD;
// break ties by the smaller candidate index x, then (within a candidate) the
// smaller medoid slot. Returns true if (cand_x, cand_m) should replace (best_x,·).
inline bool better_swap(double cand_delta, int cand_x, double best_delta, int best_x)
{
  if (cand_x < 0) return false;
  if (cand_delta < best_delta) return true;
  return cand_delta == best_delta && (best_x < 0 || cand_x < best_x);
}

// ---------------------------------------------------------------------------
// FastPAM1Naive SWAP — the pre-5.1 baseline. Single best (medoid_out, candidate_in)
// swap per pass, but the per-candidate gain uses the NAIVE nested O(N·k) loop:
//   for point p, for medoid m:
//     if m == nearest[p]: delta_m += min(second_dist[p], d_xp) - nearest_dist[p]
//     else:               delta_m += min(0, d_xp - nearest_dist[p])
// ⇒ O(N²·k) per iteration. Kept as the bench baseline and the digit-identity
// oracle for the O(N)-decomposition FastPAM1 below.
// ---------------------------------------------------------------------------
void pam1_naive_swap_impl(Problem& prob, int N, int k,
                          std::vector<int>& medoids, std::vector<bool>& is_medoid,
                          std::vector<int>& nearest, std::vector<double>& nearest_dist,
                          std::vector<double>& second_dist, int max_iter,
                          int& iter, bool& converged)
{
  const double eps = 1e-9 * std::max(1.0, compute_total_cost(nearest_dist));
  for (iter = 0; iter < max_iter; ++iter) {
    double best_delta = 0.0;
    int best_m_idx = -1, best_x_new = -1;
    std::exception_ptr failure;
    int failure_candidate = N;

    const int swap_chunk = dtwc::omp_chunk_size(N);
    #pragma omp parallel
    {
      std::vector<double> local_delta_m(k);
      double local_best_delta = 0.0;
      int local_best_m_idx = -1, local_best_x_new = -1;

      #pragma omp for schedule(dynamic, swap_chunk)
      for (int x = 0; x < N; ++x) {
        if (is_medoid[x]) continue;
        try {
          std::fill(local_delta_m.begin(), local_delta_m.end(), 0.0);
          for (int p = 0; p < N; ++p) {
            const double d_xp = core::detail::require_finite_candidate_distance(
              prob.dist_by_ind(p, x), "fast_pam", p, x);
            const int nearest_m = nearest[p];
            for (int m = 0; m < k; ++m) {
              if (m == nearest_m)
                local_delta_m[m] += std::min(second_dist[p], d_xp) - nearest_dist[p];
              else if (d_xp - nearest_dist[p] < 0.0)
                local_delta_m[m] += d_xp - nearest_dist[p];
            }
          }
          for (int m = 0; m < k; ++m)             // smallest m wins ties (strict <)
            if (better_swap(local_delta_m[m], x, local_best_delta, local_best_x_new)) {
              local_best_delta = local_delta_m[m];
              local_best_m_idx = m;
              local_best_x_new = x;
            }
        } catch (...) {
#pragma omp critical(dtwc_medoid_candidate_failure)
          {
            if (x < failure_candidate) {
              failure_candidate = x;
              failure = std::current_exception();
            }
          }
        }
      }
      #pragma omp critical
      {
        if (better_swap(local_best_delta, local_best_x_new, best_delta, best_x_new)) {
          best_delta = local_best_delta;
          best_m_idx = local_best_m_idx;
          best_x_new = local_best_x_new;
        }
      }
    } // end omp parallel

    if (failure) std::rethrow_exception(failure);
    if (best_x_new < 0 || best_delta >= -eps) { converged = true; break; }

    is_medoid[medoids[best_m_idx]] = false;
    is_medoid[best_x_new] = true;
    medoids[best_m_idx] = best_x_new;
    compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);
  }
}

// Removal loss ρ[m] = Σ_{o: nearest[o]=m} (second_dist[o] − nearest_dist[o]) ≥ 0:
// the extra total cost if medoid slot m were removed and each of its points
// reassigned to its current second-nearest medoid. O(N).
void update_removal_loss(int N, const std::vector<int>& nearest,
                         const std::vector<double>& nearest_dist,
                         const std::vector<double>& second_dist, std::vector<double>& rho)
{
  std::fill(rho.begin(), rho.end(), 0.0);
  for (int o = 0; o < N; ++o)
    rho[nearest[o]] += second_dist[o] - nearest_dist[o];
}

/// Best medoid slot to swap out for candidate x_c, and the resulting ΔTD.
struct SwapEval { double change; int best_m; };

// Evaluate, for candidate x_c = xj, ΔTD(m, xj) = acc + ploss[m] for every medoid
// slot m in ONE O(N) pass, then return the most negative over m (Algorithm 3
// decomposition, derived in the file header comment above). O(N + k).
//
// Deliberately SEQUENTIAL. FasterPAM is eager — swaps within a sweep depend on
// earlier ones — so this runs once per candidate (up to N times per sweep). An
// OpenMP region here would pay fork/join + a k-wide reduction PER candidate, and
// the O(N) body is far too small to amortise that (measured ~10× SLOWER than the
// sequential loop at N=1000). FasterPAM wins by doing ~1/k the work, not by using
// cores; the parallelism lives in the post-swap refresh (compute_nearest_and_second,
// run only on accepted swaps). Very-large-N is the CLARA subsample path, not this.
SwapEval find_best_swap(Problem& prob, int N, int k, int xj,
                        const std::vector<double>& rho, const std::vector<int>& nearest,
                        const std::vector<double>& nearest_dist,
                        const std::vector<double>& second_dist,
                        std::vector<double>& ploss)
{
  ploss.assign(rho.begin(), rho.end()); // reuse caller's buffer (no per-candidate alloc)
  double acc = 0.0;                     // shared benefit of adding x_c (Case A over all points)
  for (int o = 0; o < N; ++o) {
    const double doj = core::detail::require_finite_candidate_distance(
      prob.dist_by_ind(xj, o), "fast_pam", o, xj);
    const double d1 = nearest_dist[o];
    if (doj < d1) {
      acc += doj - d1;                             // x_c becomes o's nearest
      ploss[nearest[o]] += d1 - second_dist[o];    // correct Case-A over-count (≤ 0)
    } else if (doj < second_dist[o]) {
      ploss[nearest[o]] += doj - second_dist[o];   // only helps if o's medoid removed (≤ 0)
    }
  }

  int best_m = 0;
  double best = ploss[0];
  for (int m = 1; m < k; ++m)
    if (ploss[m] < best) { best = ploss[m]; best_m = m; }
  return { acc + best, best_m };
}

// ---------------------------------------------------------------------------
// FastPAM1 SWAP (Schubert & Rousseeuw 2021, Algorithm 2) — the O(N)-decomposition
// form. Same single-best-swap RESULT as the naive baseline (digit-identical via
// better_swap ordering), but each candidate's ΔTD over ALL medoids comes from the
// O(N) find_best_swap pass, so an iteration is O(N²) — parallel over candidates —
// instead of O(N²·k). Best swap per iteration, then refresh nearest/second/ρ.
// ---------------------------------------------------------------------------
void fastpam1_swap_impl(Problem& prob, int N, int k,
                        std::vector<int>& medoids, std::vector<bool>& is_medoid,
                        std::vector<int>& nearest, std::vector<double>& nearest_dist,
                        std::vector<double>& second_dist, int max_iter,
                        int& iter, bool& converged)
{
  std::vector<double> rho(k);
  update_removal_loss(N, nearest, nearest_dist, second_dist, rho);
  const double eps = 1e-9 * std::max(1.0, compute_total_cost(nearest_dist));

  for (iter = 0; iter < max_iter; ++iter) {
    double best_delta = 0.0;
    int best_m_idx = -1, best_x_new = -1;
    std::exception_ptr failure;
    int failure_candidate = N;

    const int swap_chunk = dtwc::omp_chunk_size(N);
    #pragma omp parallel
    {
      std::vector<double> ploss(k);   // per-thread scratch, reused across candidates
      double local_best_delta = 0.0;
      int local_best_m_idx = -1, local_best_x_new = -1;

      #pragma omp for schedule(dynamic, swap_chunk)
      for (int x = 0; x < N; ++x) {
        if (is_medoid[x]) continue;
        try {
          const SwapEval e = find_best_swap(
            prob, N, k, x, rho, nearest, nearest_dist, second_dist, ploss);
          if (better_swap(e.change, x, local_best_delta, local_best_x_new)) {
            local_best_delta = e.change;
            local_best_m_idx = e.best_m;
            local_best_x_new = x;
          }
        } catch (...) {
#pragma omp critical(dtwc_medoid_candidate_failure)
          {
            if (x < failure_candidate) {
              failure_candidate = x;
              failure = std::current_exception();
            }
          }
        }
      }
      #pragma omp critical
      {
        if (better_swap(local_best_delta, local_best_x_new, best_delta, best_x_new)) {
          best_delta = local_best_delta;
          best_m_idx = local_best_m_idx;
          best_x_new = local_best_x_new;
        }
      }
    } // end omp parallel

    if (failure) std::rethrow_exception(failure);
    if (best_x_new < 0 || best_delta >= -eps) { converged = true; break; }

    is_medoid[medoids[best_m_idx]] = false;
    is_medoid[best_x_new] = true;
    medoids[best_m_idx] = best_x_new;
    compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);
    update_removal_loss(N, nearest, nearest_dist, second_dist, rho);
  }
}

// ---------------------------------------------------------------------------
// FasterPAM SWAP (Schubert & Rousseeuw 2021, Algorithm 3). Eager: cycle through
// candidate points; for each, find the best medoid to swap it in for in O(N)
// (find_best_swap) and perform the swap immediately if it lowers TD, then refresh
// nearest/second and ρ. A full sweep of N candidates with no accepted swap ⇒
// converged. Per-sweep cost O(N²) vs FastPAM1's O(N²·k); accepted swaps (few over
// the whole run) each refresh state in O(N·k) via compute_nearest_and_second.
//
// State refresh after a swap uses the full (parallel) recompute rather than the
// paper's O(N) incremental do_swap: identical result, and since accepted swaps
// total O(k) over the run the extra work is negligible next to O(N²) per sweep.
// ---------------------------------------------------------------------------
void fasterpam_swap_impl(Problem& prob, int N, int k,
                         std::vector<int>& medoids, std::vector<bool>& is_medoid,
                         std::vector<int>& nearest, std::vector<double>& nearest_dist,
                         std::vector<double>& second_dist, int max_iter,
                         int& iter, bool& converged)
{
  std::vector<double> rho(k);
  update_removal_loss(N, nearest, nearest_dist, second_dist, rho);
  std::vector<double> ploss(k); // scratch, reused across candidates (no per-candidate alloc)

  // Accept only improvements below this threshold: guards against float noise
  // cycling. Tied to the cost scale so it is meaningful for any distance range.
  const double eps = 1e-9 * std::max(1.0, compute_total_cost(nearest_dist));

  for (iter = 0; iter < max_iter; ++iter) {
    bool any_swap = false;
    for (int j = 0; j < N; ++j) {
      if (is_medoid[j]) continue;
      const SwapEval e = find_best_swap(prob, N, k, j, rho, nearest, nearest_dist, second_dist, ploss);
      if (e.change < -eps) {
        is_medoid[medoids[e.best_m]] = false;
        medoids[e.best_m] = j;
        is_medoid[j] = true;
        compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);
        update_removal_loss(N, nearest, nearest_dist, second_dist, rho);
        any_swap = true;
      }
    }
    if (!any_swap) { converged = true; break; }
  }
}

} // anonymous namespace


core::ClusteringResult fast_pam_swap(Problem& prob, const std::vector<int>& initial_medoids,
                                     int max_iter, PAMVariant variant)
{
  validate_pam_variant(variant);
  const int N = algorithms::detail::checked_fast_pam_point_count(
    prob.size(), "fast_pam_swap");
  algorithms::detail::validate_medoids(initial_medoids, N);

  prob.fillDistanceMatrix();

  std::vector<int> medoids = initial_medoids;
  const int k = static_cast<int>(medoids.size());

  std::vector<bool> is_medoid(N, false);
  for (int m : medoids) is_medoid[m] = true;

  std::vector<int> nearest(N);
  std::vector<double> nearest_dist(N);
  std::vector<double> second_dist(N);
  compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);

  int iter = 0;
  bool converged = false;
  if (k == 1) {
    // Degenerate: with one medoid there is no second-nearest (second_dist = +inf),
    // so the removal-loss decomposition is undefined (ρ = inf, corrections = −inf
    // ⇒ NaN). The single-medoid optimum is simply argmin_x Σ_o d(x, o); compute it
    // directly in O(N²), smallest index winning ties. Handles all three variants.
    double best_cost = std::numeric_limits<double>::max();
    int best_x = medoids[0];
    bool best_present = false;
    std::exception_ptr failure;
    int failure_candidate = N;
    const int chunk = dtwc::omp_chunk_size(N);
    #pragma omp parallel
    {
      double loc_cost = std::numeric_limits<double>::max();
      int loc_x = medoids[0];
      bool loc_present = false;
      #pragma omp for schedule(dynamic, chunk) nowait
      for (int x = 0; x < N; ++x) {
        try {
          core::detail::OrderedMedoidObjective candidate_cost("fast_pam");
          for (int o = 0; o < N; ++o) {
            candidate_cost.add(core::detail::require_finite_candidate_distance(
              prob.dist_by_ind(x, o), "fast_pam", o, x));
          }
          const double c = candidate_cost.value();
          if (!loc_present || c < loc_cost || (c == loc_cost && x < loc_x)) {
            loc_cost = c;
            loc_x = x;
            loc_present = true;
          }
        } catch (...) {
#pragma omp critical(dtwc_medoid_candidate_failure)
          {
            if (x < failure_candidate) {
              failure_candidate = x;
              failure = std::current_exception();
            }
          }
        }
      }
      #pragma omp critical
      {
        if (loc_present
            && (!best_present || loc_cost < best_cost
                || (loc_cost == best_cost && loc_x < best_x))) {
          best_cost = loc_cost;
          best_x = loc_x;
          best_present = true;
        }
      }
    }
    if (failure) std::rethrow_exception(failure);
    medoids[0] = best_x;
    compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);
    converged = true;
  } else {
    switch (variant) {
      case PAMVariant::FastPAM1Naive:
        pam1_naive_swap_impl(prob, N, k, medoids, is_medoid, nearest, nearest_dist,
                             second_dist, max_iter, iter, converged);
        break;
      case PAMVariant::FastPAM1:
        fastpam1_swap_impl(prob, N, k, medoids, is_medoid, nearest, nearest_dist,
                           second_dist, max_iter, iter, converged);
        break;
      case PAMVariant::FasterPAM:
        fasterpam_swap_impl(prob, N, k, medoids, is_medoid, nearest, nearest_dist,
                            second_dist, max_iter, iter, converged);
        break;
      default:
        throw std::logic_error("fast_pam_swap: unreachable PAMVariant");
    }
  }

  core::ClusteringResult result;
  result.medoid_indices = medoids;
  result.labels.assign(nearest.begin(), nearest.end());
  result.total_cost = compute_total_cost(nearest_dist);
  result.iterations = iter;
  result.converged = converged;

  // 2.0 result write-back (Task 1.6): store labels/medoids/k back into `prob` so
  // pure-C++ users get the same state the Python/MATLAB wrappers wired by hand —
  // scores::silhouette(prob) etc. then work with NO manual wiring.
  prob.set_n_clusters(k);
  prob.centroids_ind = result.medoid_indices;
  prob.clusters_ind  = result.labels;
  return result;
}


core::ClusteringResult fast_pam(Problem& prob, int n_clusters, int max_iter)
{
  const auto plan = algorithms::detail::resolve_fast_pam_plan(
    prob.size(), n_clusters, "fast_pam");

  prob.fillDistanceMatrix();

  // -------------------------------------------------------------------------
  // BUILD phase: initialize medoids using K-means++. Temporarily set prob's
  // cluster count, run the existing initializer, copy medoids, restore state.
  // -------------------------------------------------------------------------
  const int orig_Nc = prob.cluster_size();
  const auto orig_centroids = prob.centroids_ind;
  const auto orig_clusters = prob.clusters_ind;

  prob.set_numberOfClusters(plan.n_clusters);
  init::Kmeanspp(prob);
  std::vector<int> medoids = prob.centroids_ind;

  prob.set_numberOfClusters(orig_Nc);
  prob.centroids_ind = orig_centroids;
  prob.clusters_ind = orig_clusters;

  // Default: the O(N)-decomposition FastPAM1 (Task 5.1) — same result as the old
  // naive O(N²·k) swap but O(N²) per iteration and parallel over candidates, so it
  // never regresses the common small-k / large-N case. FasterPAM (eager) wins at
  // large k but is sequential; callers pick it explicitly via fast_pam_swap.
  return fast_pam_swap(prob, medoids, max_iter, PAMVariant::FastPAM1);
}

core::ClusteringResult fast_pam_seeded(Problem& prob, int n_clusters,
                                       std::uint64_t random_seed, int max_iter)
{
  const auto plan = algorithms::detail::resolve_fast_pam_plan(
    prob.size(), n_clusters, "fast_pam_seeded");
  const int N = plan.n_points;
  prob.fillDistanceMatrix();

  std::mt19937_64 rng(random_seed);
  std::vector<int> medoids{static_cast<int>(core::portable_bounded(
    rng, static_cast<std::uint64_t>(N)))};
  medoids.reserve(static_cast<std::size_t>(plan.n_clusters));
  std::vector<double> distances(static_cast<std::size_t>(N),
                                std::numeric_limits<double>::infinity());
  while (static_cast<int>(medoids.size()) < plan.n_clusters) {
    for (int i = 0; i < N; ++i)
      distances[static_cast<std::size_t>(i)] = std::min(
        distances[static_cast<std::size_t>(i)], prob.dist_by_ind(medoids.back(), i));
    for (int medoid : medoids) distances[static_cast<std::size_t>(medoid)] = 0.0;
    // This is k-median++ D-sampling: PAM minimizes a sum of DTW distances, so
    // the sampling weight is the current nearest objective contribution d.
    // Barycenter k-means uses D^2-sampling because its `align_squared` values
    // are already squared-local-cost objective contributions. Squaring this
    // vector would instead bias a different (sum-of-squares) PAM objective.
    const auto weights = core::distance_sampling_weights(
      distances, medoids, "fast_pam_seeded");
    int chosen = 0;
    if (weights.total <= 0.0) {
      while (std::find(medoids.begin(), medoids.end(), chosen) != medoids.end()) ++chosen;
    } else {
      chosen = static_cast<int>(core::portable_weighted_index(
        weights.values.begin(), weights.values.end(), weights.total, rng));
      if (std::find(medoids.begin(), medoids.end(), chosen) != medoids.end()) {
        chosen = 0;
        while (std::find(medoids.begin(), medoids.end(), chosen) != medoids.end()) ++chosen;
      }
    }
    medoids.push_back(chosen);
  }
  return fast_pam_swap(prob, medoids, max_iter, PAMVariant::FastPAM1);
}

} // namespace dtwc
