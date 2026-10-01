/**
 * @file fast_pam.cpp
 * @brief FasterPAM k-medoids SWAP (Schubert & Rousseeuw 2021).
 *
 * @details Reference: Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager
 *   k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
 *   algorithms." *Information Systems* 101:101804 (arXiv:2008.05171). BUILD uses
 *   the existing K-means++; the eager FasterPAM SWAP uses the O(N) ΔTD
 *   decomposition derived below.
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
 * ±inf ⇒ NaN, so k=1 is special-cased to a direct argmin in swap_phase.)
 * Cross-checked against the paper (Alg. 3–4) and the Rust `kmedoids` reference.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../core/portable_random.hpp"
#include "../core/distance_sampling_weights.hpp"
#include "../initialisation.hpp"
#include "../base/parallelisation.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
  const std::vector<index_t>& medoids,
  index_t N,
  std::vector<index_t>& nearest,
  std::vector<double>& nearest_dist,
  std::vector<double>& second_dist)
{
  const auto k = static_cast<index_t>(medoids.size());

  // Lock-free by design: each index writes only to nearest[p], nearest_dist[p],
  // and second_dist[p] at its own index p — no two threads access the same element.
  auto assign_point = [&](std::size_t index) {
    const auto p = static_cast<index_t>(index);
    double best = std::numeric_limits<double>::max();
    double second_best = std::numeric_limits<double>::max();
    index_t best_idx = 0;
    bool has_best = false;
    bool has_second = false;

    for (index_t m = 0; m < k; ++m) {
      const double d = prob.dist_by_ind(p, medoids[m]);
      // A medoid tied with another medoid (a duplicate series) serves itself,
      // or its own cluster would be published empty.
      if (!has_best || d < best || (d == best && medoids[m] == p)) {
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
  };
  run_openmp(assign_point, static_cast<std::size_t>(N));
}

/**
 * @brief Compute the total cost (sum of nearest-medoid distances).
 */
double compute_total_cost(const std::vector<double>& nearest_dist)
{
  return core::detail::ordered_medoid_objective(
    nearest_dist, "fast_pam");
}

// Removal loss ρ[m] = Σ_{o: nearest[o]=m} (second_dist[o] − nearest_dist[o]) ≥ 0:
// the extra total cost if medoid slot m were removed and each of its points
// reassigned to its current second-nearest medoid. O(N).
void update_removal_loss(index_t N, const std::vector<index_t>& nearest,
                         const std::vector<double>& nearest_dist,
                         const std::vector<double>& second_dist, std::vector<double>& rho)
{
  std::fill(rho.begin(), rho.end(), 0.0);
  for (index_t o = 0; o < N; ++o)
    rho[nearest[o]] += second_dist[o] - nearest_dist[o];
}

/// Best medoid slot to swap out for candidate x_c, and the resulting ΔTD.
struct SwapEval { double change; index_t best_m; };

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
SwapEval find_best_swap(Problem& prob, index_t N, index_t xj,
                        const std::vector<double>& rho, const std::vector<index_t>& nearest,
                        const std::vector<double>& nearest_dist,
                        const std::vector<double>& second_dist,
                        std::vector<double>& ploss)
{
  ploss.assign(rho.begin(), rho.end()); // reuse caller's buffer (no per-candidate alloc)
  double acc = 0.0;                     // shared benefit of adding x_c (Case A over all points)
  for (index_t o = 0; o < N; ++o) {
    const double doj = prob.dist_by_ind(xj, o);
    const double d1 = nearest_dist[o];
    if (doj < d1) {
      acc += doj - d1;                             // x_c becomes o's nearest
      ploss[nearest[o]] += d1 - second_dist[o];    // correct Case-A over-count (≤ 0)
    } else if (doj < second_dist[o]) {
      ploss[nearest[o]] += doj - second_dist[o];   // only helps if o's medoid removed (≤ 0)
    }
  }

  index_t best_m = 0;
  double best = ploss[0];
  for (std::size_t m = 1; m < ploss.size(); ++m)
    if (ploss[m] < best) { best = ploss[m]; best_m = static_cast<index_t>(m); }
  return { acc + best, best_m };
}

// ---------------------------------------------------------------------------
// FasterPAM SWAP (Schubert & Rousseeuw 2021, Algorithm 3). Eager: cycle through
// candidate points; for each, find the best medoid to swap it in for in O(N)
// (find_best_swap) and perform the swap immediately if it lowers TD, then refresh
// nearest/second and ρ. A full sweep of N candidates with no accepted swap ⇒
// converged. Per-sweep cost O(N²); accepted swaps (few over
// the whole run) each refresh state in O(N·k) via compute_nearest_and_second.
//
// State refresh after a swap uses the full (parallel) recompute rather than the
// paper's O(N) incremental do_swap: identical result, and since accepted swaps
// total O(k) over the run the extra work is negligible next to O(N²) per sweep.
// ---------------------------------------------------------------------------
void fasterpam_swap_impl(Problem& prob, index_t N, index_t k,
                         std::vector<index_t>& medoids, std::vector<bool>& is_medoid,
                         std::vector<index_t>& nearest, std::vector<double>& nearest_dist,
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
    for (index_t j = 0; j < N; ++j) {
      if (is_medoid[j]) continue;
      const SwapEval e = find_best_swap(prob, N, j, rho, nearest, nearest_dist, second_dist, ploss);
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

/// Point count of a Problem that can hold `n_clusters` medoids, after the arguments
/// are checked. max_iter = 0 is meaningful (BUILD only, no SWAP); a negative count
/// is not, and every binding inherits this refusal.
index_t checked_point_count(const Problem& prob, index_t n_clusters, int max_iter, const char* caller)
{
  if (max_iter < 0)
    throw InvalidInput(std::string(caller) + ": max_iter must be at least 0 (0 returns the BUILD "
                       "medoids without a SWAP); got " + std::to_string(max_iter) + ".");
  const index_t n = prob.size();
  if (n == 0)
    throw InvalidInput(std::string(caller) + ": Problem has no data points.");
  if (n_clusters <= 0 || n_clusters > n)
    throw InvalidInput(std::string(caller) + ": n_clusters must be in [1, N]. Got n_clusters="
                       + std::to_string(n_clusters) + ", N=" + std::to_string(n) + ".");
  return n;
}

/// SWAP phase from the BUILD medoids; writes the result back into `prob`.
core::ClusteringResult swap_phase(Problem& prob, std::vector<index_t> medoids, int max_iter)
{
  const index_t N = prob.size();
  const auto k = static_cast<index_t>(medoids.size());

  std::vector<bool> is_medoid(N, false);
  for (index_t m : medoids) is_medoid[m] = true;

  std::vector<index_t> nearest(N);
  std::vector<double> nearest_dist(N);
  std::vector<double> second_dist(N);
  compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);

  int iter = 0;
  bool converged = false;
  if (k == 1) {
    // Degenerate: with one medoid there is no second-nearest (second_dist = +inf),
    // so the removal-loss decomposition is undefined (ρ = inf, corrections = −inf
    // ⇒ NaN). The single-medoid optimum is simply argmin_x Σ_o d(x, o); compute it
    // directly in O(N²), smallest index winning ties. Each candidate writes its
    // own cost; the argmin is taken serially.
    std::vector<double> candidate_cost(static_cast<std::size_t>(N));
    auto total_distance = [&](std::size_t index) {
      const auto x = static_cast<index_t>(index);
      core::detail::OrderedMedoidObjective cost;
      for (index_t o = 0; o < N; ++o) cost.add(prob.dist_by_ind(x, o));
      candidate_cost[index] = cost.value();
    };
    run_openmp(total_distance, static_cast<std::size_t>(N));
    medoids[0] = static_cast<index_t>(
      std::min_element(candidate_cost.begin(), candidate_cost.end()) - candidate_cost.begin());
    compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);
    converged = true;
  } else {
    fasterpam_swap_impl(prob, N, k, medoids, is_medoid, nearest, nearest_dist,
                        second_dist, max_iter, iter, converged);
  }

  core::ClusteringResult result;
  result.medoid_indices = medoids;
  result.labels = std::move(nearest);
  result.total_cost = compute_total_cost(nearest_dist);
  result.iterations = iter;
  result.converged = converged;

  prob.set_result(result); // scores::silhouette(prob) etc. read it back
  return result;
}

} // anonymous namespace


core::ClusteringResult fast_pam(Problem& prob, index_t n_clusters, int max_iter)
{
  (void)checked_point_count(prob, n_clusters, max_iter, "fast_pam");
  prob.fill_distance_matrix();

  // -------------------------------------------------------------------------
  // BUILD phase: initialize medoids using K-means++. Temporarily set prob's
  // cluster count, run the existing initializer, copy medoids, restore state.
  // -------------------------------------------------------------------------
  const index_t orig_Nc = prob.n_clusters();
  const auto orig_centroids = prob.centroids_ind;
  const auto orig_clusters = prob.clusters_ind;

  prob.set_n_clusters(n_clusters);
  init::Kmeanspp(prob);
  std::vector<index_t> medoids = prob.centroids_ind;

  prob.set_n_clusters(orig_Nc);
  prob.centroids_ind = orig_centroids;
  prob.clusters_ind = orig_clusters;

  return swap_phase(prob, std::move(medoids), max_iter);
}

core::ClusteringResult fast_pam_seeded(Problem& prob, index_t n_clusters,
                                       std::uint64_t random_seed, int max_iter)
{
  const index_t N = checked_point_count(prob, n_clusters, max_iter, "fast_pam_seeded");
  prob.fill_distance_matrix();

  std::mt19937_64 rng(random_seed);
  std::vector<index_t> medoids{static_cast<index_t>(core::portable_bounded(
    rng, static_cast<std::uint64_t>(N)))};
  medoids.reserve(static_cast<std::size_t>(n_clusters));
  std::vector<double> distances(static_cast<std::size_t>(N),
                                std::numeric_limits<double>::infinity());
  while (static_cast<index_t>(medoids.size()) < n_clusters) {
    for (index_t i = 0; i < N; ++i)
      distances[static_cast<std::size_t>(i)] = std::min(
        distances[static_cast<std::size_t>(i)], prob.dist_by_ind(medoids.back(), i));
    for (index_t medoid : medoids) distances[static_cast<std::size_t>(medoid)] = 0.0;
    // This is k-median++ D-sampling: PAM minimizes a sum of DTW distances, so
    // the sampling weight is the current nearest objective contribution d.
    // Barycenter k-means uses D^2-sampling because its `align_squared` values
    // are already squared-local-cost objective contributions. Squaring this
    // vector would instead bias a different (sum-of-squares) PAM objective.
    const auto weights = core::distance_sampling_weights(
      distances, medoids, "fast_pam_seeded");
    index_t chosen = 0;
    if (weights.total <= 0.0) {
      while (std::find(medoids.begin(), medoids.end(), chosen) != medoids.end()) ++chosen;
    } else {
      chosen = static_cast<index_t>(core::portable_weighted_index(
        weights.values.begin(), weights.values.end(), weights.total, rng));
      if (std::find(medoids.begin(), medoids.end(), chosen) != medoids.end()) {
        chosen = 0;
        while (std::find(medoids.begin(), medoids.end(), chosen) != medoids.end()) ++chosen;
      }
    }
    medoids.push_back(chosen);
  }
  return swap_phase(prob, std::move(medoids), max_iter);
}

} // namespace dtwc
