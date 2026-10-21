/**
 * @file fast_pam.cpp
 * @brief FasterPAM k-medoids SWAP (Schubert & Rousseeuw 2021).
 *
 * @details Reference: Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager
 *   k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
 *   algorithms." *Information Systems* 101:101804 (arXiv:2008.05171). BUILD is
 *   k-medoids++ (core::kmedoids_pp); the eager FasterPAM SWAP uses the O(N) ΔTD
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
#include "../core/distance_sampling_weights.hpp" // kmedoids_pp
#include "../base/parallelisation.hpp"

#include <algorithm>
#include <array>
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
 * @param matrix        The packed distance matrix's values, at its width.
 * @param medoids       Current medoid indices.
 * @param N             Number of data points.
 * @param nearest       [out] Index into medoids array of nearest medoid for each point.
 * @param nearest_dist  [out] Distance to nearest medoid for each point.
 * @param second_dist   [out] Distance to second-nearest medoid for each point.
 */
template <typename T>
void compute_nearest_and_second(
  const T* matrix,
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
      const index_t medoid = medoids[m]; // read once: cl 19.50 read it again for the tie below (2026-10-11)
      const double d = matrix[core::tri_index(static_cast<std::size_t>(p), static_cast<std::size_t>(medoid))];
      // A medoid tied with another medoid (a duplicate series) serves itself,
      // or its own cluster would be published empty.
      if (!has_best || d < best || (d == best && medoid == p)) {
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

constexpr auto G = static_cast<index_t>(core::group_rows);

// Evaluate, for each candidate x_c = a + g of a group of W, ΔTD(m, x_c) = acc + ploss[m] for every medoid slot m in
// ONE O(N) pass, then the most negative over m (Algorithm 3 decomposition, derived in the file header comment above);
// out[g] for g < n. O(N + k) per candidate. Each candidate's sums run on one thread, in point order, through the
// operations of the one-candidate loop (W = 1), so its (change, slot) is the same whatever group, run or thread count
// evaluates it.
template <index_t W, typename T>
void evaluate_group(const T *matrix, index_t N, index_t a, index_t n, const std::vector<double> &rho,
                    const std::vector<index_t> &nearest, const std::vector<double> &nearest_dist,
                    const std::vector<double> &second_dist, SwapEval *out)
{
  const auto k = static_cast<index_t>(rho.size());
  thread_local std::vector<double> scratch; // ploss[m * W + g]; grows, never shrinks
  if (scratch.size() < static_cast<std::size_t>(k * W)) scratch.resize(static_cast<std::size_t>(k * W));
  double *const ploss = scratch.data();
  for (index_t m = 0; m < k; ++m)
    for (index_t g = 0; g < W; ++g) ploss[m * W + g] = rho[m];
  const index_t *const near = nearest.data();
  const double *const d1s = nearest_dist.data();
  const double *const d2s = second_dist.data();
  const index_t a0 = std::min(a, N - W); // lane g is candidate a0 + g: a short last group drops its first lanes
  std::array<double, W> total;           // acc of each lane: the shared benefit of adding x_c (Case A, all points)
  if constexpr (W == 1) {                // the loop find_best_swap ran before groups
    double acc = 0.0;
    for (index_t o = 0; o < N; ++o) {
      const double doj = matrix[core::tri_index(static_cast<std::size_t>(a), static_cast<std::size_t>(o))];
      const double d1 = d1s[o];
      if (doj < d1) {
        acc += doj - d1;               // x_c becomes o's nearest
        ploss[near[o]] += d1 - d2s[o]; // correct Case-A over-count (≤ 0)
      } else if (doj < d2s[o]) {
        ploss[near[o]] += doj - d2s[o]; // only helps if o's medoid removed (≤ 0)
      }
    }
    total[0] = acc;
  } else {
    // D(a0 + g, o) without a register per lane (GCC 13 kept the W rows a0 + g in registers and read pointers from
    // the stack): with hi = max(a0 + g, o), tri_index = hi (hi + 1) / 2 + min = hi (hi - 1) / 2 + (a0 + g) + o.
    // A select on values, not std::max: through its references GCC branched on each comparison of o - a0 with g and
    // threaded them.
    const auto distance = [&](index_t g, index_t o) {
      const index_t d = o - a0;
      const auto hi = static_cast<std::size_t>(a0 + (d > g ? d : g));
      return matrix[hi * (hi - 1) / 2 + static_cast<std::size_t>(a0 + g + o)];
    };
    std::array<double, W> acc{};
    for (index_t o = 0; o < N; ++o) {
      const double d2 = d2s[o];
      double *const pl = ploss + near[o] * W;
      const auto update = [&](index_t g, double doj, double d1) { // the one-candidate loop's branches
        if (doj < d1) {
          acc[g] += doj - d1;
          pl[g] += d1 - d2;
        } else if (doj < d2) {
          pl[g] += doj - d2;
        }
      };
      // Half the lanes at a time, their distances read before d1 as the one-candidate loop reads its one: clang ranks a
      // sum's terms by where they were read, so it groups (doj + acc) - d1 as there. A half's W / 2 distances and
      // the W accumulators fit the 16 vector registers; d1 is read again for the second half.
      constexpr index_t H = W / 2;
      double doj[W];
      for (index_t g = 0; g < H; ++g) doj[g] = distance(g, o);
      const double d1 = d1s[o];
      for (index_t g = 0; g < H; ++g) update(g, doj[g], d1);
      for (index_t g = H; g < W; ++g) doj[g] = distance(g, o);
      const double d1_again = d1s[o];
      for (index_t g = H; g < W; ++g) update(g, doj[g], d1_again);
    }
    total = acc; // copied whole after the loop, so acc stays in registers through it (GCC 13)
  }

  for (index_t g = 0; g < n; ++g) {
    const index_t lane = a - a0 + g;
    index_t best_m = 0;
    double best = ploss[lane];
    for (index_t m = 1; m < k; ++m)
      if (ploss[m * W + lane] < best) { best = ploss[m * W + lane]; best_m = m; }
    out[g] = { total[lane] + best, best_m };
  }
}

// Below this many points the sweep evaluates one candidate at a time on the calling thread, the sequential sweep
// itself: there a parallel region, or a group's evaluations dropped after each swap, cost more than they save.
constexpr index_t parallel_from = 2000;

// ---------------------------------------------------------------------------
// FasterPAM SWAP (Schubert & Rousseeuw 2021, Algorithm 3). Eager: cycle through
// candidate points; for each, find the best medoid to swap it in for in O(N)
// and perform the swap immediately if it lowers TD, then refresh
// nearest/second and ρ. A full sweep of N candidates with no accepted swap ⇒
// converged. Per-sweep cost O(N²); accepted swaps (few over
// the whole run) each refresh state in O(N·k) via compute_nearest_and_second.
//
// From parallel_from points on, the candidates are evaluated in runs: consecutive groups of G in one parallel loop,
// each against the state at the run's start, which is the sequential sweep's state up to the run's first accepted
// swap. The first candidate of the run (index order) that lowers TD is accepted, the evaluations after it are dropped,
// and the next run starts after it. So the accepted swaps, and the medoids, labels, cost and iterations, are the
// sequential sweep's for any run length and thread count. A run is T groups after a swap (T threads) and doubles, up
// to 8T, while none is accepted; on one thread it is one group. Below parallel_from a run is one candidate.
//
// State refresh after a swap uses the full (parallel) recompute rather than the
// paper's O(N) incremental do_swap: identical result, and since accepted swaps
// total O(k) over the run the extra work is negligible next to O(N²) per sweep.
//
// `values` are the matrix's values at its width: a float32 matrix's are widened as they are read, every sum in double.
// ---------------------------------------------------------------------------
template <typename T>
void fasterpam_swap_impl(const core::DistanceMatrix& matrix, const T* values, index_t N, index_t k,
                         std::vector<index_t>& medoids, std::vector<bool>& is_medoid,
                         std::vector<index_t>& nearest, std::vector<double>& nearest_dist,
                         std::vector<double>& second_dist, int max_iter,
                         int& iter, bool& converged)
{
  std::vector<double> rho(k);
  update_removal_loss(N, nearest, nearest_dist, second_dist, rho);

  // Accept only improvements below this threshold: guards against float noise
  // cycling. Tied to the cost scale so it is meaningful for any distance range.
  const double eps = 1e-9 * std::max(1.0, compute_total_cost(nearest_dist));

  const bool grouped = N >= parallel_from;
  const index_t width = grouped ? G : 1; // candidates a group holds
  const index_t threads = grouped ? get_max_threads() : 1;
  const index_t longest = threads > 1 ? 8 * threads : 1; // groups in a run
  std::vector<SwapEval> evals(static_cast<std::size_t>(longest * width)); // a run's candidates
  for (iter = 0; iter < max_iter; ++iter) {
    bool any_swap = false;
    index_t groups = threads;
    for (index_t j0 = 0; j0 < N;) {
      if (!grouped && is_medoid[j0]) { // the sequential sweep skips a medoid unevaluated
        ++j0;
        continue;
      }
      const index_t j1 = std::min(N, j0 + groups * width);
      if (grouped) // a mapped matrix starts reading the next run's rows (a hint; nothing on the heap)
        matrix.prefetch_rows(static_cast<std::size_t>(j1),
                             static_cast<std::size_t>(std::min(N, j1 + std::min(2 * groups, longest) * G)));
      auto run_group = [&](std::size_t q) {
        const index_t a = j0 + static_cast<index_t>(q) * width;
        if (grouped)
          evaluate_group<G>(values, N, a, std::min(G, j1 - a), rho, nearest, nearest_dist, second_dist,
                            evals.data() + q * G);
        else
          evaluate_group<1>(values, N, a, 1, rho, nearest, nearest_dist, second_dist, evals.data() + q);
      };
      run_openmp(run_group, static_cast<std::size_t>((j1 - j0 + width - 1) / width), threads > 1, 8);

      index_t j = j0;
      while (j < j1 && (is_medoid[j] || !(evals[j - j0].change < -eps))) ++j;
      if (j == j1) {
        j0 = j1;
        groups = std::min(2 * groups, longest);
        continue;
      }
      const SwapEval e = evals[j - j0];
      is_medoid[medoids[e.best_m]] = false;
      medoids[e.best_m] = j;
      is_medoid[j] = true;
      compute_nearest_and_second(values, medoids, N, nearest, nearest_dist, second_dist);
      update_removal_loss(N, nearest, nearest_dist, second_dist, rho);
      any_swap = true;
      j0 = j + 1;
      groups = threads;
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

  int iter = 0;
  bool converged = false;
  const core::DistanceMatrix &matrix = prob.distance_matrix();
  matrix.visit([&](const auto *values) {
    compute_nearest_and_second(values, medoids, N, nearest, nearest_dist, second_dist);
    if (k == 1) {
      // Degenerate: with one medoid there is no second-nearest (second_dist = +inf),
      // so the removal-loss decomposition is undefined (ρ = inf, corrections = −inf
      // ⇒ NaN). The single-medoid optimum is simply argmin_x Σ_o d(x, o); compute it
      // directly in O(N²), smallest index winning ties. A task sums a group of
      // candidates, each in point order, and writes their costs; the argmin is taken serially.
      std::vector<double> candidate_cost(static_cast<std::size_t>(N));
      auto total_distance = [&](std::size_t q) {
        const auto a = static_cast<index_t>(q) * G;
        core::detail::OrderedMedoidObjective cost[G];
        core::for_each_point(values, N, a, [&](index_t, const double *d) {
          for (index_t g = 0; g < G; ++g) cost[g].add(d[g]);
        });
        for (index_t g = 0; g < std::min(G, N - a); ++g)
          candidate_cost[static_cast<std::size_t>(a + g)] = cost[g].value();
      };
      run_openmp(total_distance, static_cast<std::size_t>((N + G - 1) / G));
      medoids[0] = static_cast<index_t>(
        std::min_element(candidate_cost.begin(), candidate_cost.end()) - candidate_cost.begin());
      compute_nearest_and_second(values, medoids, N, nearest, nearest_dist, second_dist);
      converged = true;
    } else {
      fasterpam_swap_impl(matrix, values, N, k, medoids, is_medoid, nearest, nearest_dist,
                          second_dist, max_iter, iter, converged);
    }
  });

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


core::ClusteringResult fast_pam(Problem& prob, index_t n_clusters, int max_iter, std::uint64_t seed)
{
  const index_t N = checked_point_count(prob, n_clusters, max_iter, "fast_pam");
  prob.fill_distance_matrix();

  std::mt19937_64 rng(seed);
  // BUILD reads N·k entries: from the typed values, as the swap (cl 19.50 reloaded get()'s pointer at each read).
  auto medoids = prob.distance_matrix().visit([&](const auto *matrix) {
    return core::kmedoids_pp(
      N, n_clusters, rng,
      [matrix](index_t m, index_t i) {
        return matrix[core::tri_index(static_cast<std::size_t>(m), static_cast<std::size_t>(i))];
      },
      "fast_pam");
  });
  return swap_phase(prob, std::move(medoids), max_iter);
}

} // namespace dtwc
