/**
 * @file clarans.cpp
 * @brief Implementation of CLARANS randomized k-medoids clustering.
 *
 * @details Implements the CLARANS neighborhood search from:
 *   Ng, R.T. & Han, J. (2002). "CLARANS: A method for clustering objects for
 *   spatial data mining." IEEE Transactions on Knowledge and Data Engineering,
 *   14(5), 1003–1016.  https://doi.org/10.1109/TKDE.2002.1033770
 *
 * Key design choices:
 *   - Strictly improving swaps only (delta < -1e-12). Neutral swaps are
 *     rejected to avoid cycles and ensure termination.
 *   - Deterministic: each restart seeds from opts.random_seed + restart index.
 *   - Hard DTW budget: max_dtw_evals guards against unbounded runtime.
 *   - After accepting a swap the full assignment is recomputed. The extra
 *     O(N*k) lookups hit the lazy distance cache (distByInd), so repeated
 *     lookups are free after the first evaluation.
 *   - Auto max_neighbor = max(250, (int)(0.0125 * k * (N - k))), matching
 *     the original CLARANS paper's recommended parameterization.
 *
 * @warning Experimental. Not exposed in CLI. Use FastCLARA for large N.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include "clarans.hpp"
#include "../Problem.hpp"
#include "../core/portable_random.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace dtwc::algorithms {

core::ClusteringResult clarans(Problem &prob, const CLARANSOptions &opts)
{
  // 64-bit size: the old `int N = static_cast<int>(prob.size())` silently
  // truncated for size() > INT_MAX (audit handoff-2026-06-01:24). Loop
  // counters stay `int` (signed-vs-signed against N -> no signed/unsigned
  // warning); the one int-typed RNG boundary below narrows N explicitly.
  const int64_t N = static_cast<int64_t>(prob.size());
  static_assert(sizeof(N) >= 8, "N must stay 64-bit so static_cast<int>(size()) cannot truncate (audit R4).");
  if (N <= 0)
    throw std::runtime_error("clarans: no data points in problem");
  if (opts.n_clusters <= 0 || opts.n_clusters > N)
    throw std::runtime_error(
      "clarans: n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(opts.n_clusters) + ", N=" + std::to_string(N));

  const int k = opts.n_clusters;

  // Auto max_neighbor: matches CLARANS paper heuristic.
  const int max_nb = (opts.max_neighbor < 0)
                       ? std::max(250, static_cast<int>(0.0125 * k * (N - k)))
                       : opts.max_neighbor;

  int64_t dtw_evals = 0;
  const bool has_budget = (opts.max_dtw_evals > 0);

  core::ClusteringResult best;
  best.total_cost = std::numeric_limits<double>::max();

  for (int restart = 0; restart < opts.num_local; ++restart) {
    if (has_budget && dtw_evals >= opts.max_dtw_evals) break;

    // Seed each restart deterministically: avoids inter-restart correlation
    // while keeping the whole run reproducible from opts.random_seed.
    std::mt19937_64 rng(
      static_cast<std::uint64_t>(opts.random_seed)
      + static_cast<std::uint64_t>(restart));

    // -----------------------------------------------------------------------
    // 1. Random initial medoids (sample k from [0, N) without replacement).
    // Per-restart heap allocations (indices, medoids, labels, nearest_dist)
    // are amortised: restart-count is bounded by num_local (typically 5-10)
    // and dominated by the O(max_nb * N) inner work below.
    // -----------------------------------------------------------------------
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    core::portable_shuffle(indices.begin(), indices.end(), rng);

    std::vector<int> medoids(indices.begin(), indices.begin() + k);
    std::sort(medoids.begin(), medoids.end()); // canonical order

    // -----------------------------------------------------------------------
    // 2. Initial full assignment: O(N*k) distance lookups.
    // -----------------------------------------------------------------------
    std::vector<int> labels(N);
    std::vector<double> nearest_dist(N);
    double total_cost = 0.0;

    for (int p = 0; p < N; ++p) {
      double best_d = std::numeric_limits<double>::max();
      int best_m = 0;
      for (int m = 0; m < k; ++m) {
        double d = prob.distByInd(p, medoids[m]);
        ++dtw_evals;
        if (d < best_d) {
          best_d = d;
          best_m = m;
        }
      }
      labels[p] = best_m;
      nearest_dist[p] = best_d;
      total_cost += best_d;
    }

    // -----------------------------------------------------------------------
    // 3. CLARANS swap loop.
    // -----------------------------------------------------------------------
    std::unordered_set<int> medoid_set(medoids.begin(), medoids.end());
    int neighbor_count = 0;

    // When k == N every point is already a medoid — no swap is possible.
    // Skip the swap loop entirely; the result is optimal by definition.
    const bool all_medoids = (k == N);

    int total_swaps = 0;
    const int max_total_swaps = max_nb * 10; // hard upper bound on total iterations
    while (!all_medoids && neighbor_count < max_nb && total_swaps < max_total_swaps) {
      ++total_swaps;
      if (has_budget && dtw_evals >= opts.max_dtw_evals) break;

      // Pick a random medoid slot to potentially remove.
      const int m_idx = static_cast<int>(core::portable_bounded(
        rng, static_cast<std::uint64_t>(k)));
      const int m_out = medoids[m_idx];

      // Pick a random non-medoid candidate to insert.
      int x_in;
      do {
        // x_in is an int point index; this randomized search is int-bound.
        x_in = static_cast<int>(core::portable_bounded(
          rng, static_cast<std::uint64_t>(N)));
      } while (medoid_set.count(x_in));

      // ------------------------------------------------------------------
      // Evaluate the cost delta of swapping m_out for x_in.
      //
      // For each point p:
      //   - If p's current nearest medoid is m_idx (the removed medoid):
      //       p must reassign. New nearest is min(dist(p, x_in),
      //       best distance among remaining medoids).
      //       These remaining-medoid lookups are cached; no new DTW evals.
      //   - Otherwise:
      //       p can optionally switch to x_in if it is closer.
      //       Only the dist(p, x_in) call is a new DTW eval.
      // ------------------------------------------------------------------
      double delta = 0.0;

      for (int p = 0; p < N; ++p) {
        const double d_new = prob.distByInd(p, x_in);
        ++dtw_evals;

        if (labels[p] == m_idx) {
          // p was assigned to the medoid being removed.
          // Find its new best across all remaining medoids + x_in.
          double best_remaining = d_new;
          for (int mm = 0; mm < k; ++mm) {
            if (mm == m_idx) continue;
            // distByInd is lazy-cached; repeated lookups are O(1).
            double d = prob.distByInd(p, medoids[mm]);
            if (d < best_remaining) best_remaining = d;
          }
          delta += best_remaining - nearest_dist[p];
        } else {
          // p retains its current medoid unless x_in is strictly closer.
          if (d_new < nearest_dist[p]) {
            delta += d_new - nearest_dist[p];
          }
          // else no change for this point.
        }
      }

      if (delta < -1e-12) {
        // Strictly improving swap — accept.
        medoid_set.erase(m_out);
        medoid_set.insert(x_in);
        medoids[m_idx] = x_in;

        // Recompute full assignment after the swap.
        // All distances hit the lazy cache; no new DTW evals counted.
        total_cost = 0.0;
        for (int p = 0; p < N; ++p) {
          double best_d = std::numeric_limits<double>::max();
          int best_m = 0;
          for (int m = 0; m < k; ++m) {
            double d = prob.distByInd(p, medoids[m]);
            if (d < best_d) {
              best_d = d;
              best_m = m;
            }
          }
          labels[p] = best_m;
          nearest_dist[p] = best_d;
          total_cost += best_d;
        }

        neighbor_count = 0; // Reset non-improving counter.
      } else {
        ++neighbor_count;
      }
    } // end swap loop

    // -----------------------------------------------------------------------
    // 4. Track best result across restarts.
    // -----------------------------------------------------------------------
    if (total_cost < best.total_cost) {
      best.labels = labels;
      best.medoid_indices = medoids;
      best.total_cost = total_cost;
      best.converged = all_medoids || (neighbor_count >= max_nb);
      best.iterations = restart + 1;
    }
  } // end restarts

  // 2.0 result write-back (Task 1.6): store labels/medoids/k into `prob` so
  // scoring functions work with NO manual wiring (mirrors the binding auto-wire
  // at _dtwcpp_core.cpp:744-747, which Phase 2 deletes).
  prob.set_n_clusters(opts.n_clusters);
  prob.centroids_ind = best.medoid_indices;
  prob.clusters_ind = best.labels;

  return best;
}

} // namespace dtwc::algorithms
