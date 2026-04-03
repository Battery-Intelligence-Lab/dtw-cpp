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

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace dtwc::algorithms {

core::ClusteringResult clarans(Problem& prob, const CLARANSOptions& opts)
{
    const int N = static_cast<int>(prob.size());
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
        std::mt19937 rng(static_cast<unsigned>(opts.random_seed) + static_cast<unsigned>(restart));

        // -----------------------------------------------------------------------
        // 1. Random initial medoids (sample k from [0, N) without replacement).
        // -----------------------------------------------------------------------
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<int> medoids(indices.begin(), indices.begin() + k);
        std::sort(medoids.begin(), medoids.end()); // canonical order

        // -----------------------------------------------------------------------
        // 2. Initial full assignment: O(N*k) distance lookups.
        // -----------------------------------------------------------------------
        std::vector<int>    labels(N);
        std::vector<double> nearest_dist(N);
        double total_cost = 0.0;

        for (int p = 0; p < N; ++p) {
            double best_d = std::numeric_limits<double>::max();
            int    best_m = 0;
            for (int m = 0; m < k; ++m) {
                double d = prob.distByInd(p, medoids[m]);
                ++dtw_evals;
                if (d < best_d) { best_d = d; best_m = m; }
            }
            labels[p]       = best_m;
            nearest_dist[p] = best_d;
            total_cost     += best_d;
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
            const int m_idx = std::uniform_int_distribution<int>(0, k - 1)(rng);
            const int m_out = medoids[m_idx];

            // Pick a random non-medoid candidate to insert.
            int x_in;
            do {
                x_in = std::uniform_int_distribution<int>(0, N - 1)(rng);
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
                    int    best_m = 0;
                    for (int m = 0; m < k; ++m) {
                        double d = prob.distByInd(p, medoids[m]);
                        if (d < best_d) { best_d = d; best_m = m; }
                    }
                    labels[p]       = best_m;
                    nearest_dist[p] = best_d;
                    total_cost     += best_d;
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
            best.labels          = labels;
            best.medoid_indices  = medoids;
            best.total_cost      = total_cost;
            best.converged       = all_medoids || (neighbor_count >= max_nb);
            best.iterations      = restart + 1;
        }
    } // end restarts

    return best;
}

} // namespace dtwc::algorithms
