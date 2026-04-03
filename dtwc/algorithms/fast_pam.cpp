/**
 * @file fast_pam.cpp
 * @brief Implementation of FastPAM1 k-medoids clustering algorithm.
 *
 * @details Implements the FastPAM1 SWAP phase from:
 *   Schubert, E. & Rousseeuw, P.J. (2021). "Fast and eager k-medoids clustering:
 *   O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms."
 *   JMLR, 22(1), 4653-4688.
 *
 * The algorithm has two phases:
 *   BUILD: Initialize medoids (uses K-means++ from the existing codebase).
 *   SWAP:  Repeatedly find the best (medoid_out, candidate_in) swap that
 *          reduces total cost, using nearest/second-nearest caching for
 *          O(N) per swap candidate evaluation.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../initialisation.hpp"
#include "../parallelisation.hpp"
#include "../settings.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
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
  const std::vector<int>& medoids,
  int N,
  std::vector<int>& nearest,
  std::vector<double>& nearest_dist,
  std::vector<double>& second_dist)
{
  const int k = static_cast<int>(medoids.size());

  // Lock-free by design: each iteration writes only to nearest[p], nearest_dist[p],
  // and second_dist[p] at its own index p — no two threads access the same element.
#pragma omp parallel for schedule(static)
  for (int p = 0; p < N; ++p) {
    double best = std::numeric_limits<double>::max();
    double second_best = std::numeric_limits<double>::max();
    int best_idx = 0;

    for (int m = 0; m < k; ++m) {
      double d = prob.distByInd(p, medoids[m]);
      if (d < best) {
        second_best = best;
        best = d;
        best_idx = m;
      } else if (d < second_best) {
        second_best = d;
      }
    }

    nearest[p] = best_idx;
    nearest_dist[p] = best;
    second_dist[p] = second_best;
  }
}

/**
 * @brief Compute the total cost (sum of nearest-medoid distances).
 */
double compute_total_cost(const std::vector<double>& nearest_dist)
{
  return std::reduce(nearest_dist.begin(), nearest_dist.end(), 0.0);
}

} // anonymous namespace


core::ClusteringResult fast_pam(Problem& prob, int n_clusters, int max_iter)
{
  const int N = static_cast<int>(prob.size());

  if (N <= 0) {
    throw std::runtime_error("fast_pam: Problem has no data points.");
  }
  if (n_clusters <= 0 || n_clusters > N) {
    throw std::runtime_error(
      "fast_pam: n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(n_clusters) + ", N=" + std::to_string(N) + ".");
  }

  // Ensure the full distance matrix is computed up front for performance.
  prob.fillDistanceMatrix();

  // -------------------------------------------------------------------------
  // BUILD phase: initialize medoids using K-means++.
  // We temporarily set prob's cluster count and use the existing initializer,
  // then copy out the medoid indices and restore prob's state.
  // -------------------------------------------------------------------------
  const int orig_Nc = prob.cluster_size();
  const auto orig_centroids = prob.centroids_ind;
  const auto orig_clusters = prob.clusters_ind;

  prob.set_numberOfClusters(n_clusters);
  init::Kmeanspp(prob);
  std::vector<int> medoids = prob.centroids_ind;

  // Restore prob's state so we don't leave side effects.
  prob.set_numberOfClusters(orig_Nc);
  prob.centroids_ind = orig_centroids;
  prob.clusters_ind = orig_clusters;

  // -------------------------------------------------------------------------
  // Precompute nearest / second-nearest medoid for each point.
  // -------------------------------------------------------------------------
  std::vector<int> nearest(N);
  std::vector<double> nearest_dist(N);
  std::vector<double> second_dist(N);

  compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);

  // Build a boolean lookup for "is this point a medoid?"
  std::vector<bool> is_medoid(N, false);
  for (int m : medoids) {
    is_medoid[m] = true;
  }

  // -------------------------------------------------------------------------
  // SWAP phase (FastPAM1).
  //
  // For each iteration, evaluate all (medoid_out, candidate_in) pairs.
  // For a given pair, the change in total cost (delta) is computed in O(N)
  // using the nearest/second-nearest arrays:
  //
  //   For each point p:
  //     if nearest[p] == medoid_out_idx:
  //       // p loses its medoid -- goes to second nearest or candidate
  //       delta += min(second_dist[p], dist(p, candidate)) - nearest_dist[p]
  //     else:
  //       // p keeps its medoid, but might prefer the candidate
  //       delta += min(0, dist(p, candidate) - nearest_dist[p])
  //
  // Accept the swap with the most negative delta. Repeat until convergence.
  // -------------------------------------------------------------------------

  bool converged = false;
  int iter = 0;
  const int k = static_cast<int>(medoids.size());

  for (; iter < max_iter; ++iter) {
    double best_delta = 0.0;  // Only accept strictly negative deltas.
    int best_m_idx = -1;      // Index into medoids[] of the medoid to remove.
    int best_x_new = -1;      // Data point index of the candidate to add.

    // Evaluate all (medoid_out, candidate_in) pairs.
    // Lock-free by design: each thread uses thread-local delta_m buffers and
    // local best-swap trackers. Only the final reduction uses omp critical.
    // The outer loop over candidates x is embarrassingly parallel:
    // each candidate accumulates into its own delta_m buffer.
    #pragma omp parallel
    {
      std::vector<double> local_delta_m(k);
      double local_best_delta = 0.0;
      int local_best_m_idx = -1;
      int local_best_x_new = -1;

      #pragma omp for schedule(dynamic, 16)
      for (int x = 0; x < N; ++x) {
        if (is_medoid[x]) continue;

        std::fill(local_delta_m.begin(), local_delta_m.end(), 0.0);

        for (int p = 0; p < N; ++p) {
          double d_xp = prob.distByInd(p, x);
          int nearest_m = nearest[p];

          for (int m = 0; m < k; ++m) {
            if (m == nearest_m) {
              local_delta_m[m] += std::min(second_dist[p], d_xp) - nearest_dist[p];
            } else {
              double improvement = d_xp - nearest_dist[p];
              if (improvement < 0.0) {
                local_delta_m[m] += improvement;
              }
            }
          }
        }

        for (int m = 0; m < k; ++m) {
          if (local_delta_m[m] < local_best_delta) {
            local_best_delta = local_delta_m[m];
            local_best_m_idx = m;
            local_best_x_new = x;
          }
        }
      }

      // Reduce: find global best swap across all threads.
      #pragma omp critical
      {
        if (local_best_delta < best_delta) {
          best_delta = local_best_delta;
          best_m_idx = local_best_m_idx;
          best_x_new = local_best_x_new;
        }
      }
    } // end omp parallel

    if (best_m_idx < 0) {
      // No improving swap found -- converged.
      converged = true;
      break;
    }

    // Perform the swap.
    int old_medoid = medoids[best_m_idx];
    is_medoid[old_medoid] = false;
    is_medoid[best_x_new] = true;
    medoids[best_m_idx] = best_x_new;

    // Recompute nearest / second-nearest after the swap.
    compute_nearest_and_second(prob, medoids, N, nearest, nearest_dist, second_dist);
  }

  // -------------------------------------------------------------------------
  // Build result.
  // -------------------------------------------------------------------------
  core::ClusteringResult result;
  result.medoid_indices = medoids;
  result.labels.assign(nearest.begin(), nearest.end());

  result.total_cost = compute_total_cost(nearest_dist);
  result.iterations = iter;
  result.converged = converged;

  return result;
}

} // namespace dtwc
