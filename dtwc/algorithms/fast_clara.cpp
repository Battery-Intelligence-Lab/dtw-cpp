/**
 * @file fast_clara.cpp
 * @brief Implementation of FastCLARA: scalable k-medoids via subsampling + FastPAM.
 *
 * @details For each of n_samples subsamples:
 *   1. Draw sample_size random indices from [0, N).
 *   2. Create a sub-Problem containing only the sampled series.
 *   3. Run FastPAM on the sub-Problem to find medoids.
 *   4. Map sub-Problem medoid indices back to original dataset indices.
 *   5. Assign ALL N points to the nearest medoid (computing only N*k distances).
 *   6. Track the result with the lowest total cost across all subsamples.
 *
 * References:
 *   - Kaufman & Rousseeuw (1990), "Finding Groups in Data."
 *   - Schubert & Rousseeuw (2021), JMLR 22(1), 4653-4688.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @author Claude (generated)
 * @date 08 Apr 2026
 */

#include "fast_clara.hpp"
#include "fast_pam.hpp"
#include "../Problem.hpp"
#include "../settings.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dtwc::algorithms {

namespace {

/**
 * @brief Compute effective CLARA sample size per Schubert & Rousseeuw (2021) recommendation.
 *
 * @param k Number of clusters.
 * @param N Total number of data points.
 * @return Effective sample size: max(40 + 2*k, min(N, 10*k + 100)).
 */
int clara_sample_size(int k, int N)
{
  return std::max(40 + 2 * k, std::min(N, 10 * k + 100));
}

/**
 * @brief Assign all N points to the nearest medoid, computing only N*k distances.
 *
 * @param prob          Original Problem with all N series.
 * @param medoid_indices Indices (in full dataset) of the k medoids.
 * @param[out] labels   Cluster assignment per point [0, k).
 * @return Total cost (sum of distances to nearest medoid).
 */
double assign_all_points(
  Problem& prob,
  const std::vector<int>& medoid_indices,
  std::vector<int>& labels)
{
  const int N = static_cast<int>(prob.size());
  const int k = static_cast<int>(medoid_indices.size());
  labels.resize(N);

  double total_cost = 0.0;

  // We compute DTW on-the-fly via distByInd (lazy: computes and caches).
  // This only touches N*k entries, not the full N^2 matrix.
  for (int p = 0; p < N; ++p) {
    double best_dist = std::numeric_limits<double>::max();
    int best_label = 0;

    for (int m = 0; m < k; ++m) {
      double d = prob.distByInd(p, medoid_indices[m]);
      if (d < best_dist) {
        best_dist = d;
        best_label = m;
      }
    }

    labels[p] = best_label;
    total_cost += best_dist;
  }

  return total_cost;
}

} // anonymous namespace


core::ClusteringResult fast_clara(Problem& prob, const CLARAOptions& opts)
{
  const int N = static_cast<int>(prob.size());

  if (N <= 0) {
    throw std::runtime_error("fast_clara: Problem has no data points.");
  }
  if (opts.n_clusters <= 0 || opts.n_clusters > N) {
    throw std::runtime_error(
      "fast_clara: n_clusters must be in [1, N]. Got n_clusters="
      + std::to_string(opts.n_clusters) + ", N=" + std::to_string(N) + ".");
  }
  if (opts.n_samples <= 0) {
    throw std::runtime_error("fast_clara: n_samples must be > 0.");
  }

  // Determine effective sample size.
  int sample_size = opts.sample_size;
  if (sample_size < 0) {
    sample_size = clara_sample_size(opts.n_clusters, N); // Schubert & Rousseeuw 2021
  }
  // Clamp to [k, N].
  sample_size = std::max(sample_size, opts.n_clusters);
  sample_size = std::min(sample_size, N);

  // If sample_size >= N, just run FastPAM on the full dataset.
  if (sample_size >= N) {
    return fast_pam(prob, opts.n_clusters, opts.max_iter);
  }

  std::mt19937 rng(opts.random_seed);

  // All indices [0, N).
  std::vector<int> all_indices(N);
  std::iota(all_indices.begin(), all_indices.end(), 0);

  core::ClusteringResult best_result;
  best_result.total_cost = std::numeric_limits<double>::max();

  for (int s = 0; s < opts.n_samples; ++s) {
    // 1. Draw a random subsample of indices.
    std::vector<int> sample_indices(all_indices.begin(), all_indices.end());
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(sample_size);

    // 2. Create a sub-Problem with zero-copy span views into parent data.
    std::vector<std::span<const data_t>> sub_spans;
    std::vector<std::string_view> sub_names;
    sub_spans.reserve(sample_size);
    sub_names.reserve(sample_size);

    for (int idx : sample_indices) {
      sub_spans.push_back(prob.series(idx));        // O(1), no data copy
      sub_names.push_back(prob.series_name(idx));   // O(1), no string copy
    }

    Problem sub_prob("clara_subsample_" + std::to_string(s));
    // Copy all relevant settings from the original problem.
    sub_prob.band = prob.band;
    sub_prob.variant_params = prob.variant_params;
    sub_prob.missing_strategy = prob.missing_strategy;
    sub_prob.distance_strategy = prob.distance_strategy;
    sub_prob.verbose = prob.verbose;
    sub_prob.set_view_data(Data(std::move(sub_spans), std::move(sub_names), prob.data.ndim));

    // 3. Run FastPAM on the sub-Problem.
    auto sub_result = fast_pam(sub_prob, opts.n_clusters, opts.max_iter);

    // 4. Map sub-Problem medoid indices back to full dataset indices.
    std::vector<int> full_medoids(opts.n_clusters);
    for (int m = 0; m < opts.n_clusters; ++m) {
      full_medoids[m] = sample_indices[sub_result.medoid_indices[m]];
    }

    // 5. Assign ALL N points to the nearest medoid.
    std::vector<int> labels;
    double total_cost = assign_all_points(prob, full_medoids, labels);

    // 6. Track the best result.
    if (total_cost < best_result.total_cost) {
      best_result.labels = std::move(labels);
      best_result.medoid_indices = std::move(full_medoids);
      best_result.total_cost = total_cost;
      best_result.iterations = sub_result.iterations;
      best_result.converged = sub_result.converged;
    }
  }

  return best_result;
}

} // namespace dtwc::algorithms
