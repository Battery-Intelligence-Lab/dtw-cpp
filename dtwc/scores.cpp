/**
 * @file scores.cpp
 * @brief Header file for calculating different types of scores in clustering algorithms.
 *
 * This file contains the declarations of functions used for calculating different types
 * of scores, focusing primarily on the silhouette score for clustering analysis. The
 * silhouette score is a measure of how well an object lies within its cluster and is
 * a common method to evaluate the validity of a clustering solution.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "scores.hpp"
#include "Problem.hpp"
#include "parallelisation.hpp"

#include <algorithm>      // for std::max
#include <cmath>          // for std::log
#include <cstddef>
#include <cstdint>        // for int64_t
#include <iostream>
#include <limits>         // for std::numeric_limits
#include <stdexcept>      // for std::runtime_error
#include <unordered_map>
#include <utility>        // for pair
#include <vector>

namespace dtwc::scores {

/**
 * @brief Calculates the silhouette score for each data point in a given clustering problem.
 *
 * The silhouette score is a measure of how similar an object is to its own cluster (cohesion)
 * compared to other clusters (separation). The score ranges from -1 to 1, where a high value
 * indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
 *
 * @param prob The clustering problem instance, which contains the data points, cluster indices, and centroids.
 * @return std::vector<double> A vector of silhouette scores for each data point.
 *
 * @note Requires that the data has already been clustered; if not, it will prompt the user to cluster the data first.
 * @see https://en.wikipedia.org/wiki/Silhouette_(clustering) for more information on silhouette scoring.
 */
std::vector<double> silhouette(Problem &prob)
{
  const auto Nb = prob.size();         //!< Number of profiles
  const auto Nc = prob.cluster_size(); //!< Number of clusters

  std::vector<double> silhouettes(Nb, -1); //!< Silhouette scores for each profile initialised to -1

  if (prob.centroids_ind.empty()) {
    std::cout << "Please cluster the data before calculating silhouette!" << std::endl;
    return silhouettes;
  }

  prob.fillDistanceMatrix(); //!< We need all pairwise distance for silhouette score.

  auto oneTask = [&](size_t i_b) {
    const auto i_c = prob.clusters_ind[i_b];

    thread_local std::vector<std::pair<int, double>> mean_distances(Nc);
    mean_distances.assign(Nc, { 0, 0 });

    for (auto i : Range(prob.size())) {
      mean_distances[prob.clusters_ind[i]].first++;
      mean_distances[prob.clusters_ind[i]].second += prob.distByInd(static_cast<int>(i), static_cast<int>(i_b));
    }


    if (mean_distances[i_c].first == 1) // If the profile is the only member of the cluster
      silhouettes[i_b] = 0;
    else {
      auto min = std::numeric_limits<double>::max();
      for (int i = 0; i < Nc; i++) // Finding means:
        if (i == i_c)
          mean_distances[i].second /= (mean_distances[i].first - 1);
        else {
          mean_distances[i].second /= mean_distances[i].first;
          min = std::min(min, mean_distances[i].second);
        }

      silhouettes[i_b] = (min - mean_distances[i_c].second) / std::max(min, mean_distances[i_c].second);
    }
  };

  dtwc::run(oneTask, prob.size());

  return silhouettes;
}

/**
 * @brief Calculates the Davies-Bouldin index for a given clustering problem.
 *
 * The Davies-Bouldin index is a measure of the average similarity between clusters and the
 * dissimilarity between clusters. It is used to evaluate the quality of a clustering solution,
 * with a lower value indicating better separation between clusters.
 *
 * @param prob The clustering problem instance, which contains the data points, cluster indices, and centroids.
 * @return double The Davies-Bouldin index.
 *
 * @note Requires that the data has already been clustered; throws std::runtime_error if centroids are not set.
 * @see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index for more information on the Davies-Bouldin index.
 */
double daviesBouldinIndex(Problem &prob)
{
  const auto Nc = prob.cluster_size(); //!< Number of clusters

  if (prob.centroids_ind.empty()) {
    throw std::runtime_error("Cluster before calculating DBI");
  }

  prob.fillDistanceMatrix(); //!< We need all pairwise distances for the Davies-Bouldin index.

  // Compute within-cluster scatter S_i = (1/|C_i|) * sum_{x in C_i} d(x, medoid_i)
  std::vector<double> scatter(Nc, 0.0);
  std::vector<int> cluster_counts(Nc, 0);
  for (auto i : Range(prob.size())) {
    int ci = prob.clusters_ind[i];
    scatter[ci] += prob.distByInd(static_cast<int>(i), prob.centroids_ind[ci]);
    cluster_counts[ci]++;
  }
  for (int c = 0; c < Nc; ++c) {
    if (cluster_counts[c] > 0)
      scatter[c] /= cluster_counts[c];
  }

  // Compute DBI = (1/k) * sum_i max_{j != i} R_ij
  // where R_ij = (S_i + S_j) / d(medoid_i, medoid_j)
  double dbi = 0.0;
  for (int i = 0; i < Nc; ++i) {
    double max_ratio = 0.0;
    for (int j = 0; j < Nc; ++j) {
      if (i != j) {
        double d_ij = prob.distByInd(prob.centroids_ind[i], prob.centroids_ind[j]);
        if (d_ij > 0) {
          double ratio = (scatter[i] + scatter[j]) / d_ij;
          max_ratio = std::max(max_ratio, ratio);
        }
      }
    }
    dbi += max_ratio;
  }
  return dbi / Nc;
}

/**
 * @brief Computes the Dunn Index for a clustering.
 *
 * Dunn = min(inter-cluster distance) / max(intra-cluster diameter).
 * Higher values indicate better-separated, more compact clusters.
 *
 * @param prob The clustered problem instance.
 * @return double Dunn index, or infinity if max intra-cluster diameter is zero.
 */
double dunnIndex(Problem &prob)
{
  if (prob.centroids_ind.empty())
    throw std::runtime_error("Cluster before calculating Dunn Index");

  prob.fillDistanceMatrix();

  const auto N = static_cast<int>(prob.size());

  double min_inter = std::numeric_limits<double>::max();
  double max_intra = 0.0;

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      double d = prob.distByInd(i, j);
      if (prob.clusters_ind[i] == prob.clusters_ind[j]) {
        // Same cluster: contributes to intra-cluster diameter
        max_intra = std::max(max_intra, d);
      } else {
        // Different clusters: contributes to inter-cluster distance
        min_inter = std::min(min_inter, d);
      }
    }
  }

  if (max_intra == 0.0)
    return std::numeric_limits<double>::infinity();

  return min_inter / max_intra;
}

/**
 * @brief Computes the total inertia (within-cluster sum of distances to medoids).
 *
 * Inertia = sum_i d(i, medoid_of_cluster(i)).
 * Lower values indicate tighter clusters.
 *
 * @param prob The clustered problem instance.
 * @return double Total inertia.
 */
double inertia(Problem &prob)
{
  if (prob.centroids_ind.empty())
    throw std::runtime_error("Cluster before calculating inertia");

  prob.fillDistanceMatrix();

  double total = 0.0;
  for (auto i : Range(prob.size())) {
    int medoid = prob.centroids_ind[prob.clusters_ind[i]];
    total += prob.distByInd(static_cast<int>(i), medoid);
  }
  return total;
}

/**
 * @brief Computes the medoid-adapted Calinski-Harabasz Index.
 *
 * CH = (B / (k-1)) / (W / (N-k))
 * where B is the between-cluster scatter and W is the within-cluster scatter,
 * both computed using squared distances to medoids rather than Euclidean
 * distances to means.
 *
 * @param prob The clustered problem instance.
 * @return double Calinski-Harabasz index.
 */
double calinskiHarabaszIndex(Problem &prob)
{
  if (prob.centroids_ind.empty())
    throw std::runtime_error("Cluster before calculating Calinski-Harabasz Index");

  const auto N = static_cast<int>(prob.size());
  const auto k = prob.cluster_size();

  if (k <= 1)
    throw std::runtime_error("Calinski-Harabasz Index requires at least 2 clusters");
  if (N <= k)
    throw std::runtime_error("Calinski-Harabasz Index requires more points than clusters");

  prob.fillDistanceMatrix();

  // Find overall medoid: point with minimum sum of distances to all other points
  int overall_medoid = 0;
  double min_row_sum = std::numeric_limits<double>::max();
  for (int i = 0; i < N; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < N; ++j)
      row_sum += prob.distByInd(i, j);
    if (row_sum < min_row_sum) {
      min_row_sum = row_sum;
      overall_medoid = i;
    }
  }

  // Within-cluster scatter W = sum_c sum_{x in c} d(x, medoid_c)^2
  double W = 0.0;
  for (int i = 0; i < N; ++i) {
    int medoid_c = prob.centroids_ind[prob.clusters_ind[i]];
    double d = prob.distByInd(i, medoid_c);
    W += d * d;
  }

  // Between-cluster scatter B = sum_c |c| * d(medoid_c, overall_medoid)^2
  std::vector<int> cluster_counts(k, 0);
  for (int i = 0; i < N; ++i)
    cluster_counts[prob.clusters_ind[i]]++;

  double B = 0.0;
  for (int c = 0; c < k; ++c) {
    double d = prob.distByInd(prob.centroids_ind[c], overall_medoid);
    B += cluster_counts[c] * d * d;
  }

  return (B / (k - 1)) / (W / (N - k));
}

/**
 * @brief Computes the Adjusted Rand Index between two label assignments.
 *
 * ARI measures the similarity between two clusterings, adjusted for chance.
 * ARI = 1.0 for identical labelings, ~0.0 for random labelings.
 *
 * @param labels_true Ground-truth cluster labels.
 * @param labels_pred Predicted cluster labels.
 * @return double ARI value.
 * @throws std::invalid_argument if label vectors have different sizes.
 */
double adjustedRandIndex(const std::vector<int> &labels_true,
                         const std::vector<int> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw std::invalid_argument("adjustedRandIndex: label vectors must have the same length");

  const auto n = static_cast<int64_t>(labels_true.size());

  // Build contingency table using pair keys
  std::unordered_map<int, int> a_counts, b_counts;
  std::unordered_map<int64_t, int> contingency;

  for (int i = 0; i < static_cast<int>(labels_true.size()); ++i) {
    a_counts[labels_true[i]]++;
    b_counts[labels_pred[i]]++;
    // Encode pair as a single 64-bit key (assumes labels fit in 32-bit int)
    int64_t key = (static_cast<int64_t>(labels_true[i]) << 32) | static_cast<uint32_t>(labels_pred[i]);
    contingency[key]++;
  }

  // C(x,2) = x*(x-1)/2
  auto c2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };

  int64_t sum_cij2 = 0;
  for (auto &kv : contingency)
    sum_cij2 += c2(kv.second);

  int64_t sum_ai2 = 0;
  for (auto &kv : a_counts)
    sum_ai2 += c2(kv.second);

  int64_t sum_bj2 = 0;
  for (auto &kv : b_counts)
    sum_bj2 += c2(kv.second);

  int64_t cn2 = c2(n);

  // expected = sum_ai2 * sum_bj2 / C(n,2)
  double expected = (cn2 > 0) ? static_cast<double>(sum_ai2) * static_cast<double>(sum_bj2) / static_cast<double>(cn2) : 0.0;
  double max_val = 0.5 * (static_cast<double>(sum_ai2) + static_cast<double>(sum_bj2));
  double numerator = static_cast<double>(sum_cij2) - expected;
  double denominator = max_val - expected;

  if (denominator == 0.0)
    return 1.0; // Perfect agreement (or degenerate case)

  return numerator / denominator;
}

/**
 * @brief Computes the Normalized Mutual Information between two label assignments.
 *
 * NMI = MI / (0.5 * (H_true + H_pred))
 * where MI is the mutual information and H_true, H_pred are the marginal entropies.
 *
 * @param labels_true Ground-truth cluster labels.
 * @param labels_pred Predicted cluster labels.
 * @return double NMI in [0, 1]. Returns 1.0 if both labelings are constant.
 * @throws std::invalid_argument if label vectors have different sizes.
 */
double normalizedMutualInformation(const std::vector<int> &labels_true,
                                   const std::vector<int> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw std::invalid_argument("normalizedMutualInformation: label vectors must have the same length");

  const auto n = static_cast<int>(labels_true.size());
  if (n == 0) return 0.0;

  const double inv_n = 1.0 / n;

  std::unordered_map<int, int> a_counts, b_counts;
  std::unordered_map<int64_t, int> contingency;

  for (int i = 0; i < n; ++i) {
    a_counts[labels_true[i]]++;
    b_counts[labels_pred[i]]++;
    int64_t key = (static_cast<int64_t>(labels_true[i]) << 32) | static_cast<uint32_t>(labels_pred[i]);
    contingency[key]++;
  }

  // Marginal entropies
  double H_true = 0.0;
  for (auto &kv : a_counts) {
    double p = kv.second * inv_n;
    H_true -= p * std::log(p);
  }

  double H_pred = 0.0;
  for (auto &kv : b_counts) {
    double p = kv.second * inv_n;
    H_pred -= p * std::log(p);
  }

  // Mutual information
  double MI = 0.0;
  for (auto &kv : contingency) {
    int row_key = static_cast<int>(kv.first >> 32);
    int col_key = static_cast<int>(kv.first & 0xFFFFFFFF);
    double p_ij = kv.second * inv_n;
    double p_i = a_counts[row_key] * inv_n;
    double p_j = b_counts[col_key] * inv_n;
    MI += p_ij * std::log(p_ij / (p_i * p_j));
  }

  double denom = 0.5 * (H_true + H_pred);
  if (denom == 0.0) return 1.0;

  return MI / denom;
}

} // namespace dtwc::scores
