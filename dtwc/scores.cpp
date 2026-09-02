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
#include "error.hpp"
#include "parallelisation.hpp"

#include <algorithm>      // for std::max, std::count_if
#include <cmath>          // for std::log
#include <cstddef>
#include <cstdint>        // for int64_t
#include <iostream>
#include <limits>         // for std::numeric_limits
#include <stdexcept>      // for std::invalid_argument
#include <string>         // for std::to_string
#include <unordered_map>
#include <utility>        // for pair
#include <vector>

namespace dtwc::scores {

namespace {

/**
 * @brief Member count of every declared cluster id, validating the labelling.
 *
 * All internal validity indices are defined on the REALISED partition — the set
 * of labels that actually occur — not on the declared `n_clusters`. A
 * declared-but-empty cluster has no members, hence no scatter, no medoid and no
 * b(i) contribution; counting it corrupts the 1/k normaliser and lets the
 * "at least 2 clusters" guards pass vacuously.
 */
std::vector<int> cluster_counts_checked(const Problem &prob, const char *who)
{
  const auto N = prob.size();
  const int Nc = prob.n_clusters();
  if (prob.clusters_ind.size() != N)
    throw InvalidInput(std::string(who) + ": clusters_ind holds "
                             + std::to_string(prob.clusters_ind.size()) + " labels for "
                             + std::to_string(N) + " points.");

  std::vector<int> counts(static_cast<std::size_t>(std::max(Nc, 0)), 0);
  for (auto i : Range(N)) {
    const int c = prob.clusters_ind[i];
    if (c < 0 || c >= Nc)
      throw InvalidInput(std::string(who) + ": label " + std::to_string(c)
                               + " on point " + std::to_string(i) + " is outside [0, "
                               + std::to_string(Nc) + ").");
    ++counts[static_cast<std::size_t>(c)];
  }
  return counts;
}

/// Number of non-empty clusters in `counts`.
int n_realised_clusters(const std::vector<int> &counts)
{
  return static_cast<int>(std::count_if(counts.begin(), counts.end(),
                                        [](int c) { return c > 0; }));
}

/// Shared guard: an internal index that compares clusters needs at least two
/// NON-EMPTY ones. Throws UndefinedScore (not the plain InvalidInput of the
/// other guards) so a save path that must survive an undefined index can catch
/// exactly this case. Returns the realised count.
int require_two_realised(const std::vector<int> &counts, const char *who, const char *why)
{
  const int realised = n_realised_clusters(counts);
  if (realised < 2)
    throw UndefinedScore(std::string(who) + " requires at least 2 non-empty clusters; "
                                + why + " Got " + std::to_string(realised)
                                + " realised cluster(s) out of " + std::to_string(counts.size())
                                + " declared.");
  return realised;
}

} // anonymous namespace


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
  const auto Nc = prob.n_clusters(); //!< Number of clusters

  std::vector<double> silhouettes(Nb, -1); //!< Silhouette scores for each profile initialised to -1

  if (prob.centroids_ind.empty()) {
    std::cout << "Please cluster the data before calculating silhouette!" << '\n';
    return silhouettes;
  }

  // s(i) is only defined when a SECOND non-empty cluster supplies b(i);
  // otherwise `min` stays at DBL_MAX and (min - a)/min evaluates to ~ +1.0 — a
  // "perfect" score for one cluster. sklearn raises here for the same reason.
  const auto counts = cluster_counts_checked(prob, "silhouette");
  require_two_realised(counts, "silhouette",
                       "b(i) is the mean distance to the nearest OTHER cluster, which does not "
                       "exist for a single realised cluster.");

  prob.fill_distance_matrix(); //!< We need all pairwise distance for silhouette score.

  auto oneTask = [&](size_t i_b) {
    const auto i_c = prob.clusters_ind[i_b];

    std::vector<std::pair<int, double>> mean_distances(Nc, { 0, 0 });

    for (auto i : Range(prob.size())) {
      mean_distances[prob.clusters_ind[i]].first++;
      mean_distances[prob.clusters_ind[i]].second += prob.dist_by_ind(static_cast<int>(i), static_cast<int>(i_b));
    }


    if (mean_distances[i_c].first == 1) // If the profile is the only member of the cluster
      silhouettes[i_b] = 0;
    else {
      auto min = std::numeric_limits<double>::max();
      for (int i = 0; i < Nc; i++) // Finding means:
        if (i == i_c)
          mean_distances[i].second /= (mean_distances[i].first - 1);
        else if (mean_distances[i].first > 0) {
          mean_distances[i].second /= mean_distances[i].first;
          min = std::min(min, mean_distances[i].second);
        }

      // a = b = 0 is 0/0; Rousseeuw's convention is s(i) = 0 (neither well nor
      // badly placed). A NaN here would poison any downstream mean.
      const double a = mean_distances[i_c].second;
      const double denom = std::max(min, a);
      silhouettes[i_b] = (denom > 0.0) ? (min - a) / denom : 0.0;
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
 * @note Requires that the data has already been clustered; throws dtwc::InvalidInput if centroids are not set.
 * @see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index for more information on the Davies-Bouldin index.
 */
double davies_bouldin(Problem &prob)
{
  const auto Nc = prob.n_clusters(); //!< Number of clusters

  if (prob.centroids_ind.empty()) {
    throw InvalidInput("Cluster before calculating DBI");
  }

  // The Davies-Bouldin index is undefined for a single cluster: R_ij needs a
  // second cluster (j != i) to form any similarity ratio, so the max_{j!=i}
  // loop below finds nothing and the index silently collapses to 0. Reject
  // Nc < 2 with a clear error instead (audit handoff-2026-06-01:25).
  if (Nc < 2)
    throw InvalidInput(
      "davies_bouldin requires at least 2 clusters; the Davies-Bouldin index "
      "is undefined for a single cluster (no inter-cluster separation).");

  // k, i and j range over the REALISED clusters: an empty one has no meaningful
  // S_c or medoid, yet would still divide the final sum.
  const auto cluster_counts = cluster_counts_checked(prob, "davies_bouldin");
  const int k_realised = require_two_realised(
    cluster_counts, "davies_bouldin",
    "R_ij needs a second cluster j != i, so the index is undefined "
    "for a single realised cluster.");

  prob.fill_distance_matrix(); //!< We need all pairwise distances for the Davies-Bouldin index.

  // Compute within-cluster scatter S_i = (1/|C_i|) * sum_{x in C_i} d(x, medoid_i)
  std::vector<double> scatter(Nc, 0.0);
  for (auto i : Range(prob.size())) {
    int ci = prob.clusters_ind[i];
    scatter[ci] += prob.dist_by_ind(static_cast<int>(i), prob.centroids_ind[ci]);
  }
  for (int c = 0; c < Nc; ++c) {
    if (cluster_counts[c] > 0)
      scatter[c] /= cluster_counts[c];
  }

  // Compute DBI = (1/k) * sum_i max_{j != i} R_ij
  // where R_ij = (S_i + S_j) / M_ij and M_ij = d(medoid_i, medoid_j).
  double dbi = 0.0;
  for (int i = 0; i < Nc; ++i) {
    if (cluster_counts[i] == 0) continue; // skip empty clusters
    double max_ratio = 0.0;
    for (int j = 0; j < Nc; ++j) {
      if (i == j || cluster_counts[j] == 0) continue;
      const double d_ij = prob.dist_by_ind(prob.centroids_ind[i], prob.centroids_ind[j]);
      const double combined_scatter = scatter[i] + scatter[j];
      // M_ij == 0 must NOT skip the pair. Davies & Bouldin (1979) require R_ij to
      // be strictly decreasing in M_ij with R_ij = 0 iff S_i = S_j = 0, so the
      // M_ij -> 0 limit is +inf when the clusters have any spread and 0 in the
      // 0/0 case (their axiom 3). Skipping it reported the worst configuration
      // — coincident medoids with real spread — as a perfect DBI = 0.
      double ratio;
      if (d_ij > 0.0)
        ratio = combined_scatter / d_ij;
      else
        ratio = (combined_scatter > 0.0) ? std::numeric_limits<double>::infinity() : 0.0;
      max_ratio = std::max(max_ratio, ratio);
    }
    dbi += max_ratio;
  }
  return dbi / k_realised;
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
double dunn(Problem &prob)
{
  if (prob.centroids_ind.empty())
    throw InvalidInput("Cluster before calculating Dunn Index");

  // The Dunn index is min(inter-cluster distance) / max(intra-cluster diameter).
  // With a single cluster there are no inter-cluster pairs, so min_inter stays
  // at numeric_limits::max() and the result is a meaningless huge value (or
  // +inf). The count must be of REALISED clusters: on declared n_clusters,
  // Nc = 3 with only label 0 in use passed the guard and returned ~1.8e308.
  const auto counts = cluster_counts_checked(prob, "dunn");
  require_two_realised(counts, "dunn",
                       "the index is a ratio of inter- to intra-cluster distances and no "
                       "inter-cluster pair exists.");

  prob.fill_distance_matrix();

  const auto N = static_cast<int>(prob.size());

  double min_inter = std::numeric_limits<double>::max();
  double max_intra = 0.0;

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      const double d = prob.dist_by_ind(i, j);
      if (prob.clusters_ind[i] == prob.clusters_ind[j]) 
        max_intra = std::max(max_intra, d); // Same cluster: contributes to intra-cluster diameter
       else 
        min_inter = std::min(min_inter, d); // Different clusters: contributes to inter-cluster distance
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
    throw InvalidInput("Cluster before calculating inertia");

  prob.fill_distance_matrix();

  double total = 0.0;
  for (auto i : Range(prob.size())) {
    int medoid = prob.centroids_ind[prob.clusters_ind[i]];
    total += prob.dist_by_ind(static_cast<int>(i), medoid);
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
double calinski_harabasz(Problem &prob)
{
  if (prob.centroids_ind.empty())
    throw InvalidInput("Cluster before calculating Calinski-Harabasz Index");

  const auto N = static_cast<int>(prob.size());
  const auto Nc = prob.n_clusters();

  // k is the number of REALISED clusters: it sets both the (k-1) and the (N-k)
  // degrees of freedom, so an empty declared cluster would bias both.
  const auto cluster_counts = cluster_counts_checked(prob, "calinski_harabasz");
  const int k = n_realised_clusters(cluster_counts);

  if (k <= 1)
    throw InvalidInput("Calinski-Harabasz Index requires at least 2 clusters");
  if (N <= k)
    throw InvalidInput("Calinski-Harabasz Index requires more points than clusters");

  prob.fill_distance_matrix();

  // Find overall medoid: point with minimum sum of distances to all other points
  int overall_medoid = 0;
  double min_row_sum = std::numeric_limits<double>::max();
  for (int i = 0; i < N; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < N; ++j)
      row_sum += prob.dist_by_ind(i, j);
    if (row_sum < min_row_sum) {
      min_row_sum = row_sum;
      overall_medoid = i;
    }
  }

  // Within-cluster scatter W = sum_c sum_{x in c} d(x, medoid_c)^2
  double W = 0.0;
  for (int i = 0; i < N; ++i) {
    int medoid_c = prob.centroids_ind[prob.clusters_ind[i]];
    double d = prob.dist_by_ind(i, medoid_c);
    W += d * d;
  }

  // Between-cluster scatter B = sum_c |c| * d(medoid_c, overall_medoid)^2
  double B = 0.0;
  for (int c = 0; c < Nc; ++c) {
    if (cluster_counts[c] == 0) continue; // an empty cluster has no medoid
    double d = prob.dist_by_ind(prob.centroids_ind[c], overall_medoid);
    B += cluster_counts[c] * d * d;
  }

  if (W == 0.0)
    return std::numeric_limits<double>::infinity();
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
double adjusted_rand(const std::vector<int> &labels_true,
                     const std::vector<int> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw std::invalid_argument("adjusted_rand: label vectors must have the same length");

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
double normalized_mutual_info(const std::vector<int> &labels_true,
                              const std::vector<int> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw std::invalid_argument("normalized_mutual_info: label vectors must have the same length");

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
