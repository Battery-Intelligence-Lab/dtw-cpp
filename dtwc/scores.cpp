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
#include "base/error.hpp"
#include "base/parallelisation.hpp"
#include "types/Range.hpp"

#include <algorithm>      // for std::max, std::count_if
#include <cmath>          // for std::log
#include <cstddef>
#include <cstdint>        // for int64_t
#include <limits>         // for std::numeric_limits
#include <map>
#include <stdexcept>
#include <string>         // for std::to_string
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
std::vector<index_t> cluster_counts_checked(const Problem &prob, const char *who)
{
  const auto N = prob.size();
  const index_t Nc = prob.n_clusters();

  std::vector<index_t> counts(static_cast<std::size_t>(std::max<index_t>(Nc, 0)), 0);
  for (auto i : Range(N)) {
    const index_t c = prob.clusters_ind[i];
    if (c < 0 || c >= Nc)
      throw InvalidInput(std::string(who) + ": label " + std::to_string(c)
                               + " on point " + std::to_string(i) + " is outside [0, "
                               + std::to_string(Nc) + ").");
    ++counts[static_cast<std::size_t>(c)];
  }
  return counts;
}

/// Number of non-empty clusters in `counts`.
index_t n_realised_clusters(const std::vector<index_t> &counts)
{
  return static_cast<index_t>(std::count_if(counts.begin(), counts.end(),
                                            [](index_t c) { return c > 0; }));
}

/// Shared guard: an internal index that compares clusters needs at least two
/// NON-EMPTY ones. Throws UndefinedScore (not the plain InvalidInput of the
/// other guards) so a save path that must survive an undefined index can catch
/// exactly this case. Returns the realised count.
index_t require_two_realised(const std::vector<index_t> &counts, const char *who, const char *why)
{
  const index_t realised = n_realised_clusters(counts);
  if (realised < 2)
    throw UndefinedScore(std::string(who) + " requires at least 2 non-empty clusters; "
                                + why + " Got " + std::to_string(realised)
                                + " realised cluster(s) out of " + std::to_string(counts.size())
                                + " declared.");
  return realised;
}

// The scores' loops over the packed distances at their width (`matrix`, from DistanceMatrix::visit); every sum runs
// in double. Function templates, not generic lambdas: in a generic lambda cl 19.50 read the captures through the
// closure, even a constexpr one, and left the loops without the unrolling and FMA contraction it gives them here, so
// Calinski-Harabasz's last bits moved (2026-10-11).

/// The silhouette of each point of a clustered `prob` into `silhouettes`; `counts` holds each cluster's size.
template <typename T>
void silhouette_of(const T *matrix, const Problem &prob, const std::vector<index_t> &counts,
                   std::vector<double> &silhouettes)
{
  const auto Nb = prob.size();
  const auto Nc = prob.n_clusters();

  // A task scores a group of points (core::for_each_point): each point's distance sum to each cluster runs over
  // the points in order.
  constexpr auto G = static_cast<index_t>(core::group_rows);
  const index_t *const label = prob.clusters_ind.data();
  auto oneTask = [&](size_t q) {
    const auto first = static_cast<index_t>(q) * G;
    thread_local std::vector<double> sums; // sums[cluster * G + g]; grows, never shrinks
    sums.assign(static_cast<size_t>(Nc * G), 0.0);
    core::for_each_point(matrix, Nb, first, [&](index_t i, const double *d) {
      double *const s = sums.data() + label[i] * G;
      for (index_t g = 0; g < G; ++g) s[g] += d[g];
    });

    for (index_t g = 0; g < std::min(G, Nb - first); ++g) {
      const index_t i_b = first + g;
      const index_t i_c = label[i_b];
      if (counts[i_c] == 1) // If the profile is the only member of the cluster
        silhouettes[i_b] = 0;
      else {
        auto min = std::numeric_limits<double>::max();
        double a = 0.0;
        for (index_t i = 0; i < Nc; i++) // Finding means:
          if (i == i_c)
            a = sums[i * G + g] / (counts[i] - 1);
          else if (counts[i] > 0)
            min = std::min(min, sums[i * G + g] / counts[i]);

        // a = b = 0 is 0/0; Rousseeuw's convention is s(i) = 0 (neither well nor
        // badly placed). A NaN here would poison any downstream mean.
        const double denom = std::max(min, a);
        silhouettes[i_b] = (denom > 0.0) ? (min - a) / denom : 0.0;
      }
    }
  };

  dtwc::run(oneTask, static_cast<size_t>((Nb + G - 1) / G));
}

/// The Davies-Bouldin index of a clustered `prob` with `k_realised` non-empty clusters of sizes `cluster_counts`.
template <typename T>
double davies_bouldin_of(const T *matrix, const Problem &prob, const std::vector<index_t> &cluster_counts,
                         index_t k_realised)
{
  const auto Nc = prob.n_clusters();

  // Compute within-cluster scatter S_i = (1/|C_i|) * sum_{x in C_i} d(x, medoid_i)
  std::vector<double> scatter(Nc, 0.0);
  for (auto i : Range(prob.size())) {
    const index_t ci = prob.clusters_ind[i];
    scatter[ci] += matrix[core::tri_index(i, static_cast<std::size_t>(prob.centroids_ind[ci]))];
  }
  for (index_t c = 0; c < Nc; ++c) {
    if (cluster_counts[c] > 0)
      scatter[c] /= cluster_counts[c];
  }

  // Compute DBI = (1/k) * sum_i max_{j != i} R_ij
  // where R_ij = (S_i + S_j) / M_ij and M_ij = d(medoid_i, medoid_j).
  double dbi = 0.0;
  for (index_t i = 0; i < Nc; ++i) {
    if (cluster_counts[i] == 0) continue; // skip empty clusters
    double max_ratio = 0.0;
    for (index_t j = 0; j < Nc; ++j) {
      if (i == j || cluster_counts[j] == 0) continue;
      const double d_ij = matrix[core::tri_index(static_cast<std::size_t>(prob.centroids_ind[i]),
                                                 static_cast<std::size_t>(prob.centroids_ind[j]))];
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

/// The Dunn index of a clustered `prob`.
template <typename T>
double dunn_of(const T *matrix, const Problem &prob)
{
  // Packed row r holds the pairs (r, c), c < r, in one stretch: the rows go in blocks, in parallel, each block with
  // its own extremes. min and max pick a value and never round, so the order of the pairs changes neither.
  const index_t N = prob.size();
  const index_t *const label = prob.clusters_ind.data();
  constexpr index_t block_rows = 64;
  const index_t blocks = (N + block_rows - 1) / block_rows;
  std::vector<double> block_min(static_cast<size_t>(blocks)), block_max(static_cast<size_t>(blocks));
  auto block = [&](size_t b) {
    double lo = std::numeric_limits<double>::max();
    double hi = 0.0;
    const index_t r1 = std::min(N, static_cast<index_t>(b + 1) * block_rows);
    for (index_t r = static_cast<index_t>(b) * block_rows; r < r1; ++r) {
      const T *const row = matrix + core::packed_size(static_cast<size_t>(r));
      const index_t lr = label[r];
      for (index_t c = 0; c < r; ++c) {
        if (label[c] == lr)
          hi = std::max<double>(hi, row[c]); // Same cluster: contributes to intra-cluster diameter
        else
          lo = std::min<double>(lo, row[c]); // Different clusters: contributes to inter-cluster distance
      }
    }
    block_min[b] = lo;
    block_max[b] = hi;
  };
  run_openmp(block, static_cast<size_t>(blocks));

  double min_inter = std::numeric_limits<double>::max();
  double max_intra = 0.0;
  for (index_t b = 0; b < blocks; ++b) {
    min_inter = std::min(min_inter, block_min[b]);
    max_intra = std::max(max_intra, block_max[b]);
  }

  if (max_intra == 0.0)
    return std::numeric_limits<double>::infinity();

  return min_inter / max_intra;
}

/// The inertia of a clustered `prob`: each point's distance to its medoid, summed in point order.
template <typename T>
double inertia_of(const T *matrix, const Problem &prob)
{
  double total = 0.0;
  for (auto i : Range(prob.size())) {
    const index_t medoid = prob.centroids_ind[prob.clusters_ind[i]];
    total += matrix[core::tri_index(i, static_cast<std::size_t>(medoid))];
  }
  return total;
}

/// The medoid-adapted Calinski-Harabasz index of a clustered `prob` with `k` non-empty clusters of sizes
/// `cluster_counts`. The values' pointer comes by reference: by value, clang 21 reloaded a constant of the row sum
/// once per 8 entries, which it keeps in a register here (2026-10-11).
template <typename T>
double calinski_harabasz_of(const T *const &matrix, const Problem &prob, const std::vector<index_t> &cluster_counts,
                            index_t k)
{
  const index_t N = prob.size();
  const auto Nc = prob.n_clusters();

  // Find overall medoid: point with minimum sum of distances to all other points
  index_t overall_medoid = 0;
  double min_row_sum = std::numeric_limits<double>::max();
  for (index_t i = 0; i < N; ++i) {
    double row_sum = 0.0;
    for (index_t j = 0; j < N; ++j)
      row_sum += matrix[core::tri_index(static_cast<std::size_t>(i), static_cast<std::size_t>(j))];
    if (row_sum < min_row_sum) {
      min_row_sum = row_sum;
      overall_medoid = i;
    }
  }

  // Within-cluster scatter W = sum_c sum_{x in c} d(x, medoid_c)^2
  double W = 0.0;
  for (index_t i = 0; i < N; ++i) {
    const index_t medoid_c = prob.centroids_ind[prob.clusters_ind[i]];
    double d = matrix[core::tri_index(static_cast<std::size_t>(i), static_cast<std::size_t>(medoid_c))];
    W += d * d;
  }

  // Between-cluster scatter B = sum_c |c| * d(medoid_c, overall_medoid)^2
  const index_t *const centroid = prob.centroids_ind.data(); // GCC 13 reloaded it at each cluster (2026-10-11)
  double B = 0.0;
  for (index_t c = 0; c < Nc; ++c) {
    if (cluster_counts[c] == 0) continue; // an empty cluster has no medoid
    double d = matrix[core::tri_index(static_cast<std::size_t>(centroid[c]),
                                       static_cast<std::size_t>(overall_medoid))];
    B += static_cast<double>(cluster_counts[c]) * d * d;
  }

  if (W == 0.0)
    return std::numeric_limits<double>::infinity();
  return (B / (k - 1)) / (W / (N - k));
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
 * @throws InvalidInput if the Problem has not been clustered.
 * @see https://en.wikipedia.org/wiki/Silhouette_(clustering) for more information on silhouette scoring.
 */
std::vector<double> silhouette(Problem &prob)
{
  const auto Nb = prob.size();         //!< Number of profiles

  // v1.0.0 printed a line and returned N copies of -1, a valid-looking score.
  prob.require_clustered("silhouette");

  std::vector<double> silhouettes(Nb, -1); //!< Silhouette scores for each profile initialised to -1

  // s(i) is only defined when a SECOND non-empty cluster supplies b(i);
  // otherwise `min` stays at DBL_MAX and (min - a)/min evaluates to ~ +1.0 — a
  // "perfect" score for one cluster. sklearn raises here for the same reason.
  const auto counts = cluster_counts_checked(prob, "silhouette");
  require_two_realised(counts, "silhouette",
                       "b(i) is the mean distance to the nearest OTHER cluster, which does not "
                       "exist for a single realised cluster.");

  prob.fill_distance_matrix(); //!< We need all pairwise distance for silhouette score.
  prob.distance_matrix().visit([&](const auto *matrix) { silhouette_of(matrix, prob, counts, silhouettes); });
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

  prob.require_clustered("davies_bouldin");

  // The Davies-Bouldin index is undefined for a single cluster: R_ij needs a
  // second cluster (j != i) to form any similarity ratio, so the max_{j!=i}
  // loop below finds nothing and the index silently collapses to 0. Reject
  // Nc < 2 with a clear error instead.
  if (Nc < 2)
    throw InvalidInput(
      "davies_bouldin requires at least 2 clusters; the Davies-Bouldin index "
      "is undefined for a single cluster (no inter-cluster separation).");

  // k, i and j range over the REALISED clusters: an empty one has no meaningful
  // S_c or medoid, yet would still divide the final sum.
  const auto cluster_counts = cluster_counts_checked(prob, "davies_bouldin");
  const index_t k_realised = require_two_realised(
    cluster_counts, "davies_bouldin",
    "R_ij needs a second cluster j != i, so the index is undefined "
    "for a single realised cluster.");

  prob.fill_distance_matrix(); //!< We need all pairwise distances for the Davies-Bouldin index.

  return prob.distance_matrix().visit(
    [&](const auto *matrix) { return davies_bouldin_of(matrix, prob, cluster_counts, k_realised); });
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
  prob.require_clustered("dunn");

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
  return prob.distance_matrix().visit([&](const auto *matrix) { return dunn_of(matrix, prob); });
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
  prob.require_clustered("inertia");

  prob.fill_distance_matrix();

  return prob.distance_matrix().visit([&](const auto *matrix) { return inertia_of(matrix, prob); });
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
  prob.require_clustered("calinski_harabasz");

  const index_t N = prob.size();

  // k is the number of REALISED clusters: it sets both the (k-1) and the (N-k)
  // degrees of freedom, so an empty declared cluster would bias both.
  const auto cluster_counts = cluster_counts_checked(prob, "calinski_harabasz");
  const index_t k = n_realised_clusters(cluster_counts);

  if (k <= 1)
    throw InvalidInput("Calinski-Harabasz Index requires at least 2 clusters");
  if (N <= k)
    throw InvalidInput("Calinski-Harabasz Index requires more points than clusters");

  prob.fill_distance_matrix();

  return prob.distance_matrix().visit(
    [&](const auto *matrix) { return calinski_harabasz_of(matrix, prob, cluster_counts, k); });
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
 * @throws InvalidInput if label vectors have different sizes.
 */
double adjusted_rand(const std::vector<index_t> &labels_true,
                     const std::vector<index_t> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw InvalidInput("adjusted_rand: label vectors must have the same length");

  const auto n = static_cast<index_t>(labels_true.size());

  // Contingency table keyed by the label pair: any two labels count apart.
  std::map<index_t, index_t> a_counts, b_counts;
  std::map<std::pair<index_t, index_t>, index_t> contingency;

  for (std::size_t i = 0; i < labels_true.size(); ++i) {
    a_counts[labels_true[i]]++;
    b_counts[labels_pred[i]]++;
    contingency[{ labels_true[i], labels_pred[i] }]++;
  }

  // C(x,2) = x*(x-1)/2
  auto c2 = [](index_t x) -> index_t { return x * (x - 1) / 2; };

  index_t sum_cij2 = 0;
  for (auto &kv : contingency)
    sum_cij2 += c2(kv.second);

  index_t sum_ai2 = 0;
  for (auto &kv : a_counts)
    sum_ai2 += c2(kv.second);

  index_t sum_bj2 = 0;
  for (auto &kv : b_counts)
    sum_bj2 += c2(kv.second);

  index_t cn2 = c2(n);

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
 * @throws InvalidInput if label vectors have different sizes.
 */
double normalized_mutual_info(const std::vector<index_t> &labels_true,
                              const std::vector<index_t> &labels_pred)
{
  if (labels_true.size() != labels_pred.size())
    throw InvalidInput("normalized_mutual_info: label vectors must have the same length");

  if (labels_true.empty()) return 0.0;

  const double inv_n = 1.0 / static_cast<double>(labels_true.size());

  // Contingency table keyed by the label pair: any two labels count apart.
  std::map<index_t, index_t> a_counts, b_counts;
  std::map<std::pair<index_t, index_t>, index_t> contingency;

  for (std::size_t i = 0; i < labels_true.size(); ++i) {
    a_counts[labels_true[i]]++;
    b_counts[labels_pred[i]]++;
    contingency[{ labels_true[i], labels_pred[i] }]++;
  }

  // Marginal entropies
  double H_true = 0.0;
  for (auto &kv : a_counts) {
    double p = static_cast<double>(kv.second) * inv_n;
    H_true -= p * std::log(p);
  }

  double H_pred = 0.0;
  for (auto &kv : b_counts) {
    double p = static_cast<double>(kv.second) * inv_n;
    H_pred -= p * std::log(p);
  }

  // Mutual information
  double MI = 0.0;
  for (auto &kv : contingency) {
    double p_ij = static_cast<double>(kv.second) * inv_n;
    double p_i = static_cast<double>(a_counts[kv.first.first]) * inv_n;
    double p_j = static_cast<double>(b_counts[kv.first.second]) * inv_n;
    MI += p_ij * std::log(p_ij / (p_i * p_j));
  }

  double denom = 0.5 * (H_true + H_pred);
  if (denom == 0.0) return 1.0;

  return MI / denom;
}

double score(Problem &prob, std::string_view name)
{
  std::string key(name);
  std::transform(key.begin(), key.end(), key.begin(),
                 [](unsigned char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c); });
  if (key == "silhouette") {
    const auto values = silhouette(prob);
    double sum = 0.0;
    for (const double value : values) sum += value;
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
  }
  if (key == "davies_bouldin") return davies_bouldin(prob);
  if (key == "dunn") return dunn(prob);
  if (key == "calinski_harabasz") return calinski_harabasz(prob);
  if (key == "inertia") return inertia(prob);
  throw InvalidInput("Result::score: unknown score '" + std::string(name)
                     + "'. Valid scores: silhouette, davies_bouldin, dunn, calinski_harabasz, inertia.");
}

} // namespace dtwc::scores
