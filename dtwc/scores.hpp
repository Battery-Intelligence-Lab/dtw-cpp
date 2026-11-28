/**
 * @file scores.hpp
 * @brief Internal and external clustering validity scores.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "base/settings.hpp" // index_t

#include <string_view>
#include <vector>

namespace dtwc {
class Problem;
namespace scores {
  /// Compute per-point silhouette scores in [-1, 1] for a clustered @p prob (fills distance matrix if needed).
  /// @throws InvalidInput if not clustered; UndefinedScore if fewer than 2 non-empty clusters exist.
  std::vector<double> silhouette(Problem &prob);
  /// Compute the Davies-Bouldin index (lower indicates better separation) for a clustered @p prob.
  /// @throws InvalidInput if not clustered; UndefinedScore if fewer than 2 non-empty clusters exist.
  double davies_bouldin(Problem &prob);

  /// Compute the Dunn index (min inter-cluster distance / max intra-cluster diameter; higher is better).
  /// @throws InvalidInput if not clustered; UndefinedScore if fewer than 2 non-empty clusters exist.
  double dunn(Problem &prob);
  /// Compute total within-cluster inertia (sum of distances from each point to its medoid).
  /// @throws InvalidInput if @p prob has not been clustered.
  double inertia(Problem &prob);
  /// Compute the medoid-adapted Calinski-Harabasz index (ratio of between- to within-cluster variance; higher is better).
  /// @throws InvalidInput if not clustered or N <= k; UndefinedScore if fewer than 2 non-empty clusters exist.
  double calinski_harabasz(Problem &prob);

  /// The score `name` names, ASCII case aside: silhouette (the mean over the
  /// series), davies_bouldin, dunn, calinski_harabasz or inertia.
  /// @throws InvalidInput for any other name.
  double score(Problem &prob, std::string_view name);

  /// Compute the Adjusted Rand Index (ARI) between ground-truth and predicted cluster labels (1.0 = exact match).
  /// @throws InvalidInput if @p labels_true and @p labels_pred have different sizes.
  double adjusted_rand(const std::vector<index_t> &labels_true,
                       const std::vector<index_t> &labels_pred);
  /// Compute the arithmetic-mean Normalized Mutual Information (NMI) in [0, 1] between two labelings.
  /// @throws InvalidInput if @p labels_true and @p labels_pred have different sizes.
  double normalized_mutual_info(const std::vector<index_t> &labels_true,
                                const std::vector<index_t> &labels_pred);

} // namespace scores

} // namespace dtwc
