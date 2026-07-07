/**
 * @file scores.hpp
 * @brief Header file for calculating different types of scores in clustering algorithms.
 *
 * @details This file contains the declarations of functions used for calculating different types
 * of scores, focusing primarily on the silhouette score for clustering analysis. The
 * silhouette score is a measure of how well an object lies within its cluster and is
 * a common method to evaluate the validity of a clustering solution.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include <vector>

namespace dtwc {
class Problem; // Pre-definition
namespace scores {
  // Canonical 2.0 names (snake_case; the redundant Index/Information noun is
  // dropped per the frozen API contract §2.4). 1.x camelCase names survive as
  // [[deprecated]] inline shims below and are removed in 3.0.
  std::vector<double> silhouette(Problem &prob);
  double davies_bouldin(Problem &prob);

  double dunn(Problem &prob);
  double inertia(Problem &prob);
  double calinski_harabasz(Problem &prob);

  double adjusted_rand(const std::vector<int> &labels_true,
                       const std::vector<int> &labels_pred);
  double normalized_mutual_info(const std::vector<int> &labels_true,
                                const std::vector<int> &labels_pred);

  // ---- Deprecated 1.x aliases (forward to canonical; removed in 3.0) ----
  [[deprecated("use scores::davies_bouldin")]]
  inline double daviesBouldinIndex(Problem &prob) { return davies_bouldin(prob); }

  [[deprecated("use scores::dunn")]]
  inline double dunnIndex(Problem &prob) { return dunn(prob); }

  [[deprecated("use scores::calinski_harabasz")]]
  inline double calinskiHarabaszIndex(Problem &prob) { return calinski_harabasz(prob); }

  [[deprecated("use scores::adjusted_rand")]]
  inline double adjustedRandIndex(const std::vector<int> &labels_true,
                                  const std::vector<int> &labels_pred)
  {
    return adjusted_rand(labels_true, labels_pred);
  }

  [[deprecated("use scores::normalized_mutual_info")]]
  inline double normalizedMutualInformation(const std::vector<int> &labels_true,
                                            const std::vector<int> &labels_pred)
  {
    return normalized_mutual_info(labels_true, labels_pred);
  }

} // namespace scores

} // namespace dtwc
