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

#include "base/settings.hpp" // index_t

#include <vector>

namespace dtwc {
class Problem; // Pre-definition
namespace scores {
  // snake_case names; the redundant Index/Information noun is dropped.
  std::vector<double> silhouette(Problem &prob);
  double davies_bouldin(Problem &prob);

  double dunn(Problem &prob);
  double inertia(Problem &prob);
  double calinski_harabasz(Problem &prob);

  double adjusted_rand(const std::vector<index_t> &labels_true,
                       const std::vector<index_t> &labels_pred);
  double normalized_mutual_info(const std::vector<index_t> &labels_true,
                                const std::vector<index_t> &labels_pred);

} // namespace scores

} // namespace dtwc
