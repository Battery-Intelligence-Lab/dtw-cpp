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
  std::vector<double> silhouette(Problem &prob);
  double davies_bouldin(Problem &prob);

  double dunn(Problem &prob);
  double inertia(Problem &prob);
  double calinski_harabasz(Problem &prob);

  /// The score `name` names, ASCII case aside: silhouette (the mean over the
  /// series), davies_bouldin, dunn, calinski_harabasz or inertia.
  /// @throws InvalidInput for any other name.
  double score(Problem &prob, std::string_view name);

  double adjusted_rand(const std::vector<index_t> &labels_true,
                       const std::vector<index_t> &labels_pred);
  double normalized_mutual_info(const std::vector<index_t> &labels_true,
                                const std::vector<index_t> &labels_pred);

} // namespace scores

} // namespace dtwc
