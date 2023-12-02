/*
 * scores.hpp
 *
 * Calculating different type of scores

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Problem.hpp"
#include "parallelisation.hpp"

#include <iostream>

namespace dtwc::scores {

auto silhouette(Problem &prob)
{
  // For explanation, see: https://en.wikipedia.org/wiki/Silhouette_(clustering)

  const auto Nb = prob.data.size();    // Number of profiles
  const auto Nc = prob.cluster_size(); // Number of clusters

  std::vector<double> silhouettes(Nb);

  if (prob.centroids_ind.empty()) {
    std::cout << "Please cluster the data before calculating silhouette!\n";
    return silhouettes;
  }


  auto oneTask = [&, N = Nb](size_t i_b) {
    auto i_c = prob.clusters_ind[i_b];

    if (prob.cluster_members[i_c].size() == 1)
      silhouettes[i_b] = 0;
    else {
      thread_local std::vector<double> mean_distances(Nc);

      for (size_t i = 0; i < Nb; i++)
        mean_distances[prob.clusters_ind[i]] += prob.distByInd(i, i_b);

      auto min = std::numeric_limits<double>::max();
      for (size_t i = 0; i < Nc; i++) // Finding means:
        if (i == i_c)
          mean_distances[i] /= (prob.cluster_members[i].size() - 1);
        else {
          mean_distances[i] /= prob.cluster_members[i].size();
          min = std::min(min, mean_distances[i]);
        }

      silhouettes[i_b] = (min - mean_distances[i_c]) / std::max(min, mean_distances[i_c]);
    }
  };

  dtwc::run(oneTask, Nb);

  return silhouettes;
}

} // namespace dtwc::scores
