/**
 * @file scores.hpp
 * @brief Header file for calculating different types of scores in clustering algorithms.
 *
 * This file contains the declarations of functions used for calculating different types
 * of scores, focusing primarily on the silhouette score for clustering analysis. The
 * silhouette score is a measure of how well an object lies within its cluster and is
 * a common method to evaluate the validity of a clustering solution.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Problem.hpp"
#include "parallelisation.hpp"

#include <iostream>
#include <vector>
#include <cstddef>

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
auto silhouette(Problem &prob)
{
  const auto Nb = prob.data.size();    //!< Number of profiles
  const auto Nc = prob.cluster_size(); //!< Number of clusters

  std::vector<double> silhouettes(Nb);

  if (prob.centroids_ind.empty()) {
    std::cout << "Please cluster the data before calculating silhouette!" << std::endl;
    return silhouettes;
  }

  auto oneTask = [&](size_t i_b) {
    auto i_c = prob.clusters_ind[i_b];

    if (prob.cluster_members[i_c].size() == 1)
      silhouettes[i_b] = 0;
    else {
      thread_local std::vector<double> mean_distances(Nc);

      for (int i = 0; i < Nb; i++)
        mean_distances[prob.clusters_ind[i]] += prob.distByInd(i, i_b);

      auto min = std::numeric_limits<double>::max();
      for (int i = 0; i < Nc; i++) // Finding means:
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
