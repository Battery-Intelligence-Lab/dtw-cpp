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
 * @author Volkan Kumtepeli, Becky Perriment
 */

#include "scores.hpp"
#include "Problem.hpp"
#include "parallelisation.hpp"

#include <iostream>
#include <vector>
#include <cstddef>
#include <utility> // for pair

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
      mean_distances[prob.clusters_ind[i]].second += prob.distByInd(i, i_b);
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

} // namespace dtwc::scores
