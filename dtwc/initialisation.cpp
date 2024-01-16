/*
 * initialisation.cpp
 *
 * Header file for initialisation functions.

 *  Created on: 19 Jan 2021
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "initialisation.hpp"
#include "settings.hpp"        // for randGenerator
#include "parallelisation.hpp" // for run
#include "Problem.hpp"
#include "types/Range.hpp" // for Range

#include <cstddef>   // for size_t
#include <algorithm> // for sample
#include <cassert>   // for assert
#include <iterator>  // for back_inserter
#include <limits>    // for numeric_limits
#include <random>    // for discrete_distribution, uniform_int_di...
#include <vector>    // for vector

namespace dtwc::init {

void random(Problem &prob)
{
  const auto Nc = prob.cluster_size();

  if (Nc <= 0)
    throw std::runtime_error("init::random has failed. Number of clusters is " + std::to_string(Nc) + ", but it should be greater than zero.\n");

  std::vector<int> candidate_centroids;
  candidate_centroids.reserve(prob.cluster_size());
  auto range = Range(prob.size());
  std::sample(range.begin(), range.end(), std::back_inserter(candidate_centroids), Nc, randGenerator);

  prob.set_clusters(candidate_centroids);
}

void Kmeanspp(Problem &prob)
{
  // First cluster is selected at random, others are selected based on distance.
  const auto Nc = prob.cluster_size();

  if (Nc <= 0)
    throw std::runtime_error("init::Kmeanspp has failed. Number of clusters is " + std::to_string(Nc) + ", but it should be greater than zero.\n");

  prob.centroids_ind.clear();

  std::uniform_int_distribution<size_t> d(0, prob.size() - 1);
  std::vector<int> candidate_centroids;
  candidate_centroids.reserve(Nc);

  candidate_centroids.push_back(d(randGenerator));

  std::vector<data_t> distances(prob.size(), std::numeric_limits<data_t>::max());

  auto distTask = [&](int i_p) {
    distances[i_p] = std::min(distances[i_p], prob.distByInd(candidate_centroids.back(), i_p));
  };

  for (int i = 1; i < Nc; i++) {
    dtwc::run(distTask, prob.size());
    std::discrete_distribution<> dd(distances.begin(), distances.end());
    candidate_centroids.push_back(dd(randGenerator));
  }

  prob.set_clusters(candidate_centroids);
}


} // namespace dtwc::init