#include <stdexcept> // Add missing include directive for the <stdexcept> header file

/**
 * @file initialisation.cpp
 *
 * @brief Implementation of initialization algorithms for clustering problems.
 * This file includes functions for random initialization and K-means++ initialization
 * of clusters in a clustering problem.
 *
 * @details
 * The functions defined in this file provide means to initialize cluster centroids
 * for a given Problem instance. The initialization is a critical step in clustering
 * algorithms, impacting their performance and outcomes. Two methods are implemented:
 * 1. Random initialization, where cluster centroids are randomly chosen.
 * 2. K-means++ initialization, which is a smarter way to initialize centroids
 *    by considering distances between data points.
 *
 * @note
 * It is assumed that the Problem class and its associated functions and members
 * are defined elsewhere and are being properly included in this file.
 *
 * @date 19 Jan 2021
 * @author Volkan Kumtepeli, Becky Perriment
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

/**
 * @brief Randomly initializes the cluster centroids for a given problem.
 *
 * @param prob Reference to the Problem object whose clusters are to be initialized.
 *
 * @exception std::runtime_error if the number of clusters (Nc) is non-positive.
 *
 * @details
 * This function randomly selects cluster centroids from the range of data indices.
 * It first checks if the number of clusters is valid (greater than zero), reserves
 * space for candidate centroids, clears any existing cluster assignments, and then
 * randomly selects centroids.
 */
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

/**
 * @brief Initialises cluster centroids using the K-means++ algorithm.
 *
 * @param prob Reference to the Problem object whose clusters are to be initialized.
 *
 * @exception std::runtime_error if the number of clusters (Nc) is non-positive.
 *
 * @details
 * Implements the K-means++ algorithm for initializing clusters. The first centroid
 * is chosen randomly, and subsequent centroids are chosen based on the distance
 * from existing centroids. This method aims to provide a better initial condition
 * for clustering algorithms, potentially leading to better final clusters.
 */
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