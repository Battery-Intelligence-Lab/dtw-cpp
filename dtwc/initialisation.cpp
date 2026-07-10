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
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "initialisation.hpp"
#include "settings.hpp"        // for randGenerator
#include "parallelisation.hpp" // for run
#include "Problem.hpp"
#include "types/Range.hpp" // for Range

#include <cstddef>   // for size_t
#include <algorithm> // for sample
#include <cassert>   // for assert
#include <limits>    // for numeric_limits
#include <numeric>   // for iota
#include <random>    // for discrete_distribution, uniform_int_di...
#include <vector>    // for vector

namespace dtwc::init {

namespace {

template <typename URBG>
void random_with_engine(Problem &prob, URBG &rng)
{
  const auto Nc = prob.n_clusters();

  if (Nc <= 0)
    throw std::runtime_error("init::random has failed. Number of clusters is " + std::to_string(Nc) + ", but it should be greater than zero.\n");
  if (prob.size() == 0 || static_cast<std::size_t>(Nc) > prob.size())
    throw std::runtime_error("init::random requires 1 <= number of clusters <= number of series");

  std::vector<int> candidate_centroids(prob.size());
  std::iota(candidate_centroids.begin(), candidate_centroids.end(), 0);
  std::shuffle(candidate_centroids.begin(), candidate_centroids.end(), rng);
  candidate_centroids.resize(static_cast<std::size_t>(Nc));

  prob.set_clusters(candidate_centroids);
}

template <typename URBG>
void kmeanspp_with_engine(Problem &prob, URBG &rng)
{
  // First cluster is selected at random, others are selected based on distance.
  const auto Nc = prob.n_clusters();

  if (Nc <= 0)
    throw std::runtime_error("init::Kmeanspp has failed. Number of clusters is " + std::to_string(Nc) + ", but it should be greater than zero.\n");
  if (prob.size() == 0 || static_cast<std::size_t>(Nc) > prob.size())
    throw std::runtime_error("init::Kmeanspp requires 1 <= number of clusters <= number of series");

  prob.centroids_ind.clear();

  std::uniform_int_distribution<int> d(0, static_cast<int>(prob.size() - 1));
  std::vector<int> candidate_centroids;
  candidate_centroids.reserve(Nc);

  candidate_centroids.push_back(d(rng));

  std::vector<data_t> distances(prob.size(), std::numeric_limits<data_t>::max());

  // Prime the lazy DenseDistanceMatrix allocation and DTW-function rebind on
  // the caller thread before `run()` enters OpenMP. Direct public calls to
  // Kmeanspp do not necessarily come through Problem::fill_distance_matrix().
  // Without this serial first lookup, workers can race in the lazy rebind and
  // corrupt the shared distance callable before their disjoint matrix writes.
  if (!prob.is_distance_matrix_filled() && prob.size() > 1) {
    const int first_centroid = candidate_centroids.front();
    const int anchor = (first_centroid == 0) ? 1 : 0;
    (void)prob.dist_by_ind(first_centroid, anchor);
  }

  auto distTask = [&](size_t i_p) {
    distances[i_p] = std::min(distances[i_p], prob.dist_by_ind(candidate_centroids.back(), static_cast<int>(i_p)));
  };

  for (int i = 1; i < Nc; i++) {
    dtwc::run(distTask, prob.size());
    std::discrete_distribution<> dd(distances.begin(), distances.end());
    candidate_centroids.push_back(static_cast<int>(dd(rng)));
  }

  prob.set_clusters(candidate_centroids);
}

} // namespace

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
  random_with_engine(prob, randGenerator);
}

void random_seeded(Problem &prob, std::uint64_t random_seed)
{
  std::mt19937_64 rng(random_seed);
  random_with_engine(prob, rng);
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
  kmeanspp_with_engine(prob, randGenerator);
}

void Kmeanspp_seeded(Problem &prob, std::uint64_t random_seed)
{
  std::mt19937_64 rng(random_seed);
  kmeanspp_with_engine(prob, rng);
}


} // namespace dtwc::init
