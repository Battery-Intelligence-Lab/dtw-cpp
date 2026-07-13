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
#include "core/portable_random.hpp"
#include "core/distance_sampling_weights.hpp"
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

int first_unselected(std::size_t size, const std::vector<int> &selected)
{
  for (std::size_t candidate = 0; candidate < size; ++candidate) {
    if (std::find(selected.begin(), selected.end(),
                  static_cast<int>(candidate)) == selected.end())
      return static_cast<int>(candidate);
  }
  throw std::logic_error("initialization exhausted all candidate indices");
}

template <typename Shuffle>
void random_with(Problem &prob, Shuffle &shuffle)
{
  const auto Nc = prob.n_clusters();

  if (Nc <= 0)
    throw std::runtime_error("init::random has failed. Number of clusters is " + std::to_string(Nc) + ", but it should be greater than zero.\n");
  if (prob.size() == 0 || static_cast<std::size_t>(Nc) > prob.size())
    throw std::runtime_error("init::random requires 1 <= number of clusters <= number of series");

  std::vector<int> candidate_centroids(prob.size());
  std::iota(candidate_centroids.begin(), candidate_centroids.end(), 0);
  shuffle(candidate_centroids.begin(), candidate_centroids.end());
  candidate_centroids.resize(static_cast<std::size_t>(Nc));

  prob.set_clusters(candidate_centroids);
}

template <typename FirstIndex, typename WeightedIndex>
void kmeanspp_with(Problem &prob, FirstIndex &first_index,
                   WeightedIndex &weighted_index)
{
  // First cluster is selected at random, others are selected based on distance.
  const auto Nc = prob.n_clusters();

  if (Nc <= 0)
    throw std::runtime_error("init::Kmeanspp has failed. Number of clusters is " + std::to_string(Nc) + ", but it should be greater than zero.\n");
  if (prob.size() == 0 || static_cast<std::size_t>(Nc) > prob.size())
    throw std::runtime_error("init::Kmeanspp requires 1 <= number of clusters <= number of series");

  prob.centroids_ind.clear();

  std::vector<int> candidate_centroids;
  candidate_centroids.reserve(Nc);

  candidate_centroids.push_back(first_index(prob.size()));

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
    const auto weights = core::distance_sampling_weights(
      distances, candidate_centroids, "init::Kmeanspp");
    candidate_centroids.push_back(
      weighted_index(weights.values, weights.total, candidate_centroids));
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
  auto shuffle = [](auto first, auto last) {
    std::shuffle(first, last, randGenerator);
  };
  random_with(prob, shuffle);
}

void random_seeded(Problem &prob, std::uint64_t random_seed)
{
  std::mt19937_64 rng(random_seed);
  auto shuffle = [&rng](auto first, auto last) {
    core::portable_shuffle(first, last, rng);
  };
  random_with(prob, shuffle);
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
  auto first_index = [](std::size_t size) {
    std::uniform_int_distribution<int> distribution(
      0, static_cast<int>(size - 1));
    return distribution(randGenerator);
  };
  auto weighted_index = [](const auto &distances, double total,
                           const std::vector<int> &selected) {
    // std::discrete_distribution requires a positive total weight. Identical
    // series (and equal signed dissimilarities after translation) legitimately
    // leave every unselected weight at zero, so complete the distinct medoid
    // set deterministically instead of constructing an invalid distribution.
    if (total <= 0.0)
      return first_unselected(distances.size(), selected);
    std::discrete_distribution<int> distribution(
      distances.begin(), distances.end());
    const int candidate = distribution(randGenerator);
    return std::find(selected.begin(), selected.end(), candidate)
             == selected.end()
         ? candidate
         : first_unselected(distances.size(), selected);
  };
  kmeanspp_with(prob, first_index, weighted_index);
}

void Kmeanspp_seeded(Problem &prob, std::uint64_t random_seed)
{
  std::mt19937_64 rng(random_seed);
  auto first_index = [&rng](std::size_t size) {
    return static_cast<int>(core::portable_bounded(
      rng, static_cast<std::uint64_t>(size)));
  };
  auto weighted_index = [&rng](const auto &distances, double total,
                               const std::vector<int> &selected) {
    if (total <= 0.0) {
      return first_unselected(distances.size(), selected);
    }
    int candidate = static_cast<int>(core::portable_weighted_index(
      distances.begin(), distances.end(), total, rng));
    if (std::find(selected.begin(), selected.end(), candidate)
        != selected.end()) {
      candidate = first_unselected(distances.size(), selected);
    }
    return candidate;
  };
  kmeanspp_with(prob, first_index, weighted_index);
}


} // namespace dtwc::init
