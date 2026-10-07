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
 * Each runs from a seed: the one-argument functions draw it from
 * dtwc::randGenerator, and Lloyd's restarts pass their own (Problem::init_with_seed,
 * defined here beside the two initialisers it seeds).
 *
 * @date 19 Jan 2021
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "initialisation.hpp"
#include "base/error.hpp"
#include "core/portable_random.hpp"
#include "core/distance_sampling_weights.hpp" // kmedoids_pp
#include "base/random_engine.hpp"             // for randGenerator
#include "base/settings.hpp"
#include "Problem.hpp"

#include <cstddef> // for size_t
#include <cstdint> // for uint64_t
#include <numeric> // for iota
#include <random>  // for mt19937_64
#include <vector>  // for vector

namespace dtwc::init {

namespace {

void random_from_seed(Problem &prob, std::uint64_t seed)
{
  const auto Nc = prob.n_clusters();

  if (prob.size() == 0 || Nc > prob.size())
    throw InvalidInput("init::random requires 1 <= number of clusters <= number of series");

  std::vector<index_t> candidate_centroids(static_cast<std::size_t>(prob.size()));
  std::iota(candidate_centroids.begin(), candidate_centroids.end(), index_t{ 0 });
  std::mt19937_64 rng(seed);
  core::portable_shuffle(candidate_centroids.begin(), candidate_centroids.end(), rng);
  candidate_centroids.resize(static_cast<std::size_t>(Nc));

  prob.set_clusters(candidate_centroids);
}

void kmeanspp_from_seed(Problem &prob, std::uint64_t seed)
{
  const auto Nc = prob.n_clusters();

  if (prob.size() == 0 || Nc > prob.size())
    throw InvalidInput("init::Kmeanspp requires 1 <= number of clusters <= number of series");

  prob.centroids_ind.clear();
  prob.fill_distance_matrix(); // the seeding below only reads it

  std::mt19937_64 rng(seed);
  prob.set_clusters(core::kmedoids_pp(
    prob.size(), Nc, rng,
    [&prob](index_t c, index_t i) { return prob.dist_by_ind(c, i); },
    "init::Kmeanspp"));
}

} // namespace

/**
 * @brief Randomly initializes the cluster centroids for a given problem.
 *
 * @param prob Reference to the Problem object whose clusters are to be initialized.
 *
 * @exception InvalidInput if the number of clusters (Nc) is not in [1, number of series].
 *
 * @details Shuffles the series indices from a seed drawn from dtwc::randGenerator
 * and keeps the first Nc as the centroids.
 */
void random(Problem &prob) { random_from_seed(prob, randGenerator()); }

/**
 * @brief Initialises cluster centroids using the K-means++ algorithm.
 *
 * @param prob Reference to the Problem object whose clusters are to be initialized.
 *
 * @exception InvalidInput if the number of clusters (Nc) is not in [1, number of series].
 *
 * @details
 * Implements the K-means++ algorithm for initializing clusters (core::kmedoids_pp,
 * from a seed drawn from dtwc::randGenerator). The first centroid
 * is chosen randomly, and subsequent centroids are chosen based on the distance
 * from existing centroids. This method aims to provide a better initial condition
 * for clustering algorithms, potentially leading to better final clusters.
 */
void Kmeanspp(Problem &prob) { kmeanspp_from_seed(prob, randGenerator()); }

} // namespace dtwc::init

namespace dtwc {

/// Lloyd's restart from `seed`: init::random and init::Kmeanspp run from that seed
/// instead of drawing one. `init_fun` is a public extension point, and an arbitrary
/// callback has no seed parameter, so it is invoked unchanged and owns its own RNG.
void Problem::init_with_seed(std::uint64_t seed)
{
  using initializer_t = void (*)(Problem &);
  const auto *target = init_fun.target<initializer_t>();
  if (target != nullptr && *target == &init::random)
    init::random_from_seed(*this, seed);
  else if (target != nullptr && *target == &init::Kmeanspp)
    init::kmeanspp_from_seed(*this, seed);
  else
    init();
}

} // namespace dtwc
