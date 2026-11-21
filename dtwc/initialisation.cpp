/**
 * @file initialisation.cpp
 * @brief Medoid initialisation (random and k-medoids++) and Problem::init_with_seed.
 *
 * @date 19 Jan 2021
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "initialisation.hpp"
#include "base/error.hpp"
#include "core/portable_random.hpp"
#include "core/distance_sampling_weights.hpp"
#include "base/random_engine.hpp"
#include "base/settings.hpp"
#include "Problem.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

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
  // N·k reads: from the typed values (fast_pam's BUILD).
  prob.set_clusters(prob.distance_matrix().visit([&](const auto *matrix) {
    return core::kmedoids_pp(
      prob.size(), Nc, rng,
      [matrix](index_t c, index_t i) {
        return matrix[core::tri_index(static_cast<std::size_t>(c), static_cast<std::size_t>(i))];
      },
      "init::Kmeanspp");
  }));
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
