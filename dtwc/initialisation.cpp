/*
 * initialisation.cpp
 *
 * Source file for initialisation functions.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "settings.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <random>

namespace dtwc::initialisation {

// template <typename data_t>
// auto init_random(const std::vector<std::vector<data_t>> &sequences, int N)
// {
//   // N = number of clusters:
//   std::vector<std::vector<data_t>> centroids_vec;
//   std::sample(sequences.begin(), sequences.end(), std::back_inserter(centroids_vec), 5, randGenerator);

//   return centroids_vec;
// }

template <typename data_t, typename Tfun>
auto init_Kmeanspp(const std::vector<std::vector<data_t>> &sequences, int N, Tfun &distanceFun)
{
  // distance should be DTWlike function that take two sequences and give the distance between them.
  // First one is slected at random, others are selected based on distance.
  assert(N > 0);

  if (N == 1)
    return init_random(sequences, N);

  // else
  std::vector<std::vector<data_t>> centroids_vec;
  std::vector<size_t> centroids_ind(N, 0);

  std::vector<data_t> distances(sequences.size(), std::numeric_limits<data_t>::max());

  int gen{ -1 };
  std::uniform_int_distribution<> d(0, sequences.size() - 1);
  gen = d(randGenerator);

  for (int i = 0; i < N - 1; i++) {
    centroids_ind[i] = gen;
    centroids_vec.push_back(sequences[gen]);
    for (size_t j = 0; j != sequences.size(); j++) {
      const data_t dist = distanceFun(sequences[gen], sequences[j]);
      distances[j] = std::min(distances[j], dist);
    }

    std::discrete_distribution<> d(distances.begin(), distances.end());
    gen = d(randGenerator);
  }
  centroids_ind.back() = gen;
  centroids_vec.push_back(sequences[gen]);

  return centroids_vec;
}

std::vector<size_t> init_random_ind(size_t seqSize, int Nc)
{
  // N = number of clusters:
  std::vector<size_t> centroids_ind, all_ind(seqSize);
  std::iota(all_ind.begin(), all_ind.end(), 0);
  std::sample(all_ind.begin(), all_ind.end(), std::back_inserter(centroids_ind), Nc, randGenerator);
  return centroids_ind;
}

} // namespace dtwc::initialisation