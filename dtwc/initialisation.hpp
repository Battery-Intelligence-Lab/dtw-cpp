/*
 * initialisation.cpp
 *
 * Header file for initialisation functions.

 *  Created on: 19 Jan 2021
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"        // for randGenerator
#include "parallelisation.hpp" // for run

#include <cstddef>   // for size_t
#include <algorithm> // for sample
#include <cassert>   // for assert
#include <iterator>  // for back_inserter
#include <limits>    // for numeric_limits
#include <random>    // for discrete_distribution, uniform_int_di...
#include <vector>    // for vector

namespace dtwc::initialisation {

template <typename data_t>
auto init_random(const std::vector<std::vector<data_t>> &sequences, int Nc)
{
  // Nc = number of clusters:
  std::vector<std::vector<data_t>> centroids_vec;
  std::sample(sequences.begin(), sequences.end(), std::back_inserter(centroids_vec), 5, randGenerator);

  return centroids_vec;
}

template <typename data_t, typename Tfun>
auto init_Kmeanspp(const std::vector<std::vector<data_t>> &sequences, int Nc, Tfun &distanceFun)
{
  // distance should be DTWlike function that take two sequences and give the distance between them.
  // First one is slected at random, others are selected based on distance.
  assert(Nc > 0);

  if (Nc == 1)
    return init_random(sequences, Nc);

  // else
  std::vector<std::vector<data_t>> centroids_vec;
  std::vector<size_t> centroids_ind(Nc, 0);

  std::vector<data_t> distances(sequences.size(), std::numeric_limits<data_t>::max());

  int gen{ -1 };
  std::uniform_int_distribution<> d(0, sequences.size() - 1);
  gen = d(randGenerator);

  for (int i = 0; i < Nc - 1; i++) {
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

std::vector<size_t> init_random_ind(size_t Nb, int Nc);


template <typename data_t, typename Tfun>
auto init_Kmeanspp_ind(size_t Nb, int Nc, Tfun &distancebyIndFunc)
{
  // distance should be DTWlike function that take two sequences and give the distance between them.
  // First one is slected at random, others are selected based on distance.
  assert(Nc > 0);

  if (Nc == 1) return init_random_ind(Nb, Nc);

  // else
  std::vector<size_t> centroids_ind(Nc);
  std::vector<data_t> distances(Nb, std::numeric_limits<data_t>::max());

  int gen{ -1 };
  std::uniform_int_distribution<> d(0, Nb - 1);
  gen = d(randGenerator);

  for (int i = 0; i < Nc - 1; i++) {
    centroids_ind[i] = gen;
    auto distTask = [&](int i_p) {
      const auto dist = distancebyIndFunc(gen, i_p);
      distances[i_p] = std::min(distances[i_p], dist);
    };

    dtwc::run(distTask, Nb);

    std::discrete_distribution<> d(distances.begin(), distances.end());
    gen = d(randGenerator);
  }
  centroids_ind.back() = gen;

  return centroids_ind;
}


} // namespace dtwc::initialisation