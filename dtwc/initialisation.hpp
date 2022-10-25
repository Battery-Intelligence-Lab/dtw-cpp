

// Vk 2021.01.19

#pragma once

#include "utility.hpp"

#include <iostream>
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <iterator>
#include <memory>

#include <cassert>


namespace dtwc::Initialisation {

template <typename Tdata>
auto init_random(const std::vector<std::vector<Tdata>> &sequences, int N)
{
  // N = number of clusters:

  std::vector<std::vector<Tdata>> centroids_vec;
  std::sample(sequences.begin(), sequences.end(), std::back_inserter(centroids_vec), 5, randGenerator);

  return centroids_vec;
}

template <typename Tdata, typename Tfun>
auto init_Kmeanspp(const std::vector<std::vector<Tdata>> &sequences, int N, Tfun &distanceFun)
{
  // distance should be DTWlike function that take two sequences and give the distance between them.
  // First one is slected at random, others are selected based on distance.
  assert(N > 0);

  if (N == 1)
    return init_random(sequences, N);

  // else
  std::vector<std::vector<Tdata>> centroids_vec;
  std::vector<int> centroids_ind(N, -1);

  std::vector<Tdata> distances(sequences.size(), std::numeric_limits<Tdata>::max());

  int gen{ -1 };
  std::uniform_int_distribution<> d(0, sequences.size() - 1);
  gen = d(randGenerator);

  for (int i = 0; i < N - 1; i++) {
    centroids_ind[i] = gen;
    centroids_vec.push_back(sequences[gen]);
    for (size_t j = 0; j != sequences.size(); j++) {
      const Tdata dist = distanceFun(sequences[gen], sequences[j]);
      distances[j] = std::min(distances[j], dist);
    }

    std::discrete_distribution<> d(distances.begin(), distances.end());
    gen = d(randGenerator);
  }
  centroids_ind.back() = gen;
  centroids_vec.push_back(sequences[gen]);

  return centroids_vec;
}

auto init_random_ind(size_t seqSize, int N)
{
  // N = number of clusters:
  std::vector<int> centroids_ind, all_ind(seqSize);
  std::iota(all_ind.begin(), all_ind.end(), 0);
  std::sample(all_ind.begin(), all_ind.end(), std::back_inserter(centroids_ind), N, randGenerator);
  return centroids_ind;
}

template <typename Tdata, typename Tfun>
auto init_Kmeanspp_ind(size_t seqSize, int N, Tfun &distancebyIndFunc)
{
  // distance should be DTWlike function that take two sequences and give the distance between them.
  // First one is slected at random, others are selected based on distance.
  assert(N > 0);

  if (N == 1) return init_random_ind(seqSize, N);

  // else
  std::vector<int> centroids_ind(N, -1);
  std::vector<Tdata> distances(seqSize, std::numeric_limits<Tdata>::max());

  int gen{ -1 };
  std::uniform_int_distribution<> d(0, seqSize - 1);
  gen = d(randGenerator);

  for (int i = 0; i < N - 1; i++) {
    centroids_ind[i] = gen;
    auto distTask = [&](int i_p) {
      const auto dist = distancebyIndFunc(gen, i_p);
      distances[i_p] = std::min(distances[i_p], dist);
    };

    dtwc::run(distTask, seqSize);

    std::discrete_distribution<> d(distances.begin(), distances.end());
    gen = d(randGenerator);
  }
  centroids_ind.back() = gen;

  return centroids_ind;
}


} // namespace dtwc::Initialisation