/**
 * @file initialisation.hpp
 * @brief Header file for initialisation functions.
 *
 * This file contains the declarations of initialisation functions for the dtwc namespace.
 *
 * @date 19 Jan 2021
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include <cstdint>

namespace dtwc {
class Problem;
namespace init {
  /// Initialise the medoids randomly: the series shuffled from `seed`, the first n_clusters() kept. One seed
  /// gives the same medoids on every platform. Lloyd's restarts call it with their seed (Problem::init_fun).
  void random(Problem &prob, std::uint64_t seed);
  /// Initialise the medoids by k-medoids++ (core::kmedoids_pp) from `seed`. Lloyd's restarts call it with
  /// their seed (Problem::init_fun).
  void Kmeanspp(Problem &prob, std::uint64_t seed);
} // namespace init
} // namespace dtwc
