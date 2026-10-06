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

namespace dtwc {
class Problem;
namespace init {
  /// Initialise the medoids randomly. Seeded by one draw of dtwc::randGenerator;
  /// Lloyd's restarts seed it themselves (Problem::init_fun).
  void random(Problem &prob);
  /// Initialise the medoids by k-medoids++ (core::kmedoids_pp). Seeded by one draw
  /// of dtwc::randGenerator; Lloyd's restarts seed it themselves (Problem::init_fun).
  void Kmeanspp(Problem &prob);
} // namespace init
} // namespace dtwc
