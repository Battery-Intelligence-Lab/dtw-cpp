/*
 * initialisation.hpp
 *
 * Header file for initialisation functions.

 *  Created on: 19 Jan 2021
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

namespace dtwc {
class Problem;
namespace init {
  void random(Problem &prob);   // Random centroids initialisation
  void Kmeanspp(Problem &prob); // Kmeanspp centroids initialisation
} // namespace init
} // namespace dtwc