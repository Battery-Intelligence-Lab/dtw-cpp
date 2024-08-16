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
  void random(Problem &prob);   //!< This function initializes the centroids randomly.
  void Kmeanspp(Problem &prob); //!< This function initializes the centroids using the K-means++ algorithm.
} // namespace init
} // namespace dtwc