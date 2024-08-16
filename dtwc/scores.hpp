/**
 * @file scores.hpp
 * @brief Header file for calculating different types of scores in clustering algorithms.
 *
 * @details This file contains the declarations of functions used for calculating different types
 * of scores, focusing primarily on the silhouette score for clustering analysis. The
 * silhouette score is a measure of how well an object lies within its cluster and is
 * a common method to evaluate the validity of a clustering solution.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include <vector>

namespace dtwc {
class Problem; // Pre-definition
namespace scores {
  std::vector<double> silhouette(Problem &prob);

} // namespace scores

} // namespace dtwc
