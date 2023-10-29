/*
 * mip.hpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Problem.hpp"
#include "settings.hpp"
#include "utility.hpp"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {
class Problem;

void MIP_clustering_byGurobi(Problem &prob);
void MIP_clustering_byGurobi_relaxed(Problem &prob);

void MIP_clustering_bySparseSimplex(Problem &prob);
void MIP_clustering_byDenseSimplex(Problem &prob);

} // namespace dtwc
