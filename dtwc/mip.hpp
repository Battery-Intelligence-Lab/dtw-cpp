/*
 * mip.hpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

namespace dtwc {
class Problem;

void MIP_clustering_byGurobi(Problem &prob);

void MIP_clustering_byHiGHS(Problem &prob);

} // namespace dtwc
