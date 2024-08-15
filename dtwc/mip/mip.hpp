/**
 * @file mip.hpp
 * @brief Collecting mixed-integer program functions.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 06 Nov 2022
 */

#pragma once

namespace dtwc {
class Problem;

void MIP_clustering_byGurobi(Problem &prob);
void MIP_clustering_byHiGHS(Problem &prob);

} // namespace dtwc
