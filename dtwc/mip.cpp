/*
 * mip.cpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "mip.hpp"
#include "Problem.hpp"
#include "settings.hpp"
#include "utility.hpp"
#include "solver/Simplex.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {

void MIP_clustering_bySimplex(Problem &prob)
{
  std::cout << "Simplex is being called!" << std::endl;
  dtwc::Clock clk; // Create a clock object

  prob.clear_clusters();

  thread_local dtwc::solver::Simplex simplexSolver(prob);

  std::cout << "Problem formulation finished in " << clk << '\n';
  simplexSolver.gomoryAlgorithm();
  std::cout << "Problem solution finished in " << clk << '\n';

  auto [solution, copt] = simplexSolver.getResults();

  fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);

  const auto Nb = prob.data.size();

  for (ind_t i{ 0 }; i < Nb; i++)
    if (solution[i * (Nb + 1)] > 0.5)
      prob.centroids_ind.push_back(i);

  prob.clusters_ind = std::vector<ind_t>(Nb);

  ind_t i_cluster = 0;
  for (auto i : prob.centroids_ind) {
    prob.cluster_members.emplace_back();
    for (size_t j{ 0 }; j < Nb; j++)
      if (solution[i * Nb + j] > 0.5) {
        prob.clusters_ind[j] = i_cluster;
        prob.cluster_members.back().push_back(j);
      }

    i_cluster++;
  }
}


} // namespace dtwc
