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
#include "solver/SparseSimplex.hpp"
#include "solver/LP.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {

void extract_solution(Problem &prob, auto &solution)
{
  prob.clear_clusters();
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

template <typename T>
void MIP_clustering_bySimplex(Problem &prob)
{
  std::cout << "Simplex is being called!" << std::endl;
  dtwc::Clock clk; // Create a clock object


  thread_local auto simplexSolver = T(prob);

  std::cout << "Problem formulation finished in " << clk << '\n';
  simplexSolver.gomoryAlgorithm();
  std::cout << "Problem solution finished in " << clk << '\n';

  auto [solution, copt] = simplexSolver.getResults();

  fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);

  extract_solution(prob, solution);
}

void MIP_clustering_bySparseSimplex(Problem &prob)
{
  MIP_clustering_bySimplex<dtwc::solver::SparseSimplex>(prob);
}

void MIP_clustering_byDenseSimplex(Problem &prob)
{
  MIP_clustering_bySimplex<dtwc::solver::Simplex>(prob);
}

void MIP_clustering_byOSLP(Problem &prob)
{

  dtwc::solver::LP lp;
  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();
  lp.setSize(Nb, Nc);

  // Alias variables.
  auto &q = lp.getQvec();
  auto &w_sol = lp.getSolution();

  std::fill(w_sol.begin(), w_sol.end(), 0.0);

  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      q[i + j * Nb] = prob.distByInd_scaled(i, j);

  lp.maxIterations = 3000;
  lp.numItrConv = 200;
  lp.epsAbs = 1e-4;
  lp.epsRel = 1e-4;

  // lp.solve();
  lp.solve_LU();

  extract_solution(prob, lp.getSolution());
}
} // namespace dtwc
