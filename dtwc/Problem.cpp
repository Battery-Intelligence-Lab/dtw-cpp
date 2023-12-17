/*
 * Problem.cpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "Problem.hpp"
#include "mip.hpp"             // for MIP_clustering_byGurobi
#include "parallelisation.hpp" // for run
#include "scores.hpp"          // for silhouette
#include "settings.hpp"        // for data_t, randGenerator, band, isDebug
#include "warping.hpp"         // for dtwBanded, dtwFull
#include "types/Range.hpp"     // for Range

#include <algorithm> // for max_element, min, min_element, sample
#include <iomanip>   // for operator<<, setprecision
#include <iostream>  // for cout
#include <iterator>  // for back_insert_iterator, back_inserter
#include <limits>    // for numeric_limits
#include <new>       // for bad_alloc
#include <random>    // for mt19937, discrete_distribution, unifo...
#include <string>    // for allocator, char_traits, operator+
#include <utility>   // for pair
#include <vector>    // for vector, operator==

namespace dtwc {

void Problem::clear_clusters()
{
  centroids_ind.clear();
  clusters_ind.clear();
  cluster_members.clear();
}

void Problem::resize()
{
  cluster_members.resize(Nc);
  centroids_ind.resize(Nc);
  clusters_ind.resize(data.size());
}

void Problem::set_numberOfClusters(int Nc_)
{
  Nc = Nc_;
  resize();
}

bool Problem::set_solver(Solver solver_)
{
  if (solver_ == Solver::Gurobi) {
#ifdef DTWC_ENABLE_GUROBI
    mipSolver = Solver::Gurobi;
    return true;
#else
    std::cout << "Solver Gurobi is not available; therefore using default solver\n";
    mipSolver = settings::DEFAULT_MIP_SOLVER;
    return false;
#endif
  }

  mipSolver = solver_;
  return true;
}


double Problem::distByInd(int i, int j)
{
  if (distMat(i, j) < 0) {
    if constexpr (settings::band == 0)
      distMat(j, i) = distMat(i, j) = dtwFull_L<data_t>(p_vec(i), p_vec(j));
    else
      distMat(j, i) = distMat(i, j) = dtwBanded<data_t>(p_vec(i), p_vec(j), settings::band);
  }
  return distMat(i, j);
}

void Problem::fillDistanceMatrix()
{
  auto oneTask = [&, N = data.size()](size_t i_linear) {
    size_t i{ i_linear / N }, j{ i_linear % N };
    if (i <= j)
      distByInd(i, j);
  };

  run(oneTask, data.size() * data.size());
  maxDistance();
}

data_t Problem::maxDistance()
{
  if (maxDist < 0)
    maxDist = *std::max_element(distMat.data.begin(), distMat.data.end());

  return maxDist;
}

void Problem::printClusters()
{
  std::cout << "Clusters: ";
  for (auto ind : centroids_ind)
    std::cout << get_name(ind) << ' ';

  std::cout << '\n';

  for (int i{ 0 }; i < Nc; i++) {
    std::cout << get_name(centroids_ind[i]) << " has: ";

    for (auto member : cluster_members[i])
      std::cout << get_name(member) << " ";

    std::cout << '\n';
  }
}

void Problem::writeClusters()
{
  auto file_name = name + "_Nc_" + std::to_string(Nc) + ".csv";

  std::ofstream myFile(output_folder / file_name, std::ios_base::out);

  myFile << "Clusters:\n";

  for (int i{ 0 }; i < Nc; i++) {
    if (i != 0) myFile << ',';

    myFile << p_names(centroids_ind[i]);
  }

  myFile << "\n\n"
         << "Data" << ',' << "its cluster\n";

  for (int i{ 0 }; i < data.size(); i++)
    myFile << p_names(i) << ',' << p_names(centroids_ind[clusters_ind[i]]) << '\n';

  myFile << "Procedure is completed with cost: " << findTotalCost() << '\n';

  myFile.close();
}


void Problem::writeSilhouettes()
{
  auto silhouettes = scores::silhouette(*this);

  std::string silhouette_name{ name + "_silhouettes_Nc_" };

  silhouette_name += std::to_string(Nc) + ".csv";

  std::ofstream myFile(output_folder / silhouette_name, std::ios_base::out);

  myFile << "Silhouettes:\n";
  for (int i{ 0 }; i < data.size(); i++)
    myFile << p_names(i) << ',' << silhouettes[i] << '\n';

  myFile.close();
}


void Problem::cluster()
{
  switch (method) {
  case Method::Kmedoids:
    cluster_by_kMedoidsPAM();
    break;
  case Method::MIP:
    cluster_by_MIP();
    break;
  }
}


void Problem::cluster_and_process()
{
  cluster();
  printClusters(); // Prints to screen.
  writeDistanceMatrix();
  writeClusters(); // Prints to file.
  writeSilhouettes();
}

void Problem::init_random()
{
  centroids_ind.clear();
  auto range = Range(data.size());
  std::sample(range.begin(), range.end(), std::back_inserter(centroids_ind), Nc, randGenerator);
}

void Problem::init_Kmeanspp()
{
  // First cluster is slected at random, others are selected based on distance.
  centroids_ind.clear();

  std::uniform_int_distribution<size_t> d(0, data.size() - 1);
  centroids_ind.push_back(d(randGenerator));

  std::vector<data_t> distances(data.size(), std::numeric_limits<data_t>::max());

  auto distTask = [&](int i_p) {
    distances[i_p] = std::min(distances[i_p], distByInd(centroids_ind.back(), i_p));
  };

  for (int i = 1; i < Nc; i++) {
    dtwc::run(distTask, data.size());
    std::discrete_distribution<> dd(distances.begin(), distances.end());
    centroids_ind.push_back(dd(randGenerator));
  }
}

void Problem::cluster_by_MIP()
{
  switch (mipSolver) {
  case Solver::Gurobi:
    MIP_clustering_byGurobi(*this);
    break;
  case Solver::HiGHS:
    MIP_clustering_byHiGHS(*this);
    break;
  }
}

void Problem::distributeClusters()
{
  for (auto &member : cluster_members)
    member.clear();

  for (size_t i = 0; i < clusters_ind.size(); i++)
    cluster_members[clusters_ind[i]].push_back(i);
}

void Problem::assignClusters()
{
  auto assignClustersTask = [&](int i_p) // i_p -> index of points
  {
    double minDist{ 1e9 };
    int minInd{}, c_ind{};
    for (auto i_c : centroids_ind) { // #TODO test if runs correctly as we changed previous algorithm.
      const auto distNew = distByInd(i_p, i_c);
      if (distNew < minDist) {
        minDist = distNew;
        minInd = c_ind;
      }
      c_ind++;
    }

    clusters_ind[i_p] = minInd;
  };

  clusters_ind.resize(data.size()); // Resize before assigning.
  run(assignClustersTask, data.size());

  distributeClusters();
}

void Problem::distanceInClusters()
{
  auto distanceInClustersTask = [&](int i_p) {
    const int clusterNo = clusters_ind[i_p];
    for (auto otherPointInd : cluster_members[clusterNo])
      if (i_p <= otherPointInd)
        distByInd(i_p, otherPointInd);
  };

  run(distanceInClustersTask, data.size());
}

void Problem::calculateMedoids()
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  std::vector<data_t> clusterCosts(centroids_ind.size(), maxValue);
  auto findBetterMedoidTask = [&](int i_p) // i_p is point index.
  {
    const auto i_c = clusters_ind[i_p];
    data_t sum{ 0 };
    for (auto member : cluster_members[i_c])
      sum += distByInd(i_p, member);

    if (sum < clusterCosts[i_c]) {
      clusterCosts[i_c] = sum;
      centroids_ind[i_c] = i_p;
    }
  };

  run(findBetterMedoidTask, data.size());
}

void Problem::cluster_by_kMedoidsPAM()
{
  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();

  for (int i_rand = 0; i_rand < N_repetition; i_rand++) {
    std::cout << "Metoid initialisation is started.\n";
    init_random(); // Use random initialisation
    // init_Kmeanspp(); // Use not random init.

    std::cout << "Metoid initialisation is finished. "
              << Nc << " medoids are initialised.\n";

    std::cout << "Start clustering:\n";


    auto [status, total_cost] = cluster_by_kMedoidsPAM_single(i_rand);

    if (status == 0)
      std::cout << "Medoids are same for last two iterations, algorithm is converged!\n";
    else if (status == -1)
      std::cout << "Maximum iteration is reached before medoids are converged!\n";

    std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';

    if (total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
    }
  }

  std::ofstream bestRepFile(output_folder / (name + "_bestRepetition.csv"), std::ios_base::out);
  bestRepFile << best_rep << '\n';
  bestRepFile.close();

  std::cout << "Best repetition: " << best_rep << '\n';
}

std::pair<int, double> Problem::cluster_by_kMedoidsPAM_single(int rep)
{
  auto oldmedoids = centroids_ind;

  int status = -1;
  const auto outPath = output_folder / (this->name + "medoids_rep_" + std::to_string(rep) + ".csv");
  std::ofstream medoidsFile(outPath, std::ios_base::out);

  if (!medoidsFile.good()) {
    std::cout << "Failed to open file in path: " << outPath << '\n'
              << "Program is exiting." << std::endl;

    throw 1;
  }

  for (int i = 0; i < maxIter; i++) {

    std::cout << "Medoids: ";
    for (auto medoid : centroids_ind) {
      std::cout << get_name(medoid) << ' ';
      medoidsFile << get_name(medoid) << ',';
    }

    medoidsFile << '\n';

    assignClusters();
    std::cout << "Test 1" << std::endl;

    std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
              << findTotalCost() << ".\n"; // Uses clusters_ind to find cost.

    printClusters();
    distanceInClusters(); // Just populates distByInd matrix ahead.
    calculateMedoids();   // Changes centroids_ind

    if (oldmedoids == centroids_ind) {
      status = 0;
      break;
    }

    oldmedoids = centroids_ind;
  }

  auto total_cost = findTotalCost();
  std::cout << "Procedure is completed with cost: " << total_cost << '\n';
  medoidsFile << "Procedure is completed with cost: " << total_cost << '\n';

  medoidsFile.close();

  return std::pair(status, total_cost);
}

double Problem::findTotalCost()
{
  double sum = 0;
  for (int i = 0; i < data.size(); i++) {
    const auto i_p = centroids_ind[clusters_ind[i]];
    if constexpr (settings::isDebug)
      std::cout << "Distance between " << i << " and closest cluster " << i_p
                << " which is: " << distByInd(i, i_p) << "\n";

    sum += distByInd(i, i_p); // #TODO should cost be square or like this?
  }

  return sum;
}

void Problem::writeMedoidMembers(int iter, int rep)
{
  const std::string medoid_name = "medoidMembers_Nc_" + std::to_string(Nc) + "_rep_"
                                  + std::to_string(rep) + "_iter_" + std::to_string(iter) + ".csv";

  std::ofstream medoidMembers(output_folder / medoid_name, std::ios_base::out);
  for (auto &members : cluster_members) {
    for (auto member : members)
      medoidMembers << get_name(member) << ',';
    medoidMembers << '\n';
  }
  medoidMembers.close();
}

void Problem::writeDataOrder(fs::path out_folder)
{
  std::ofstream out(out_folder / (name + "_dataOrder.csv"), std::ios_base::out);

  for (int i = 0; i < data.size(); i++)
    out << i << ',' << p_names(i) << '\n';

  out.close();
}

} // namespace dtwc