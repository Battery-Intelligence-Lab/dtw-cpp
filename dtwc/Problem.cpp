/*
 * Problem.cpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "Problem.hpp"
#include "mip.hpp"
#include "settings.hpp"
#include "utility.hpp"
#include "fileOperations.hpp"
//#include "initialisation.hpp"
#include "timing.hpp"
#include "scores.hpp"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>
#include <utility>

namespace dtwc {

void Problem::clear_clusters()
{
  centroids_ind.clear();
  clusters_ind.clear();
  cluster_members.clear();
}

double Problem::distByInd(int i, int j)
{
  if (distMat(i, j) < 0) {
    if constexpr (settings::band == 0) {
      distMat(j, i) = distMat(i, j) = dtwFun_L<data_t>(p_vec(i), p_vec(j));
    } else {
      distMat(j, i) = distMat(i, j) = dtwFunBanded_Act<data_t>(p_vec(i), p_vec(j), settings::band); // dtwFunBanded_Act_L faster and more accurate.
    }
  }
  return distMat(i, j);
}

void Problem::fillDistanceMatrix()
{
  auto oneTask = [&, N = data.size()](size_t i_linear) {
    thread_local TestNumberOfThreads a{};
    size_t i{ i_linear / N }, j{ i_linear % N };
    if (i <= j)
      distByInd(i, j);
  };

  run(oneTask, data.size() * data.size());
}

void Problem::printClusters()
{
  std::cout << "Clusters: ";
  for (auto ind : centroids_ind)
    std::cout << get_name(ind) << ' ';

  std::cout << '\n';

  for (size_t i{ 0 }; i < Nc; i++) {
    std::cout << get_name(centroids_ind[i]) << " has: ";

    for (auto member : cluster_members[i])
      std::cout << get_name(member) << " ";

    std::cout << '\n';
  }
}

void Problem::writeClusters(std::string file_name)
{
  file_name = name + "_" + file_name + "_Nc_" + std::to_string(Nc) + ".csv";


  std::ofstream myFile(output_folder / file_name, std::ios_base::out);

  myFile << "Clusters:\n";

  for (size_t i{ 0 }; i < Nc; i++) {
    if (i != 0)
      myFile << ',';

    myFile << p_names(centroids_ind[i]);
  }

  myFile << "\n\n";
  myFile << "Data" << ',' << "its cluster\n";

  for (size_t i{ 0 }; i < data.size(); i++) {
    myFile << p_names(i) << ',' << p_names(centroids_ind[clusters_ind[i]]) << '\n';
  }

  myFile << "Procedure is completed with cost: " << findTotalCost() << '\n';

  myFile.close();
}


void Problem::writeSilhouettes()
{
  auto silhouettes = scores::silhouette(*this);

  std::string name{ this->name + "_silhouettes_" };

  name += std::to_string(Nc) + ".csv";

  std::ofstream myFile(output_folder / name, std::ios_base::out);

  myFile << "Silhouettes:\n";
  for (size_t i{ 0 }; i < data.size(); i++) {
    myFile << p_names(i) << ',' << silhouettes[i] << '\n';
  }

  myFile.close();
}


void Problem::init_random()
{
  centroids_ind.clear();
  auto range = Range(0, data.size());
  std::sample(range.begin(), range.end(), std::back_inserter(centroids_ind), Nc, randGenerator);
}

void Problem::init_Kmeanspp()
{
  // First cluster is slected at random, others are selected based on distance.
  centroids_ind.clear();

  std::uniform_int_distribution<ind_t> d(0, data.size() - 1);
  centroids_ind.push_back(d(randGenerator));

  std::vector<data_t> distances(data.size(), std::numeric_limits<data_t>::max());

  auto distTask = [&](int i_p) {
    distances[i_p] = std::min(distances[i_p], distByInd(centroids_ind.back(), i_p));
  };

  for (size_t i = 1; i < Nc; i++) {
    dtwc::run(distTask, data.size());
    std::discrete_distribution<> dd(distances.begin(), distances.end());
    centroids_ind.push_back(dd(randGenerator));
  }
}

void Problem::cluster_by_MIP()
{
  if constexpr (settings::is_OSQP)
    MIP_clustering_byOSQP(*this);
  else if (settings::is_relaxed)
    MIP_clustering_byGurobi_relaxed(*this);
  else
    MIP_clustering_byGurobi(*this);
}

void Problem::distributeClusters()
{
  for (auto &member : cluster_members)
    member.clear();

  for (ind_t i = 0; i < clusters_ind.size(); i++)
    cluster_members[clusters_ind[i]].push_back(i);
}

void Problem::assignClusters()
{
  auto assignClustersTask = [&](int i_p) // i_p -> index of points
  {
    auto dist = [&](int i_c) { return distByInd(i_p, i_c); };
    auto compare = [&dist](auto i_1, auto i_2) { return dist(i_1) < dist(i_2); };
    auto it = std::min_element(centroids_ind.begin(), centroids_ind.end(), compare);
    clusters_ind[i_p] = std::distance(centroids_ind.begin(), it);
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

  std::vector<data_t> clusterCosts(centroids_ind.size(), maxValue<data_t>);
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

void Problem::cluster_by_kMedoidsPAM_repetetive(int N_repetition, int maxIter)
{
  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();

  for (int i_rand = 0; i_rand < N_repetition; i_rand++) {
    std::cout << "Metoid initialisation is started.\n";
    init_random(); // Use random initialisation
    // init_Kmeanspp(); // Use not random init.

    std::cout << "Metoid initialisation is finished. "
              << Nc << " metoids are initialised.\n";

    std::cout << "Start clustering:\n";


    auto [status, total_cost] = cluster_by_kMedoidsPAM(i_rand, maxIter);

    if (status == 0)
      std::cout << "Metoids are same for last two iterations, algorithm is converged!\n";
    else if (status == -1)
      std::cout << "Maximum iteration is reached before metoids are converged!\n";

    std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';

    if (total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
    }
  }

  std::ofstream bestRepFile(output_folder / "bestRepetition.csv", std::ios_base::out);
  bestRepFile << best_rep << '\n';
  bestRepFile.close();

  std::cout << "Best repetition: " << best_rep << '\n';
}


std::pair<int, double> Problem::cluster_by_kMedoidsPAM(int rep, int maxIter)
{
  auto oldmedoids = centroids_ind;

  int status = -1;
  std::ofstream medoidsFile(output_folder / ("medoids_rep_" + std::to_string(rep) + ".csv"), std::ios_base::out);
  for (int i = 0; i < maxIter; i++) {

    std::cout << "Medoids: ";
    for (auto medoid : centroids_ind) {
      std::cout << get_name(medoid) << ' ';
      medoidsFile << get_name(medoid) << ',';
    }

    medoidsFile << '\n';

    assignClusters();
    std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
              << findTotalCost() << ".\n"; // Uses clusters_ind to find cost.

    printClusters();

    writeMedoidMembers(i, rep);

    distanceInClusters(); // Just populates distByInd matrix ahead.
    calculateMedoids();   // Changes centroids_ind

    if (aremedoidsSame(oldmedoids, centroids_ind)) {
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
  for (size_t i = 0; i < data.size(); i++) {
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
  std::string name = "medoidMembers_Nc_" + std::to_string(Nc) + "_rep_"
                     + std::to_string(rep) + "_iter_" + std::to_string(iter) + ".csv";

  std::ofstream medoidMembers(output_folder / name, std::ios_base::out);
  for (auto &members : cluster_members) {
    for (auto member : members)
      medoidMembers << get_name(member) << ',';
    medoidMembers << '\n';
  }
  medoidMembers.close();
}

} // namespace dtwc