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

namespace dtwc {


void Problem::clear_clusters()
{
  centroids_ind.clear();
  clusters_ind.clear();
  cluster_members.clear();
}

double Problem::DTWdistByInd(int i, int j)
{
  if (DTWdist(i, j) < 0) {
    if constexpr (settings::band == 0) {
      DTWdist(j, i) = DTWdist(i, j) = dtwFun_L<data_t>(p_vec[i], p_vec[j]);
    } else {
      DTWdist(j, i) = DTWdist(i, j) = dtwFunBanded_Act<data_t>(p_vec[i], p_vec[j], settings::band); // dtwFunBanded_Act_L faster and more accurate.
    }
  }
  return DTWdist(i, j);
}

void Problem::fillDistanceMatrix()
{
  auto oneTask = [&, N = Nb](size_t i_linear) {
    thread_local TestNumberOfThreads a{};
    size_t i{ i_linear / N }, j{ i_linear % N };
    if (i <= j)
      DTWdistByInd(i, j);
  };

  run(oneTask, Nb * Nb);
}

void Problem::printClusters()
{
  std::cout << "Clusters: ";
  for (auto ind : centroids_ind)
    std::cout << p_names[ind] << ' ';

  std::cout << '\n';

  for (size_t i{ 0 }; i < Nc; i++) {
    auto centroid = centroids_ind[i];
    std::cout << "Cluster " << p_names[centroid] << " has: ";

    for (auto member : cluster_members[i])
      std::cout << p_names[member] << " ";
    std::cout << '\n';
  }
}

void Problem::writeClusters(std::string &file_name)
{
  file_name += "_" + std::to_string(Nc) + ".csv";

  std::ofstream myFile(settings::resultsPath + file_name, std::ios_base::out);

  myFile << "Clusters:\n";

  for (size_t i{ 0 }; i < Nc; i++) {
    if (i != 0)
      myFile << ',';

    myFile << p_names[centroids_ind[i]];
  }

  myFile << "\n\n";
  myFile << "Data" << ',' << "its cluster\n";

  for (size_t i{ 0 }; i < p_names.size(); i++) {
    myFile << p_names[i] << ',' << p_names[centroids_ind[clusters_ind[i]]] << '\n';
  }

  myFile.close();
}


void Problem::load_data_fromFolder(std::string_view folder_path, int Ndata, bool print)
{
  std::tie(p_vec, p_names) = load_data<data_t>(folder_path, Ndata, print);

  Nb = p_vec.size();
  DTWdist = dtwc::VecMatrix<data_t>(Nb, Nb, -1);
}

void Problem::cluster_byMIP()
{
  MIP_clustering_byGurobi(*this);
  printClusters();
}

void Problem::write_silhouettes()
{
  auto silhouettes = scores::silhouette(*this);

  std::string name{ "silhouettes_" };

  name += std::to_string(Nc) + ".csv";

  std::ofstream myFile(settings::resultsPath + name, std::ios_base::out);

  myFile << "Silhouettes:\n";
  for (size_t i{ 0 }; i < Nb; i++) {
    myFile << p_names[i] << ',' << silhouettes[i] << '\n';
  }

  myFile.close();
}


void Problem::init_random()
{
  assert(Nc > 0);
  clear_clusters();
  std::uniform_int_distribution<ind_t> distrib(0, Nb - 1);

  for (auto _ : Range(Nc))
    centroids_ind.push_back(distrib(randGenerator));
}

void Problem::init_Kmeanspp()
{
  // First cluster is slected at random, others are selected based on distance.
  clear_clusters();

  std::uniform_int_distribution<ind_t> d(0, Nb - 1);
  centroids_ind.push_back(d(randGenerator));

  std::vector<data_t> distances(Nb, std::numeric_limits<data_t>::max());

  auto distTask = [&](int i_p) {
    distances[i_p] = std::min(distances[i_p], DTWdistByInd(centroids_ind.back(), i_p));
  };

  for (size_t i = 1; i < centroids_ind.size(); i++) {
    dtwc::run(distTask, Nb);
    std::discrete_distribution<> d(distances.begin(), distances.end());
    centroids_ind.push_back(d(randGenerator));
  }
}

} // namespace dtwc
