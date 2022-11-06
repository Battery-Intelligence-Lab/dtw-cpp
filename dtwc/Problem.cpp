/*
 * Problem.cpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "Problem.hpp"
#include "Clusters.hpp"
#include "mip.hpp"
#include "settings.hpp"
#include "utility.hpp"
#include "fileOperations.hpp"
//#include "initialisation.hpp"
#include "timing.hpp"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {

double Problem::DTWdistByInd(int i, int j)
{
  if (DTWdist(i, j) < 0) {
    if constexpr (settings::band == 0) {
      DTWdist(j, i) = DTWdist(i, j) = dtwFun_L<Tdata>(p_vec[i], p_vec[j]);
    } else {
      DTWdist(j, i) = DTWdist(i, j) = dtwFunBanded_Act<Tdata>(p_vec[i], p_vec[j], settings::band); // dtwFunBanded_Act_L faster and more accurate.
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

  dtwc::run(oneTask, Nb * Nb);
}

void Problem::load_data_fromFolder(std::string_view folder_path, int Ndata, bool print)
{
  std::tie(p_vec, p_names) = load_data<Tdata>(folder_path, Ndata, print);

  Nb = p_vec.size();
  DTWdist = dtwc::VecMatrix<Tdata>(Nb, Nb, -1);
}

void Problem::cluster_byMIP()
{

  //MIP_clustering_byGurobi(*this);
  //clusters.print_wNames(p_names);
}

auto Problem::calculate_silhouette()
{

  // For explanation, see: https://en.wikipedia.org/wiki/Silhouette_(clustering)
  const auto Nc = clusters.size();

  if (clusters.centroids_ind.empty())
    std::cout << "Please cluster the data before calculating silhouette!\n";

  std::vector<double> silhouettes(Nb);

  auto oneTask = [&, N = Nb](size_t i_b) {
    auto i_c = clusters.clusters_ind[i_b];

    if (clusters.cluster_members[i_c].size() == 1)
      silhouettes[i_b] = 0;
    else {
      thread_local std::vector<double> mean_distances(Nc);

      for (size_t i = 0; i < Nb; i++)
        mean_distances[clusters.clusters_ind[i]] += DTWdistByInd(i, i_b);

      auto min = std::numeric_limits<double>::max();
      for (size_t i = 0; i < Nc; i++) // Finding means:
        if (i == i_c)
          mean_distances[i] /= (clusters.cluster_members[i].size() - 1);
        else {
          mean_distances[i] /= clusters.cluster_members[i].size();
          min = std::min(min, mean_distances[i]);
        }

      silhouettes[i_b] = (min - mean_distances[i_c]) / std::max(min, mean_distances[i_c]);
    }
  };

  dtwc::run(oneTask, Nb);

  return silhouettes;
}

void Problem::write_silhouettes()
{
  const auto Nc = clusters.size();
  auto silhouettes = calculate_silhouette();

  std::string name{ "silhouettes_" };

  name += std::to_string(Nc) + ".csv";

  std::ofstream myFile(settings::resultsPath + name, std::ios_base::out);

  myFile << "Silhouettes:\n";
  for (size_t i{ 0 }; i < Nb; i++) {
    myFile << p_names[i] << ',' << silhouettes[i] << '\n';
  }

  myFile.close();
}


} // namespace dtwc
