/*
 * Clusters.hpp
 *
 * Encapsulating Clusters in a class.

 *  Created on: 04 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Problem.hpp"
#include "settings.hpp"
#include "utility.hpp"
//#include "fileOperations.hpp"
//#include "initialisation.hpp"
#include "timing.hpp"
#include "gurobi_c++.h"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {
class Clusters
{
public:
  std::vector<unsigned int> centroids_ind;                // indices of cluster centroids.
  std::vector<unsigned int> clusters_ind;                 // which point belongs to which cluster.
  std::vector<std::vector<unsigned int>> cluster_members; // Members of each clusters!

  size_t Nc{ 4 }; // Number of clusters.

  auto clear()
  {
    centroids_ind.clear();
    clusters_ind.clear();
    cluster_members.clear();
  }

  auto set_size(size_t Nc_)
  {
    Nc = Nc_;
    clear();
  }

  auto size() const { return Nc; }

  void print_wNames(std::vector<std::string> &p_names)
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

  void write_wNames(std::string file_name, std::vector<std::string> &p_names)
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
};


} // namespace dtwc
