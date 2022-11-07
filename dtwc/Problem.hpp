/*
 * Problem.hpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 19 Oct 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

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

class Problem
{
  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;
  VecMatrix<data_t> DTWdist;


  ind_t Nc{ 1 }; // Number of clusters.
  ind_t Nb;      // Number of data points

public:
  std::vector<ind_t> centroids_ind;                // indices of cluster centroids.
  std::vector<ind_t> clusters_ind;                 // which point belongs to which cluster.
  std::vector<std::vector<ind_t>> cluster_members; // Members of each clusters!

  // Getters and setters:
  auto &getDistanceMatrix() { return DTWdist; }

  void clear_clusters();

  auto set_numberOfClusters(ind_t Nc_)
  {
    assert(Nc_ > 0);
    Nc = Nc_;
  }

  auto size() const { return Nb; }
  auto cluster_size() const { return Nc; }

  double DTWdistByInd(int i, int j);
  void fillDistanceMatrix();

  void writeDistanceMatrix(const std::string &name) { writeMatrix(DTWdist, name); }

  void load_data_fromFolder(std::string_view folder_path, int Ndata = -1, bool print = false);

  void cluster_byMIP();

  void printClusters();
  void writeClusters(std::string &file_name);

  auto calculate_silhouette();

  void write_silhouettes();

  // Initialisation of clusters:
  void init_random();
  void init_Kmeanspp();
};


} // namespace dtwc