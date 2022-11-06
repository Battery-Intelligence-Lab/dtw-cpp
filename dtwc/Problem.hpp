/*
 * Problem.hpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 19 Oct 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

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

class Problem
{
  std::vector<std::vector<Tdata>> p_vec;
  std::vector<std::string> p_names;
  VecMatrix<Tdata> DTWdist;


  size_t Nb; // Number of data points

public:
  Clusters clusters;

  // Getters and setters:

  auto &getDistanceMatrix() { return DTWdist; }

  auto set_numberOfClusters(size_t Nc_) { clusters.set_size(Nc_); }

  auto size() const { return Nb; }
  auto cluster_size() const { return clusters.size(); }

  double DTWdistByInd(int i, int j);
  void fillDistanceMatrix();

  void writeDistanceMatrix(const std::string &name) { writeMatrix(DTWdist, name); }

  void load_data_fromFolder(std::string_view folder_path, int Ndata = -1, bool print = false);

  void cluster_byMIP();

void writeClusters(std::string &file_name) { clusters.write_wNames(file_name, p_names); }

  auto calculate_silhouette();

  void write_silhouettes();
};


} // namespace dtwc