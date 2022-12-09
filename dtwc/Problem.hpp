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
#include "Data.hpp"
#include "DataLoader.hpp"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>
#include <utility>


namespace dtwc {

class Problem
{
  ind_t Nc{ 1 }; // Number of clusters.
  VecMatrix<data_t> distMat;
  data_t maxDist{ -1 };

public:
  bool writeAsFileNames{ settings::writeAsFileNames };
  fs::path output_folder{ settings::resultsPath };
  std::string name{}; // Problem name
  Data data;

  std::vector<ind_t> centroids_ind;                // indices of cluster centroids.
  std::vector<ind_t> clusters_ind;                 // which point belongs to which cluster.
  std::vector<std::vector<ind_t>> cluster_members; // Members of each clusters!


  // Constructors:
  Problem() = default;
  Problem(std::string_view name_) : name{ name_ } {}
  Problem(std::string_view name_, DataLoader &loader_) : name{ name_ }, data{ loader_.load() }
  {
    distMat = dtwc::VecMatrix<data_t>(data.size(), data.size(), -1);
  }

  // Getters and setters:
  auto &getDistanceMatrix() { return distMat; }

  auto cluster_size() const { return Nc; }
  auto &p_names(size_t i) { return data.p_names[i]; } // Alias not to write data. everytime.
  auto &p_vec(size_t i) { return data.p_vec[i]; }     // Alias not to write data. everytime.

  void clear_clusters();

  void resize()
  {
    cluster_members.resize(Nc);
    centroids_ind.resize(Nc);
    clusters_ind.resize(data.size());
  }

  auto set_numberOfClusters(ind_t Nc_)
  {
    assert(Nc_ > 0);
    Nc = Nc_;
    resize();
  }

  std::string get_name(ind_t i) { return writeAsFileNames ? p_names(i) : std::to_string(i); }

  data_t maxDistance();


  data_t distByInd(int i, int j);
  data_t distByInd_scaled(int i, int j) { return distByInd(i, j) * 10.0 / (maxDistance()); };
  void fillDistanceMatrix();
  void printDistanceMatrix() { getDistanceMatrix().print(); }

  void writeDistanceMatrix(const std::string &name_) { writeMatrix(getDistanceMatrix(), name_, output_folder); }
  void writeDistanceMatrix() { writeDistanceMatrix(name + "_distanceMatrix.csv"); }

  void printClusters();
  void writeClusters();

  void writeMedoidMembers(int iter, int rep = 0);

  auto calculate_silhouette();

  void writeSilhouettes();

  // Initialisation of clusters:
  void init_random();
  void init_Kmeanspp();

  // Clustering functions:
  void cluster_by_MIP();
  std::pair<int, double> cluster_by_kMedoidsPAM(int rep, int maxIter = 100);
  void cluster_by_kMedoidsPAM_repetetive(int N_repetition, int maxIter = 100);

  // Aux
  double findTotalCost();
  void assignClusters();
  void distributeClusters();
  void distanceInClusters();

  void calculateMedoids();

  void writeDataOrder(fs::path out_folder = settings::resultsPath)
  {
    std::ofstream out(out_folder / (name + "_dataOrder.csv"), std::ios_base::out);

    for (size_t i = 0; i < data.size(); i++)
      out << i << ',' << p_names(i) << '\n';

    out.close();
  }
};


} // namespace dtwc