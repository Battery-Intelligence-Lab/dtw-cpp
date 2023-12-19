/*
 * Problem.hpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 19 Oct 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Data.hpp"            // for Data
#include "DataLoader.hpp"      // for DataLoader
#include "fileOperations.hpp"  // for writeMatrix, readMatrix
#include "settings.hpp"        // for data_t, resultsPath, writeAsFileNames
#include "types/VecMatrix.hpp" // for VecMatrix
#include "enums/enums.hpp"     // for using Enum types.
#include "initialisation.hpp"  // for init functions

#include <cstddef>     // for size_t
#include <filesystem>  // for operator/, path
#include <ostream>     // for operator<<, basic_ostream, ofstream
#include <string>      // for char_traits, operator+, operator<<
#include <string_view> // for string_view
#include <utility>     // for pair
#include <vector>      // for vector, allocator
#include <type_traits> // std::decay_t
#include <functional>  // std::function
#include <iostream>

#include <Eigen/Dense> //

namespace dtwc {

class Problem
{
  int Nc{ 1 }; // Number of clusters.
  Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> distMat;
  data_t maxDist{ -1 };
  Solver mipSolver{ settings::DEFAULT_MIP_SOLVER };

  // Private functions:
  std::pair<int, double> cluster_by_kMedoidsPAM_single(int rep);

public:
  bool writeAsFileNames{ settings::writeAsFileNames };
  Method method{ Method::Kmedoids };
  int maxIter{ 100 };    // Maximum number of iteration for iterative-methods
  int N_repetition{ 1 }; // Repetition for iterative-methods.
  int band{ settings::band };

  std::function<void(Problem &)> init_fun{ init::random }; // Initialisation function.

  std::decay_t<decltype(settings::resultsPath)> output_folder{ settings::resultsPath };
  std::string name{}; // Problem name
  Data data;

  std::vector<int> centroids_ind;                // indices of cluster centroids.
  std::vector<int> clusters_ind;                 // which point belongs to which cluster.
  std::vector<std::vector<int>> cluster_members; // Members of each clusters!

  // Constructors:
  Problem() = default;
  Problem(std::string_view name_) : name{ name_ } {}
  Problem(std::string_view name_, DataLoader &loader_)
    : name{ name_ }, data{ loader_.load() }
  {
    refreshDistanceMatrix();
  }

  auto size() const { return data.size(); }
  auto cluster_size() const { return Nc; }
  auto &p_names(size_t i) { return data.p_names[i]; } // Alias not to write data. everytime.
  auto &p_vec(size_t i) { return data.p_vec[i]; }     // Alias not to write data. everytime.

  void refreshDistanceMatrix() { distMat.setConstant(size(), size(), -1); }

  // Getters and setters:
  auto &getDistanceMatrix() { return distMat; }

  template <typename T>
  auto readDistanceMatrix(const T &distMat_path) { readMatrix(distMat, distMat_path); } // Reads distance matrix from file.

  void clear_clusters();
  void resize();

  void set_numberOfClusters(int Nc_);
  void set_clusters(std::vector<int> &candidate_centroids);
  bool set_solver(dtwc::Solver solver_);

  std::string get_name(size_t i) { return writeAsFileNames ? p_names(i) : std::to_string(i); }

  data_t maxDistance();

  data_t distByInd(int i, int j);
  data_t distByInd_scaled(int i, int j) { return distByInd(i, j) * 2.0 / (maxDistance()); };
  void fillDistanceMatrix();
  void printDistanceMatrix() { std::cout << getDistanceMatrix() << '\n'; }

  void writeDistanceMatrix(const std::string &name_) { writeMatrix(getDistanceMatrix(), name_, output_folder); }
  void writeDistanceMatrix() { writeDistanceMatrix(name + "_distanceMatrix.csv"); }

  void printClusters();
  void writeClusters();

  void writeMedoidMembers(int iter, int rep = 0);

  auto calculate_silhouette();

  void writeSilhouettes();

  // Initialisation of clusters:
  void init_random() { init::random(*this); }
  void init_Kmeanspp() { init::Kmeanspp(*this); }
  void init() { init_fun(*this); }

  // Clustering functions:
  void cluster();
  void cluster_by_MIP();
  void cluster_by_kMedoidsPAM();

  void cluster_and_process();

  // Auxillary
  double findTotalCost();
  void assignClusters();
  void distributeClusters();
  void distanceInClusters();

  void calculateMedoids();
};


} // namespace dtwc