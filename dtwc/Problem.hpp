/**
 * @file Problem.hpp
 * @brief Encapsulates the DTWC (Dynamic Time Warping Clustering) problem in a class.
 *
 * @details This file contains the definition of the Problem class used in DTWC applications.
 * It includes various methods for manipulating and analyzing clusters.
 *
 * @date Created on 19 Oct 2022
 * @author Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Data.hpp"           // for Data
#include "DataLoader.hpp"     // for DataLoader
#include "fileOperations.hpp" // for writeMatrix, readMatrix
#include "settings.hpp"       // for data_t, resultsPath
#include "enums/enums.hpp"    // for using Enum types.
#include "initialisation.hpp" // for init functions

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

#include <armadillo>

namespace dtwc {

/**
 * @class Problem
 * @brief Class representing a problem in DTWC.
 *
 * @details This class encapsulates all the functionalities and data structures required to solve
 * a dynamic time warping clustering problem. It includes methods for initialising clusters,
 * calculating distances, clustering, and writing results.
 */
class Problem
{
public:
  using distMat_t = arma::Mat<double>;
  using path_t = std::decay_t<decltype(settings::resultsPath)>;

private:
  int Nc{ 1 };                                      /*!< Number of clusters. */
  distMat_t distMat;                                /*!< Distance matrix. */
  Solver mipSolver{ settings::DEFAULT_MIP_SOLVER }; /*!< Solver for MIP. */

  bool is_distMat_filled{ false }; /*!< Flag indicating if the distance matrix is filled. */

  // Private functions:
  std::pair<int, double> cluster_by_kMedoidsPAM_single(int rep);

  void writeBestRep(int best_rep);
  void writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost);
  void distanceInClusters();

public:
  Method method{ Method::Kmedoids };         /*!< Clustering method. */
  int maxIter{ 100 };                        /*!< Maximum number of iteration for iterative-methods. */
  int N_repetition{ 1 };                     /*!< Repetition for iterative-methods. */
  int band{ settings::DEFAULT_BAND_LENGTH }; /*!< Band length for Sakoe-Chiba band, -1 for full DTW. */

  std::function<void(Problem &)> init_fun{ init::random }; /*!< Initialisation function. */

  path_t output_folder{ settings::resultsPath }; /*!< Output folder for results. */
  std::string name{};                            /*!< Problem name. */
  Data data;                                     /*!< Data associated with the problem. */

  std::vector<int> clusters_ind;  //!< Indices of which point belongs to which cluster. [0,Np]
  std::vector<int> centroids_ind; //!< indices of cluster centroids. [0, Np)

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
  auto &get_name(size_t i) { return data.p_names[i]; }
  auto const &get_name(size_t i) const { return data.p_names[i]; }

  auto &p_vec(size_t i) { return data.p_vec[i]; }
  auto const &p_vec(size_t i) const { return data.p_vec[i]; }

  void refreshDistanceMatrix();
  void resize();

  // Getters and setters:
  void readDistanceMatrix(const fs::path &distMat_path);
  void set_numberOfClusters(int Nc_);
  void set_clusters(std::vector<int> &candidate_centroids);
  bool set_solver(dtwc::Solver solver_);

  data_t maxDistance() const { return distMat.max(); }
  data_t distByInd(int i, int j);
  bool isDistanceMatrixFilled() const { return is_distMat_filled; }

  void fillDistanceMatrix();
  void printDistanceMatrix() const;

  void writeDistanceMatrix(const std::string &name_) const;
  void writeDistanceMatrix() const { writeDistanceMatrix(name + "_distanceMatrix.csv"); }

  void printClusters() const;
  void writeClusters();

  void writeMedoidMembers(int iter, int rep = 0) const;
  void writeSilhouettes();

  // Initialisation of clusters:
  void init() { init_fun(*this); }

  // Clustering functions:
  void cluster();
  void cluster_by_MIP();
  void cluster_by_kMedoidsPAM();

  void cluster_and_process();

  // Auxillary
  double findTotalCost();
  void assignClusters();

  void calculateMedoids();
};


} // namespace dtwc