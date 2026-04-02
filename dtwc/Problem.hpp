/**
 * @file Problem.hpp
 * @brief Encapsulates the DTWC (Dynamic Time Warping Clustering) problem in a class.
 *
 * @details This file contains the definition of the Problem class used in DTWC applications.
 * It includes various methods for manipulating and analyzing clusters.
 *
 * @date 19 Oct 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "Data.hpp"           // for Data
#include "DataLoader.hpp"     // for DataLoader
#include "fileOperations.hpp" // for load_batch_file, readFile
#include "settings.hpp"       // for data_t, resultsPath
#include "enums/enums.hpp"    // for using Enum types.
#include "initialisation.hpp" // for init functions
#include "core/dtw_options.hpp" // for DTWVariant

#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t
#include <filesystem>  // for operator/, path
#include <ostream>     // for operator<<, basic_ostream, ofstream
#include <string>      // for char_traits, operator+, operator<<
#include <string_view> // for string_view
#include <utility>     // for pair
#include <vector>      // for vector, allocator
#include <type_traits> // std::decay_t
#include <functional>  // std::function
#include <iostream>

#include "core/distance_matrix.hpp"

namespace dtwc {

/// Strategy for computing the pairwise distance matrix.
enum class DistanceMatrixStrategy {
  Auto,       ///< Choose best strategy automatically
  BruteForce, ///< Parallel brute-force (no lower-bound pruning)
  Pruned,     ///< Parallel with LB_Kim + LB_Keogh early-abandon
  GPU         ///< CUDA GPU (requires DTWC_HAS_CUDA)
};

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
  using distMat_t = core::DenseDistanceMatrix;
  using path_t = std::filesystem::path;

  /// DTW distance function type: computes distance between two series.
  using dtw_fn_t = std::function<data_t(const std::vector<data_t> &, const std::vector<data_t> &)>;

private:
  int Nc{ 1 };                                      /*!< Number of clusters. */
  distMat_t distMat;                                /*!< Distance matrix. */
  Solver mipSolver{ settings::DEFAULT_MIP_SOLVER }; /*!< Solver for MIP. */
  dtw_fn_t dtw_fn_;                                 /*!< DTW distance function (set by rebind_dtw_fn). */

  bool is_distMat_filled{ false }; /*!< Flag indicating if the distance matrix is filled. */

  void rebind_dtw_fn(); ///< Rebind dtw_fn_ based on current variant_params and band.
  void fillDistanceMatrix_BruteForce(); ///< Brute-force parallel distance matrix fill.

  // Private functions:
  std::pair<int, double> cluster_by_kMedoidsLloyd_single(int rep);

  void writeBestRep(int best_rep);
  void writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost);
  void distanceInClusters();

public:
  Method method{ Method::Kmedoids };         /*!< Clustering method. */
  int maxIter{ 100 };                        /*!< Maximum number of iteration for iterative-methods. */
  int N_repetition{ 1 };                     /*!< Repetition for iterative-methods. */
  int band{ settings::DEFAULT_BAND_LENGTH }; /*!< Band length for Sakoe-Chiba band, -1 for full DTW. */
  core::DTWVariantParams variant_params;     /*!< DTW variant selection and parameters. */
  core::MissingStrategy missing_strategy = core::MissingStrategy::Error; /*!< Strategy for handling NaN values in series. */
  DistanceMatrixStrategy distance_strategy{ DistanceMatrixStrategy::Auto }; /*!< Distance matrix strategy. */
  bool verbose{ false };                     /*!< Print progress messages for long-running operations. */

  std::function<void(Problem &)> init_fun{ init::random }; /*!< Initialisation function. */

  path_t output_folder{ settings::paths::resultsPath }; /*!< Output folder for results. */
  std::string name{};                            /*!< Problem name. */
  Data data;                                     /*!< Data associated with the problem. */

  std::vector<int> clusters_ind;  //!< Indices of which point belongs to which cluster. [0,Nc)
  std::vector<int> centroids_ind; //!< indices of cluster centroids. [0, Np)

  // Constructors:
  Problem() { rebind_dtw_fn(); }
  Problem(std::string_view name_) : name{ name_ } { rebind_dtw_fn(); }
  Problem(std::string_view name_, DataLoader &loader_)
    : name{ name_ }, data{ loader_.load() }
  {
    refreshDistanceMatrix(); // also calls rebind_dtw_fn()
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
  int centroid_of(int i_p) const { return centroids_ind[clusters_ind[i_p]]; } // [0, Np) Get the centroid of the cluster of i_p

  void readDistanceMatrix(const fs::path &distMat_path);
  void set_numberOfClusters(int Nc_);
  void set_clusters(std::vector<int> &candidate_centroids);
  bool set_solver(dtwc::Solver solver_);

  void set_data(dtwc::Data data_)
  {
    data = data_;
    refreshDistanceMatrix();
  }

  /// Set DTW variant and rebind the distance function.
  void set_variant(core::DTWVariant v);
  void set_variant(core::DTWVariantParams params);

  data_t maxDistance() const { return distMat.max(); }
  data_t distByInd(int i, int j);
  bool isDistanceMatrixFilled() const { return is_distMat_filled; }

  /// Access the underlying distance matrix (const).
  const distMat_t &distance_matrix() const { return distMat; }
  /// Access the underlying distance matrix (mutable).
  distMat_t &distance_matrix() { return distMat; }
  /// Mark the distance matrix as filled (e.g., after loading from checkpoint).
  void set_distance_matrix_filled(bool filled) { is_distMat_filled = filled; }

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
  void cluster_by_kMedoidsLloyd();

  void cluster_and_process();

  // Auxillary
  double findTotalCost();
  void assignClusters();

  void calculateMedoids();
};


} // namespace dtwc
