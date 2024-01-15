/**
 * @file Problem.cpp
 * @brief Implementation of the DTWC (Dynamic Time Warping Clustering) problem encapsulated in a class.
 *
 * @details This file includes the implementation of the Problem class, which contains methods for clustering,
 * initializing clusters, calculating distances, and other functionalities related to the DTWC problem.
 *
 * @date Created on: 06 Nov 2022
 * @author Volkan Kumtepeli, Becky Perriment
 */

#include "Problem.hpp"
#include "mip.hpp"             // for MIP_clustering_byGurobi
#include "parallelisation.hpp" // for run
#include "scores.hpp"          // for silhouette
#include "settings.hpp"        // for data_t, randGenerator, band, isDebug
#include "warping.hpp"         // for dtwBanded, dtwFull
#include "types/Range.hpp"     // for Range
#include "initialisation.hpp"  // For initialisation functions


#include <algorithm> // for max_element, min, min_element, sample
#include <iomanip>   // for operator<<, setprecision
#include <iostream>  // for cout
#include <iterator>  // for back_insert_iterator, back_inserter
#include <limits>    // for numeric_limits
#include <random>    // for mt19937, discrete_distribution, unifo...
#include <string>    // for allocator, char_traits, operator+
#include <utility>   // for pair
#include <vector>    // for vector, operator==

namespace dtwc {

/**
 * @brief Clears existing cluster information.
 *
 * @details Resets the centroids indices, cluster indices, and members of each cluster to their default states.
 */
void Problem::clear_clusters()
{
  centroids_ind.clear();
  clusters_ind.clear();
  cluster_members.clear();
}

/**
 * @brief Resizes data structures based on the current number of clusters.
 *
 * @details Adjusts the size of cluster_members, centroids_ind, and clusters_ind arrays based on the current
 * value of Nc (number of clusters).
 */
void Problem::resize()
{
  cluster_members.resize(Nc);
  centroids_ind.resize(Nc);
  clusters_ind.resize(size());
}

/**
 * @brief Sets the number of clusters for the problem.
 *
 * @param Nc_ The number of clusters to set.
 * @throws std::runtime_error if the size of candidate_centroids is not equal to Nc.
 */
void Problem::set_numberOfClusters(int Nc_)
{
  Nc = Nc_;
  resize();
}

/**
 * @brief Sets the initial centroids for clustering.
 *
 * @param candidate_centroids A vector containing the indices of candidate centroids.
 */
void Problem::set_clusters(std::vector<int> &candidate_centroids)
{
  if (candidate_centroids.size() != Nc)
    throw std::runtime_error("Set cluster has failed as number of centroids is not same as the number of indices in candidate centroids vector.\n");

  centroids_ind = candidate_centroids;
}

/**
 * @brief Sets the solver to be used for clustering.
 *
 * @param solver_ The solver to use.
 * @return True if the solver is set successfully,
 * False otherwise (e.g., if Gurobi is not available and a default solver is used instead).
 */
bool Problem::set_solver(Solver solver_)
{
  if (solver_ == Solver::Gurobi) {
#ifdef DTWC_ENABLE_GUROBI
    mipSolver = Solver::Gurobi;
    return true;
#else
    std::cout << "Solver Gurobi is not available; therefore using default solver\n";
    mipSolver = settings::DEFAULT_MIP_SOLVER;
    return false;
#endif
  }

  mipSolver = solver_;
  return true;
}

/**
 * @brief Prints the current distance matrix to the standard output.
 * @details Outputs the distance matrix in a human-readable format, useful for debugging and verification.
 */
void Problem::printDistanceMatrix() const { std::cout << distMat << '\n'; }

/**
 * @brief Refreshes the distance matrix.
 * @details Resets the distance matrix and marks it as not filled. This is necessary when the data has changed,
 * requiring a re-calculation of distances.
 */
void Problem::refreshDistanceMatrix()
{
  distMat.set_size(size(), size());
  distMat.fill(-1);
  is_distMat_filled = false;
}

/**
 *@brief Retrieves or calculates the distance between two points by their indices.
 *@param i Index of the first point.
 *@param j Index of the second point.
 *@return The distance between the two points.
 */
double Problem::distByInd(int i, int j)
{
  if (distMat(i, j) < 0)
    distMat(j, i) = distMat(i, j) = dtwBanded(p_vec(i), p_vec(j), band);

  return distMat(i, j);
}

/**
 * @brief Fills the distance matrix by computing distances between all pairs of points.
 * @details Populates the distance matrix using the DTW banded algorithm. This operation is parallelized for efficiency.
 */
void Problem::fillDistanceMatrix()
{
  auto oneTask = [&, N = data.size()](size_t i_linear) {
    size_t i{ i_linear / N }, j{ i_linear % N };
    if (i <= j)
      distByInd(i, j);
  };

  std::cout << "Distance matrix is being filled!" << std::endl;
  run(oneTask, data.size() * data.size());
  is_distMat_filled = true;
  std::cout << "Distance matrix has been filled!" << std::endl;
}

/**
 * @brief Performs clustering based on the specified method.
 * @details Chooses between different clustering methods (K-medoids or MIP) and performs the clustering accordingly.
 */
void Problem::cluster()
{
  switch (method) {
  case Method::Kmedoids:
    cluster_by_kMedoidsPAM();
    break;
  case Method::MIP:
    cluster_by_MIP();
    break;
  }
}

/**
 * @brief Executes the clustering process and additional post-processing tasks.
 * @details Performs clustering, then prints and writes the cluster results, including silhouettes, to files.
 */
void Problem::cluster_and_process()
{
  cluster();
  printClusters(); // Prints to screen.
  writeDistanceMatrix();
  writeClusters(); // Prints to file.
  writeSilhouettes();
}

/**
 *@brief Clusters the data using Mixed Integer Programming (MIP) based on the chosen solver.
 *@details Uses either Gurobi or HiGHS solver for MIP clustering, depending on the solver set in the Problem instance.
 */
void Problem::cluster_by_MIP()
{
  switch (mipSolver) {
  case Solver::Gurobi:
    MIP_clustering_byGurobi(*this);
    break;
  case Solver::HiGHS:
    MIP_clustering_byHiGHS(*this);
    break;
  }
}

/**
 * @brief Distributes the data points into their respective clusters.
 * @details Assigns each data point to the cluster of the nearest centroid, then updates the cluster_members array.
 */
void Problem::distributeClusters()
{
  for (auto &member : cluster_members)
    member.clear();

  for (size_t i = 0; i < clusters_ind.size(); i++)
    cluster_members[clusters_ind[i]].push_back(i);
}

/**
 * @brief Assigns each data point to the nearest cluster centroid.
 * @details Iterates over each data point, calculating its distance to each centroid, and assigns it to the nearest one.
 */
void Problem::assignClusters()
{
  auto assignClustersTask = [&](int i_p) // i_p -> index of points
  {
    double minDist{ 1e9 };
    int minInd{}, c_ind{};
    for (auto i_c : centroids_ind) { // #TODO test if runs correctly as we changed previous algorithm.
      const auto distNew = distByInd(i_p, i_c);
      if (distNew < minDist) {
        minDist = distNew;
        minInd = c_ind;
      }
      c_ind++;
    }

    clusters_ind[i_p] = minInd;
  };

  clusters_ind.resize(data.size()); // Resize before assigning.
  run(assignClustersTask, data.size());

  distributeClusters();
}

/**
 * @brief Calculates the pairwise distances within each cluster.
 * @details Iterates through each data point, determining its cluster and calculating the distance to other points
 * within the same cluster. This method populates the distance matrix with these intra-cluster distances.
 */
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

/**
 * @brief Calculates and updates the medoids of each cluster.
 * @details This function iterates through each data point and calculates the total cost of designating that point
 * as the medoid of its cluster. The point with the minimum total cost is set as the new medoid for that cluster.
 */
void Problem::calculateMedoids()
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  std::vector<data_t> clusterCosts(centroids_ind.size(), maxValue);
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

/**
 * @brief Performs the clustering using the k-Medoids PAM (Partitioning Around Medoids) algorithm.
 * @details Executes the PAM clustering algorithm with multiple repetitions, each time initializing medoids randomly.
 * The repetition yielding the lowest total cost is chosen as the best solution.
 */
void Problem::cluster_by_kMedoidsPAM()
{
  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();

  for (int i_rand = 0; i_rand < N_repetition; i_rand++) {
    std::cout << "Metoid initialisation is started.\n";
    init();

    std::cout << "Metoid initialisation is finished. "
              << Nc << " medoids are initialised.\n"
              << "Start clustering:\n";

    auto [status, total_cost] = cluster_by_kMedoidsPAM_single(i_rand);

    if (status == 0)
      std::cout << "Medoids are same for last two iterations, algorithm is converged!\n";
    else if (status == -1)
      std::cout << "Maximum iteration is reached before medoids are converged!\n";

    if (total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
    }
    std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';
  }

  writeBestRep(best_rep);
}

/**
 * @brief Executes a single iteration of the k-Medoids PAM clustering.
 * @details This function performs a single iteration of the k-Medoids PAM algorithm, updating the medoids and clusters,
 * and calculating the total cost for this iteration.
 * @param rep The current repetition number.
 * @return A pair containing the status (whether the algorithm converged or not) and the total cost of clustering for this repetition.
 */
std::pair<int, double> Problem::cluster_by_kMedoidsPAM_single(int rep)
{
  if (centroids_ind.empty()) init(); //<! Initialise if not initialised.

  auto oldmedoids = centroids_ind;

  int status = -1;
  std::vector<std::vector<int>> centroids_all;

  for (int i = 0; i < maxIter; i++) {

    std::cout << "Medoids: ";
    for (auto medoid : centroids_ind)
      std::cout << get_name(medoid) << ' ';

    centroids_all.push_back(centroids_ind);

    assignClusters();

    std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
              << findTotalCost() << ".\n"; // Uses clusters_ind to find cost.

    printClusters();
    distanceInClusters(); // Just populates distByInd matrix ahead.
    calculateMedoids();   // Changes centroids_ind

    if (oldmedoids == centroids_ind) {
      status = 0;
      break;
    }

    oldmedoids = centroids_ind;
  }

  const double total_cost = findTotalCost();
  std::cout << "Procedure is completed with cost: " << total_cost << '\n';
  writeMedoids(centroids_all, rep, total_cost);
  return std::pair(status, total_cost);
}

/**
 * @brief Calculates the total cost of the current clustering solution.
 * @details Computes the sum of the distances between each point and its closest medoid.
 * This serves as a measure of the quality of the current clustering solution.
 * @return The total cost of the clustering.
 */
double Problem::findTotalCost()
{
  double sum = 0;
  for (int i = 0; i < data.size(); i++) {
    const auto i_p = centroids_ind[clusters_ind[i]];
    if constexpr (settings::isDebug)
      std::cout << "Distance between " << i << " and closest cluster " << i_p
                << " which is: " << distByInd(i, i_p) << "\n";

    sum += distByInd(i, i_p); // #TODO should cost be square or like this?
  }

  return sum;
}

} // namespace dtwc