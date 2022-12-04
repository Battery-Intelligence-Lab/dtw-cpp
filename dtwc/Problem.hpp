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
#include <utility>


namespace dtwc {

class Problem
{
  VecMatrix<data_t> DTWdist;


  ind_t Nc{ 1 }; // Number of clusters.
  ind_t Nb{ 0 }; // Number of data points

public:
  bool writeAsFileNames{ settings::writeAsFileNames };
  fs::path output_folder{ settings::resultsPath };
  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  std::vector<ind_t> centroids_ind;                // indices of cluster centroids.
  std::vector<ind_t> clusters_ind;                 // which point belongs to which cluster.
  std::vector<std::vector<ind_t>> cluster_members; // Members of each clusters!

  // Getters and setters:
  auto &getDistanceMatrix() { return DTWdist; }

  void clear_clusters();
  void resize()
  {
    cluster_members.resize(Nc);
    centroids_ind.resize(Nc);
    clusters_ind.resize(Nb);
  }

  auto set_numberOfClusters(ind_t Nc_)
  {
    assert(Nc_ > 0);
    Nc = Nc_;
    resize();
  }

  std::string get_name(ind_t i) { return writeAsFileNames ? p_names[i] : std::to_string(i); }

  auto size() const { return Nb; }
  auto cluster_size() const { return Nc; }

  double DTWdistByInd(int i, int j);
  void fillDistanceMatrix();
  void printDistanceMatrix() { DTWdist.print(); }

  void writeDistanceMatrix(const std::string &name) { writeMatrix(DTWdist, name, output_folder); }

  void load_data_fromFolder(std::string_view folder_path, int Ndata = -1, bool print = false);
  void load_data_fromVec(std::vector<std::vector<data_t>> &&p_vec_new, std::vector<std::string> &&p_names_new);


  void printClusters();
  void writeClusters(std::string file_name);
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
};


} // namespace dtwc