/*
 * Problem.hpp
 *
 * Encapsulating DTWC problem in a class.

 *  Created on: 19 Oct 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "settings.hpp"
#include "utility.hpp"
#include "fileOperations.hpp"
#include "initialisation.hpp"
#include "timing.hpp"
#include "gurobi_c++.h"


#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {
template <typename Tdata>
class Problem
{
  std::vector<std::vector<Tdata>> p_vec;
  std::vector<std::string> p_names;
  VecMatrix<Tdata> DTWdist;

  std::vector<int> centroids_ind;                // indices of cluster centroids.
  std::vector<int> clusters_ind;                 // which point belongs to which cluster.
  std::vector<std::vector<int>> cluster_members; // Members of each clusters!

  int Nb;      // Number of data points
  int Nc{ 4 }; // Number of clusters.

public:
  // Getters and setters:

  auto &getDistanceMatrix() { return DTWdist; }

  auto clear_clusters()
  {
    centroids_ind.clear();
    clusters_ind.clear();
    cluster_members.clear();
  }

  auto set_numberOfClusters(int Nc_)
  {
    Nc = Nc_;
    clear_clusters();
  }


  double DTWdistByInd(int i, int j)
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

  void fillDistanceMatrix()
  {
    auto oneTask = [&, N = Nb](size_t i_linear) {
      thread_local TestNumberOfThreads a{};
      size_t i{ i_linear / N }, j{ i_linear % N };
      if (i <= j)
        DTWdistByInd(i, j);
    };

    dtwc::run(oneTask, Nb * Nb);
  }

  void writeDistanceMatrix(const std::string &name) { writeMatrix(DTWdist, name); }

  void load_data_fromFolder(std::string_view folder_path, int Ndata = -1, bool print = false)
  {
    std::tie(p_vec, p_names) = load_data<Tdata>(folder_path, Ndata, print);

    Nb = p_vec.size();
    DTWdist = dtwc::VecMatrix<Tdata>(Nb, Nb, -1);
  }


  void cluster_byMIP()
  {
    clear_clusters();

    try {
      GRBEnv env = GRBEnv();
      GRBModel model = GRBModel(env);

      // Create variables
      std::unique_ptr<GRBVar[]> isCluster{ model.addVars(Nb, GRB_BINARY) };
      std::unique_ptr<GRBVar[]> w{ model.addVars(Nb * Nb, GRB_BINARY) };

      for (size_t i{ 0 }; i < Nb; i++) {
        GRBLinExpr lhs = 0;
        for (size_t j{ 0 }; j < Nb; j++) {
          lhs += w[j + i * Nb];
        }
        model.addConstr(lhs, '=', 1.0);
      }


      for (size_t j{ 0 }; j < Nb; j++)
        for (size_t i{ 0 }; i < Nb; i++)
          model.addConstr(w[i + j * Nb] <= isCluster[i]);

      {
        GRBLinExpr lhs = 0;
        for (size_t i{ 0 }; i < Nb; i++)
          lhs += isCluster[i];

        model.addConstr(lhs == Nc); // There should be Nc clusters.
      }

      // Set objective
      GRBLinExpr obj = 0;
      for (size_t j{ 0 }; j < Nb; j++)
        for (size_t i{ 0 }; i < Nb; i++)
          obj += w[i + j * Nb] * DTWdistByInd(i, j);

      model.setObjective(obj, GRB_MINIMIZE);
      std::cout << "Finished setting up the MILP problem." << std::endl;

      model.optimize();

      // for (size_t i{ 0 }; i < Nb; i++)
      //   std::cout << isCluster[i].get(GRB_StringAttr_VarName) << " "
      //             << isCluster[i].get(GRB_DoubleAttr_X) << '\n';

      // std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

      for (int i{ 0 }; i < Nb; i++)
        if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
          centroids_ind.push_back(i);

      std::cout << "Clusters: ";
      for (auto ind : centroids_ind)
        std::cout << p_names[ind] << ' ';

      std::cout << '\n';


      clusters_ind = std::vector<int>(Nb);

      int i_cluster = 0;
      for (auto i : centroids_ind) {
        cluster_members.emplace_back();
        std::cout << "Cluster " << p_names[i] << " has: ";
        for (size_t j{ 0 }; j < Nb; j++)
          if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.5) {
            std::cout << p_names[j] << " ";
            clusters_ind[j] = i_cluster;
            cluster_members.back().push_back(j);
          }

        std::cout << '\n';
        i_cluster++;
      }

    } catch (GRBException &e) {
      std::cout << "Error code = " << e.getErrorCode() << std::endl
                << e.getMessage() << std::endl;
    } catch (...) {
      std::cout << "Exception during Gurobi optimisation" << std::endl;
    }
  }


  void writeClusters(std::string name)
  {

    // for (int i = 0; i < Nc; i++) {
    //   for (size_t j = 0; j < cluster_members[i].size(); j++)
    //     std::cout << cluster_members[i][j] << ' ';
    //   std::cout << '\n';
    // }

    name += "_" + std::to_string(Nc) + ".csv";

    std::ofstream myFile(settings::resultsPath + name, std::ios_base::out);

    myFile << "Clusters:\n";

    for (int i{ 0 }; i < Nc; i++) {
      if (i != 0)
        myFile << ',';

      myFile << p_names[centroids_ind[i]];
    }

    myFile << "\n\n";
    myFile << "Data" << ',' << "its cluster\n";

    for (int i{ 0 }; i < Nb; i++) {
      myFile << p_names[i] << ',' << p_names[centroids_ind[clusters_ind[i]]] << '\n';
    }

    myFile.close();
  }

  auto calculate_silhouette()
  {
    // For explanation, see: https://en.wikipedia.org/wiki/Silhouette_(clustering)

    if (centroids_ind.empty())
      std::cout << "Please cluster the data before calculating silhouette!\n";

    std::vector<double> silhouettes(Nb);

    auto oneTask = [&, N = Nb](size_t i_b) {
      auto i_c = clusters_ind[i_b];

      if (cluster_members[i_c].size() == 1)
        silhouettes[i_b] = 0;
      else {
        thread_local std::vector<double> mean_distances(Nc);

        for (size_t i = 0; i < Nb; i++)
          mean_distances[clusters_ind[i]] += DTWdistByInd(i, i_b);


        auto min = std::numeric_limits<double>::max();
        for (size_t i = 0; i < Nc; i++) // Finding means:
          if (i == i_c)
            mean_distances[i] /= (cluster_members[i].size() - 1);
          else {
            mean_distances[i] /= cluster_members[i].size();
            min = std::min(min, mean_distances[i]);
          }

        silhouettes[i_b] = (min - mean_distances[i_c]) / std::max(min, mean_distances[i_c]);
      }
    };

    dtwc::run(oneTask, Nb);

    return silhouettes;
  }

  void write_silhouettes()
  {

    auto silhouettes = calculate_silhouette();

    std::string name{ "silhouettes_" };

    name += std::to_string(Nc) + ".csv";

    std::ofstream myFile(settings::resultsPath + name, std::ios_base::out);

    myFile << "Silhouettes:\n";
    for (int i{ 0 }; i < Nb; i++) {
      myFile << p_names[i] << ',' << silhouettes[i] << '\n';
    }

    myFile.close();
  }
};


} // namespace dtwc