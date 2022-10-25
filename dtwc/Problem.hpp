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

namespace dtwc {
template <typename Tdata>
class Problem
{
  std::vector<std::vector<Tdata>> p_vec;
  std::vector<std::string> p_names;
  VecMatrix<Tdata> DTWdist;

  std::vector<int> centroids_ind; // indices of cluster centroids.
  std::vector<int> clusters_ind;  // which point belongs to which cluster.

  int Nb;      // Number of data points
  int Nc{ 4 }; // Number of clusters.

public:
  // Getters and setters:

  auto &getDistanceMatrix() { return DTWdist; }

  auto set_numberOfClusters(int Nc_)
  {
    Nc = Nc_;
    centroids_ind.clear();
    clusters_ind.clear();
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


      for (size_t i{ 0 }; i < Nb; i++)
        std::cout << isCluster[i].get(GRB_StringAttr_VarName) << " "
                  << isCluster[i].get(GRB_DoubleAttr_X) << '\n';

      std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;


      for (int i{ 0 }; i < Nb; i++)
        if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
          centroids_ind.push_back(i);

      std::cout << "Clusters: ";
      for (auto ind : centroids_ind)
        std::cout << p_names[ind] << ' ';

      std::cout << '\n';


      clusters_ind = std::vector<int>(Nb);


      for (auto i : centroids_ind) {
        std::cout << "Cluster " << p_names[i] << " has: ";
        for (size_t j{ 0 }; j < Nb; j++)
          if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.5) {
            std::cout << p_names[j] << " ";
            clusters_ind[j] = i;
          }

        std::cout << '\n';
      }


    } catch (GRBException &e) {
      std::cout << "Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
    } catch (...) {
      std::cout << "Exception during optimization" << std::endl;
    }
  }


  void writeClusters(const std::string &name)
  {
    std::ofstream myFile(settings::resultsPath + name, std::ios_base::out);

    myFile << "Clusters:\n";
    for (int i{ 0 }; i < Nc; i++) {
      if (i != 0)
        myFile << ',';

      myFile << p_names[centroids_ind[i]];
    }

    myFile << "\n\n";
    myFile << "Data" << ',' << "its cluster\n";

    for (int i{ 0 }; i < Nb; i++)
      myFile << p_names[i] << ',' << p_names[clusters_ind[i]] << '\n';


    myFile.close();
  }
};


} // namespace dtwc