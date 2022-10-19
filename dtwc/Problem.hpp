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

  int Nb;   // Number of data points
  int Nc{ 4 }; // Number of clusters.

public:
  // Getters and setters:

  auto &getDistanceMatrix() { return DTWdist; }

  auto set_numberOfClusters(int Nc_)
  {
    Nc = Nc_;
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
      size_t i{ i_linear / N }, j{ i_linear % N };
      if (i <= j)
        DTWdistByInd(i, j);
    };

    dtwc::run(oneTask, Nb * Nb);
  }

  void writeAllDistances(const std::string &name) { writeMatrix(DTWdist, name); }

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
      std::unique_ptr<GRBVar[]> w{model.addVars(Nb * Nb, GRB_BINARY)};

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

    } catch (GRBException& e) {
      std::cout << "Error code = " << e.getErrorCode() << std::endl;
      std::cout << e.getMessage() << std::endl;
    } catch (...) {
      std::cout << "Exception during optimization" << std::endl;
    }
  }
};
} // namespace dtwc