/*
 * mip.hpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "mip.hpp"
#include "Problem.hpp"
#include "settings.hpp"
#include "utility.hpp"
#include "gurobi_c++.h"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {

void MIP_clustering_byGurobi(Problem &prob)
{
  auto &clusters = prob.clusters;

  const auto Nb = prob.size();
  const auto Nc = clusters.size();

  clusters.clear();
  
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
        obj += w[i + j * Nb] * prob.DTWdistByInd(i, j);

    model.setObjective(obj, GRB_MINIMIZE);
    std::cout << "Finished setting up the MILP problem." << std::endl;

    model.optimize();

    for (unsigned int i{ 0 }; i < Nb; i++)
      if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
        clusters.centroids_ind.push_back(i);


    clusters.clusters_ind = std::vector<unsigned int>(Nb);

    unsigned int i_cluster = 0;
    for (auto i : clusters.centroids_ind) {
      clusters.cluster_members.emplace_back();
      for (size_t j{ 0 }; j < Nb; j++)
        if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.5) {
          clusters.clusters_ind[j] = i_cluster;
          clusters.cluster_members.back().push_back(j);
        }

      i_cluster++;
    }

  } catch (GRBException &e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl
              << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Unknown Exception during Gurobi optimisation" << std::endl;
  }
}


} // namespace dtwc
