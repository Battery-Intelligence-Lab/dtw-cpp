/*
 * mip.hpp
 *
 * Encapsulating mixed-integer program functions in a class.

 *  Created on: 06 Nov 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
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
  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();

  prob.clear_clusters();

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
        obj += w[i + j * Nb] * prob.distByInd(i, j);

    model.setObjective(obj, GRB_MINIMIZE);
    std::cout << "Finished setting up the MILP problem." << std::endl;

    model.optimize();

    for (ind_t i{ 0 }; i < Nb; i++)
      if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
        prob.centroids_ind.push_back(i);


    prob.clusters_ind = std::vector<ind_t>(Nb);

    ind_t i_cluster = 0;
    for (auto i : prob.centroids_ind) {
      prob.cluster_members.emplace_back();
      for (size_t j{ 0 }; j < Nb; j++)
        if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.5) {
          prob.clusters_ind[j] = i_cluster;
          prob.cluster_members.back().push_back(j);
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


void MIP_clustering_byGurobi_relaxed(Problem &prob)
{
  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();

  prob.clear_clusters();

  try {
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    std::unique_ptr<GRBVar[]> isCluster{ model.addVars(Nb, GRB_CONTINUOUS) };
    std::unique_ptr<GRBVar[]> w{ model.addVars(Nb * Nb, GRB_CONTINUOUS) };

    for (size_t i{ 0 }; i < Nb; i++) {
      GRBLinExpr lhs = 0;
      for (size_t j{ 0 }; j < Nb; j++) {
        lhs += w[j + i * Nb];
        model.addConstr(w[j + i * Nb] <= 1); // For relaxed version.
        model.addConstr(w[j + i * Nb] >= 0); // For relaxed version.
      }
      model.addConstr(lhs, '=', 1.0);
    }


    for (size_t j{ 0 }; j < Nb; j++)
      for (size_t i{ 0 }; i < Nb; i++)
        model.addConstr(w[i + j * Nb] <= isCluster[i]);

    {
      GRBLinExpr lhs = 0;
      for (size_t i{ 0 }; i < Nb; i++) {
        lhs += isCluster[i];
        model.addConstr(isCluster[i] <= 1); // For relaxed version.
        model.addConstr(isCluster[i] >= 0); // For relaxed version.
      }

      model.addConstr(lhs == Nc); // There should be Nc clusters.
    }

    // Set objective
    GRBLinExpr obj = 0;
    for (size_t j{ 0 }; j < Nb; j++)
      for (size_t i{ 0 }; i < Nb; i++)
        obj += w[i + j * Nb] * prob.distByInd(i, j);

    model.setObjective(obj, GRB_MINIMIZE);
    std::cout << "Finished setting up the MILP problem." << std::endl;

    model.optimize();

    for (ind_t i{ 0 }; i < Nb; i++)
      if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
        prob.centroids_ind.push_back(i);


    prob.clusters_ind = std::vector<ind_t>(Nb);

    ind_t i_cluster = 0;
    for (auto i : prob.centroids_ind) {
      prob.cluster_members.emplace_back();
      for (size_t j{ 0 }; j < Nb; j++)
        if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.5) {
          prob.clusters_ind[j] = i_cluster;
          prob.cluster_members.back().push_back(j);
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
