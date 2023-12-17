/*
 * mip.cpp
 *
 * Encapsulating Gurobi-related mixed-integer programming functions in a class.

 *  Created on: 25 Dec 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 *
 */

#include "mip.hpp"
#include "../Problem.hpp"
#include "../settings.hpp"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>


#ifdef DTWC_ENABLE_GUROBI

#include "gurobi_c++.h"

#endif


namespace dtwc {
void MIP_clustering_byGurobi(Problem &prob)
{
#ifdef DTWC_ENABLE_GUROBI

  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();

  prob.clear_clusters();

  try {
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    std::unique_ptr<GRBVar[]> w{ model.addVars(Nb * Nb, GRB_BINARY) };

    for (int i{ 0 }; i < Nb; i++) {
      GRBLinExpr lhs = 0;
      for (int j{ 0 }; j < Nb; j++) {
        lhs += w[j + i * Nb];
      }
      model.addConstr(lhs, '=', 1.0);
    }


    for (int j{ 0 }; j < Nb; j++)
      for (int i{ 0 }; i < Nb; i++)
        model.addConstr(w[i + j * Nb] <= w[i * (Nb + 1)]);
    {
      GRBLinExpr lhs = 0;
      for (int i{ 0 }; i < Nb; i++)
        lhs += w[i * (Nb + 1)];

      model.addConstr(lhs == Nc); // There should be Nc clusters.
    }

    // Set objective
    GRBLinExpr obj = 0;
    for (int j{ 0 }; j < Nb; j++)
      for (int i{ 0 }; i < Nb; i++)
        obj += w[i + j * Nb] * prob.distByInd_scaled(i, j);

    model.setObjective(obj, GRB_MINIMIZE);

    // model.set(GRB_IntParam_NumericFocus, 3); // Much numerics
    // model.set(GRB_IntParam_Method, 1);       // simplex
    model.set(GRB_DoubleParam_MIPGap, 1e-5); // Default 1e-4
    // model.set(GRB_IntParam_Threads, 3); // Set to dual simplex?
    // model.set(GRB_IntParam_Cuts, 3); // More cuts? -> not very effective.

    std::cout << "Finished setting up the MILP problem." << std::endl;


    model.optimize();

    for (int i{ 0 }; i < Nb; i++)
      if (w[i * (Nb + 1)].get(GRB_DoubleAttr_X) > 0.5)
        prob.centroids_ind.push_back(i);

    prob.clusters_ind.resize(Nb);

    int i_cluster = 0;
    for (auto i : prob.centroids_ind) {
      prob.cluster_members.emplace_back();
      for (int j{ 0 }; j < Nb; j++)
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
#else
  std::cout << "Gurobi solver is not activated but is being used!" << std::endl;
#endif
}

} // namespace dtwc