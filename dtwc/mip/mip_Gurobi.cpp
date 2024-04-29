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
#include "../types/types.hpp" // for Range


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

  const auto Nb(prob.size()), Nc(prob.cluster_size());
  prob.centroids_ind.clear();

  try {
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    std::unique_ptr<GRBVar[]> w{ model.addVars(Nb * Nb, GRB_BINARY) };

    for (auto i : Range(Nb)) {
      GRBLinExpr lhs = 0;
      for (auto j : Range(Nb))
        lhs += w[j + i * Nb];

      model.addConstr(lhs, '=', 1.0);
    }


    for (auto j : Range(Nb))
      for (auto i : Range(Nb))
        model.addConstr(w[i + j * Nb] <= w[i * (Nb + 1)]);

    {
      GRBLinExpr lhs = 0;
      for (auto i : Range(Nb))
        lhs += w[i * (Nb + 1)];

      model.addConstr(lhs == Nc); // There should be Nc clusters.
    }

    prob.fillDistanceMatrix(); // We need full distance matrix before MIP clustering.
    const auto scaling_factor = std::max(prob.maxDistance() / 2.0, 1.0);
    // Set objective
    GRBLinExpr obj = 0;
    for (auto j : Range(Nb))
      for (auto i : Range(Nb))
        obj += w[i + j * Nb] * prob.distByInd(i, j) / scaling_factor;

    model.setObjective(obj, GRB_MINIMIZE);

    model.set(GRB_IntParam_NumericFocus, 3); // Much numerics
    model.set(GRB_DoubleParam_MIPGap, 1e-5); // Default 1e-4

    std::cout << "Finished setting up the MILP problem." << std::endl;

    model.optimize();

    for (auto i : Range(Nb))
      if (w[i * (Nb + 1)].get(GRB_DoubleAttr_X) > 0.5)
        prob.centroids_ind.push_back(i);

    prob.clusters_ind.resize(Nb);

    for (auto i : Range(prob.cluster_size()))
      for (auto j : Range(Nb))
        if (w[prob.centroids_ind[i] + j * Nb].get(GRB_DoubleAttr_X) > 0.5)
          prob.clusters_ind[j] = i;

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