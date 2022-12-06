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
#include "osqp.h"

#include <vector>
#include <string_view>
#include <memory>
#include <limits>

namespace dtwc {


void MIP_clustering_byOSQP(Problem &prob)
{

  std::cout << "OSQP has been called!" << std::endl;
  // This is not an actual MIP solver; however, it relies on problem having turtley unimodular matrices!
  const auto Nb = prob.data.size();
  const auto Nc = prob.cluster_size();

  prob.clear_clusters();
  try {
    c_int n = Nb * (Nb + 1);                       // total states w, isCluster
    c_int m = (Nb * Nb + Nb) + (Nb * Nb) + Nb + 1; // number of constraints

    c_float P_x[1] = { 0 };
    c_int P_nnz = 0;
    c_int P_i[1] = { 0 };

    auto P_p = new c_int[n + 1];

    for (int i = 0; i < n; i++)
      P_p[i] = 0;

    P_p[n] = 1;


    auto l = new c_float[m];
    auto u = new c_float[m];


    size_t i = 0;
    size_t i_until = (Nb * Nb + Nb + Nb * Nb);
    for (; i < i_until; i++) {
      l[i] = 0.0;
      u[i] = 1.0;
    }

    i_until += Nb;
    for (; i < i_until; i++)
      l[i] = u[i] = 1.0;

    i_until += 1;
    for (; i < i_until; i++)
      l[i] = u[i] = Nc;


    c_int A_nnz = (Nb * Nb + Nb) + 2 * Nb * Nb + (Nb * Nb + Nb);

    // Every column has 3 non-zero. So there should be (N^2 + N)*3, A_i

    auto A_i = new c_int[A_nnz];
    auto A_p = new c_int[n + 1];
    auto A_x = new c_float[A_nnz];

    A_p[0] = 0;

    for (size_t j_out = 0; j_out < (Nb + 1); j_out++) // Columns
      for (size_t j_in = 0; j_in < Nb; j_in++)        // inner block
      {
        auto j = j_out * Nb + j_in;


        if (j_out < Nb) {
          A_p[j + 1] = A_p[j] + 3;

          A_i[A_p[j] + 0] = j;
          A_i[A_p[j] + 1] = n + j;
          A_i[A_p[j] + 2] = n + Nb * Nb + j_out;

          A_x[A_p[j] + 0] = 1;
          A_x[A_p[j] + 1] = -1;
          A_x[A_p[j] + 2] = 1;
        } else {

          A_p[j + 1] = A_p[j] + 2 + Nb;

          A_i[A_p[j] + 0] = j;
          A_x[A_p[j] + 0] = 1;

          for (int k = 0; k < Nb; k++) {
            A_i[A_p[j] + 1 + k] = n + j_in + k * Nb;
            A_x[A_p[j] + 1 + k] = 1;
          }

          A_i[A_p[j] + Nb + 1] = n + Nb * Nb + j_out;
          A_x[A_p[j] + Nb + 1] = 1;
        }
      }

    auto q = new c_float[n];

    for (size_t j{ 0 }; j < Nb; j++)
      for (size_t i{ 0 }; i < Nb; i++)
        q[i + j * Nb] = prob.distByInd(i, j);


    for (size_t i{ 0 }; i < Nb; i++)
      q[Nb * Nb + i] = 0;

    // Exitflag
    c_int exitflag = 0;

    // Workspace structures
    OSQPWorkspace *work;
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPData *data = (OSQPData *)c_malloc(sizeof(OSQPData));

    // --------------------------------

    // Populate data
    if (data) {
      data->n = n;
      data->m = m;
      data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
      data->q = q;
      data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
      data->l = l;
      data->u = u;
    }

    // Define solver settings as default
    if (settings) osqp_set_default_settings(settings);

    // Additional settings by Vk:
    settings->max_iter = 100000;
    settings->eps_abs = 1e-4;
    settings->eps_rel = 1e-4;


    // Setup workspace
    exitflag = osqp_setup(&work, data, settings);

    // Solve Problem
    osqp_solve(work);


    // for (size_t i = 0; i < work->data->n; i++)
    //   std::cout << work->solution->x[i] << '\n';


    // ----- Retrieve solutions START ------
    for (ind_t i{ 0 }; i < Nb; i++) {
      auto isCentroid_i = work->solution->x[Nb * Nb + i];

      if (isCentroid_i > 0.75) {
        prob.centroids_ind.push_back(i);
        if (isCentroid_i < 0.9)
          std::cout << "OSQP may not have the most accurate solution ever for centroid " << isCentroid_i << '\n';
      } else if (isCentroid_i > 0.25) // Should not happen!
      {
        std::cerr << "Centroid " << i << " has value of " << isCentroid_i << " which should not happen for turtley unimodular matrices!\n";
        throw 10000; // #TODO more meaningful error codes?
      }
    }

    prob.clusters_ind = std::vector<ind_t>(Nb);

    ind_t i_cluster = 0;
    for (auto i : prob.centroids_ind) {
      prob.cluster_members.emplace_back();
      for (size_t j{ 0 }; j < Nb; j++)
        if (work->solution->x[i + j * Nb] > 0.75) {
          if (work->solution->x[i + j * Nb] < 0.9)
            std::cout << "OSQP may not have the most accurate solution ever for weight " << work->solution->x[i + j * Nb] << '\n';
          prob.clusters_ind[j] = i_cluster;
          prob.cluster_members.back().push_back(j);
        } else if (work->solution->x[i + j * Nb] > 0.25) {
          std::cerr << "Weight " << i + j * Nb << " has value of " << work->solution->x[i + j * Nb] << " which should not happen for turtley unimodular matrices!\n";
          throw 10000; // #TODO more meaningful error codes?
        }

      i_cluster++;
    }
    // ----------  Retrieve solutions END ---------


    // Clean workspace
    // osqp_cleanup(work);
    if (data) {
      if (data->A) c_free(data->A);
      if (data->P) c_free(data->P);
      c_free(data);
    }
    if (settings) c_free(settings);


  } catch (...) {
    std::cout << "OSQP problem" << std::endl;
  }
}


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
  std::cout << "Gurobi-relaxed has been called!" << std::endl;

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
      if (isCluster[i].get(GRB_DoubleAttr_X) > 0.9)
        prob.centroids_ind.push_back(i);
      else if (isCluster[i].get(GRB_DoubleAttr_X) > 0.1) {
        std::cerr << "Cluster " << i << " has value of " << isCluster[i].get(GRB_DoubleAttr_X) << " which should not happen for turtley unimodular matrices!\n";
        throw 10000; // #TODO more meaningful error codes?
      }

    prob.clusters_ind = std::vector<ind_t>(Nb);

    ind_t i_cluster = 0;
    for (auto i : prob.centroids_ind) {
      prob.cluster_members.emplace_back();
      for (size_t j{ 0 }; j < Nb; j++)
        if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.9) {
          prob.clusters_ind[j] = i_cluster;
          prob.cluster_members.back().push_back(j);
        } else if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.1) {
          std::cerr << "Weight " << i + j * Nb << " has value of " << w[i + j * Nb].get(GRB_DoubleAttr_X) << " which should not happen for turtley unimodular matrices!\n";
          throw 10000; // #TODO more meaningful error codes?
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
