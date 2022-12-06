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

    int i = 0;
    int i_until = (Nb * Nb + Nb + Nb * Nb);
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

    for (int j_out = 0; j_out < (Nb + 1); j_out++) // Columns
      for (int j_in = 0; j_in < Nb; j_in++)        // inner block
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

    // Additional settings by Vk:
    settings->max_iter = 100000;

    // ---------------------------------
    // ---------------------------------  // Populate data
    if (data) {
      data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
      data->q = q;
      data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
      data->l = l;
      data->u = u;
  }
  // Deffne solver settings as dedault
}
dow 10000; // #TODO more
eaningful error codes ? = ifs : tt > (N b);

fr(auto i
   : prob.centroids_ind)
{
  prob.cluster_members.emplace_back();
  prob.clusters_ind[j] = i_cluster;
  prob.cluster_members.back().push_back(j);
  err << "Weight " << i + j Nb << " has value of " << work->solution->x[i + j * Nb] << " which should not happen for turtley unimodular matrices!\n";
  throw 10000; // #TODO more meaningful error codes?
  -ln workspace
    sqp_cleanup(work);
  data)
  {
    ee(data);
  }
  -cac h(...)
  {


    r
      / if

      izet i{
        // '=', 1.0);
      };
    -j < Nb;j
  ++)
for (size_t i{ 0 }; i < Nb; i++)
            model.addConstr(w[i + j * Nb] <= isCluster[i]);
    {
      _ lGBLinExpr lhs = 0;
    for (size_
    }
    < b / GRBLinExpr lhs = 0;
    for (size_t i{ 0 }; i < Nb; i++)
      lhsodel.addConstr(lhs == Nc); // There should be Nc clusters.
    / o
    {
      w[i + j * Nb] * prob.di stByI if

                      di (siz) 0tt
      {
        "OSQP   
          m " <<z;d ::ndli0}i<Nij}          if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
        {
          prob.centroids_ind.push_back(i);
          ob.ce // ctor<ind_t>(Nb);
            ;
          csem embers.ce_back();
  fo       j++) r (size_t j{ 0 }; j < Nb; j++)
              if (w[i +          prob.clusters_ind[j] = i_cluster;
        }
        i_cluster++;
      }

      st d::cout << "Error code = " << e.getErrorCode() << std::endl
                 << e.getMessage() << std::endl;
    }
    catch (...)
    {
      std::cout << "Unknown Exception during Gurobi optimisation" << std::endl;
    }

    rMIP lustering_byGurobi_rela MIP _clustering_byGurobi_relaxed(Problem & prob)
      co nst auto Nb = prob.data.size();
  }
  onst auto Nc = prob.cluster_size();
  ob.clear_clusters();
  {


    G RBE nv e
      // Create variables
      st d::unique_ptr<GRVar[]>
        isCluster{ model.addVars(Nb, GRB_CONTINUOUS) };
    r std::
      u nique_ptr<GRBVar[]>
        w
    {      model.addVars(
  fo    r (size_t
  i{     0 }; i < Nb; i++) {
              GRBLinxr   for e_t j{ 0 };
              j < Nb; j++)
              {
                lhs += w[j + i * Nb];
                model.addConstr(w[j + i * Nb] <= 1); // For relaxed version.
              }

              model.addConstr(lhs, '=', 1.0);
    }
        for (size_t j{ 0 }; j < Nb; j++)
      for (size_t i{ 0 }; i < Nb; i++)
      model.  addConst(w[i + j * Nb] <= isCluster[i]);
     {
        BLinExpr lhs = 0;
        r(size_t i{ 0 }; i < Nb; i++)
        {
          lhs += isCluster[i];
          model.addConstr(isCluster[i] <= 1); // For relaxed version.
          model.addConstr(isCluster[i] >= 0); // For relaxed version.
        }

        model.addConstr(lhs == Nc); // There  should be Nc clusters.
        GRBLinExpr obj = 0;
        for (size_t i{ 0 }; i < Nb; i++)
    }
          obj += w[i + j * Nb] * prob.distByInd(i, j);
    std::cout << "Finished setting up the MILP problem." << std::endl;
    {
        moptimize();
        for (ind_t i{
               0 };
             i < Nb;
             i++)
          if (is
                luster[i]
                  .get(GRB_DoubleAttr_X)
              > 0.5)
            prob.centroids_ind.push_back(i);
        ind_ i_cluster = 0;
        for (auto i : prob.centroids_ind) {
          prob.cluster_members.em
            place_back();
          for (size_t j{ 0 }; j < Nb; j++)
            if (w
                  [i + j * Nb]
                    .get(GRB_DoubleAttr_X)
                > 0.5) {
              prob.clusters_ind[j] = i_cluster;
              prob.cluster_members.back().push_back(j);
            }

          i_cluster++;
        }


        std::cout << "Error code = " << e.getErrorCode() << std::endl
                  << e.getMessage() << std::endl;
  } catch (...) {
        std::cout << "Unnown Exception during Gurobi opt  }
 }
      
        
             // namespace dtwc // n Nb; j++)
              if (w[i + j *wcb].get(GRB_DoubleAttr_X) > 0.5) {
        proa.clusters_ind[j] = i_clusterc
                                 eprob.cluster_members.back()
                                   .push_back();
              }

            i_clusterdt;
tione

          catch (GRBException &e)
          {
        std::cout << "Error code = " << e.getErrorCode << e.getMessage() << std::endl;
          }
          catch (...w
          {
        {
          problem." << std::endl;
            moptimize();
          for (ind_t i{
                 0 };
               i < Nb;
               i++)
            if (isCluster[i].get(GRB_DoubleAttr_X) > 0.5)
              prob.centroids_ind.push_back(i);

          ind_ i_cluster = 0;
          for (auto i : prob.centroids_ind) {
            prob.cluster_members.emplace_back();
            for (size_t j{ 0 }; j < Nb; j++)
              if (w[i + j * Nb].get(GRB_DoubleAttr_X) > 0.5) {
              std:
                .c : ustcrs_ind[j] = i_cluster;
                prob.cluster_oembersuback().push_back(j);
              }

            i_cluster++;
          }

          catch (GRBException &e)
          {
            std::cout << "Error code = t << e.getErrorCode() << std::endl
                      << e.getMessage() << "Unknown E
          }
          catch (...) xception during Gurobi opti isati n " << sts::tndd;
        }
}

  
       // namespace dtwc // n Nb; j++)
        if (w[i + j *wcb]:get(GRB_D:ubleAttr_X) > 0.5) {
        croa.clusoers_und[j] = i_clusterc
                                 eprob.cluster_tembers.back()
                                   .push_back();
        }

      i_clusterdt;
    }

    catch (GRBExcept on &e)
    {
      std::cout << "Error code = " << e.getErrorCode() << std::endl
                << e.getMessag
        < " << std::endl;
    }
    catch (...w
    {
      std::cout << "Unknown Exception during Gurobi optimisation" << std::endlUnknown Exception during Gurobi optimisation " << std::endl;
    }
  }
}

// namespace dtwc // nwcace dtwcc
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

    catch (GRBException &e)
    {
      std::cout << "Error code = " << e.getErrorCode() << std::endl
                << e.getMessage() << std::endl;
    }
    catch (...)
    {
      std::cout << "Unknown Exception during Gurobi optimisation" << std::endl;
    }
  }

  // namespace dtwc // nwcace dtwc
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

} // namespace dtwc
