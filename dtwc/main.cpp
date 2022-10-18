#include "dtwc.hpp"
#include "gurobi_c++.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>


int main()
{

  using namespace dtwc;
  dtwc::Clock clk;
  int Ndata_max = 100; // Load 10 data maximum.

  auto [p_vec, p_names] = load_data<Tdata, true>(settings::path, Ndata_max);

  std::cout << "Data loading finished at " << clk << "\n";

  dtwc::VecMatrix<Tdata> DTWdist(p_vec.size(), p_vec.size(), -1); // For distance memoization.

  // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating

  auto DTWdistByInd = [&DTWdist, p_vec = p_vec](int i, int j) {
    if (DTWdist(i, j) < 0) {
      if constexpr (settings::band == 0) {
        DTWdist(j, i) = DTWdist(i, j) = dtwFun_L<Tdata>(p_vec[i], p_vec[j]);
      } else {
        DTWdist(j, i) = DTWdist(i, j) = dtwFunBanded_Act<Tdata>(p_vec[i], p_vec[j], settings::band); // dtwFunBanded_Act_L faster and more accurate.
      }
    }
    return DTWdist(i, j);
  };

  fillDistanceMatrix(DTWdistByInd, p_vec.size()); // Otherwise takes time.

  std::string DistMatrixName = "DTW_matrix.csv";
  writeMatrix(DTWdist, DistMatrixName);
  // DTWdist.print();
  std::cout << "Finished calculating distances " << clk << "\n";
  std::cout << "Band used " << settings::band << "\n\n\n";


  auto Nb = p_vec.size();
  int Nc = 4;

  try {
    GRBEnv env = GRBEnv();

    GRBModel model = GRBModel(env);

    // Create variables

    GRBVar *isCluster = model.addVars(Nb, GRB_BINARY);
    GRBVar *w = model.addVars(Nb * Nb, GRB_BINARY);
    std::cout << "Finished creating w and is Cluster " << clk << "\n";


    for (size_t i{ 0 }; i < Nb; i++) {
      GRBLinExpr lhs = 0;
      for (size_t j{ 0 }; j < Nb; j++) {
        lhs += w[j + i * Nb];
      }
      model.addConstr(lhs, '=', 1.0);
    }
    std::cout << "Finished Only one cluster can be assigned " << clk << "\n";


    for (size_t j{ 0 }; j < Nb; j++)
      for (size_t i{ 0 }; i < Nb; i++)
        model.addConstr(w[i + j * Nb] <= isCluster[i]);

    std::cout << "Finished if w of ith data is activated then it is a cluster.  " << clk << "\n";

    {
      GRBLinExpr lhs = 0;
      for (size_t i{ 0 }; i < Nb; i++)
        lhs += isCluster[i];

      model.addConstr(lhs == Nc); // There should be Nc clusters.
    }
    std::cout << "Finished There should be Nc clusters.   " << clk << "\n";


    // Set objective

    GRBLinExpr obj = 0;
    for (size_t j{ 0 }; j < Nb; j++)
      for (size_t i{ 0 }; i < Nb; i++)
        obj += w[i + j * Nb] * DTWdistByInd(i, j);

    std::cout << "Finished OBJ.   " << clk << "\n";

    model.setObjective(obj, GRB_MINIMIZE);

    // First optimize() call will fail - need to set NonConvex to 2
    std::cout << "Finished setting up the MILP problem " << clk << "\n";
    model.optimize();

    for (size_t i{ 0 }; i < Nb; i++)
      std::cout << isCluster[i].get(GRB_StringAttr_VarName) << " "
                << isCluster[i].get(GRB_DoubleAttr_X) << '\n';


    std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
  }

  std::cout << "Finished all tasks " << clk << "\n";


} //
