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

    std::vector<GRBVar> w_vec;
    w_vec.reserve(Nb * Nb);

    for (size_t i{ 0 }; i < (Nb * Nb); i++)
      w_vec.push_back(model.addVar(0.0, 1.0, 0.0, GRB_BINARY, ""));

    dtwc::VecMatrix<GRBVar> w(Nb, Nb, std::move(w_vec));


    for (size_t i{ 0 }; i < Nb; i++) {
      GRBLinExpr lhs = 0;
      for (size_t j{ 0 }; j < Nb; j++) {
        lhs += w(j, i);
      }
      model.addConstr(lhs, '=', 1.0);
    }

    for (size_t i{ 0 }; i < Nb; i++)
      for (size_t j{ 0 }; j < Nb; j++)
        model.addConstr(w(i, j) <= isCluster[i]);


    {
      GRBLinExpr lhs = 0;
      for (size_t i{ 0 }; i < Nb; i++)
        lhs += isCluster[i];

      model.addConstr(lhs == Nc); // There should be Nc clusters.
    }


    // Set objective

    GRBLinExpr obj = 0;
    for (size_t i{ 0 }; i < Nb; i++)
      for (size_t j{ 0 }; j < Nb; j++)
        obj += w(i, j) * DTWdistByInd(i, j);

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
