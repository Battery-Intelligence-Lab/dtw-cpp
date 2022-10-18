#include "dtwc.hpp"
#include <gurobi_c++.h>

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>


int main(){

  try {
    GRBEnv env = GRBEnv();

    GRBModel model = GRBModel(env);

    // Create variables

    GRBVar x = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x");
    GRBVar y = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "y");
    GRBVar z = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "z");

    // Set objective

    GRBLinExpr obj = x;
    model.setObjective(obj, GRB_MAXIMIZE);

    // Add linear constraint: x + y + z <= 10

    model.addConstr(x + y + z <= 10, "c0");

    // Add bilinear inequality constraint: x * y <= 2

    model.addQConstr(x * y <= 2, "bilinear0");

    // Add bilinear equality constraint: y * z == 1

    model.addQConstr(x * z + y * z == 1, "bilinear1");

    // First optimize() call will fail - need to set NonConvex to 2

    try {
      model.optimize();
      assert(0);
    } catch (GRBException e) {
      std::cout << "Failed (as expected)" << std::endl;
    }

    model.set(GRB_IntParam_NonConvex, 2);
    model.optimize();

    std::cout << x.get(GRB_StringAttr_VarName) << " "
              << x.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << y.get(GRB_StringAttr_VarName) << " "
              << y.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << z.get(GRB_StringAttr_VarName) << " "
              << z.get(GRB_DoubleAttr_X) << std::endl;

    // Constrain x to be integral and solve again
    x.set(GRB_CharAttr_VType, GRB_INTEGER);
    model.optimize();

    std::cout << x.get(GRB_StringAttr_VarName) << " "
              << x.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << y.get(GRB_StringAttr_VarName) << " "
              << y.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << z.get(GRB_StringAttr_VarName) << " "
              << z.get(GRB_DoubleAttr_X) << std::endl;

    std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
  }


 
  // using namespace dtwc;
  // dtwc::Clock clk;
  //   int Ndata_max = 10; // Load 10 data maximum.

  //   auto [p_vec, p_names] = load_data<Tdata, true>(settings::path, Ndata_max);

  //   std::cout << "Data loading finished at " << clk << "\n";

  //   dtwc::VecMatrix<Tdata> DTWdist(p_vec.size(), p_vec.size(), -1); // For distance memoization.

  //   // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating

  //   auto DTWdistByInd = [&DTWdist, p_vec = p_vec](int i, int j) {
  //   if (DTWdist(i, j) < 0) {
  //     if constexpr (settings::band == 0) {
  //       DTWdist(j, i) = DTWdist(i, j) = dtwFun_L<Tdata>(p_vec[i], p_vec[j]);
  //     } else {
  //       DTWdist(j, i) = DTWdist(i, j) = dtwFunBanded_Act<Tdata>(p_vec[i], p_vec[j], settings::band); // dtwFunBanded_Act_L faster and more accurate.
  //     }
  //   }
  //   return DTWdist(i, j);
  // };

  //   fillDistanceMatrix(DTWdistByInd, p_vec.size()); // Otherwise takes time.

  //   std::string DistMatrixName = "DTW_matrix.csv";
  // writeMatrix(DTWdist, DistMatrixName);
  // // DTWdist.print();
  // std::cout << "Finished all tasks in " << clk << "\n";
  // std::cout << "Band used " << settings::band << "\n\n\n";
} //
