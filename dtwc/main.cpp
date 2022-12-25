#include "dtwc.hpp"
#include "examples.hpp"
#include "solver/LP.hpp"
#include "../benchmark/benchmark_main.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>


int main()
{
  // Here are some examples. You can either take the contents of example functions into main or modify and run them.
  dtwc::Clock clk; // Create a clock object

  // dtwc::examples::cluster_byKmeans_single();
  // dtwc::examples::cluster_byMIP_single();
  // dtwc::examples::cluster_byMIP_multiple();

  dtwc::solver::ConstraintOperator op(3);

  std::vector<double> x_in_Nx{ 0.4273, 0.9873, 0.9884, 0.6498, 0.8156, 0.1178, 0.1487, 0.0820, 0.9530 };

  std::vector<double> x_in_Nm{ 0.1952, 0.0419, 0.0609, 0.0216, 0.2926, 0.8275, 0.9503, 0.6279, 0.0015, 0.8404, 0.2998, 0.5949, 0.5313, 0.0954, 0.2934, 0.9077, 0.7570, 0.7960, 0.8599, 0.8053, 0.5468, 0.5003 };

  std::vector<double> x_out_Nm, x_out_Nx;


  // Result for operator At:

  //   2.9944
  // 0.6020
  // 0.3259
  // 0.2956
  // 2.6550
  // 1.3394
  // 0.5894
  // 0.4177
  // 1.9369

  // result for operator At
  //       0.4273
  //   0.9873
  //   0.9884
  //   0.6498
  //   0.8156
  //   0.1178
  //   0.1487
  //   0.0820
  //   0.9530
  //        0
  //  -0.1717
  //  -0.0354
  //  -0.2225
  //        0
  //   0.8352
  //   0.2786
  //   0.7336
  //        0
  //   2.4030
  //   1.5832
  //   1.1837
  //   2.1959


  // v for rho 1, sigma 1

  // 5.5096
  // 4.5493
  // 4.4152
  // 3.1053
  // 5.9722
  // 0.9836
  // 1.2025
  // 0.6141
  // 6.0854

  // op.A(x_out_Nm, x_in_Nx);
  // op.At(x_out_Nx, x_in_Nm);

  // std::cout << "x_out_Nm: ";
  // for (auto x : x_out_Nm)
  //   std::cout << x << ", ";

  // std::cout << '\n';

  // std::cout << "x_out_Nx: ";
  // for (auto x : x_out_Nx)
  //   std::cout << x << ", ";

  // std::cout << '\n';

  // op.V(x_out_Nx, x_in_Nx, 1, 1);
  // std::cout << "x_out_V: ";
  // for (auto x : x_out_Nx)
  //   std::cout << x << ", ";


  dtwc::Problem prob;

  // prob.readDistanceMatrix(dtwc::settings::dataPath / "test" / "nonUnimodular_1_Nc_2.csv");
  prob.readDistanceMatrix(dtwc::settings::dataPath / "test" / "AllGestureWiimoteX_dist_50.csv");


  prob.getDistanceMatrix().resize(50, 50);

  const auto Nb = prob.getDistanceMatrix().rows();
  prob.data.Nb = Nb;

  size_t Nc = 5;

  prob.set_numberOfClusters(Nc);

  // dtwc::MIP_clustering_byGurobi_relaxed(prob);
  // dtwc::MIP_clustering_byOSQP(prob);


  dtwc::solver::LP lp;
  lp.maxIterations = 15000;
  lp.numItrConv = 50;
  lp.epsAbs = 1e-3;
  lp.epsRel = 1e-3;

  lp.setSize(Nb, Nc);

  auto &q = lp.getQvec();

  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      q[i + j * Nb] = prob.distByInd_scaled(i, j);


  auto &w_sol = lp.getSolution();
  for (size_t j{ 0 }; j < Nb; j++)
    for (size_t i{ 0 }; i < Nb; i++)
      w_sol[i + j * Nb] = 1;

  lp.int_solve();
  std::cout << "cost: " << lp.cost() << '\n';

  std::cout << "Finished all tasks " << clk << "\n";

  std::ofstream w_sol_out(dtwc::settings::resultsPath / "test" / "AllGestureWiimoteX_sol_250.csv");

  for (size_t j{ 0 }; j < Nb; j++) {
    for (size_t i{ 0 }; i < Nb; i++)
      w_sol_out << w_sol[i + j * Nb] << ',';

    w_sol_out << '\n';
  }


  // dtwc::benchmarks::run_all();


  //  dtwc::examples::cluster_byKmeans_single(); // -> Not properly working
}
