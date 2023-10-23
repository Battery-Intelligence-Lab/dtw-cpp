#include "dtwc.hpp"
#include "examples.hpp"
#include "solver/LP.hpp"
#include "solver/Simplex.hpp"
#include "../benchmark/benchmark_main.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

int main()
{

  // Here are some examples. You can either take the contents of example functions into main or modify and run them.
  dtwc::Clock clk; // Create a clock object

  // Define A matrix
  Eigen::MatrixXd A(2, 4);
  A << -4, 6, 1, 0,
    1, 1, 0, 1;

  // Define b vector
  Eigen::VectorXd b(2);
  b << 5,
    5;

  // Define c vector
  Eigen::VectorXd c(4);
  c << 1, -2, 0, 0;

  // Output the values to verify
  std::cout << "Matrix A:\n"
            << A << std::endl;
  std::cout << "Vector b:\n"
            << b << std::endl;
  std::cout << "Vector c:\n"
            << c << std::endl;


  dtwc::solver::Simplex mySimplexProblem(A, b, c);
  mySimplexProblem.gomoryAlgorithm();

  auto [solution, copt] = mySimplexProblem.getResults();

  fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);

  // dtwc::examples::cluster_byKmeans_single();
  // dtwc::examples::cluster_byMIP_single();
  // dtwc::examples::cluster_byMIP_multiple();

  // int Ndata_max = 300; // Load 300 data maximum.
  // auto Nc = 2;         // Number of clusters

  // dtwc::DataLoader dl{ dtwc::settings::dataPath / "test" / "nonUnimodular_1_Nc_2.csv" };

  // dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  // std::cout << "Data loading finished at " << clk << "\n";

  // prob.fillDistanceMatrix();
  // prob.writeDistanceMatrix();

  // std::cout << "Finished calculating distances " << clk << std::endl;
  // std::cout << "Band used " << dtwc::settings::band << "\n\n\n";

  // prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  // prob.cluster_by_MIP();         // Uses MILP to do clustering.
  // prob.printClusters();          // Prints to screen.
  // prob.writeClusters();          // Prints to file.
  // prob.writeSilhouettes();

  // std::cout << "Finished all tasks " << clk << "\n";

  // prob.readDistanceMatrix(dtwc::settings::dataPath / "test" / "AllGestureWiimoteX_dist_50.csv");


  // prob.getDistanceMatrix().resize(50, 50);

  // const auto Nb = prob.getDistanceMatrix().rows();
  // prob.data.Nb = Nb;

  // size_t Nc = 5;

  // prob.set_numberOfClusters(Nc);

  // // dtwc::MIP_clustering_byGurobi_relaxed(prob);
  // // dtwc::MIP_clustering_byOSQP(prob);


  // dtwc::solver::LP lp;
  // lp.maxIterations = 15000;
  // lp.numItrConv = 10;
  // lp.epsAbs = 1e-4;
  // lp.epsRel = 1e-4;

  // lp.setSize(Nb, Nc);

  // auto &q = lp.getQvec();

  // for (size_t j{ 0 }; j < Nb; j++)
  //   for (size_t i{ 0 }; i < Nb; i++)
  //     q[i + j * Nb] = prob.distByInd_scaled(i, j);

  // auto &w_sol = lp.getSolution();
  // for (size_t j{ 0 }; j < Nb; j++)
  //   for (size_t i{ 0 }; i < Nb; i++)
  //     w_sol[i + j * Nb] = 1;

  // lp.int_solve();
  // std::cout << "cost: " << lp.cost() << '\n';

  std::cout << "Finished all tasks " << clk << "\n";

  // std::ofstream w_sol_out(dtwc::settings::resultsPath / "test" / "AllGestureWiimoteX_sol_250.csv");

  // for (size_t j{ 0 }; j < Nb; j++) {
  //   for (size_t i{ 0 }; i < Nb; i++)
  //     w_sol_out << w_sol[i + j * Nb] << ',';

  //   w_sol_out << '\n';
  // }


  // dtwc::benchmarks::run_all();


  //  dtwc::examples::cluster_byKmeans_single(); // -> Not properly working
}
