#include "dtwc.hpp"
#include "examples.hpp"
#include "solver/Simplex.hpp"
#include "solver/SparseSimplex.hpp"
#include "../benchmark/benchmark_main.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "solver/EqualityConstraints.hpp"
#include "solver/test.hpp"


int main()
{
  auto [eq, c] = dtwc::solver::get_prob_small();

  dtwc::solver::SparseSimplex prob_small(eq, c);

  prob_small.gomoryAlgorithm();
  auto [solution_small, copt_small] = prob_small.getResults();

  fmt::println("Solution: {} and Copt = [{}]\n", solution_small, copt_small);


  // Define A matrix using triplet format for setting non-zero entries


  // Here are some examples. You can either take the contents of example functions into main or modify and run them.
  dtwc::Clock clk; // Create a clock object


  // dtwc::solver::Simplex mySimplexProblem(A, b, c);
  // mySimplexProblem.gomoryAlgorithm();

  // auto [solution, copt] = mySimplexProblem.getResults();

  // fmt::println("Solution: {} and Copt = [{}]\n", solution, copt);


  // int Ndata_max = 100; // Load 300 data maximum.
  // auto Nc = 4;         // Number of clusters

  // dtwc::DataLoader dl{ dtwc::settings::dataPath / "dummy", Ndata_max };

  // dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  // std::cout << "Data loading finished at " << clk << "\n";


  // // prob.fillDistanceMatrix();
  // // prob.writeDistanceMatrix();
  // prob.readDistanceMatrix(dtwc::settings::resultsPath / "DTW_MILP_results_distanceMatrix.csv");

  // std::cout << "Finished calculating distances " << clk << std::endl;
  // std::cout << "Band used " << dtwc::settings::band << "\n\n\n";

  // prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  // prob.cluster_by_MIP();         // Uses MILP to do clustering.

  // prob.printClusters(); // Prints to screen.
  // // prob.writeClusters(); // Prints to file.
  // // prob.writeSilhouettes();


  // dtwc::examples::cluster_byKmeans_single();
  //  dtwc::examples::cluster_byMIP_single();
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
  // dtwc::benchmarks::run_all();

  std::cout << "Finished all tasks " << clk << "\n";

  // std::ofstream w_sol_out(dtwc::settings::resultsPath / "test" / "AllGestureWiimoteX_sol_250.csv");

  // for (size_t j{ 0 }; j < Nb; j++) {
  //   for (size_t i{ 0 }; i < Nb; i++)
  //     w_sol_out << w_sol[i + j * Nb] << ',';

  //   w_sol_out << '\n';
  // }


  //  dtwc::examples::cluster_byKmeans_single(); // -> Not properly working
}
