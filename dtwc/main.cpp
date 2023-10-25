#include "dtwc.hpp"
#include "examples.hpp"
#include "solver/Simplex.hpp"
#include "../benchmark/benchmark_main.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

int main()
{

    // Define A matrix using triplet format for setting non-zero entries
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(4); // number of non-zero elements in A

    tripletList.push_back(T(0,0,-4));  // row 0, col 0, value -4
    tripletList.push_back(T(0,1,6));   // row 0, col 1, value 6
    tripletList.push_back(T(0,2,1));   // row 0, col 2, value 1
    tripletList.push_back(T(1,0,1));   // row 1, col 0, value 1
    tripletList.push_back(T(1,1,1));   // row 1, col 1, value 1
    tripletList.push_back(T(1,3,1));   // row 1, col 3, value 1

    Eigen::SparseMatrix<double> A(2, 4);
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    std::cout << A << '\n';

    A.row(1)  += A.row(2);


  // Here are some examples. You can either take the contents of example functions into main or modify and run them.
  dtwc::Clock clk; // Create a clock object

  // // Define A matrix
  // Eigen::MatrixXd A(2, 4);
  // A << -4, 6, 1, 0,
  //   1, 1, 0, 1;

  // // Define b vector
  // Eigen::VectorXd b(2);
  // b << 5,
  //   5;

  // // Define c vector
  // Eigen::VectorXd c(4);
  // c << 1, -2, 0, 0;

  // // Output the values to verify
  // std::cout << "Matrix A:\n"
  //           << A << std::endl;
  // std::cout << "Vector b:\n"
  //           << b << std::endl;
  // std::cout << "Vector c:\n"
  //           << c << std::endl;


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
