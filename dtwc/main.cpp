#include "dtwc.hpp"
#include <iostream>
#include <fstream>
#include <string_view>

#include <armadillo>

int main()
{
  std::cout << "Main has started!" << std::endl;
  {
    std::vector<double> x{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, y{ 1, 2, 3, 4, 5, 6 };
    dtwc::dtwBanded(x, y, 2);
  }
  std::cout << "dtwBanded has finished!" << std::endl;

  // using namespace dtwc;

  // dtwc::Clock clk; // Create a clock object
  // std::string probName = "DTW_kMeans_results";

  // auto Nc = 3; // Number of clusters

  // dtwc::DataLoader dl{ settings::dataPath / "dummy" };
  // dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  // dtwc::Problem prob{ probName, dl }; // Create a problem.
  // prob.maxIter = 100;

  // prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  // prob.N_repetition = 5;

  // prob.set_solver(dtwc::Solver::Gurobi);

  // prob.fillDistanceMatrix();
  // // prob.writeDistanceMatrix();

  // // prob.cluster_by_MIP();

  // // prob.printClusters(); // Prints to screen.
  // // prob.writeClusters(); // Prints to file.
  // // prob.writeSilhouettes();

  // std::cout << "Finished all tasks " << clk << "\n";
}
