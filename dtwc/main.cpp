#include "dtwc.hpp"
#include <iostream>
#include <fstream>
#include <string_view>

#include <armadillo>

int main()
{
  using namespace dtwc;

  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3; // Number of clusters

  dtwc::DataLoader dl{ settings::dataPath / "dummy" };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.maxIter = 100;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.N_repetition = 5;

  prob.set_solver(dtwc::Solver::Gurobi);
  prob.band = 5;

  prob.fillDistanceMatrix();
  prob.writeDistanceMatrix();

  prob.cluster_by_MIP();

  prob.printClusters(); // Prints to screen.
  prob.writeClusters(); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}
