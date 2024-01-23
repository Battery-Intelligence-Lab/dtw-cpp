#include "dtwc.hpp"
#include <iostream>
#include <fstream>
#include <string_view>

#include <armadillo>

int main()
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3;        // Number of clusters
  int Ndata_max = 20; // Load maximum 20 of data.
  dtwc::DataLoader dl{ dtwc::settings::dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.maxIter = 100;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.N_repetition = 5;         // Repeat the iterative algorithm

  prob.set_solver(dtwc::Solver::Gurobi); // MIP solver type.
  prob.band = -1;                        // Sakoe chiba band length.

  prob.cluster_by_MIP();

  prob.writeDistanceMatrix();

  prob.printClusters(); // Prints to screen.
  prob.writeClusters(); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}
