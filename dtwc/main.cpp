/**
 * @file main.cpp
 * @brief Example driver for DTWC++ demonstrating basic clustering usage.
 * @author Volkan Kumtepeli
 */

#include "dtwc.hpp"
#include <iostream>
#include <fstream>
#include <string_view>

int main()
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3;        // Number of clusters
  int Ndata_max = 20; // Load maximum 20 of data.
  // Note: Run this from the project root directory, or specify an absolute path
  dtwc::DataLoader dl{ std::filesystem::path("data") / "dummy", Ndata_max };
  dl.start_column(1).start_row(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.set_max_iter(100);

  prob.set_n_clusters(Nc); // Nc = number of clusters.
  prob.set_n_repetitions(5);         // Repeat the iterative algorithm

  prob.set_solver(dtwc::Solver::Gurobi); // MIP solver type.
  prob.band = -1;                        // Sakoe chiba band length.

  prob.cluster_by_mip();

  prob.write_distance_matrix();

  prob.print_clusters(); // Prints to screen.
  prob.write_clusters(); // Prints to file.
  prob.write_silhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}
