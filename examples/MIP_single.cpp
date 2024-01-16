/*!
 * @file MIP_single.cpp
 * @brief Demonstration of DTWC++ library usage for clustering problems.
 *
 * This program demonstrates the use of the DTWC++ library to solve clustering problems.
 * It includes creating a clock object, loading data, setting up the problem parameters,
 * and executing clustering algorithms.
 *
 * @date 04 Nov 2022
 * @author Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <filesystem> // for operator/, path
#include <iostream>   // for operator<<, ostream, basic_ostream, cout
#include <string>     // for allocator, string, char_traits

int main()
{
  dtwc::Clock clk; // Create a clock object

  int Ndata_max = 300; // Load 300 data maximum.
  auto Nc = 6;         // Number of clusters

  dtwc::DataLoader dl{ dtwc::settings::dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  std::cout << "Data loading finished at " << clk << "\n";

  prob.fillDistanceMatrix();

  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << dtwc::settings::band << "\n\n\n";

  prob.method = dtwc::Method::MIP;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.cluster_and_process();

  std::cout << "Finished all tasks " << clk << "\n";

  return EXIT_SUCCESS;
}