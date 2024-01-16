/*!
 * @file MIP_multiple.cpp
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

  int Ndata_max = 100;         // Load 100 data maximum.
  auto Nc = dtwc::Range(3, 6); // Clustering for Nc = 3,4,5. Range function like Python so 6 is not included.

  dtwc::DataLoader dl{ dtwc::settings::dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  std::cout << "Data loading finished at " << clk << "\n";

  // prob.readDistanceMatrix("../matlab/DTWdist_band_all.csv"); // Comment out if recalculating the matrix.
  prob.fillDistanceMatrix();
  prob.writeDistanceMatrix();
  // prob.printDistanceMatrix();
  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << dtwc::settings::band << "\n\n\n";


  std::string reportName = "DTW_MILP_results";

  // Calculate for number of clusters Nc = 3,4,5;
  for (auto nc : Nc) {
    std::cout << "\n\nClustering by MIP for Number of clusters : " << nc << '\n';
    prob.set_numberOfClusters(nc); // Nc = number of clusters.
    prob.cluster_by_MIP();         // Uses MILP to do clustering.
    prob.writeClusters();
    prob.writeSilhouettes();
  }

  std::cout << "Finished all tasks " << clk << "\n";

  return EXIT_SUCCESS;
}