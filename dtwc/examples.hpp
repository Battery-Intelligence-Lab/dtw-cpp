/*
 * examples.hpp
 *
 * Example uses

 *  Created on: 04 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "dtwc.hpp"

#include <filesystem>  // for operator/, path
#include <iostream>    // for operator<<, ostream, basic_ostream, cout
#include <string>      // for allocator, string, char_traits
#include <string_view> // for string_view

namespace dtwc::examples {

inline void cluster_byKmeans_single()
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3; // Number of clusters

  dtwc::DataLoader dl{ settings::dataPath / "dummy" };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.maxIter = 100;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.N_repetition = 5;

  prob.cluster_by_kMedoidsPAM();

  prob.printClusters(); // Prints to screen.
  prob.writeClusters(); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}


inline void cluster_byMIP_single()
{
  dtwc::Clock clk; // Create a clock object

  int Ndata_max = 300; // Load 300 data maximum.
  auto Nc = 6;         // Number of clusters

  dtwc::DataLoader dl{ settings::dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  std::cout << "Data loading finished at " << clk << "\n";

  prob.fillDistanceMatrix();

  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << settings::band << "\n\n\n";

  prob.method = Method::MIP;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.cluster_and_process();

  std::cout << "Finished all tasks " << clk << "\n";
}

inline void cluster_byMIP_multiple()
{
  dtwc::Clock clk; // Create a clock object

  int Ndata_max = 100;         // Load 100 data maximum.
  auto Nc = dtwc::Range(3, 6); // Clustering for Nc = 3,4,5. Range function like Python so 6 is not included.

  dtwc::DataLoader dl{ settings::dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  std::cout << "Data loading finished at " << clk << "\n";

  // prob.readDistanceMatrix("../matlab/DTWdist_band_all.csv"); // Comment out if recalculating the matrix.
  prob.fillDistanceMatrix();
  prob.writeDistanceMatrix();
  // prob.printDistanceMatrix();
  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << settings::band << "\n\n\n";


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
}

} // namespace dtwc::examples