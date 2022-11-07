/*
 * examples.hpp
 *
 * Example uses

 *  Created on: 04 Nov 2022
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "dtwc.hpp"

#include <iostream>
#include <string>

namespace dtwc::examples {


void cluster_byKmeans_single()
{
  dtwc::Clock clk; // Create a clock object
  std::string reportName = "DTW_kMeans_results";


  int Ndata_max = 7; // Load 100 data maximum.
  auto Nc = 2;       // Number of clusters

  int N_repetition = 3;
  int maxIter = 100;

  dtwc::Problem prob; // Create a problem.

  prob.load_data_fromFolder("../../data/dummy", Ndata_max);
  std::cout << "Data loading finished at " << clk << "\n";

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.cluster_by_kMedoidsPAM_repetetive(N_repetition, maxIter);

  prob.printClusters();           // Prints to screen.
  prob.writeClusters(reportName); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}


void cluster_byMIP_single()
{
  dtwc::Clock clk; // Create a clock object

  int Ndata_max = 100; // Load 100 data maximum.
  auto Nc = 4;         // Number of clusters

  dtwc::Problem prob; // Create a problem.

  prob.load_data_fromFolder("../../data/dummy", Ndata_max);
  std::cout << "Data loading finished at " << clk << "\n";

  // // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating
  prob.fillDistanceMatrix();

  std::string DistMatrixName = "DTW_matrix.csv";

  prob.writeDistanceMatrix(DistMatrixName);
  // prob.getDistanceMatrix().print();
  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << settings::band << "\n\n\n";


  std::string reportName = "DTW_MILP_results";

  prob.set_numberOfClusters(Nc);  // Nc = number of clusters.
  prob.cluster_by_MIP();          // Uses MILP to do clustering.
  prob.printClusters();           // Prints to screen.
  prob.writeClusters(reportName); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}

void cluster_byMIP_multiple()
{
  dtwc::Clock clk; // Create a clock object

  int Ndata_max = 100;         // Load 100 data maximum.
  auto Nc = dtwc::Range(3, 6); // Clustering for Nc = 3,4,5. Range function like Python so 6 is not included.

  dtwc::Problem prob; // Create a problem.

  prob.load_data_fromFolder("../../data/dummy", Ndata_max);
  std::cout << "Data loading finished at " << clk << "\n";

  // // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating
  prob.fillDistanceMatrix();

  std::string DistMatrixName = "DTW_matrix.csv";

  prob.writeDistanceMatrix(DistMatrixName);
  // prob.getDistanceMatrix().print();
  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << settings::band << "\n\n\n";


  std::string reportName = "DTW_MILP_results";

  // Calculate for number of clusters Nc = 3,4,5;
  for (auto nc : Nc) {
    std::cout << "\n\nClustering by MIP for Number of clusters : " << nc << '\n';
    prob.set_numberOfClusters(nc); // Nc = number of clusters.
    prob.cluster_by_MIP();         // Uses MILP to do clustering.
    prob.writeClusters(reportName);
    prob.writeSilhouettes();
  }

  std::cout << "Finished all tasks " << clk << "\n";
}

} // namespace dtwc::examples