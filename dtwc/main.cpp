#include "dtwc.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>


int main()
{
  dtwc::Clock clk;
  int Ndata_max = 100; // Load 10 data maximum.
  int Nc = 4;          // Number of clusters

  dtwc::Problem<Tdata> prob; // Create a problem.

  prob.load_data_fromFolder("../../data/dummy", Ndata_max);
  std::cout << "Data loading finished at " << clk << "\n";

  // // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating
  prob.fillDistanceMatrix();

  std::string DistMatrixName = "DTW_matrix.csv";

  prob.writeDistanceMatrix(DistMatrixName);
  // prob.getDistanceMatrix().print();
  std::cout << "Finished calculating distances " << clk << "\n";
  std::cout << "Band used " << settings::band << "\n\n\n";

  prob.set_numberOfClusters(Nc); // 4 clusters.

  prob.cluster_byMIP(); // Uses MILP to do clustering.

  std::string reportName = "DTW_MILP_results.csv";
  prob.writeClusters(reportName);

  std::cout << "Finished all tasks " << clk << "\n";
} //
