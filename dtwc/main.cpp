#include "dtwc.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>


int main()
{
  dtwc::Clock clk;
  int Ndata_max = 100; // Load 10 data maximum.

  dtwc::Problem<Tdata> prob; // Create a problem.

  prob.load_data_fromFolder("../../data/dummy", Ndata_max);
  std::cout << "Data loading finished at " << clk << "\n";

  // // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating
  prob.fillDistanceMatrix();

  std::string DistMatrixName = "DTW_matrix.csv";

  prob.writeAllDistances(DistMatrixName);
  prob.getDistanceMatrix().print();
  std::cout << "Finished calculating distances " << clk << "\n";
  std::cout << "Band used " << settings::band << "\n\n\n";

  prob.set_numberOfClusters(4); // 4 clusters.

  prob.clusterMIP();

  std::cout << "Finished all tasks " << clk << "\n";
} //
