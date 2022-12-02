#include "dtwc.hpp"
#include "examples.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>


int main()
{
  // Here are some examples. You can either take the contents of example functions into main or modify and run them.
  // dtwc::examples::cluster_byMIP_multiple();
  dtwc::Clock clk; // Create a clock object
  auto Nc = 3;     // Number of clusters

  dtwc::Problem prob; // Create a problem.

  auto [p_vec, p_names] = dtwc::load_tsv<data_t>("../../data/benchmark/UCRArchive_2018/UMD/UMD_TEST.tsv");

  prob.load_data_fromVec(std::move(p_vec), std::move(p_names));

  // // readMatrix(DTWdist, "../matlab/DTWdist_band_all.csv"); // Comment out if recalculating
  prob.fillDistanceMatrix();
  prob.writeDistanceMatrix("UMD_test_matrix.csv");

  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << settings::band << "\n\n\n";


  std::string reportName = "DTW_MILP_results";

  prob.set_numberOfClusters(Nc);  // Nc = number of clusters.
  prob.cluster_by_MIP();          // Uses MILP to do clustering.
  prob.printClusters();           // Prints to screen.
  prob.writeClusters(reportName); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
  //  dtwc::examples::cluster_byKmeans_single(); // -> Not properly working
}
