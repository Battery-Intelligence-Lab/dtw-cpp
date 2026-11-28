/*!
 * @file MIP_multiple.cpp
 * @brief Demonstration of DTWC++ library usage for clustering problems.
 *
 * This program demonstrates the use of the DTWC++ library to solve clustering problems.
 * It includes creating a clock object, loading data, setting up the problem parameters,
 * and executing clustering algorithms.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 04 Nov 2022
 */

#include <dtwc.hpp>

#include <filesystem> // for operator/, path
#include <iostream>   // for operator<<, ostream, basic_ostream, cout
#include <string>     // for allocator, string, char_traits

int main(int argc, char **argv)
{
  dtwc::Clock clk; // Create a clock object

  int Ndata_max = 100; // Load 100 data maximum.

  // The data folder: the first argument, else ./data (run from the project root).
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "./data";

  dtwc::DataLoader dl{ data_dir / "dummy", Ndata_max };
  dl.start_column(1).start_row(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob("DTW_MILP_results", dl); // Create a problem.

  std::cout << "Data loading finished at " << clk << "\n";

  // prob.read_distance_matrix("../matlab/DTWdist_band_all.csv"); // Comment out if recalculating the matrix.
  prob.fill_distance_matrix();
  // prob.print_distance_matrix();
  std::cout << "Finished calculating distances " << clk << std::endl;
  std::cout << "Band used " << prob.distance().band << "\n\n\n";


  std::string reportName = "DTW_MILP_results";

  // Calculate for number of clusters Nc = 3,4,5;
  for (dtwc::index_t nc = 3; nc <= 5; ++nc) {
    std::cout << "\n\nClustering by MIP for Number of clusters : " << nc << '\n';
    prob.set_n_clusters(nc); // Nc = number of clusters.
    prob.cluster_by_mip();         // Uses MILP to do clustering.
    // The files dtwc_cl writes, one folder under ./results for each number of clusters.
    dtwc::detail::write_result_files(prob, std::filesystem::path{ "./results" } / ("Nc_" + std::to_string(nc)), true,
                                     &std::cout);
  }

  std::cout << "Finished all tasks " << clk << "\n";

  return EXIT_SUCCESS;
}
