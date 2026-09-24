/*!
 * @file Kmedoids_single.cpp
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

int main(int argc, char **argv)
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3; // Number of clusters

  // The data folder: the first argument, else ./data (run from the project root).
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "./data";

  dtwc::DataLoader dl{ data_dir / "dummy" };
  dl.start_column(1).start_row(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.set_max_iter(100);

  prob.set_n_clusters(Nc); // Nc = number of clusters.
  prob.set_n_repetitions(5);

  prob.cluster_by_kmedoids_lloyd();

  prob.print_clusters(); // Prints to screen.
  prob.write_clusters(); // Prints to file.
  prob.write_silhouettes();

  std::cout << "Finished all tasks " << clk << "\n";

  return EXIT_SUCCESS;
}
