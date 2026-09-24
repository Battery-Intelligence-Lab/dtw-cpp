#include "dtwc.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream> // was reaching this TU through settings.hpp (C-21a)

int main(int argc, char **argv)
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3;        // Number of clusters
  int Ndata_max = 20; // Load maximum 20 of data.

  // The data folder: the first argument, else ./data (run from the project root).
  const std::filesystem::path data_dir = argc > 1 ? argv[1] : "./data";
  dtwc::DataLoader dl{ data_dir / "dummy", Ndata_max };
  dl.start_column(1).start_row(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.set_max_iter(100);

  prob.set_n_clusters(Nc); // Nc = number of clusters.
  prob.set_n_repetitions(5);         // Repeat the iterative algorithm

  // MIP solver type. false: the solver is not in this build (Gurobi without
  // DTWC_ENABLE_GUROBI), and the Problem would fall back to HiGHS.
  if (!prob.set_solver(dtwc::Solver::HiGHS)) {
    std::cerr << "The requested MIP solver is not in this build of DTWC++.\n";
    return EXIT_FAILURE;
  }
  prob.band = -1; // Sakoe chiba band length.

  prob.cluster_by_mip();

  prob.write_distance_matrix();

  prob.print_clusters(); // Prints to screen.
  prob.write_clusters(); // Prints to file.
  prob.write_silhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}
