#include "dtwc.hpp"

int main()
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3;        // Number of clusters
  int Ndata_max = 20; // Load maximum 20 of data.

  // For path, we have several predefined variables:
  // dtwc::settings::dtwc_folder   = source folder of the DTW-C++ library
  // dtwc::settings::dtwc_dataPath = data folder inside the source folder of DTW-C++ library.
  // dtwc::settings::root_folder   = folder with the main CMakeLists.txt, hence your main project folder.
  // dtwc::settings::dataPath      = data folder inside your main project folder, hence if you have `data` folder inside your main.

  // Here as we use the "dummy" data inside dtwc library, we use `dtwc_dataPath`
  dtwc::DataLoader dl{ dtwc::settings::dtwc_dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.maxIter = 100;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.N_repetition = 5;         // Repeat the iterative algorithm

  prob.set_solver(dtwc::Solver::HiGHS); // MIP solver type.
  prob.band = -1;                       // Sakoe chiba band length.

  prob.cluster_by_MIP();

  prob.writeDistanceMatrix();

  prob.printClusters(); // Prints to screen.
  prob.writeClusters(); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";
}
