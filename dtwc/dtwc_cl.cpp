/*
 * dtwc_cl.cpp
 *
 * Command line interface for DTWC++
 *
 * Created on: 11 Dec 2023
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "dtwc.hpp"

#include <iostream>
#include <string>
#include <CLI/CLI.hpp>

// Declarations of the auxillary functions:
dtwc::Range str_to_range(std::string str);

int main(int argc, char **argv)
{
  auto app_description = "A C++ library for fast Dynamic Time Wrapping Clustering";
  auto default_name = "DTWC++_results";

  // Input parameters:
  std::string Nc_str;
  std::string probName{ "dtwc" };
  std::string inputPath{ "../data/dummy" };
  std::string outPath{ "." };
  std::string method{ "kMedoids" };
  std::string solver{ "HiGHS" };
  std::string distMatPath{ "" };

  int maxIter{ dtwc::settings::DEFAULT_MAX_ITER };
  int skipRows{0}, skipCols{0};
  int N_repetition{ 1 };
  int bandWidth{-1};

  CLI::App app{ app_description };

  app.add_option("--Nc,--clusters,--number_of_clusters", Nc_str, "Number range in the format i..j or single number i");
  app.add_option("--name,--probName", probName, "Name of the clustering problem");
  app.add_option("-i,--in,--input", inputPath, "Input file or folder");
  app.add_option("-o,--out,--output", outPath, "Output folder");
  app.add_option("--skipRows", skipRows, "First N rows to skip (default = 0)");
  app.add_option("--skipCols,--skipColumns", skipCols, "First N columns to skip (default = 0)");
  app.add_option("--maxIter,--iter", maxIter, "Maximum iteration for iterative algorithms");
  app.add_option("--method", method, "Method (kMedoids or MIP)");
  app.add_option("--repeat,--Nrepeat,--Nrepetition,--Nrep", N_repetition, "Number of repetitions for Kmedoids.");
  app.add_option("--solver,--mip_solver,--mipSolver", solver, "Number of repetitions for Kmedoids.");
  app.add_option("--bandwidth,--bandw,--bandlength", bandWidth, "Width of the band used.");
  app.add_option("--distMat,--distance_matrix,--distances", distMatPath, "Path for distance matrix.");

  std::cout << "Arguments are being parsed." << std::endl;

  CLI11_PARSE(app, argc, argv);

  // Check if no arguments were provided
  if (argc == 1) {
    std::cout << app.help() << std::endl;
    return EXIT_SUCCESS;
  }

  std::cout << "Arguments are parsed." << std::endl;

  auto Nc = str_to_range(Nc_str); // dtwc::Range(3,5); 
  dtwc::Clock clk; // Create a clock object

  std::cout << "Nc_str : " << Nc_str << '\n';
  std::cout << "name : " << probName << '\n';
  std::cout << "input : " << inputPath << '\n';
  std::cout << "output path : " << outPath << '\n';
  std::cout << "Skipped rows : " << skipRows << '\n';
  std::cout << "Skipped cols : " << skipCols << '\n';
  std::cout << "Max iteration : " << maxIter << std::endl;

  dtwc::DataLoader dl{ inputPath };
  dl.startColumn(skipCols).startRow(skipRows); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  std::cout << "Data loading finished at " << clk << "\n";

  prob.maxIter = maxIter;
  prob.N_repetition = N_repetition;
  prob.output_folder = outPath;
  prob.band = bandWidth;
  try {
    if(distMatPath != "")
      prob.readDistanceMatrix(distMatPath);
  } catch (const std::exception &e) {
    std::cout << "Distance matrix could not be read! Continuing without matrix!" << std::endl;
  }


  if (solver == "HiGHS" || solver == "highs")
    prob.set_solver(dtwc::Solver::HiGHS);
  else if (solver == "Gurobi" || solver == "gurobi")
    prob.set_solver(dtwc::Solver::Gurobi);
  else
    std::cout << "MIP solver is not recognized! Continuing with the default solver.";


  if (method == "kMedoids")
    prob.method = dtwc::Method::Kmedoids;
  else if (method == "mip" || method == "MIP")
    prob.method = dtwc::Method::MIP;
  else
    std::cout << "Clustering method is not recognised! Using default clustering method: kMedoids.\n";

  for (auto nc : Nc) {
    std::cout << "\n\nClustering by " << method << " for Number of clusters : " << nc << std::endl;
    prob.set_numberOfClusters(nc); // Nc = number of clusters.
    prob.cluster_and_process();
  }

  std::cout << "Finished all tasks " << clk << std::endl;
}

// Definitions of the auxillary functions:
dtwc::Range str_to_range(std::string str)
{
  dtwc::Range range{};
  try {
    size_t pos = str.find("..");
    if (pos != std::string::npos) {
      const int start = std::stoi(str.substr(0, pos));
      const int end = std::stoi(str.substr(pos + 2)) + 1;
      range = dtwc::Range(start, end);
    } else {
      const int number = std::stoi(str);
      range = dtwc::Range(number, number + 1);
    }

  } catch (const std::exception &e) {
    std::cerr << "Error processing input: " << e.what() << std::endl;
  }

  return range;
}
