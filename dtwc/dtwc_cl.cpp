/*
 * dtwc_cl.cpp
 *
 * Command line interface for DTWC++
 *
 * Created on: 11 Dec 2023
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include "dtwc_cl.hpp"
#include "dtwc.hpp"
#include <iostream>
#include <string>
#include <CLI/CLI.hpp>


int main(int argc, char **argv)
{
  auto app_description = "A C++ library for fast Dynamic Time Wrapping Clustering";
  auto default_name = "DTWC++_results";


  // Input parameters:
  std::string Nc_str;
  std::string probName{ "dtwc" };
  std::string inputPath{ "." };
  std::string outPath{ "." };
  int maxIter{ dtwc::settings::DEFAULT_MAX_ITER };

  CLI::App app{ app_description };

  app.add_option("--Nc,--clusters,--number_of_clusters", Nc_str, "Number range in the format i..j or single number i");
  // app.add_option("--name,--probName", probName, "Name of the clustering problem");
  // app.add_option("-i,--in,--input", inputPath, "Input file or folder");
  // app.add_option("-o,--out,--output", outPath, "Output folder");
  std::cout << "Arguments are being parsed." << std::endl;

  CLI11_PARSE(app, argc, argv);

  std::cout << "Arguments are parsed." << std::endl;

  auto Nc = dtwc::cli_util::str_to_range(Nc_str);
  dtwc::Clock clk; // Create a clock object

  dtwc::DataLoader dl{ inputPath };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  std::cout << "Data loading finished at " << clk << "\n";

  prob.maxIter = maxIter;

  for (auto nc : Nc) {
    std::cout << "\n\nClustering by MIP for Number of clusters : " << nc << std::endl;
    prob.set_numberOfClusters(nc); // Nc = number of clusters.
    prob.cluster_and_process();
  }

  std::cout << "Finished all tasks " << clk << std::endl;
}