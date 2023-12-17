/*!
 * @file Kmeans_single.cpp
 * @brief Demonstration of DTWC++ library usage for clustering problems.
 *
 * This program demonstrates the use of the DTWC++ library to solve clustering problems.
 * It includes creating a clock object, loading data, setting up the problem parameters,
 * and executing clustering algorithms.
 *
 * @date 04 Nov 2022
 * @author Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include <dtwc.hpp>

#include <filesystem>  // for operator/, path
#include <iostream>    // for operator<<, ostream, basic_ostream, cout
#include <string>      // for allocator, string, char_traits
#include <string_view> // for string_view

int main()
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3; // Number of clusters

  dtwc::DataLoader dl{ dtwc::settings::dataPath / "dummy" };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.
  prob.maxIter = 100;

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.
  prob.N_repetition = 5;

  prob.cluster_by_kMedoidsPAM();

  prob.printClusters(); // Prints to screen.
  prob.writeClusters(); // Prints to file.
  prob.writeSilhouettes();

  std::cout << "Finished all tasks " << clk << "\n";

  return EXIT_SUCCESS;
}