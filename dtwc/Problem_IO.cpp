/**
 * @file Problem_IO.cpp
 * @brief Implementation of input/output functions for the Problem class.
 *
 * @details These functions handle the writing of medoids, clusters, silhouettes, and
 * distance matrices to files, as well as reading distance matrices from files.
 *
 * @date 25 Dec 2023
 * @author Volkan Kumtepeli, Becky Perriment
 */

#include "Problem.hpp"
#include "fileOperations.hpp"
#include "scores.hpp"   // for silhouette
#include "settings.hpp" // for data_t, randGenerator, band

#include <iomanip>  // for operator<<, setprecision
#include <iostream> // for cout^
#include <fstream>
#include <string> // for allocator, char_traits, operator+
#include <vector> // for vector, operator==

namespace dtwc {

/**
 *  @brief Writes the medoids and their corresponding total cost to a CSV file.
 *  @param centroids_all A vector of vectors containing all centroid indices.
 *  @param rep The current repetition number.
 *  @param total_cost The total cost associated with the medoids.
 */
void Problem::writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost)
{
  const auto outPath = output_folder / (this->name + "medoids_rep_" + std::to_string(rep) + ".csv");
  std::ofstream medoidsFile(outPath, std::ios_base::out);

  if (!medoidsFile.good()) {
    std::cout << "Failed to open file in path: " << outPath << '\n'
              << "Program is exiting." << std::endl;

    throw 1;
  }

  for (auto &c_ind : centroids_all) {
    for (auto medoid : c_ind)
      medoidsFile << get_name(medoid) << ',';

    medoidsFile << '\n';
  }

  medoidsFile << "Procedure is completed with cost: " << total_cost << '\n';
  medoidsFile.close();
}

/**
 *  @brief Prints cluster information to the standard output.
 *  @details Displays each centroid and its members.
 */
void Problem::printClusters() const
{
  std::cout << "Clusters: ";
  for (auto ind : centroids_ind)
    std::cout << get_name(ind) << ' ';

  std::cout << '\n';

  for (int i{ 0 }; i < Nc; i++) {
    std::cout << get_name(centroids_ind[i]) << " has: ";

    for (auto member : cluster_members[i])
      std::cout << get_name(member) << " ";

    std::cout << '\n';
  }
}

/**
 *  @brief Writes cluster information to a CSV file.
 *  @details The file includes cluster centroids and members, and the total cost.
 */
void Problem::writeClusters()
{
  const auto file_name = name + "_Nc_" + std::to_string(Nc) + ".csv";

  std::ofstream myFile(output_folder / file_name, std::ios_base::out);

  myFile << "Clusters:\n";

  for (int i{ 0 }; i < Nc; i++) {
    if (i != 0) myFile << ',';

    myFile << get_name(centroids_ind[i]);
  }

  myFile << "\n\n"
         << "Data" << ',' << "its cluster\n";

  for (int i{ 0 }; i < data.size(); i++)
    myFile << get_name(i) << ',' << get_name(centroids_ind[clusters_ind[i]]) << '\n';

  myFile << "Procedure is completed with cost: " << findTotalCost() << '\n';

  myFile.close();
}

/**
 *  @brief Writes silhouette scores for each data point to a CSV file.
 *  @details Calculates silhouette scores using the 'scores::silhouette' function.
 */
void Problem::writeSilhouettes()
{
  const auto silhouettes = scores::silhouette(*this);

  std::string silhouette_name{ name + "_silhouettes_Nc_" };

  silhouette_name += std::to_string(Nc) + ".csv";

  std::ofstream myFile(output_folder / silhouette_name, std::ios_base::out);

  myFile << "Silhouettes:\n";
  for (int i{ 0 }; i < data.size(); i++)
    myFile << get_name(i) << ',' << silhouettes[i] << '\n';

  myFile.close();
}

/**
 *  @brief Writes the members of each medoid to a CSV file.
 *  @param iter The current iteration number.
 *  @param rep The current repetition number.
 */
void Problem::writeMedoidMembers(int iter, int rep) const
{
  const std::string medoid_name = "medoidMembers_Nc_" + std::to_string(Nc) + "_rep_"
                                  + std::to_string(rep) + "_iter_" + std::to_string(iter) + ".csv";

  std::ofstream medoidMembers(output_folder / medoid_name, std::ios_base::out);
  for (auto &members : cluster_members) {
    for (auto member : members)
      medoidMembers << get_name(member) << ',';
    medoidMembers << '\n';
  }
  medoidMembers.close();
}

/**
 *  @brief Writes the distance matrix to a file.
 *  @param name_ The name of the output file.
 */
void Problem::writeDistanceMatrix(const std::string &name_) const
{
  writeMatrix(distMat, output_folder / name_);
}

/**
 *  @brief Writes the number of the best repetition to a file and prints it to the console.
 *  @param best_rep The best repetition number.
 */
void Problem::writeBestRep(int best_rep)
{
  std::ofstream bestRepFile(output_folder / (name + "_bestRepetition_Nc_" + std::to_string(Nc) + ".csv"), std::ios_base::out);
  bestRepFile << best_rep << '\n';
  bestRepFile.close();

  std::cout << "Best repetition: " << best_rep << '\n';
}

/**
 *  @brief Reads the distance matrix from a file.
 *  @details If the matrix cannot be read, continues without it.
 *  @param distMat_path The file path of the distance matrix.
 */
void Problem::readDistanceMatrix(const fs::path &distMat_path)
{
  try {
    readMatrix(distMat, distMat_path);
  } catch (...) {
    std::cout << "Distance matrix could not be read! Continuing without matrix!" << std::endl;
  }
}

} // namespace dtwc