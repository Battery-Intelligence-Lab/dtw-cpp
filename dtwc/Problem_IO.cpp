/**
 * @file Problem_IO.cpp
 * @brief Implementation of input/output functions for the Problem class.
 *
 * @details These functions handle the writing of medoids, clusters, silhouettes, and
 * distance matrices to files, as well as reading distance matrices from files.
 *
 * @date 25 Dec 2023
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "Problem.hpp"
#include "core/matrix_io.hpp"
#include "scores.hpp"      // for silhouette
#include "types/Range.hpp" // for Range

#include <fstream>
#include <iostream> // for cout
#include <type_traits> // for std::is_same_v, std::decay_t (visit_distmat)
#include <string>  // for allocator, char_traits, operator+
#include <vector>  // for vector, operator==

namespace dtwc {

namespace {

/// Open an output file, failing loudly: an unchecked ofstream silently produces
/// no file at all when the output folder is unwritable.
std::ofstream open_output(const std::filesystem::path &path)
{
  std::ofstream file(path, std::ios_base::out);
  if (!file.good())
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  return file;
}

/// Close an output file and report a write error instead of losing it.
void close_output(std::ofstream &file, const std::filesystem::path &path)
{
  file.close();
  if (!file.good())
    throw std::runtime_error("Write error on file: " + path.string());
}

} // namespace

/**
 *  @brief Writes the medoids and their corresponding total cost to a CSV file.
 *  @param centroids_all A vector of vectors containing all centroid indices.
 *  @param rep The current repetition number.
 *  @param total_cost The total cost associated with the medoids.
 */
void Problem::writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost)
{
  const auto outPath = output_folder_
                       / (name_ + "medoids_rep_" + std::to_string(rep) + ".csv");
  std::ofstream medoidsFile(outPath, std::ios_base::out);

  if (!medoidsFile.good()) {
    std::cout << "Failed to open file in path: " << outPath << '\n'
              << "Program is exiting." << '\n';

    throw std::runtime_error("Failed to open medoids output file: " + outPath.string());
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
void Problem::print_clusters() const
{
  std::cout << "Clusters centroids: ";
  for (auto ind : centroids_ind)
    std::cout << get_name(ind) << ' ';

  std::cout << '\n';

  for (const auto i_c : Range(Nc)) {
    std::cout << "The cluster with centroid " << get_name(centroids_ind[i_c]) << " has following members: ";

    for (const auto i_p : Range(size()))
      if (clusters_ind[i_p] == i_c)
        std::cout << get_name(i_p) << " ";

    std::cout << '\n';
  }
}

/**
 *  @brief Writes cluster information to a CSV file.
 *  @details The file includes cluster centroids and members, and the total cost.
 */
void Problem::write_clusters()
{
  const auto file_name = name_ + "_Nc_" + std::to_string(Nc) + ".csv";
  const auto path = output_folder_ / file_name;
  std::ofstream myFile = open_output(path);

  myFile << "Cluster centroids:\n";

  for (int i{ 0 }; i < Nc; i++) {
    if (i != 0) myFile << ',';

    myFile << get_name(centroids_ind[i]);
  }

  myFile << "\n\n"
         << "Data" << ',' << "its cluster\n";

  for (const auto i : Range(size()))
    myFile << get_name(i) << ',' << get_name(centroid_of(static_cast<int>(i))) << '\n';

  myFile << "Procedure is completed with cost: " << find_total_cost() << '\n';

  close_output(myFile, path);
}

/**
 *  @brief Writes silhouette scores for each data point to a CSV file.
 *  @details Calculates silhouette scores using the 'scores::silhouette' function.
 *
 *  s(i) is undefined with fewer than two realised clusters, where
 *  scores::silhouette() throws UndefinedScore. Writing output is not the place
 *  to abort a completed clustering: warn and skip the file, as the CLI does.
 *  Only that case is caught; a corrupt labelling still propagates.
 */
void Problem::write_silhouettes()
{
  std::vector<double> silhouettes;
  try {
    silhouettes = scores::silhouette(*this);
  } catch (const UndefinedScore &e) {
    std::cerr << "Warning: silhouettes skipped: " << e.what() << '\n';
    return;
  }

  std::string silhouette_name{ name_ + "_silhouettes_Nc_" };

  silhouette_name += std::to_string(n_clusters()) + ".csv";

  const auto path = output_folder_ / silhouette_name;
  std::ofstream myFile = open_output(path);

  myFile << "Silhouettes:\n";
  for (auto i : Range(size()))
    myFile << get_name(i) << ',' << silhouettes[i] << '\n';

  close_output(myFile, path);
}

/**
 *  @brief Writes the members of each medoid to a CSV file.
 *  @param iter The current iteration number.
 *  @param rep The current repetition number.
 */
void Problem::write_medoid_members(int iter, int rep) const
{
  const std::string medoid_name = "medoidMembers_Nc_" + std::to_string(Nc) + "_rep_"
                                  + std::to_string(rep) + "_iter_" + std::to_string(iter) + ".csv";

  const auto path = output_folder_ / medoid_name;
  std::ofstream medoidMembers = open_output(path);
  for (const auto i_c : Range(n_clusters())) {
    for (const auto i_p : Range(size()))
      if (clusters_ind[i_p] == i_c)
        medoidMembers << get_name(i_p) << ',';

    medoidMembers << '\n';
  }

  close_output(medoidMembers, path);
}

/**
 *  @brief Writes the distance matrix to a file.
 *  @param name_ The name of the output file. Matches the declaration in
 *         Problem.hpp; the body uses only the parameter, never the member it hides.
 */
void Problem::write_distance_matrix(const std::string &name_) const
{
  validate_mmap_cache_identity();
  validate_dense_cache_configuration();
  const auto path = output_folder_ / name_;
  visit_distmat([&](const auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      io::write_csv(m, path);
    } else {
      // MmapDistanceMatrix: data already on disk; write a CSV copy through the
      // shared formatter. F14 requires rejecting a non-finite value BEFORE the
      // destination is truncated, so the preflight is hoisted above the open
      // and the already-preflighted emitter is called directly -- operator<<
      // would repeat that O(N^2) scan. Bytes are identical either way.
      core::detail::preflight_distance_matrix_csv(m);
      std::ofstream file(
        path, std::ios::out | std::ios::binary | std::ios::trunc);
      if (!file.good())
        throw std::runtime_error("Cannot open file for writing: " + path.string());
      core::detail::write_distance_matrix_csv_preflighted(file, m);
      file.close();
      if (!file.good())
        throw std::runtime_error("Write error on file: " + path.string());
    }
  });
}

/**
 *  @brief Writes the number of the best repetition to a file and prints it to the console.
 *  @param best_rep The best repetition number.
 */
void Problem::writeBestRep(int best_rep)
{
  const auto path = output_folder_
    / (name_ + "_bestRepetition_Nc_" + std::to_string(Nc) + ".csv");
  std::ofstream bestRepFile = open_output(path);
  bestRepFile << best_rep << '\n';
  close_output(bestRepFile, path);

  std::cout << "Best repetition: " << best_rep << '\n';
}

/**
 *  @brief Reads the distance matrix from a file.
 *  @details A read failure is reported to the caller: deciding whether to
 *  continue without a precomputed matrix belongs to the caller, not the reader.
 *  Swallowing it made a failed load indistinguishable from a successful one.
 *  @param distMat_path The file path of the distance matrix.
 *  @throws std::exception if the file cannot be opened or parsed.
 */
void Problem::read_distance_matrix(const fs::path &distMat_path)
{
  ensure_dense_cache_configuration_current();
  visit_distmat([&](auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      io::read_csv(m, distMat_path);
    } else {
      throw std::runtime_error("read_distance_matrix: CSV read not supported for MmapDistanceMatrix "
                               "(use warm-start via use_mmap_distance_matrix instead).");
    }
  });
}

} // namespace dtwc
