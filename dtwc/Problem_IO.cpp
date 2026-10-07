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
#include "fileOperations.hpp" // for open_output, close_output
#include "scores.hpp"      // for silhouette
#include "types/Range.hpp" // for Range

#include <filesystem>
#include <fstream>
#include <iomanip>  // for setprecision
#include <iostream> // for cout
#include <numeric>  // for accumulate
#include <string>  // for allocator, char_traits, operator+
#include <vector>  // for vector, operator==

namespace dtwc {

/**
 *  @brief Writes the medoids and their corresponding total cost to a CSV file.
 *  @param centroids_all A vector of vectors containing all centroid indices.
 *  @param rep The current repetition number.
 *  @param total_cost The total cost associated with the medoids.
 */
void Problem::writeMedoids(std::vector<std::vector<index_t>> &centroids_all, int rep, double total_cost)
{
  const auto outPath = output_folder_
    / utf8_to_path(name_ + "medoids_rep_" + std::to_string(rep) + ".csv");
  std::ofstream medoidsFile = open_output(outPath);

  for (auto &c_ind : centroids_all) {
    for (auto medoid : c_ind)
      medoidsFile << series_name(medoid) << ',';

    medoidsFile << '\n';
  }

  medoidsFile << "Procedure is completed with cost: " << total_cost << '\n';
  close_output(medoidsFile, outPath);
}

/**
 *  @brief Prints cluster information to the standard output.
 *  @details Displays each centroid and its members.
 */
void Problem::print_clusters() const
{
  require_clustered("print_clusters");
  std::cout << "Clusters centroids: ";
  for (auto ind : centroids_ind)
    std::cout << series_name(ind) << ' ';

  std::cout << '\n';

  for (const auto i_c : Range(Nc)) {
    std::cout << "The cluster with centroid " << series_name(centroids_ind[i_c]) << " has following members: ";

    for (const auto i_p : Range(size()))
      if (clusters_ind[i_p] == i_c)
        std::cout << series_name(i_p) << " ";

    std::cout << '\n';
  }
}

/**
 *  @brief Writes cluster information to a CSV file.
 *  @details The file includes cluster centroids and members, and the total cost.
 */
void Problem::write_clusters()
{
  require_clustered("write_clusters"); // before the file is opened: a refusal leaves none behind
  const auto file_name = name_ + "_Nc_" + std::to_string(Nc) + ".csv";
  const auto path = output_folder_ / utf8_to_path(file_name);
  std::ofstream myFile = open_output(path);

  myFile << "Cluster centroids:\n";

  for (index_t i{ 0 }; i < Nc; i++) {
    if (i != 0) myFile << ',';

    myFile << series_name(centroids_ind[i]);
  }

  myFile << "\n\n"
         << "Data" << ',' << "its cluster\n";

  for (const auto i : Range(size()))
    myFile << series_name(i) << ',' << series_name(centroid_of(static_cast<index_t>(i))) << '\n';

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

  const auto path = output_folder_ / utf8_to_path(silhouette_name);
  std::ofstream myFile = open_output(path);

  myFile << "Silhouettes:\n";
  for (auto i : Range(size()))
    myFile << series_name(i) << ',' << silhouettes[i] << '\n';

  close_output(myFile, path);
}

/**
 *  @brief Writes the members of each medoid to a CSV file.
 *  @param iter The current iteration number.
 *  @param rep The current repetition number.
 */
void Problem::write_medoid_members(int iter, int rep) const
{
  require_clustered("write_medoid_members");
  const std::string medoid_name = "medoidMembers_Nc_" + std::to_string(Nc) + "_rep_"
                                  + std::to_string(rep) + "_iter_" + std::to_string(iter) + ".csv";

  const auto path = output_folder_ / utf8_to_path(medoid_name);
  std::ofstream medoidMembers = open_output(path);
  for (const auto i_c : Range(n_clusters())) {
    for (const auto i_p : Range(size()))
      if (clusters_ind[i_p] == i_c)
        medoidMembers << series_name(i_p) << ',';

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
  io::write_csv(distMat, output_folder_ / utf8_to_path(name_));
}

/**
 *  @brief Writes the number of the best repetition to a file and prints it to the console.
 *  @param best_rep The best repetition number.
 */
void Problem::writeBestRep(int best_rep)
{
  const auto path = output_folder_
    / utf8_to_path(name_ + "_bestRepetition_Nc_" + std::to_string(Nc) + ".csv");
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
 *  @throws IOError if the file cannot be opened or holds a non-numeric field;
 *          InvalidInput if it is not square and symmetric, if its size is not
 *          this Problem's series count, if a distance is ±inf, or if the matrix
 *          is memory-mapped.
 */
void Problem::read_distance_matrix(const fs::path &distMat_path)
{
  sync_band(); // the matrix is read for the band the Problem computes with
  // Values read into a mapped matrix would persist in its file under this
  // Problem's fingerprint, whatever they were computed from.
  if (distMat.is_mapped())
    throw InvalidInput("read_distance_matrix: this Problem's distance matrix is memory-mapped "
                       "(use_mmap_distance_matrix), and a CSV matrix is read into RAM only; call "
                       "refresh_distance_matrix() first, or reuse the mapped file instead.");
  // Parsed into a local so a rejected file leaves the matrix untouched. A
  // matrix of another size describes other series: it used to be kept, then
  // discarded silently at the first lookup and every distance recomputed.
  // An empty file loaded nothing, equally silently. A Problem without
  // series takes any matrix, as before.
  core::DistanceMatrix loaded;
  io::read_csv(loaded, distMat_path);
  if (size() != 0 && loaded.size() != size())
    throw InvalidInput(
      "Problem::read_distance_matrix: '" + path_to_utf8(distMat_path) + "' has "
      + std::to_string(loaded.size()) + " rows, but this Problem holds "
      + std::to_string(size()) + " series; a distance matrix has one row and "
        "one column per series, in input order. Load the matrix computed for "
        "these series, or omit it to compute the distances.");
  const bool complete = loaded.all_computed("Problem::read_distance_matrix");
  if (loaded.size() != 0) {
    distMat = std::move(loaded);
    filled_ = complete;
  }
}

void detail::write_result_files(Problem &prob, const fs::path &directory, bool complete, std::ostream *progress,
                                const std::vector<std::string> &streamed_names)
{
  const auto &labels = prob.labels();
  const auto &medoids = prob.medoids();
  // A RAM-limited Parquet run holds no series: its names are the reader's, given here.
  const bool streamed = prob.size() == 0;
  const auto series_name = [&](std::size_t i) {
    if (!streamed) return std::string(prob.series_name(i));
    return i < streamed_names.size() ? streamed_names[i] : "series_" + std::to_string(i);
  };
  const auto file_in_directory = [&](const char *suffix) { return directory / utf8_to_path(prob.name() + suffix); };

  if (complete && !streamed) prob.fill_distance_matrix();

  const auto labels_path = file_in_directory("_labels.csv");
  {
    auto out = open_output(labels_path);
    out << "name,cluster\n";
    for (std::size_t i = 0; i < labels.size(); ++i) out << series_name(i) << ',' << labels[i] << '\n';
    close_output(out, labels_path);
  }
  if (progress) *progress << "Labels written to " << labels_path << "\n";

  const auto medoids_path = file_in_directory("_medoids.csv");
  {
    auto out = open_output(medoids_path);
    out << "cluster,medoid_index,medoid_name\n";
    for (std::size_t c = 0; c < medoids.size(); ++c)
      out << c << ',' << medoids[c] << ',' << series_name(static_cast<std::size_t>(medoids[c])) << '\n';
    close_output(out, medoids_path);
  }
  if (progress) *progress << "Medoids written to " << medoids_path << "\n";

  if (streamed) {
    if (complete)
      throw InvalidInput("Result: a RAM-limited Parquet run holds no series, so it has no distance matrix or "
                         "silhouettes to save; its labels and medoids are written, with the reader's names.");
    return;
  }
  // A matrix-free run does not fill an O(N^2) matrix merely to write these files.
  if (!prob.is_distance_matrix_filled()) return;
  const auto matrix_path = file_in_directory("_distance_matrix.csv");
  io::write_csv(prob.distance_matrix(), matrix_path);
  if (progress) *progress << "Distance matrix written to " << matrix_path << "\n";

  // s(i) is undefined for one cluster, which is no reason to fail a clustering that succeeded; a
  // computed score that cannot be written is an error like any other file.
  if (medoids.size() < 2) return;
  std::vector<double> silhouettes;
  try {
    silhouettes = scores::silhouette(prob);
  } catch (const UndefinedScore &e) {
    std::cerr << "Warning: silhouettes skipped: " << e.what() << '\n';
    return;
  }
  const auto silhouettes_path = file_in_directory("_silhouettes.csv");
  {
    auto out = open_output(silhouettes_path);
    out << "name,cluster,silhouette\n";
    for (std::size_t i = 0; i < silhouettes.size(); ++i)
      out << series_name(i) << ',' << labels[i] << ',' << std::setprecision(8) << silhouettes[i] << '\n';
    close_output(out, silhouettes_path);
  }
  if (progress) {
    const double mean = silhouettes.empty() ? 0.0
                                            : std::accumulate(silhouettes.begin(), silhouettes.end(), 0.0)
                                                / static_cast<double>(silhouettes.size());
    *progress << "Silhouette scores written, mean=" << std::setprecision(4) << mean << "\n";
  }
}

} // namespace dtwc
