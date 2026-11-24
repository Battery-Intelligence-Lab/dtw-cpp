/**
 * @file Problem_IO.cpp
 * @brief Implementation of input/output functions for the Problem class.
 *
 * @details These functions print a Problem's clusters, read a distance matrix from a file,
 * and, in detail::write_result_files, write the result files of a clustered Problem.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 *
 * @date 25 Dec 2023
 */

#include "Problem.hpp"
#include "core/matrix_io.hpp"
#include "detail/result_files.hpp" // WriteTimes
#include "fileOperations.hpp" // for open_output, close_output
#include "scores.hpp"      // for silhouette

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>  // for setprecision
#include <iostream> // for cout
#include <numeric>  // for accumulate
#include <string>  // for allocator, char_traits, operator+
#include <vector>  // for vector, operator==

namespace dtwc {

void Problem::print_clusters() const
{
  require_clustered("print_clusters");
  std::cout << "Clusters centroids: ";
  for (auto ind : centroids_ind)
    std::cout << series_name(ind) << ' ';

  std::cout << '\n';

  for (std::size_t i_c = 0, n_c = static_cast<std::size_t>(Nc); i_c < n_c; ++i_c) {
    std::cout << "The cluster with centroid " << series_name(centroids_ind[i_c]) << " has following members: ";

    for (std::size_t i_p = 0, n_p = static_cast<std::size_t>(size()); i_p < n_p; ++i_p)
      if (clusters_ind[i_p] == i_c)
        std::cout << series_name(i_p) << " ";

    std::cout << '\n';
  }
}

/**
 *  @brief Reads the distance matrix from a file.
 *  @details A read failure is reported to the caller: swallowing it would make a
 *  failed load indistinguishable from a successful one, and whether to continue
 *  without a precomputed matrix is the caller's decision.
 *  @param distMat_path The file path of the distance matrix.
 *  @param not_computed What marks a pair never computed: NaN (an empty field or
 *         NaN) or Negative (also any value below zero, v1.0.0's -1).
 *  @throws IOError if the file cannot be opened or holds a non-numeric field;
 *          InvalidInput if it is not square and symmetric, if its size is not
 *          this Problem's series count, if a distance is ±inf or negative, or
 *          if the matrix is memory-mapped.
 */
void Problem::read_distance_matrix(const fs::path &distMat_path, core::NotComputed not_computed)
{
  // Values read into a mapped matrix would persist in its file under this
  // Problem's fingerprint, whatever they were computed from.
  if (distMat.is_mapped())
    throw InvalidInput("read_distance_matrix: this Problem's distance matrix is memory-mapped "
                       "(use_mmap_distance_matrix), and a CSV matrix is read into RAM only; call "
                       "refresh_distance_matrix() first, or reuse the mapped file instead.");
  // Parsed into a local so a rejected file leaves the matrix untouched. A
  // matrix of another size, an empty one included, describes other series and
  // is refused: ignoring it would recompute every distance silently at the
  // first lookup. A Problem without series takes any matrix.
  core::DistanceMatrix loaded(0, matrix_dtype_);
  io::read_csv(loaded, distMat_path, not_computed);
  if (size() != 0 && loaded.size() != size())
    throw InvalidInput(
      "Problem::read_distance_matrix: '" + path_to_utf8(distMat_path) + "' has "
      + std::to_string(loaded.size()) + " rows, but this Problem holds "
      + std::to_string(size()) + " series; a distance matrix has one row and "
        "one column per series, in input order. Load the matrix computed for "
        "these series, or omit it to compute the distances.");
  const std::string variant(name_of(core::variant_names, distance_.variant.variant));
  const bool complete = loaded.all_computed(
    "Problem::read_distance_matrix",
    "A v1.0.0 distance-matrix file marks a pair it never computed with -1: read it with --not-computed negative (C++ "
    "NotComputed::Negative, Python not_computed=\"negative\", MATLAB 'NotComputed','negative'); otherwise this is not "
    "a " + variant + " distance matrix.");
  if (loaded.size() != 0) {
    distMat = std::move(loaded);
    filled_ = complete;
  }
}

namespace {

/// `name` as one CSV field (RFC 4180): quoted, each quote doubled, where it holds a comma, a quote or a line break, so
/// a CSV reader takes it whole; any other name as it is.
std::string csv_field(std::string name)
{
  if (name.find_first_of(",\"\r\n") == std::string::npos) return name;
  std::string quoted = "\"";
  for (const char c : name) {
    if (c == '"') quoted += '"';
    quoted += c;
  }
  return quoted + '"';
}

} // namespace

void detail::write_result_files(Problem &prob, const fs::path &directory, bool complete, std::ostream *progress,
                                const std::vector<std::string> &streamed_names)
{
  write_result_files(prob, directory, complete, progress, streamed_names, nullptr);
}

void detail::write_result_files(Problem &prob, const fs::path &directory, bool complete, std::ostream *progress,
                                const std::vector<std::string> &streamed_names, WriteTimes *times)
{
  const auto &labels = prob.labels();
  const auto &medoids = prob.medoids();
  // A streamed run (Arrow IPC or Parquet) holds no series: its names are the reader's, given here.
  const bool streamed = prob.size() == 0;
  const auto series_name = [&](std::size_t i) {
    if (!streamed) return std::string(prob.series_name(i));
    return i < streamed_names.size() ? streamed_names[i] : made_name(i);
  };
  const auto file_in_directory = [&](const char *suffix) { return directory / utf8_to_path(prob.name() + suffix); };

  if (complete && !streamed) prob.fill_distance_matrix();

  const auto labels_path = file_in_directory("_labels.csv");
  {
    auto out = open_output(labels_path);
    out << "name,cluster\n";
    for (std::size_t i = 0; i < labels.size(); ++i) out << csv_field(series_name(i)) << ',' << labels[i] << '\n';
    close_output(out, labels_path);
  }
  if (progress) *progress << "Labels written to " << labels_path << "\n";

  const auto medoids_path = file_in_directory("_medoids.csv");
  {
    auto out = open_output(medoids_path);
    out << "cluster,medoid_index,medoid_name\n";
    for (std::size_t c = 0; c < medoids.size(); ++c)
      out << c << ',' << medoids[c] << ',' << csv_field(series_name(static_cast<std::size_t>(medoids[c]))) << '\n';
    close_output(out, medoids_path);
  }
  if (progress) *progress << "Medoids written to " << medoids_path << "\n";

  const auto skipped = [times](const char *why) {
    if (times) times->skipped = why;
  };
  if (streamed) {
    if (complete)
      throw InvalidInput("Result: a streamed run holds no series, so it has no distance matrix or "
                         "silhouettes to save; its labels and medoids are written, with the reader's names.");
    skipped("not computed: a streamed run holds no distance matrix");
    return;
  }
  // A matrix-free run does not fill an O(N^2) matrix merely to write these files.
  if (!prob.is_distance_matrix_filled()) {
    skipped("not computed: the method left the distance matrix unfilled, and a run does not fill it to write them");
    return;
  }
  const auto matrix_path = file_in_directory("_distance_matrix.csv");
  io::write_csv(prob.distance_matrix(), matrix_path);
  if (progress) *progress << "Distance matrix written to " << matrix_path << "\n";

  // s(i) is undefined for one cluster, which is no reason to fail a clustering that succeeded; a
  // computed score that cannot be written is an error like any other file.
  if (medoids.size() < 2) {
    skipped("not computed: one cluster");
    return;
  }
  std::vector<double> silhouettes;
  const auto wall = std::chrono::steady_clock::now();
  const double cpu = times && times->cpu_clock ? times->cpu_clock() : 0.0;
  try {
    silhouettes = scores::silhouette(prob);
  } catch (const UndefinedScore &e) {
    std::cerr << "Warning: silhouettes skipped: " << e.what() << '\n';
    if (times) times->skipped = std::string("skipped: ") + e.what();
    return;
  }
  if (times) {
    times->silhouette_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count();
    times->silhouette_cpu = times->cpu_clock ? times->cpu_clock() - cpu : 0.0;
  }
  const auto silhouettes_path = file_in_directory("_silhouettes.csv");
  {
    auto out = open_output(silhouettes_path);
    out << "name,cluster,silhouette\n";
    for (std::size_t i = 0; i < silhouettes.size(); ++i)
      out << csv_field(series_name(i)) << ',' << labels[i] << ',' << std::setprecision(8) << silhouettes[i] << '\n';
    close_output(out, silhouettes_path);
  }
  // The report's mean is -v's: one sum of the silhouettes already computed.
  const double mean = silhouettes.empty() ? 0.0
                                          : std::accumulate(silhouettes.begin(), silhouettes.end(), 0.0)
                                              / static_cast<double>(silhouettes.size());
  if (times) times->mean_silhouette = mean;
  if (progress) *progress << "Silhouette scores written, mean=" << std::setprecision(4) << mean << "\n";
}

} // namespace dtwc
