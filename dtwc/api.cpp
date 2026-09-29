/**
 * @file api.cpp
 * @brief Implementation of the DTWC++ 2.0 Tier-1 API.
 */

#include "api.hpp"

#include "Problem.hpp"
#include "cli/run.hpp"
#include "core/matrix_io.hpp"
#include "base/env.hpp"
#include "base/error.hpp"
#include "scores.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <system_error>
#include <tuple>
#include <utility>

namespace dtwc {
namespace {

std::string lower(std::string_view value)
{
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

/// The process-wide default device: what dtwc::device(name) set, CPU until then.
struct DeviceSelection
{
  Device device = Device::CPU;
  int index = 0;
};
DeviceSelection g_device;

std::string derive_name(const std::filesystem::path &path)
{
  if (path.has_filename()) {
    // UTF-8 end to end: every writer turns this name back into a path
    // component with utf8_to_path(), so the round trip is lossless.
    const auto stem = path_to_utf8(path.stem());
    if (!stem.empty()) return stem;
  }
  return "dataset";
}

void validate_skips(int skip_cols, int skip_rows)
{
  if (skip_cols < 0) throw InvalidInput("load: skip_cols must be non-negative.");
  if (skip_rows < 0) throw InvalidInput("load: skip_rows must be non-negative.");
}

std::ofstream open_output(const std::filesystem::path &path,
                          std::ios::openmode mode = std::ios::out)
{
  std::ofstream stream(path, mode);
  if (!stream.is_open()) throw IOError("Result::save: cannot open " + path.string());
  return stream;
}

/// A full disk or a file-size quota fails the writes after a successful open,
/// so the stream is checked after closing too: an open-only check left a
/// truncated file behind a save that reported success (B-05).
void close_output(std::ofstream &stream, const std::filesystem::path &path)
{
  stream.close();
  if (!stream)
    throw IOError("Result::save: cannot write " + path.string()
                  + "; the file is incomplete (disk full or file-size quota?). "
                    "Free space or save to another directory.");
}

} // namespace

Dataset::Dataset(std::filesystem::path source, int skip_cols, int skip_rows,
                 char delimiter, std::string name)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name))
{}

Dataset::Dataset(series_type source, int skip_cols, int skip_rows, char delimiter,
                 std::string name)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name))
{}

bool Dataset::is_path() const noexcept
{
  return std::holds_alternative<std::filesystem::path>(source_);
}

const std::filesystem::path &Dataset::path() const
{
  if (!is_path()) throw InvalidInput("Dataset::path: this dataset is in memory.");
  return std::get<std::filesystem::path>(source_);
}

Data Dataset::materialize_local() &&
{
  auto series = std::move(std::get<series_type>(source_));
  // One memory row is one file line, so skip_rows drops leading series exactly
  // as it drops leading lines of a batch file.
  const auto dropped = std::min<std::size_t>(
    static_cast<std::size_t>(skip_rows_), series.size());
  series.erase(series.begin(),
               series.begin() + static_cast<std::ptrdiff_t>(dropped));
  if (skip_cols_ > 0) {
    for (auto &row : series) {
      if (static_cast<std::size_t>(skip_cols_) > row.size())
        throw InvalidInput("load: skip_cols exceeds an in-memory series length.");
      row.erase(row.begin(), row.begin() + skip_cols_);
    }
  }
  std::vector<std::string> names(series.size());
  for (std::size_t i = 0; i < names.size(); ++i) names[i] = std::to_string(i);
  return Data(std::move(series), std::move(names));
}

Dataset load(const std::filesystem::path &source, int skip_cols, int skip_rows,
             char delimiter, std::string_view name)
{
  validate_skips(skip_cols, skip_rows);
  std::string resolved = name.empty() ? derive_name(source) : std::string(name);
  return Dataset(source, skip_cols, skip_rows, delimiter, std::move(resolved));
}

Dataset load(Dataset::series_type source, int skip_cols, int skip_rows,
             char delimiter, std::string_view name)
{
  validate_skips(skip_cols, skip_rows);
  return Dataset(std::move(source), skip_cols, skip_rows, delimiter,
                 name.empty() ? "dataset" : std::string(name));
}

std::string device(std::string_view name)
{
  const auto [selected, index] = detail::parse_device(name);
#if !defined(DTWC_HAS_CUDA) && !defined(DTWC_HAS_METAL)
  if (selected == Device::GPU) throw DeviceError(detail::gpu_not_built_message());
#endif
  g_device = { selected, index };
  return device();
}

std::string device()
{
  std::string out = to_string(g_device.device);
  if (g_device.device == Device::GPU && g_device.index != 0) out += ":" + std::to_string(g_device.index);
  return out;
}

Result::Result(std::shared_ptr<Problem> problem, double cost, std::string device_name,
               ClusterMethod method, int iterations, bool converged)
  : problem_(std::move(problem)), cost_(cost), device_(std::move(device_name)), method_(method),
    iterations_(iterations), converged_(converged)
{}

const std::vector<int> &Result::labels() const noexcept { return problem_->labels(); }
const std::vector<int> &Result::medoids() const noexcept { return problem_->medoids(); }

double Result::score(std::string_view name) const
{
  const std::string key = lower(name);
  if (key == "silhouette") {
    const auto values = scores::silhouette(*problem_);
    return values.empty()
      ? 0.0
      : std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
  }
  if (key == "davies_bouldin") return scores::davies_bouldin(*problem_);
  if (key == "dunn") return scores::dunn(*problem_);
  if (key == "calinski_harabasz") return scores::calinski_harabasz(*problem_);
  if (key == "inertia") return scores::inertia(*problem_);
  throw InvalidInput(
    "Result::score: unknown score '" + std::string(name)
    + "'. Valid scores: silhouette, davies_bouldin, dunn, "
      "calinski_harabasz, inertia.");
}

std::vector<double> Result::distance_matrix() const
{
  // Matrix-free methods leave the matrix unmaterialised; asking for it is an
  // explicit request for the full N*N, as score() and save() already treat it.
  problem_->fill_distance_matrix();
  const std::size_t n = problem_->size();
  std::vector<double> flat(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      flat[i * n + j] =
        problem_->dist_by_ind(static_cast<int>(i), static_cast<int>(j));

  return flat;
}

void Result::save(const std::filesystem::path &directory) const
{
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec)
    throw IOError("Result::save: cannot create '" + directory.string() + "': "
                  + ec.message());

  // problem_->name() is UTF-8 (api.cpp::derive_name); utf8_to_path keeps it so
  // on the way back to the filesystem. path::string() would re-decode it as the
  // native narrow encoding and write a mojibake filename on Windows.
  const std::string &base = problem_->name();
  const auto labels_path = directory / utf8_to_path(base + "_labels.csv");
  const auto medoids_path = directory / utf8_to_path(base + "_medoids.csv");
  const auto matrix_path = directory / utf8_to_path(base + "_distance_matrix.csv");
  const auto silhouettes_path = directory / utf8_to_path(base + "_silhouettes.csv");

  {
    auto out = open_output(labels_path);
    out << "name,cluster\n";
    for (std::size_t i = 0; i < labels().size(); ++i)
      out << problem_->series_name(i) << ',' << labels()[i] << '\n';
    close_output(out, labels_path);
  }
  {
    auto out = open_output(medoids_path);
    out << "cluster,medoid_index,medoid_name\n";
    for (std::size_t c = 0; c < medoids().size(); ++c) {
      const int idx = medoids()[c];
      out << c << ',' << idx << ','
          << problem_->series_name(static_cast<std::size_t>(idx)) << '\n';
    }
    close_output(out, medoids_path);
  }

  // save() promises the complete matrix and silhouettes. Matrix-free methods
  // retain their scaling until this explicitly requested operation.
  problem_->fill_distance_matrix();
  {
    const core::DistanceMatrix &matrix = problem_->distance_matrix();
    core::detail::preflight_distance_matrix_csv(matrix);
    auto out = open_output(
      matrix_path, std::ios::out | std::ios::binary | std::ios::trunc);
    out << matrix;
    close_output(out, matrix_path);
  }

  // s(i) is undefined with fewer than two realised clusters, where
  // scores::silhouette() throws UndefinedScore. save() must not fail a
  // clustering that succeeded: warn and skip the file, as the CLI does.
  // Result::score("silhouette") still throws — asking for the number is a
  // different contract. A corrupt labelling still propagates.
  std::vector<double> silhouette_values;
  try {
    silhouette_values = scores::silhouette(*problem_);
  } catch (const UndefinedScore &e) {
    std::cerr << "Warning: silhouettes skipped: " << e.what() << '\n';
    return;
  }
  {
    auto out = open_output(silhouettes_path);
    out << "name,cluster,silhouette\n";
    for (std::size_t i = 0; i < silhouette_values.size(); ++i)
      out << problem_->series_name(i) << ',' << labels()[i] << ','
          << std::setprecision(8) << silhouette_values[i] << '\n';
    close_output(out, silhouettes_path);
  }
}

Result cluster(const Dataset &dataset, int k, std::string_view method, int band,
               std::string_view device, int max_iter)
{
  return cluster(Dataset(dataset), k, method, band, device, max_iter);
}

Result cluster(Dataset &&dataset, int k, std::string_view method, int band,
               std::string_view device, int max_iter)
{
  Config config; // dtwc_cl's defaults for everything this signature does not name
  config.k = k;
  config.method = parse_name(cluster_method_names, method, "method");
  config.band = band;
  config.max_iter = max_iter;
  config.output.clear(); // Result::save writes; cluster() does not
  config.name = dataset.name();
  std::tie(config.device, config.gpu.device_id) =
    device.empty() ? std::pair{ g_device.device, g_device.index } : detail::parse_device(device);
  if (!dataset.is_path()) return run(config, std::move(dataset).materialize_local());
  config.input = path_to_utf8(dataset.path());
  config.skip_cols = dataset.skip_cols();
  config.skip_rows = dataset.skip_rows();
  config.delimiter = dataset.delimiter();
  return run(config);
}

} // namespace dtwc
