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
#include "fileOperations.hpp"
#include "scores.hpp"

#include <algorithm>
#include <cctype>
#include <numeric>
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

void validate_skips(index_t skip_cols, index_t skip_rows)
{
  if (skip_cols < 0) throw InvalidInput("load: skip_cols must be non-negative.");
  if (skip_rows < 0) throw InvalidInput("load: skip_rows must be non-negative.");
}

} // namespace

Dataset::Dataset(std::filesystem::path source, index_t skip_cols, index_t skip_rows,
                 char delimiter, std::string name)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name))
{}

Dataset::Dataset(series_type source, index_t skip_cols, index_t skip_rows, char delimiter,
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

Dataset load(const std::filesystem::path &source, index_t skip_cols, index_t skip_rows,
             char delimiter, std::string_view name)
{
  validate_skips(skip_cols, skip_rows);
  std::string resolved = name.empty() ? detail::default_name(source) : std::string(name);
  return Dataset(source, skip_cols, skip_rows, delimiter, std::move(resolved));
}

Dataset load(Dataset::series_type source, index_t skip_cols, index_t skip_rows,
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
               Method method, int iterations, bool converged)
  : problem_(std::move(problem)), cost_(cost), device_(std::move(device_name)), method_(method),
    iterations_(iterations), converged_(converged)
{}

const std::vector<index_t> &Result::labels() const noexcept { return problem_->labels(); }
const std::vector<index_t> &Result::medoids() const noexcept { return problem_->medoids(); }

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
  return io::to_full_matrix(std::as_const(*problem_).distance_matrix());
}

void Result::save(const std::filesystem::path &directory) const
{
  // save() promises the complete matrix and silhouettes: matrix-free methods
  // retain their scaling until this explicitly requested operation.
  // Result::score("silhouette") still throws on one cluster; asking for the
  // number is a different contract from skipping a file.
  detail::write_result_files(*problem_, directory, true);
}

Result cluster(const Dataset &dataset, index_t k, std::string_view method, int band,
               std::string_view device, int max_iter)
{
  return cluster(Dataset(dataset), k, method, band, device, max_iter);
}

Result cluster(Dataset &&dataset, index_t k, std::string_view method, int band,
               std::string_view device, int max_iter)
{
  Config config; // dtwc_cl's defaults for everything this signature does not name
  config.k = k;
  config.method = parse_name(method_names, method, "method");
  config.band = band;
  config.max_iter = max_iter;
  config.output.clear(); // Result::save writes; cluster() does not
  config.name = dataset.name();
  std::tie(config.device, config.device_index) =
    device.empty() ? std::pair{ g_device.device, g_device.index } : detail::parse_device(device);
  if (!dataset.is_path()) return run(config, std::move(dataset).materialize_local());
  config.input = path_to_utf8(dataset.path());
  config.skip_cols = dataset.skip_cols();
  config.skip_rows = dataset.skip_rows();
  config.delimiter = dataset.delimiter();
  return run(config);
}

} // namespace dtwc
