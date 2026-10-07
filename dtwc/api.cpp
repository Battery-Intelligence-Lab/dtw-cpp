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
#include "io/read_data.hpp"
#include "scores.hpp"

#include <algorithm>
#include <tuple>
#include <utility>

namespace dtwc {
namespace {

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

Result::Result(std::shared_ptr<Problem> problem, double cost, std::string device_name,
               Method method, int iterations, bool converged, std::vector<std::string> streamed_names)
  : problem_(std::move(problem)), cost_(cost), device_(std::move(device_name)), method_(method),
    iterations_(iterations), converged_(converged), streamed_names_(std::move(streamed_names))
{}

const std::vector<index_t> &Result::labels() const noexcept { return problem_->labels(); }
const std::vector<index_t> &Result::medoids() const noexcept { return problem_->medoids(); }

double Result::score(std::string_view name) const { return scores::score(*problem_, name); }

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
  detail::write_result_files(*problem_, directory, true, nullptr, streamed_names_);
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
  const std::string process_device = dtwc::device(); // used when `device` is ""
  std::tie(config.device, config.device_index) =
    detail::parse_device(device.empty() ? std::string_view(process_device) : device);
  if (!dataset.is_path()) return run(config, std::move(dataset).materialize_local());
  config.input = path_to_utf8(dataset.path());
  config.skip_cols = dataset.skip_cols();
  config.skip_rows = dataset.skip_rows();
  config.delimiter = dataset.delimiter();
  return run(config);
}

} // namespace dtwc
