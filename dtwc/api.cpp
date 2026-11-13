/**
 * @file api.cpp
 * @brief Implementation of the DTWC++ 2.0 Tier-1 API.
 * @author Volkan Kumtepeli
 */

#include "api.hpp"

#include "Problem.hpp"
#include "cdist.hpp"
#include "cli/config.hpp"
#include "cli/run.hpp"
#include "core/matrix_io.hpp"
#include "base/env.hpp"
#include "base/error.hpp"
#include "fileOperations.hpp"
#include "io/read_data.hpp"
#ifdef DTWC_HAS_ARROW
#include "io/read_arrow.hpp"
#endif
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

/// A streamed run's Result (`streamed_names` its series' names) refuses `what`: FastCLARA read its series a record
/// batch at a time within --ram-limit, and its Problem holds none, so it has no distances.
void refuse_if_streamed(const std::vector<std::string> &streamed_names, const char *what)
{
  if (streamed_names.empty()) return;
  throw InvalidInput("Result::" + std::string(what) + ": the run streamed its " + std::to_string(streamed_names.size())
                     + " series (FastCLARA read them a record batch at a time within --ram-limit) and holds none, so "
                       "it has no distances; a run whose --ram-limit holds the series in RAM, or with none, has them.");
}

} // namespace

Dataset::Dataset(std::filesystem::path source, index_t skip_cols, index_t skip_rows,
                 char delimiter, std::string name, core::Precision dtype)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name)), dtype_(dtype)
{}

Dataset::Dataset(series_type source, index_t skip_cols, index_t skip_rows, char delimiter,
                 std::string name, core::Precision dtype)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name)), dtype_(dtype)
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
             char delimiter, std::string_view name, core::Precision dtype)
{
  validate_skips(skip_cols, skip_rows);
  std::string resolved = name.empty() ? detail::default_name(source) : std::string(name);
  return Dataset(source, skip_cols, skip_rows, delimiter, std::move(resolved), dtype);
}

Dataset load(Dataset::series_type source, index_t skip_cols, index_t skip_rows,
             char delimiter, std::string_view name, core::Precision dtype)
{
  validate_skips(skip_cols, skip_rows);
  return Dataset(std::move(source), skip_cols, skip_rows, delimiter,
                 name.empty() ? "dataset" : std::string(name), dtype);
}

Result::Result(std::shared_ptr<Problem> problem, double cost, std::string device_name,
               Method method, int iterations, bool converged, std::vector<std::string> streamed_names,
               std::string report)
  : problem_(std::move(problem)), cost_(cost), device_(std::move(device_name)), method_(method),
    iterations_(iterations), converged_(converged), streamed_names_(std::move(streamed_names)),
    report_(std::move(report))
{}

const std::vector<index_t> &Result::labels() const noexcept { return problem_->labels(); }
const std::vector<index_t> &Result::medoids() const noexcept { return problem_->medoids(); }

double Result::score(std::string_view name) const
{
  refuse_if_streamed(streamed_names_, "score");
  return scores::score(*problem_, name);
}

const core::DistanceMatrix &detail::filled_matrix(const Result &result)
{
  refuse_if_streamed(result.streamed_names_, "distance_matrix");
  // Matrix-free methods leave the matrix unmaterialised; asking for it is an
  // explicit request for the full N*N, as score() and save() already treat it.
  result.problem_->fill_distance_matrix();
  return result.problem_->distance_matrix();
}

std::vector<double> Result::distance_matrix() const { return io::to_full_matrix(detail::filled_matrix(*this)); }

void Result::save(const std::filesystem::path &directory) const
{
  // save() writes the complete matrix and the silhouettes, so a matrix-free method, which left the matrix unfilled,
  // fills it here, on request. Result::score("silhouette") still throws on one cluster: asking for the number is a
  // different contract from skipping a file. A streamed run holds no series: its labels and medoids, with the
  // reader's names, as dtwc_cl writes them.
  detail::write_result_files(*problem_, directory, streamed_names_.empty(), nullptr, streamed_names_);
  // The run's report, last: a save that fails before it writes none.
  if (report_.empty()) return;
  write_text(directory / utf8_to_path(problem_->name() + "_report.txt"), report_);
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
  config.dtype = dataset.dtype(); // run() reads a path in it and converts series in memory to it
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

namespace {

std::vector<std::pair<std::string, std::string>>
to_string_pairs(std::initializer_list<SettingPair> options)
{
  std::vector<std::pair<std::string, std::string>> pairs;
  pairs.reserve(options.size());
  for (const auto &[k, v] : options) {
    pairs.emplace_back(std::string(k), std::string(v));
  }
  return pairs;
}

} // namespace

Config make_settings(std::initializer_list<SettingPair> options)
{
  Config base;
  const std::string proc_dev = dtwc::device();
  std::tie(base.device, base.device_index) = detail::parse_device(proc_dev);
  return parse_config(std::move(base), to_string_pairs(options));
}

Config make_settings(Config base, std::initializer_list<SettingPair> overrides)
{
  return parse_config(std::move(base), to_string_pairs(overrides));
}

namespace settings {
Config make(std::initializer_list<SettingPair> options) { return make_settings(options); }
Config make(Config base, std::initializer_list<SettingPair> overrides)
{
  return make_settings(std::move(base), overrides);
}
} // namespace settings

Dataset load_data(const std::filesystem::path &source, const Config &cfg)
{
  return load(source, cfg.skip_cols, cfg.skip_rows, cfg.delimiter, cfg.name, cfg.dtype);
}

Dataset load_data(Dataset::series_type source, const Config &cfg)
{
  return load(std::move(source), cfg.skip_cols, cfg.skip_rows, cfg.delimiter, cfg.name, cfg.dtype);
}

Dataset load_data(const Config &cfg)
{
  if (cfg.input.empty()) {
    throw InvalidInput("load_data: settings.input is empty; pass a source path or set 'input' in settings.");
  }
  return load_data(utf8_to_path(cfg.input), cfg);
}

Dataset load_data(const std::filesystem::path &source, index_t skip_cols, index_t skip_rows,
                  char delimiter, std::string_view name, core::Precision dtype)
{
  return load(source, skip_cols, skip_rows, delimiter, name, dtype);
}

Dataset load_data(Dataset::series_type source, index_t skip_cols, index_t skip_rows,
                  char delimiter, std::string_view name, core::Precision dtype)
{
  return load(std::move(source), skip_cols, skip_rows, delimiter, name, dtype);
}

Result do_clustering(const Dataset &dataset, const Config &cfg)
{
  return do_clustering(Dataset(dataset), cfg);
}

Result do_clustering(Dataset &&dataset, const Config &cfg)
{
  Config config = cfg;
  if (config.name.empty()) config.name = dataset.name();
  if (!dataset.is_path()) {
    if (dataset.skip_cols() == 0 && config.skip_cols > 0) dataset.skip_cols_ = config.skip_cols;
    if (dataset.skip_rows() == 0 && config.skip_rows > 0) dataset.skip_rows_ = config.skip_rows;
    if (dataset.dtype() != core::Precision::Float64 && config.dtype == core::Precision::Float64)
      config.dtype = dataset.dtype();
    config.input.clear();
    config.skip_cols = 0;
    config.skip_rows = 0;
    config.delimiter = '\0';
    config.column.clear();
    return run(config, std::move(dataset).materialize_local());
  }
  config.input = path_to_utf8(dataset.path());
  if (dataset.skip_cols() != 0 && config.skip_cols == 0) config.skip_cols = dataset.skip_cols();
  if (dataset.skip_rows() != 0 && config.skip_rows == 0) config.skip_rows = dataset.skip_rows();
  if (dataset.delimiter() != '\0' && config.delimiter == '\0') config.delimiter = dataset.delimiter();
  if (dataset.dtype() != core::Precision::Float64 && config.dtype == core::Precision::Float64)
    config.dtype = dataset.dtype();
  return run(config);
}

Result do_clustering(const Dataset &data, const Config &cfg,
                     std::initializer_list<SettingPair> overrides)
{
  return do_clustering(Dataset(data), make_settings(cfg, overrides));
}

Result do_clustering(Dataset &&data, const Config &cfg,
                     std::initializer_list<SettingPair> overrides)
{
  return do_clustering(std::move(data), make_settings(cfg, overrides));
}

void write_data(const Result &result, const std::filesystem::path &directory)
{
  result.save(directory);
}

std::vector<double> cdist(const Dataset &X, const Dataset &Y, const Config &cfg)
{
  auto read_ds = [&cfg](const Dataset &ds) -> Data {
    const index_t skip_cols = ds.skip_cols() != 0 ? ds.skip_cols() : cfg.skip_cols;
    const index_t skip_rows = ds.skip_rows() != 0 ? ds.skip_rows() : cfg.skip_rows;
    const char delimiter = ds.delimiter() != '\0' ? ds.delimiter() : cfg.delimiter;
    const core::Precision dtype =
      (ds.dtype() != core::Precision::Float64) ? ds.dtype() : cfg.dtype;
    if (!ds.is_path()) {
      Dataset copy(ds);
      copy.skip_cols_ = skip_cols;
      copy.skip_rows_ = skip_rows;
      Data out = std::move(copy).materialize_local();
      return dtype == core::Precision::Float32
               ? detail::owned_copy(out, core::Precision::Float32)
               : out;
    }
#ifdef DTWC_HAS_ARROW
    if (io::arrow_format(ds.path())) {
      return io::read_arrow(ds.path(), cfg.column, skip_cols, skip_rows, dtype);
    }
#endif
    return read_data(ds.path(), skip_cols, skip_rows, delimiter, cfg.column, dtype);
  };
  const Data x_data = read_ds(X);
  const Data y_data = read_ds(Y);
  return cdist(x_data, y_data, cfg);
}

std::vector<double> cdist(const Dataset &X, const Dataset &Y)
{
  Config cfg;
  const std::string proc_dev = dtwc::device();
  std::tie(cfg.device, cfg.device_index) = detail::parse_device(proc_dev);
  return cdist(X, Y, cfg);
}

} // namespace dtwc
