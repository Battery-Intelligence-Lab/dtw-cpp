/**
 * @file api.cpp
 * @brief Implementation of the DTWC++ 2.0 Tier-1 API.
 * @author Volkan Kumtepeli
 */

#include "api.hpp"

#include "Problem.hpp"
#include "cdist.hpp"
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

  void validate_load_options(index_t skip_cols, index_t skip_rows, std::size_t ndim)
  {
    if (skip_cols < 0) throw InvalidInput("load: skip_cols must be non-negative.");
    if (skip_rows < 0) throw InvalidInput("load: skip_rows must be non-negative.");
    if (ndim < 1) throw InvalidInput("load: ndim must be at least 1.");
  }

  Data read_dataset_path(const Dataset &ds, const std::string &column)
  {
    Data out =
#ifdef DTWC_HAS_ARROW
      io::arrow_format(ds.path())
        ? io::read_arrow(ds.path(), column, ds.skip_cols(), ds.skip_rows(), ds.dtype())
        :
#endif
        read_data(ds.path(), ds.skip_cols(), ds.skip_rows(), ds.delimiter(), column, ds.dtype());
    if (ds.ndim() > 1) {
      if (out.ndim != 1 && out.ndim != ds.ndim())
        throw InvalidInput("load: the Arrow IPC file's metadata gives ndim=" + std::to_string(out.ndim)
                           + " and ndim=" + std::to_string(ds.ndim()) + " was given; pass ndim="
                           + std::to_string(out.ndim) + ", or leave ndim out.");
      if (out.ndim == 1) {
        out.ndim = ds.ndim();
        out.validate_ndim();
      }
    }
    return out;
  }

  /// Refuse a reader option of `cfg` (skip_cols, skip_rows, delimiter, dtype) that is not at its default and differs
  /// from the one `dataset` was made with: the Dataset is read with its own, so the Config's would be dropped without
  /// a word. A Config cannot say which options were given, so its defaults stand for "not given".
  /// A Dataset holds no column: a path's read takes the Config's, as run() does, and series in memory cannot honour
  /// one.
  void require_options_made_with(const Dataset &dataset, const Config &cfg, const char *where)
  {
    const Config defaults{};
    const auto quoted = [](char c) { return c == '\0' ? std::string("''") : std::string{ '\'', c, '\'' }; };
    const auto dtype = [](core::Precision p) { return std::string(name_of(core::precision_names, p)); };
    for (const auto &[key, given, standard, made] :
         { std::tuple{ "skip_cols", std::to_string(cfg.skip_cols), std::to_string(defaults.skip_cols),
                       std::to_string(dataset.skip_cols()) },
           std::tuple{ "skip_rows", std::to_string(cfg.skip_rows), std::to_string(defaults.skip_rows),
                       std::to_string(dataset.skip_rows()) },
           std::tuple{ "delimiter", quoted(cfg.delimiter), quoted(defaults.delimiter), quoted(dataset.delimiter()) },
           std::tuple{ "dtype", dtype(cfg.dtype), dtype(defaults.dtype), dtype(dataset.dtype()) } }) {
      if (given != standard && given != made)
        throw InvalidInput(std::string(where) + ": the Dataset was made with " + key + " " + made + " and " + key
                           + " " + given + " was given; a Dataset keeps the options it was made with, so pass " + key
                           + " to load, or leave it out.");
    }
    if (!dataset.is_path()) require_reader_options(std::nullopt, 0, 0, '\0', cfg.column);
  }

  /// A streamed run's Result (`streamed_names` its series' names) refuses `what`: FastCLARA read its series a record
  /// batch at a time within --ram-limit, and its Problem holds none, so it has no distances.
  void refuse_if_streamed(const std::vector<std::string> &streamed_names, const char *what)
  {
    if (streamed_names.empty()) return;
    throw InvalidInput("Result::" + std::string(what) + ": the run streamed its " + std::to_string(streamed_names.size())
                       +
                       " series (FastCLARA read them a record batch at a time within --ram-limit) and holds none, so "
                       "it has no distances; a run whose --ram-limit holds the series in RAM, or with none, has them.");
  }

} // namespace

Dataset::Dataset(std::filesystem::path source, index_t skip_cols, index_t skip_rows,
                 char delimiter, std::string name, core::Precision dtype, std::size_t ndim)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name)), dtype_(dtype), ndim_(ndim)
{}

Dataset::Dataset(series_type source, index_t skip_cols, index_t skip_rows, char delimiter,
                 std::string name, core::Precision dtype, std::size_t ndim)
  : source_(std::move(source)), skip_cols_(skip_cols), skip_rows_(skip_rows),
    delimiter_(delimiter), name_(std::move(name)), dtype_(dtype), ndim_(ndim)
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
  return Data(std::move(series), std::move(names), ndim_);
}

Dataset load(const std::filesystem::path &source, index_t skip_cols, index_t skip_rows,
             char delimiter, std::string_view name, core::Precision dtype, std::size_t ndim)
{
  validate_load_options(skip_cols, skip_rows, ndim);
  std::string resolved = name.empty() ? detail::default_name(source) : std::string(name);
  return Dataset(source, skip_cols, skip_rows, delimiter, std::move(resolved), dtype, ndim);
}

Dataset load(Dataset::series_type source, index_t skip_cols, index_t skip_rows,
             char delimiter, std::string_view name, core::Precision dtype, std::size_t ndim)
{
  validate_load_options(skip_cols, skip_rows, ndim);
  return Dataset(std::move(source), skip_cols, skip_rows, delimiter, name.empty() ? "dataset" : std::string(name),
                 dtype, ndim);
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
  Config config; // Config's defaults (output is "", so cluster() writes nothing)
  config.k = k;
  config.method = parse_name(method_names, method, "method");
  config.band = band;
  config.max_iter = max_iter;
  std::tie(config.device, config.device_index) =
    device.empty() ? detail::parse_device(dtwc::device()) : detail::parse_device(device);
  return cluster(std::move(dataset), config);
}

Result cluster(const Dataset &dataset, const Config &cfg)
{
  return cluster(Dataset(dataset), cfg);
}

Result cluster(Dataset &&dataset, const Config &cfg)
{
  require_options_made_with(dataset, cfg, "cluster");
  Config config = cfg;
  if (config.name.empty()) config.name = dataset.name();
  config.dtype = dataset.dtype();
  if (!dataset.is_path()) { // materialize_local drops the Dataset's skips; run() refuses reader options here
    config.input.clear();
    config.skip_cols = 0;
    config.skip_rows = 0;
    config.delimiter = '\0';
    return run(config, std::move(dataset).materialize_local());
  }
  if (dataset.ndim() > 1) {
    Data loaded = read_dataset_path(dataset, cfg.column);
    config.input.clear();
    config.skip_cols = 0;
    config.skip_rows = 0;
    config.delimiter = '\0';
    return run(config, std::move(loaded));
  }
  config.input = path_to_utf8(dataset.path());
  config.skip_cols = dataset.skip_cols();
  config.skip_rows = dataset.skip_rows();
  config.delimiter = dataset.delimiter();
  return run(config);
}

std::vector<double> cdist(const Dataset &X, const Dataset &Y, const Config &cfg)
{
  auto read_ds = [&cfg](const Dataset &ds) -> Data {
    require_options_made_with(ds, cfg, "cdist");
    if (!ds.is_path()) {
      Data out = Dataset(ds).materialize_local();
      return ds.dtype() == core::Precision::Float32
               ? detail::owned_copy(out, core::Precision::Float32)
               : out;
    }
    return read_dataset_path(ds, cfg.column);
  };
  const Data x_data = read_ds(X);
  const Data y_data = read_ds(Y);
  return cdist(x_data, y_data, cfg);
}

std::vector<double> cdist(const Dataset &X, const Dataset &Y)
{
  Config cfg; // Config's defaults, on the process device
  std::tie(cfg.device, cfg.device_index) = detail::parse_device(dtwc::device());
  return cdist(X, Y, cfg);
}

} // namespace dtwc
