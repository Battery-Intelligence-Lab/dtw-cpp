/**
 * @file api.cpp
 * @brief Implementation of the DTWC++ 2.0 Tier-1 API.
 */

#include "api.hpp"

#include "DataLoader.hpp"
#include "Problem.hpp"
#include "algorithms/fast_clara.hpp"
#include "algorithms/fast_pam.hpp"
#include "algorithms/hierarchical.hpp"
#include "algorithms/one_batch_pam.hpp"
#include "core/matrix_io.hpp"
#include "detail/tier1_method_resolution.hpp"
#include "env.hpp"
#include "error.hpp"
#include "scores.hpp"
#include "settings.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <system_error>
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

std::string canonical_device_name(const Env &e)
{
  std::string out = to_string(e.device());
  if (e.device() == Device::GPU && e.device_index() != 0)
    out += ":" + std::to_string(e.device_index());
  return out;
}

std::string derive_name(const std::filesystem::path &path)
{
  if (path.has_filename()) {
    const auto stem = path.stem().string();
    if (!stem.empty()) return stem;
  }
  return "dataset";
}

void validate_common(const Dataset &dataset, int k, int max_iter)
{
  if (k <= 0) throw InvalidInput("cluster: k must be positive.");
  if (max_iter <= 0) throw InvalidInput("cluster: max_iter must be positive.");
  if (dataset.skip_cols() < 0)
    throw InvalidInput("load: skip_cols must be non-negative.");
}

std::string normalize_method(std::string_view value)
{
  std::string method = lower(value);
  std::replace(method.begin(), method.end(), '-', '_');
  if (method == "hclust") return "hierarchical";
  static constexpr std::string_view valid[] = {
    "auto", "pam", "onebatch", "clara", "kmedoids", "mip",
    "lrcore", "tadpole", "hierarchical"
  };
  if (std::find(std::begin(valid), std::end(valid), method) == std::end(valid)) {
    throw InvalidInput(
      "cluster: unknown method '" + std::string(value)
      + "'. Valid methods: auto, pam, onebatch, clara, kmedoids, mip, "
        "lrcore, tadpole, hierarchical (hclust).");
  }
  return method;
}

void configure_device(Problem &prob, Device selected, int index)
{
  validate_device(selected);
  if (selected == Device::CPU) {
    // Auto is a CPU policy (brute-force vs admissible pruning), never a GPU
    // selector, so it is safe and usually faster than forcing BruteForce.
    prob.distance_strategy = DistanceMatrixStrategy::Auto;
    return;
  }
  if (selected == Device::GPU) {
    prob.cuda_settings.device_id = index;
#if defined(DTWC_HAS_CUDA)
    prob.distance_strategy = DistanceMatrixStrategy::CUDA;
#elif defined(DTWC_HAS_METAL)
    prob.distance_strategy = DistanceMatrixStrategy::Metal;
#else
    throw DeviceError(
      "cluster: GPU was selected but this build has no GPU backend. Rebuild with "
      "-DDTWC_ENABLE_CUDA=ON or use a macOS Metal build.");
#endif
    return;
  }
  if (selected == Device::HPC)
    throw std::logic_error("configure_device: HPC handled before local setup");
  throw std::logic_error("configure_device: unreachable Device");
}

void ensure_output(std::ofstream &stream, const std::filesystem::path &path)
{
  if (!stream.is_open()) throw IOError("Result::save: cannot open " + path.string());
}

} // namespace

Dataset::Dataset(std::filesystem::path source, int skip_cols, char delimiter,
                 std::string name)
  : source_(std::move(source)), skip_cols_(skip_cols), delimiter_(delimiter),
    name_(std::move(name))
{}

Dataset::Dataset(series_type source, int skip_cols, char delimiter, std::string name)
  : source_(std::move(source)), skip_cols_(skip_cols), delimiter_(delimiter),
    name_(std::move(name))
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

Data Dataset::materialize_local() const
{
  if (is_path()) {
    DataLoader loader(path());
    loader.start_column(skip_cols_).verbosity(0);
    if (delimiter_ != 0) loader.delimiter(delimiter_);
    try {
      return loader.load_local();
    } catch (const Error &) {
      throw;
    } catch (const std::exception &e) {
      throw IOError("load: failed to read '" + path().string() + "': " + e.what());
    }
  }

  auto series = std::get<series_type>(source_);
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

Dataset load(const std::filesystem::path &source, int skip_cols, char delimiter,
             std::string_view name)
{
  if (skip_cols < 0) throw InvalidInput("load: skip_cols must be non-negative.");
  std::string resolved = name.empty() ? derive_name(source) : std::string(name);
  return Dataset(source, skip_cols, delimiter, std::move(resolved));
}

Dataset load(Dataset::series_type source, int skip_cols, char delimiter,
             std::string_view name)
{
  if (skip_cols < 0) throw InvalidInput("load: skip_cols must be non-negative.");
  return Dataset(std::move(source), skip_cols, delimiter,
                 name.empty() ? "dataset" : std::string(name));
}

std::string device(std::string_view name)
{
  env().set_device(name);
  return canonical_device_name(env());
}

std::string device() { return canonical_device_name(env()); }

Result::Result(std::shared_ptr<Problem> problem, double cost, std::string device_name)
  : problem_(std::move(problem)), cost_(cost), device_(std::move(device_name))
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

void Result::save(const std::filesystem::path &directory) const
{
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec)
    throw IOError("Result::save: cannot create '" + directory.string() + "': "
                  + ec.message());

  const auto base = directory / problem_->name();
  const auto labels_path = std::filesystem::path(base.string() + "_labels.csv");
  const auto medoids_path = std::filesystem::path(base.string() + "_medoids.csv");
  const auto matrix_path = std::filesystem::path(base.string() + "_distance_matrix.csv");
  const auto silhouettes_path =
    std::filesystem::path(base.string() + "_silhouettes.csv");

  {
    std::ofstream out(labels_path);
    ensure_output(out, labels_path);
    out << "name,cluster\n";
    for (std::size_t i = 0; i < labels().size(); ++i)
      out << problem_->series_name(i) << ',' << labels()[i] << '\n';
  }
  {
    std::ofstream out(medoids_path);
    ensure_output(out, medoids_path);
    out << "cluster,medoid_index,medoid_name\n";
    for (std::size_t c = 0; c < medoids().size(); ++c) {
      const int idx = medoids()[c];
      out << c << ',' << idx << ','
          << problem_->series_name(static_cast<std::size_t>(idx)) << '\n';
    }
  }

  // save() promises the complete matrix and silhouettes. Matrix-free methods
  // retain their scaling until this explicitly requested operation.
  problem_->fill_distance_matrix();
  std::visit([&](const auto &matrix) {
    core::detail::preflight_distance_matrix_csv(matrix);
    std::ofstream out(
      matrix_path, std::ios::out | std::ios::binary | std::ios::trunc);
    ensure_output(out, matrix_path);
    out << matrix;
    out.close();
    if (!out)
      throw IOError("Result::save: cannot write " + matrix_path.string());
  }, problem_->distance_matrix());

  const auto silhouette_values = scores::silhouette(*problem_);
  {
    std::ofstream out(silhouettes_path);
    ensure_output(out, silhouettes_path);
    out << "name,cluster,silhouette\n";
    for (std::size_t i = 0; i < silhouette_values.size(); ++i)
      out << problem_->series_name(i) << ',' << labels()[i] << ','
          << std::setprecision(8) << silhouette_values[i] << '\n';
  }
}

Result cluster(const Dataset &dataset, int k, std::string_view requested_method,
               int band, std::string_view requested_device, int max_iter)
{
  validate_common(dataset, k, max_iter);
  std::string method = normalize_method(requested_method);

  // A per-call override is validated with a local Env so it neither changes nor
  // depends on the process default. The local Env shares the configured .env
  // location and keeps the no-silent-fallback device checks.
  Device selected = env().device();
  int device_index = env().device_index();
  std::string selected_name = canonical_device_name(env());
  if (!requested_device.empty()) {
    Env local;
    local.set_env_file_dir(env().env_file_dir());
    local.set_device(requested_device);
    selected = local.device();
    device_index = local.device_index();
    selected_name = canonical_device_name(local);
  }

  if (selected == Device::HPC) {
    throw DeviceError(
      "cluster: C++ Tier-1 HPC submission is beta and requires the repository "
      "SLURM transport wrapper; use dtwcpp.cluster(..., device='hpc') or "
      "scripts/slurm/slurm_remote.sh. No local fallback was attempted.");
  }

  auto problem = std::make_shared<Problem>(dataset.name());
  problem->set_data(dataset.materialize_local());
  if (problem->size() == 0) throw InvalidInput("cluster: dataset is empty.");
  if (static_cast<std::size_t>(k) > problem->size())
    throw InvalidInput("cluster: k must not exceed the number of series.");
  problem->set_band(band);
  problem->set_max_iter(max_iter);
  configure_device(*problem, selected, device_index);

  const auto execution_target = selected == Device::GPU
    ? detail::Tier1ExecutionTarget::GPU
    : detail::Tier1ExecutionTarget::CPU;
  method = detail::resolve_tier1_method(method, problem->size(), execution_target);

  core::ClusteringResult result;
  const bool matrix_free = method == "onebatch" || method == "clara"
                        || method == "tadpole";
  if (matrix_free && selected == Device::GPU) {
    throw DeviceError(
      "cluster: method='" + method
      + "' uses a matrix-free CPU distance schedule; GPU execution is not "
        "implemented for that schedule. Use device='cpu'.");
  }
  if (!matrix_free) problem->fill_distance_matrix();

  if (method == "pam") {
    result = fast_pam_seeded(
      *problem, k, settings::DEFAULT_RANDOM_SEED, max_iter);
  } else if (method == "onebatch") {
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = k;
    options.max_iter = max_iter;
    options.random_seed = settings::DEFAULT_RANDOM_SEED;
    result = algorithms::one_batch_pam(*problem, options);
  } else if (method == "clara") {
    algorithms::CLARAOptions options;
    options.n_clusters = k;
    options.max_iter = max_iter;
    options.random_seed = settings::DEFAULT_RANDOM_SEED;
    result = algorithms::fast_clara(*problem, options);
  } else if (method == "hierarchical") {
    const auto dendrogram = algorithms::build_dendrogram(*problem);
    result = algorithms::cut_dendrogram(dendrogram, *problem, k);
  } else {
    problem->set_n_clusters(k);
    if (method == "mip") problem->set_method(Method::MIP);
    else if (method == "lrcore") problem->set_method(Method::LRCore);
    else if (method == "tadpole") problem->set_method(Method::TADPole);
    else problem->set_method(Method::Kmedoids);
    problem->cluster();
    result.labels = problem->labels();
    result.medoid_indices = problem->medoids();
    result.total_cost = problem->find_total_cost();
  }

  return Result(std::move(problem), result.total_cost, std::move(selected_name));
}

} // namespace dtwc
