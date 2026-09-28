/**
 * @file run.cpp
 * @brief dtwc::run: the checks, the method x device resolution, the load, the
 *        distance storage, the dispatch and the outputs of one clustering run.
 *
 * @details Moved from dtwc_cl.cpp's main() and api.cpp's cluster(), which both
 * call it now (IF-2 S3), so each rule has one copy. Progress lines go to stdout
 * under `verbose`; the line that announces a --resume replay is unconditional.
 *
 * @date 24 Sep 2026
 */

#include "run.hpp"

#include "../DataLoader.hpp"
#include "../Problem.hpp"
#include "../algorithms/detail/fast_clara_plan.hpp"
#include "../algorithms/fast_clara.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../algorithms/hierarchical.hpp"
#include "../algorithms/one_batch_pam.hpp"
#include "../base/error.hpp"
#include "../base/timing.hpp"
#include "../checkpoint.hpp"
#include "../fileOperations.hpp"
#include "../scores.hpp"
#ifdef DTWC_HAS_MMAP
#include "../core/mmap_data_store.hpp"
#endif
#ifdef DTWC_HAS_ARROW
#include "../io/arrow_ipc_reader.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#include "../io/parquet_reader.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dtwc {
namespace {

namespace fs = std::filesystem;
using algorithms::detail::resolve_clara_plan;

constexpr std::size_t auto_pam_max_series = 5000; ///< `auto` on the CPU: pam up to here, clara above.

std::string method_name(ClusterMethod method) { return std::string(name_of(cluster_method_names, method)); }

/// `auto` for the device and the series count; any other method as asked.
ClusterMethod resolve_method(ClusterMethod method, Device device, std::size_t n_series)
{
  if (method != ClusterMethod::Auto) return method;
  return device == Device::GPU || n_series <= auto_pam_max_series ? ClusterMethod::PAM : ClusterMethod::CLARA;
}

[[noreturn]] void refuse_gpu_method(ClusterMethod method)
{
  throw DeviceError(
    "run: method '" + method_name(method)
    + "' computes its distances on the CPU as it goes, so device 'gpu' would sit idle; the GPU fills the "
      "distance matrix that pam, kmedoids, mip, lrcore and hierarchical use (and clara when its sample covers "
      "every series). Choose one of those, or device 'cpu'. No CPU fallback was attempted.");
}

/// The Problem::cluster() route of a method, and its progress label.
std::pair<Method, const char *> problem_route(ClusterMethod method)
{
  switch (method) {
  case ClusterMethod::Kmedoids: return { Method::Kmedoids, "kMedoids Lloyd" };
  case ClusterMethod::MIP: return { Method::MIP, "MIP clustering" };
  case ClusterMethod::LRCore: return { Method::LRCore, "LR-core clustering" };
  default: return { Method::TADPole, "TADPole clustering" };
  }
}

/// Where the series come from.
enum class Source { Text, ParquetFile, ParquetDirectory, ArrowIPC, Dtws, Memory };

struct Input
{
  Source source = Source::Memory;
  fs::path path;
  std::vector<fs::path> parquet_files; ///< a Parquet directory's files, sorted
  bool parquet() const { return source == Source::ParquetFile || source == Source::ParquetDirectory; }
};

std::string lower_extension(const fs::path &path)
{
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

/// Classify the input by filesystem inspection alone, before any reader runs.
Input classify(const std::string &input_text)
{
  Input input;
  input.path = utf8_to_path(input_text);
  std::error_code ec;
  if (fs::is_directory(input.path, ec)) {
    fs::directory_iterator entries(input.path, ec);
    if (ec) throw IOError("load: cannot list '" + input_text + "': " + ec.message());
    for (const auto &entry : entries) {
      const auto ext = lower_extension(entry.path());
      if (ext == ".parquet" || ext == ".pq") input.parquet_files.push_back(entry.path());
    }
    std::sort(input.parquet_files.begin(), input.parquet_files.end());
    input.source = input.parquet_files.empty() ? Source::Text : Source::ParquetDirectory;
    return input;
  }
  const auto ext = lower_extension(input.path);
  if (ext == ".parquet" || ext == ".pq") input.source = Source::ParquetFile;
  else if (ext == ".arrow" || ext == ".ipc" || ext == ".feather") input.source = Source::ArrowIPC;
  else if (ext == ".dtws") input.source = Source::Dtws;
  else input.source = Source::Text; // anything a typed reader does not claim goes to the CSV/TSV DataLoader
  return input;
}

/// Reject an input format whose reader this binary does not contain: IOError,
/// as for `.dtws` without llfio (api-contract-2.0.md §5). The rejection must
/// exist in the build that LACKS the capability, so it sits under `#ifndef`:
/// inside `#ifdef DTWC_HAS_PARQUET` it would be absent from the
/// `DTWC_ENABLE_ARROW=OFF` build, and a Parquet file would reach the CSV
/// DataLoader, parsed as text (LESSONS F9).
void require_input_format_is_built([[maybe_unused]] Source source)
{
#ifndef DTWC_HAS_PARQUET
  if (source == Source::ParquetFile || source == Source::ParquetDirectory)
    throw IOError(
      "Parquet input (.parquet/.pq) requires a build with Arrow/Parquet "
      "(-DDTWC_ENABLE_ARROW=ON). This binary was built without Parquet "
      "support; convert the input to CSV/TSV or use an Arrow-enabled build.");
#endif
#ifndef DTWC_HAS_ARROW
  if (source == Source::ArrowIPC)
    throw IOError(
      "Arrow IPC input (.arrow/.ipc/.feather) requires a build with Arrow "
      "(-DDTWC_ENABLE_ARROW=ON). This binary was built without Arrow support; "
      "convert the input to CSV/TSV or use an Arrow-enabled build.");
#endif
#ifndef DTWC_HAS_MMAP
  if (source == Source::Dtws)
    throw IOError(
      ".dtws memory-mapped input requires a build with llfio "
      "(-DDTWC_ENABLE_LLFIO=ON). This binary was built without mmap support.");
#endif
}

/// Reject a reader option the input cannot honour: accepting and then ignoring
/// it would be a silent deceit. `--ram-limit` is the Parquet materialisation cap.
void require_input_options_apply(const Config &config, const Input &input)
{
  if (input.source == Source::Memory && !config.input.empty())
    throw InvalidInput("run: the series are passed in memory, so --input '" + config.input
                       + "' would not be read; leave input empty.");
  if (config.ram_limit != 0 && !input.parquet())
    throw InvalidInput(
      "--ram-limit caps Parquet series materialisation and cannot be honoured "
      "for this input; drop --ram-limit, or convert the series to a "
      "list-per-row Parquet file to stream them under the cap.");
  if (!config.column.empty() && !input.parquet())
    throw InvalidInput(
      "--column selects a Parquet column and cannot be honoured for this "
      "input; drop --column, or pass a .parquet/.pq file or directory.");
  if ((config.skip_rows != 0 || config.skip_cols != 0 || config.delimiter != '\0') && input.source != Source::Text)
    throw InvalidInput(
      "--skip-rows, --skip-cols and --delimiter are CSV/TSV parsing options and "
      "cannot be honoured for this input; drop them, or pass a text input.");
}

/// Read the series a path names. A read failure is an IOError naming the file;
/// other typed errors pass unchanged.
Data read_series(const Config &config, const Input &input, std::string &from)
{
  try {
    switch (input.source) {
#ifdef DTWC_HAS_PARQUET
    case Source::ParquetDirectory:
      from = " from Parquet directory";
      return io::load_parquet_directory(input.path, config.column);
    case Source::ParquetFile:
      from = " from Parquet";
      return io::load_parquet_file(input.path, config.column);
#endif
#ifdef DTWC_HAS_MMAP
    case Source::Dtws: { // copied out of the map (a mapped series store is a later step)
      from = " from .dtws cache";
      auto store = core::MmapDataStore::open(input.path);
      std::vector<std::vector<data_t>> series(store.size());
      for (std::size_t i = 0; i < series.size(); ++i) {
        const auto values = store.series(i);
        series[i].assign(values.begin(), values.end());
      }
      std::vector<std::string> names(series.size());
      const fs::path names_path = input.path.string() + ".names"; // the sidecar, when present
      std::error_code ec;
      if (fs::exists(names_path, ec)) {
        std::ifstream name_file(names_path);
        for (std::size_t i = 0; i < names.size() && std::getline(name_file, names[i]); ++i) {}
      } else {
        for (std::size_t i = 0; i < names.size(); ++i) names[i] = "series_" + std::to_string(i);
      }
      return Data(std::move(series), std::move(names), store.ndim());
    }
#endif
#ifdef DTWC_HAS_ARROW
    case Source::ArrowIPC: { // copied out of the map, as .dtws
      from = " from Arrow IPC";
      auto source = io::ArrowIPCDataSource::open(input.path);
      std::vector<std::vector<data_t>> series(source.size());
      for (std::size_t i = 0; i < series.size(); ++i) {
        const auto values = source.series(i);
        series[i].assign(values.begin(), values.end());
      }
      return Data(std::move(series), source.all_names(), source.ndim());
    }
#endif
    default: { // Text; require_input_format_is_built() refused the rest
      from.clear();
      DataLoader loader{ input.path };
      loader.start_column(config.skip_cols).start_row(config.skip_rows).verbosity(config.verbose ? 1 : 0);
      if (config.delimiter != '\0') loader.delimiter(config.delimiter);
      return loader.load_local();
    }
    }
  } catch (const IOError &e) {
    throw IOError("load: failed to read '" + config.input + "': " + e.what());
  } catch (const Error &) {
    throw;
  } catch (const std::exception &e) {
    throw IOError("load: failed to read '" + config.input + "': " + e.what());
  }
}

/// An owning Float32 copy of Float64 series.
Data convert_to_f32(const Data &data_f64)
{
  std::vector<std::vector<float>> series(data_f64.size());
  for (std::size_t i = 0; i < series.size(); ++i) {
    const auto &source = data_f64.p_vec[i];
    series[i].assign(source.begin(), source.end());
  }
  auto names = data_f64.p_names;
  return Data(std::move(series), std::move(names), data_f64.ndim);
}

#ifdef DTWC_HAS_PARQUET
std::size_t checked_parquet_series_count(std::int64_t count)
{
  if (count < 0 || static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max())
    throw InvalidInput("Parquet logical series count exceeds this platform's size limit.");
  return static_cast<std::size_t>(count);
}
#endif

/// Bind persistent distance storage once every distance-affecting setting has
/// reached the Problem. Returns the mmap cache path, or nullopt when the method
/// keeps its own storage or N is below the threshold. OneBatchPAM owns a fixed
/// O(Nm) table and never calls Problem::dist_by_ind(); non-full FastCLARA reads
/// no parent matrix. TADPole is NOT exempt: its exact distances go through
/// dist_by_ind(), whose dense cache would allocate packed O(N^2) doubles.
std::optional<fs::path> configure_distance_storage(Problem &prob, const Config &config, ClusterMethod method,
                                                   bool clara_uses_full_sample, const fs::path &cache_path)
{
  const bool checkpoint = !config.checkpoint.empty();
  const bool dist_matrix = !config.dist_matrix.empty();
  if (method == ClusterMethod::CLARA && !clara_uses_full_sample) {
    if (checkpoint || dist_matrix)
      throw InvalidInput(
        "Non-full FastCLARA does not consume a parent distance matrix; "
        "--checkpoint and --dist-matrix would load or save unused O(N^2) "
        "state. Omit those options, or request a full sample deliberately.");
    return std::nullopt;
  }
  if (method == ClusterMethod::OneBatch) return std::nullopt;
  if (config.mmap_threshold != 0 && prob.size() < config.mmap_threshold) return std::nullopt;
  if (checkpoint)
    throw InvalidInput(
      "--checkpoint uses the legacy dense CSV checkpoint format and cannot be "
      "combined with memory-mapped distance storage. The mmap cache already "
      "resumes automatically; omit --checkpoint, or raise --mmap-threshold if "
      "the dense matrix and CSV checkpoint fit in RAM.");
  if (dist_matrix)
    throw InvalidInput(
      "--dist-matrix uses a legacy dense CSV matrix and cannot be combined "
      "with memory-mapped distance storage. Omit --dist-matrix to resume the "
      "fingerprinted mmap cache, or raise --mmap-threshold if importing the "
      "dense CSV matrix fits in RAM.");
#ifdef DTWC_HAS_MMAP
  prob.use_mmap_distance_matrix(cache_path);
  return cache_path;
#else
  (void)cache_path;
  throw IOError("method='" + method_name(method) + "' at N=" + std::to_string(prob.size())
                + " requires memory-mapped distance storage because --mmap-threshold="
                + std::to_string(config.mmap_threshold)
                + " was reached, but this binary was built without mmap support. Rebuild "
                  "with -DDTWC_ENABLE_LLFIO=ON, raise --mmap-threshold only if the packed "
                  "heap matrix fits in RAM, or use --method onebatch.");
#endif
}

/// Binary v1 has no input or configuration identity, so a replay checks only
/// what can be proven without changing the frozen format.
void validate_resume_result(const core::ClusteringResult &result, std::size_t expected_series,
                            int expected_clusters)
{
  const auto fail = [](const std::string &message) { throw InvalidInput("Binary result checkpoint " + message); };
  if (result.labels.size() != expected_series)
    fail("has " + std::to_string(result.labels.size()) + " labels; current input has "
         + std::to_string(expected_series) + " series.");
  if (result.medoid_indices.size() != static_cast<std::size_t>(expected_clusters))
    fail("has " + std::to_string(result.medoid_indices.size()) + " medoids; --n-clusters requests "
         + std::to_string(expected_clusters) + ".");
  for (std::size_t i = 0; i < result.labels.size(); ++i)
    if (result.labels[i] < 0 || result.labels[i] >= expected_clusters)
      fail("label[" + std::to_string(i) + "]=" + std::to_string(result.labels[i]) + " is outside [0,"
           + std::to_string(expected_clusters) + ").");
  for (std::size_t i = 0; i < result.medoid_indices.size(); ++i) {
    const int medoid = result.medoid_indices[i];
    if (medoid < 0 || static_cast<std::size_t>(medoid) >= expected_series)
      fail("medoid[" + std::to_string(i) + "]=" + std::to_string(medoid) + " is outside [0,"
           + std::to_string(expected_series) + ").");
    for (std::size_t previous = 0; previous < i; ++previous)
      if (result.medoid_indices[previous] == medoid)
        fail("medoid index " + std::to_string(medoid) + " is duplicated.");
  }
  if (result.iterations < 0) fail("iteration count " + std::to_string(result.iterations) + " is negative.");
  if (!std::isfinite(result.total_cost)) fail("total cost is not finite.");
}

// ---- outputs ---------------------------------------------------------------

/// A series' name in the outputs. A streamed Parquet run holds no series, so
/// its names are the readers' own synthetic `series_<i>`.
std::string output_series_name(const Problem &prob, std::size_t index, std::optional<std::size_t> streamed_count)
{
  // Programming errors: a result comes from an algorithm run on this input or
  // from a --resume that validate_resume_result() checked against it.
  const std::size_t n = streamed_count.value_or(prob.size());
  if (index >= n)
    throw std::logic_error("Result index " + std::to_string(index) + " is outside the " + std::to_string(n)
                           + "-series input.");
  return streamed_count ? "series_" + std::to_string(index) : std::string(prob.series_name(index));
}

std::ofstream open_output(const fs::path &path)
{
  std::ofstream out(path);
  if (!out.is_open())
    throw IOError("Cannot open output file '" + path.string()
                  + "' for writing; check that the --output directory is writable.");
  return out;
}

/// A full disk or a file-size quota fails the writes AFTER a successful open,
/// so checking only is_open() left a truncated file behind a zero exit (B-05).
void close_output(std::ofstream &out, const fs::path &path)
{
  out.close();
  if (!out)
    throw IOError("Write error on output file '" + path.string()
                  + "': the file is incomplete (disk full or file-size quota?). Free space or choose "
                    "another --output directory, then rerun.");
}

void write_labels_csv(const fs::path &path, const Problem &prob, const core::ClusteringResult &result,
                      std::optional<std::size_t> streamed_count)
{
  const std::size_t expected = streamed_count.value_or(prob.size());
  if (result.labels.size() != expected) // programming error, as in output_series_name
    throw std::logic_error("Clustering result has " + std::to_string(result.labels.size()) + " labels for a "
                           + std::to_string(expected) + "-series input.");
  std::ofstream out = open_output(path);
  out << "name,cluster\n";
  for (std::size_t i = 0; i < result.labels.size(); ++i)
    out << output_series_name(prob, i, streamed_count) << "," << result.labels[i] << "\n";
  close_output(out, path);
}

void write_medoids_csv(const fs::path &path, const Problem &prob, const core::ClusteringResult &result,
                       std::optional<std::size_t> streamed_count)
{
  for (const int index : result.medoid_indices) { // every name before the file is truncated
    if (index < 0) throw std::logic_error("Result medoid index " + std::to_string(index) + " is negative.");
    (void)output_series_name(prob, static_cast<std::size_t>(index), streamed_count);
  }
  std::ofstream out = open_output(path);
  out << "cluster,medoid_index,medoid_name\n";
  for (int c = 0; c < result.n_clusters(); ++c) {
    const int index = result.medoid_indices[c];
    out << c << "," << index << "," << output_series_name(prob, static_cast<std::size_t>(index), streamed_count)
        << "\n";
  }
  close_output(out, path);
}

void write_silhouettes_csv(const fs::path &path, const std::vector<double> &silhouettes, const Problem &prob,
                           const core::ClusteringResult &result)
{
  std::ofstream out = open_output(path);
  out << "name,cluster,silhouette\n";
  for (std::size_t i = 0; i < silhouettes.size(); ++i)
    out << prob.series_name(i) << "," << result.labels[i] << "," << std::setprecision(8) << silhouettes[i] << "\n";
  close_output(out, path);
}

/// What a run hands to its Result.
struct Outcome
{
  std::shared_ptr<Problem> problem;
  core::ClusteringResult result;
  ClusterMethod method;
};

/// The run itself; `data` holds in-memory series, else config.input is read.
Outcome execute(const Config &config, std::optional<Data> data)
{
  // ---- 1. Every check that needs no series, before any file is touched ----
  if (!data && config.input.empty())
    throw InvalidInput("--input is required via CLI or config file (TOML or YAML)");
  if (config.k < 1)
    throw InvalidInput("-k/--n-clusters must be a positive integer, got " + std::to_string(config.k));
  if (config.n_init < 1) throw InvalidInput("--n-init must be a positive integer");
  if (config.max_iter < 1)
    throw InvalidInput("--max-iter must be a positive integer, got " + std::to_string(config.max_iter));
  if (config.checkpoint_interval != 0 && config.checkpoint.empty())
    throw InvalidInput("--checkpoint-interval requires --checkpoint <dir>.");
  if (config.resume && config.output.empty())
    throw InvalidInput("--resume replays <output>/<name>_checkpoint.bin, so it needs --output.");
  if (config.device == Device::HPC)
    throw DeviceError(
      "run: device 'hpc' submits a run to a SLURM cluster, which Python's dtwcpp.cluster(..., "
      "device='hpc') and slurm_remote.sh submit-cluster do; dtwc_cl and dtwc::run compute where they "
      "start. No local fallback was attempted.");

  auto problem = std::make_shared<Problem>(config.name);
  Problem &prob = *problem;
  // The Problem validates every distance setting as it takes it: parameter
  // domains, variant x missing strategy x metric, the MIP settings, the GPU.
  prob.set_variant(config.variant);
  prob.set_missing_strategy(config.missing);
  prob.set_metric(config.metric);
  prob.set_band(config.band);
  prob.set_max_iter(config.max_iter);
  prob.set_n_repetitions(config.n_init);
  prob.set_random_seed(config.seed);
  prob.set_tadpole_dc(config.tadpole_dc); // < 0: auto-select from a DTW subsample
  prob.set_verbose(config.verbose);
  validate_mip_settings(config.mip);
  prob.mip_settings = config.mip;
  // A false set_solver means HiGHS: --solver gurobi on a build without Gurobi
  // must not solve with HiGHS.
  if (!prob.set_solver(config.solver))
    throw SolverError("--solver " + std::string(name_of(solver_names, config.solver))
                      + " is not available: this dtwc_cl was built without Gurobi. Use --solver highs, or "
                        "rebuild with -DDTWC_ENABLE_GUROBI=ON and GUROBI_HOME set.");
  prob.set_cuda_settings(config.gpu);
  prob.set_device(config.device, config.gpu.device_id); // gpu without a GPU backend: §6.1's DeviceError
  if (config.device == Device::GPU
      && (config.method == ClusterMethod::OneBatch || config.method == ClusterMethod::TADPole))
    refuse_gpu_method(config.method);
  validate_gpu_request("run", prob.distance_strategy, config.variant, config.missing, config.dtype, config.gpu);

  algorithms::CLARAOptions clara;
  clara.n_clusters = config.k;
  clara.sample_size = config.sample_size;
  clara.n_samples = config.n_samples;
  clara.max_iter = config.max_iter;
  clara.random_seed = config.seed;
  if (config.method == ClusterMethod::CLARA) algorithms::detail::validate_clara_controls(clara, "run");

  const Input input = data ? Input{} : classify(config.input);
  require_input_format_is_built(input.source);
  require_input_options_apply(config, input);

  // A --checkpoint that cannot hold a checkpoint stops the run here, before any
  // work: the save comes after clustering, so a typo cost the whole run.
  const fs::path checkpoint_dir = utf8_to_path(config.checkpoint);
  if (!config.checkpoint.empty()) {
    std::error_code ec;
    fs::create_directories(checkpoint_dir, ec);
    if (ec || !fs::is_directory(checkpoint_dir, ec))
      throw IOError("--checkpoint '" + config.checkpoint + "' is not a usable directory ("
                    + (ec ? ec.message() : std::string("not a directory"))
                    + "); pass a directory path (it is created if missing), or omit --checkpoint.");
  }
  const fs::path output = utf8_to_path(config.output);
  if (!config.output.empty()) {
    std::error_code ec;
    fs::create_directories(output, ec);
    if (ec) throw IOError("--output '" + config.output + "' cannot be created: " + ec.message());
    prob.set_output_folder(output);
  }

  // ---- 2. The method and, from Parquet metadata when it has them, N ----
  const Clock clk;
  ClusterMethod method = config.method;
  bool stream_payload = false;
  std::size_t n_series = 0;
  bool clara_uses_full_sample = false;
  bool clara_planned = false;
  const auto plan_clara = [&] {
    if (method != ClusterMethod::CLARA || clara_planned) return;
    const auto plan = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
    clara_uses_full_sample = plan.sample_size == plan.n_points;
    if (stream_payload) algorithms::detail::validate_streaming_clara_plan(plan, "run");
    if (config.device == Device::GPU && !clara_uses_full_sample) refuse_gpu_method(method);
    clara_planned = true;
  };

#ifdef DTWC_HAS_PARQUET
  // A cap changes the load decision, so only the metadata is read here: the
  // readers map the file and its footer; no row group is decoded.
  if (input.parquet()
      && (config.ram_limit > 0 || method == ClusterMethod::Auto || method == ClusterMethod::CLARA)) {
    const auto saturating_add = [](std::size_t lhs, std::size_t rhs) {
      return rhs > std::numeric_limits<std::size_t>::max() - lhs ? std::numeric_limits<std::size_t>::max()
                                                                  : lhs + rhs;
    };
    const bool f32 = config.dtype == core::Precision::Float32;
    auto layout = detail::ParquetLayout::Directory;
    std::size_t resident_bytes = 0;
    if (input.source == Source::ParquetFile) {
      io::ParquetChunkReader metadata(input.path, config.column);
      layout = metadata.is_list_layout() ? detail::ParquetLayout::ListColumn : detail::ParquetLayout::ScalarColumn;
      n_series = checked_parquet_series_count(metadata.logical_series_count());
      resident_bytes = metadata.estimated_materialization_peak_bytes(f32);
    } else {
      for (const auto &path : input.parquet_files) {
        io::ParquetChunkReader metadata(path, config.column);
        n_series = saturating_add(n_series, checked_parquet_series_count(metadata.logical_series_count()));
        resident_bytes = saturating_add(resident_bytes, metadata.estimated_materialization_peak_bytes(f32));
      }
    }
    const auto plan = detail::plan_parquet_load(method, config.device, n_series, resident_bytes, config.ram_limit, layout);
    method = plan.method;
    stream_payload = plan.stream_payload;
    if (stream_payload) {
      clara.ram_limit_bytes = config.ram_limit;
      clara.parquet_path = input.path;
      clara.parquet_column = config.column;
      clara.use_float32 = f32;
      clara.force_parquet_streaming = true;
    }
    plan_clara();
    if (stream_payload && !config.checkpoint.empty())
      throw InvalidInput(
        "--checkpoint requires resident series data and cannot be combined "
        "with RAM-limited Parquet streaming; the binary clustering-result "
        "checkpoint is still written automatically.");
    if (stream_payload && !config.dist_matrix.empty())
      throw InvalidInput(
        "--dist-matrix requires resident series data and cannot be combined "
        "with RAM-limited Parquet streaming.");
    if (config.verbose && stream_payload)
      std::cout << "Parquet metadata selected streaming: " << n_series << " series, ~"
                << (resident_bytes / (1ULL << 20)) << " MB resident estimate exceeds the series-data cap ["
                << clk << "]\n";
  }
#endif

  // ---- 3. Load: series storage follows the device (the GPU uploads heap series) ----
  prob.set_storage_policy(config.device == Device::GPU ? core::StoragePolicy::Heap : core::StoragePolicy::Auto);
  if (!stream_payload) {
    std::string from = " in memory";
    Data series = data ? std::move(*data) : read_series(config, input, from);
    if (config.verbose) {
      std::cout << "Data loaded" << from << ": " << series.size() << " series [" << clk << "]\n";
      if (series.size() > 0) {
        std::size_t elements = 0;
        for (std::size_t i = 0; i < series.size(); ++i) elements += series.series_length(i) * series.ndim;
        std::cout << "  Data memory: ~" << (elements * sizeof(data_t) / (1ULL << 20)) << " MB (" << series.size()
                  << " series, " << (elements / series.size()) << " avg length, "
                  << name_of(core::precision_names, config.dtype) << ")\n";
      }
    }
    if (config.dtype == core::Precision::Float32 && !series.is_f32() && !series.is_view()) {
      series = convert_to_f32(series);
      if (config.verbose) std::cout << "Converted to float32 (2x memory saving)\n";
    }
    prob.set_data(std::move(series));
    n_series = prob.size();
  }
  if (config.ram_limit > 0 && config.verbose) std::cout << "Series-data RAM limit: " << config.ram_limit << " bytes\n";
  if (n_series == 0) throw InvalidInput("cluster: dataset is empty.");
  if (static_cast<std::size_t>(config.k) > n_series)
    throw InvalidInput("cluster: k must not exceed the number of series.");

  method = resolve_method(method, config.device, n_series);
  if (config.method == ClusterMethod::Auto && config.verbose)
    std::cout << "Auto-selected method: " << method_name(method) << " (N=" << n_series << ")\n";
  plan_clara();

  // ---- 4. A --resume replays a completed result instead of computing one ----
  const fs::path binary_checkpoint = output / utf8_to_path(config.name + "_checkpoint.bin");
  std::optional<core::ClusteringResult> replay;
  if (config.resume) {
    core::ClusteringResult candidate;
    if (!load_binary_checkpoint(candidate, binary_checkpoint))
      throw InvalidInput("--resume requires a readable binary result checkpoint at '" + binary_checkpoint.string()
                         + "'. Omit --resume to start a new clustering run.");
    validate_resume_result(candidate, n_series, config.k);
    replay.emplace(std::move(candidate));
  }

  // ---- 5. Distance storage, once every distance setting is in place ----
  // The mmap cache lives in the output directory, so a run that writes nothing
  // keeps its matrix in RAM. A replay creates no unused O(N^2) state, but an
  // existing cache still reopens for scoring; an imported matrix or a
  // checkpoint uses dense storage.
  if (!config.output.empty()) {
    const fs::path mmap_cache = output / utf8_to_path(config.name + "_distmat.cache");
    std::error_code ec;
    const bool reopen = replay && config.checkpoint.empty() && config.dist_matrix.empty()
                        && fs::is_regular_file(mmap_cache, ec);
    if (!replay || reopen) {
      const auto cache = configure_distance_storage(prob, config, method, clara_uses_full_sample, mmap_cache);
      if (cache && config.verbose) std::cout << "Using memory-mapped distance matrix: " << *cache << "\n";
    }
  }
  // A matrix the user supplied but that cannot be loaded is an error: going on
  // without it silently recomputed every distance and exited 0 (S-04).
  if (!config.dist_matrix.empty()) {
    try {
      prob.read_distance_matrix(utf8_to_path(config.dist_matrix));
    } catch (const std::exception &e) {
      throw IOError("--dist-matrix '" + config.dist_matrix
                    + "' cannot be loaded; fix the file, or omit --dist-matrix to compute the distances. Cause: "
                    + e.what());
    }
    if (config.verbose) std::cout << "Loaded distance matrix from " << config.dist_matrix << "\n";
  }
  if (config.checkpoint_interval != 0) { // 0: saved once, at the end
    prob.checkpoint.directory = config.checkpoint;
    prob.checkpoint.save_interval = config.checkpoint_interval;
    prob.checkpoint.enabled = true;
  }
  if (!config.checkpoint.empty()) {
    const bool resumed = load_checkpoint(prob, config.checkpoint); // false: start fresh
    if (config.verbose)
      std::cout << (resumed ? "Resumed from checkpoint: " + config.checkpoint + "\n"
                            : "No valid checkpoint found at " + config.checkpoint + ", starting fresh.\n");
  }

  // ---- 6. Cluster; the matrix methods fill through the Problem, on its device ----
  core::ClusteringResult result;
  const int k = config.k;
  if (replay) {
    result = std::move(*replay);
    std::cout << "Replaying completed result checkpoint: N=" << result.labels.size()
              << ", k=" << result.medoid_indices.size() << ", iterations=" << result.iterations
              << ", converged=" << (result.converged ? "yes" : "no") << " (requested method="
              << method_name(method) << "; binary v1 has no method provenance)\n";
  } else {
    switch (method) {
    case ClusterMethod::PAM:
      if (config.verbose) std::cout << "Running FastPAM (k=" << k << ") ...\n";
      // Restart r uses seed + r, invocation-local; the strictly lowest cost is
      // kept, so a tie keeps the earlier restart.
      result = fast_pam_seeded(prob, k, config.seed, config.max_iter);
      for (int restart = 1; restart < config.n_init; ++restart) {
        auto candidate =
          fast_pam_seeded(prob, k, std::uint64_t{ config.seed } + static_cast<std::uint64_t>(restart), config.max_iter);
        if (candidate.total_cost < result.total_cost) result = std::move(candidate);
      }
      if (config.verbose)
        std::cout << "FastPAM " << (result.converged ? "converged" : "did not converge") << " in "
                  << result.iterations << " iterations, cost=" << std::setprecision(6) << result.total_cost
                  << " [" << clk << "]\n";
      break;
    case ClusterMethod::OneBatch: {
      algorithms::OneBatchPAMOptions options;
      options.n_clusters = k;
      options.batch_size = config.batch_size;
      options.max_iter = config.max_iter;
      options.random_seed = config.seed;
      algorithms::OneBatchPAMStats stats;
      result = algorithms::one_batch_pam(prob, options, &stats);
      if (config.verbose)
        std::cout << "OneBatchPAM finished, cost=" << std::setprecision(6) << result.total_cost
                  << ", batch=" << stats.batch_size << ", distance-matrix fraction=" << std::setprecision(3)
                  << stats.full_matrix_fraction << " [" << clk << "]\n";
      break;
    }
    case ClusterMethod::CLARA:
      if (config.verbose) std::cout << "Running FastCLARA (k=" << k << ") ...\n";
      result = algorithms::fast_clara(prob, clara);
      if (config.verbose)
        std::cout << "FastCLARA finished, cost=" << std::setprecision(6) << result.total_cost << " [" << clk << "]\n";
      break;
    case ClusterMethod::Hierarchical: {
      if (config.verbose)
        std::cout << "Running hierarchical clustering (k=" << k
                  << ", linkage=" << name_of(algorithms::linkage_names, config.linkage) << ") ...\n";
      algorithms::HierarchicalOptions options;
      options.linkage = config.linkage;
      prob.fill_distance_matrix(); // the dendrogram reads every pair
      result = algorithms::cut_dendrogram(algorithms::build_dendrogram(prob, options), prob, k);
      if (config.verbose)
        std::cout << "Hierarchical clustering finished, cost=" << std::setprecision(6) << result.total_cost
                  << " [" << clk << "]\n";
      break;
    }
    // Problem::cluster()'s four: Lloyd, the exact MIP and LR-core, and
    // TADPole density-peaks with conditionally admissible LB/UB DTW pruning.
    case ClusterMethod::Kmedoids:
    case ClusterMethod::MIP:
    case ClusterMethod::LRCore:
    case ClusterMethod::TADPole: {
      const auto [problem_method, label] = problem_route(method);
      prob.set_n_clusters(k);
      prob.set_method(problem_method);
      prob.cluster();
      result.labels = prob.clusters_ind;
      result.medoid_indices = prob.centroids_ind;
      result.total_cost = prob.find_total_cost();
      // Lloyd reports its iterations; the exact methods and TADPole finish.
      result.iterations = method == ClusterMethod::Kmedoids ? prob.last_iterations() : 0;
      result.converged = method != ClusterMethod::Kmedoids || prob.last_iterations() < config.max_iter;
      if (config.verbose) std::cout << label << " finished, cost=" << result.total_cost << " [" << clk << "]\n";
      break;
    }
    case ClusterMethod::Auto: // resolved above
      throw std::logic_error("run: unresolved method auto");
    }
  }

  // ---- 7. Checkpoints first, then the outputs ----
  if (!replay && !config.output.empty()) save_binary_checkpoint(result, binary_checkpoint);
  prob.set_result(result); // the kept result, in every route and in a replay

  // Before the results, in fresh and replay runs alike, so a result write that
  // fails cannot lose the distance matrix. A save that fails is kept and raised
  // once the results are on disk, so it cannot lose them either (S-04).
  std::optional<std::string> checkpoint_failure;
  if (!config.checkpoint.empty()) {
    try {
      save_checkpoint(prob, config.checkpoint);
      if (config.verbose) std::cout << "Checkpoint saved to " << config.checkpoint << "\n";
    } catch (const std::exception &e) {
      checkpoint_failure = e.what();
    }
  }

  if (!config.output.empty()) {
    const auto streamed_count = stream_payload ? std::optional<std::size_t>{ n_series } : std::nullopt;
    const auto labels_path = output / utf8_to_path(config.name + "_labels.csv");
    write_labels_csv(labels_path, prob, result, streamed_count);
    if (config.verbose) std::cout << "Labels written to " << labels_path << "\n";
    const auto medoids_path = output / utf8_to_path(config.name + "_medoids.csv");
    write_medoids_csv(medoids_path, prob, result, streamed_count);
    if (config.verbose) std::cout << "Medoids written to " << medoids_path << "\n";

    // A matrix-free run does not fill an O(N^2) matrix merely to write these two
    // (api-contract-2.0.md, approved addendum 3).
    if (prob.is_distance_matrix_filled()) {
      prob.write_distance_matrix(config.name + "_distance_matrix.csv");
      if (config.verbose)
        std::cout << "Distance matrix written to " << output / utf8_to_path(config.name + "_distance_matrix.csv")
                  << "\n";
      // A score that cannot be computed is a warning; a computed score that
      // cannot be written is an error like any other file (B-05).
      if (k > 1) {
        std::optional<std::vector<double>> silhouettes;
        try {
          silhouettes = scores::silhouette(prob);
        } catch (const std::exception &e) {
          std::cerr << "Warning: Could not compute silhouette scores: " << e.what() << "\n";
        }
        if (silhouettes) {
          write_silhouettes_csv(output / utf8_to_path(config.name + "_silhouettes.csv"), *silhouettes, prob, result);
          const double mean = silhouettes->empty() ? 0.0
                                                   : std::accumulate(silhouettes->begin(), silhouettes->end(), 0.0)
                                                       / static_cast<double>(silhouettes->size());
          if (config.verbose) std::cout << "Silhouette scores written, mean=" << std::setprecision(4) << mean << "\n";
        }
      }
    }
  }
  if (checkpoint_failure)
    throw IOError("--checkpoint '" + config.checkpoint + "': the distance checkpoint cannot be saved (the results are "
                  "written to '" + config.output + "'); free space or pass another directory to --checkpoint, or "
                  "omit it. Cause: " + *checkpoint_failure);

  return { std::move(problem), std::move(result), method };
}

} // namespace

detail::ParquetPlan detail::plan_parquet_load(ClusterMethod method, Device device, std::size_t series_count,
                                              std::size_t estimated_resident_bytes, std::size_t ram_limit,
                                              ParquetLayout layout)
{
  if (series_count == 0) throw InvalidInput("Parquet input contains no time series.");
  const ParquetPlan plan{ resolve_method(method, device, series_count), false };
  if (ram_limit == 0 || estimated_resident_bytes <= ram_limit) return plan;
  if (plan.method != ClusterMethod::CLARA)
    throw InvalidInput("Parquet input needs approximately " + std::to_string(estimated_resident_bytes)
                       + " bytes of resident series storage, exceeding --ram-limit=" + std::to_string(ram_limit)
                       + "; method '" + method_name(plan.method)
                       + "' cannot stream it. Use --method clara with a single list-per-row Parquet file, or "
                         "raise --ram-limit.");
  if (layout == ParquetLayout::ScalarColumn)
    throw InvalidInput(
      "RAM-limited FastCLARA streaming requires list-per-row Parquet "
      "(one list cell per time series); a scalar column is one time series "
      "whose rows cannot be clustered as independent series.");
  if (layout == ParquetLayout::Directory)
    throw InvalidInput(
      "RAM-limited FastCLARA streaming currently requires a single Parquet file; "
      "a Parquet directory exceeds --ram-limit. Convert it to one list-per-row "
      "Parquet file or raise --ram-limit.");
  return { plan.method, true };
}

Result run(const Config &config)
{
  auto [problem, result, method] = execute(config, std::nullopt);
  return Result(std::move(problem), result.total_cost, device_text(config), method, result.iterations,
                result.converged);
}

Result run(const Config &config, Data data)
{
  auto [problem, result, method] = execute(config, std::move(data));
  return Result(std::move(problem), result.total_cost, device_text(config), method, result.iterations,
                result.converged);
}

} // namespace dtwc
