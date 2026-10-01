/**
 * @file run.cpp
 * @brief dtwc::run: the checks, the method x device resolution, the load, the
 *        distance storage, the dispatch and the outputs of one clustering run.
 *
 * @details Moved from dtwc_cl.cpp's main() and api.cpp's cluster(), which both
 * call it now (IF-2 S3), so each rule has one copy. Progress lines go to stdout
 * under `verbose`.
 *
 * @date 24 Sep 2026
 */

#include "run.hpp"

#include "../Problem.hpp"
#include "../algorithms/detail/fast_clara_plan.hpp"
#include "../algorithms/fast_clara.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../algorithms/hierarchical.hpp"
#include "../algorithms/one_batch_pam.hpp"
#include "../base/error.hpp"
#include "../base/timing.hpp"
#include "../checkpoint.hpp"
#include "../core/matrix_io.hpp"
#include "../fileOperations.hpp"
#include "../io/read_data.hpp"
#include "../scores.hpp"
#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#endif

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
  case ClusterMethod::TADPole: return { Method::TADPole, "TADPole clustering" };
  case ClusterMethod::Auto:
  case ClusterMethod::PAM:
  case ClusterMethod::OneBatch:
  case ClusterMethod::CLARA:
  case ClusterMethod::Hierarchical: break;
  }
  throw std::logic_error("run: method " + method_name(method) + " is not a Problem::cluster() route");
}

/// Reject a reader option the input cannot honour: accepting and then ignoring
/// it would be a silent deceit. `--ram-limit` is the Parquet materialisation cap.
/// `format` is empty for series passed in memory.
void require_input_options_apply(const Config &config, std::optional<InputFormat> format)
{
  if (!format && !config.input.empty())
    throw InvalidInput("run: the series are passed in memory, so --input '" + config.input
                       + "' would not be read; leave input empty.");
  if (config.ram_limit != 0 && format != InputFormat::Parquet)
    throw InvalidInput(
      "--ram-limit caps Parquet series materialisation and cannot be honoured "
      "for this input; drop --ram-limit, or convert the series to a "
      "list-per-row Parquet file to stream them under the cap.");
  require_reader_options(format, config.skip_cols, config.skip_rows, config.delimiter, config.column);
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

/// Bind persistent distance storage once every distance-affecting setting has
/// reached the Problem. Returns the mmap cache path, or nullopt when the method
/// keeps its own storage or N is below the threshold. OneBatchPAM owns a fixed
/// O(Nm) table and reads no parent matrix; nor does non-full FastCLARA. TADPole
/// reads the matrix when it is complete (a file or a cache from an earlier run)
/// and otherwise computes the pairs it needs.
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
  prob.set_gpu_precision(config.gpu_precision);
  prob.set_device(config.device, config.device_index); // gpu without a GPU backend: §6.1's DeviceError
  if (config.device == Device::GPU
      && (config.method == ClusterMethod::OneBatch || config.method == ClusterMethod::TADPole))
    refuse_gpu_method(config.method);
  validate_gpu_request("run", prob, config.dtype);

  algorithms::CLARAOptions clara;
  clara.n_clusters = config.k;
  clara.sample_size = config.sample_size;
  clara.n_samples = config.n_samples;
  clara.max_iter = config.max_iter;
  clara.random_seed = config.seed;
  if (config.method == ClusterMethod::CLARA) algorithms::detail::validate_clara_controls(clara, "run");

  const fs::path input = utf8_to_path(config.input);
  std::optional<InputFormat> format; // empty: the series are in memory
  if (!data) format = input_format(input);
  require_input_options_apply(config, format);

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
  if (format == InputFormat::Parquet
      && (config.ram_limit > 0 || method == ClusterMethod::Auto || method == ClusterMethod::CLARA)) {
    const bool f32 = config.dtype == core::Precision::Float32;
    std::error_code ec;
    const bool folder = fs::is_directory(input, ec);
    auto layout = detail::ParquetLayout::Directory;
    std::size_t resident_bytes = 0;
    for (const auto &file : parquet_files(input)) {
      const io::ParquetChunkReader metadata(file, config.column);
      if (!folder)
        layout = metadata.is_list_layout() ? detail::ParquetLayout::ListColumn : detail::ParquetLayout::ScalarColumn;
      n_series += static_cast<std::size_t>(metadata.logical_series_count());
      resident_bytes += metadata.estimated_materialization_peak_bytes(f32);
    }
    const auto plan = detail::plan_parquet_load(method, config.device, n_series, resident_bytes, config.ram_limit, layout);
    method = plan.method;
    stream_payload = plan.stream_payload;
    if (stream_payload) {
      clara.ram_limit_bytes = config.ram_limit;
      clara.parquet_path = input;
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

  // ---- 3. Load the series into RAM ----
  if (!stream_payload) {
    Data series = data ? std::move(*data)
                       : read_data(input, config.skip_cols, config.skip_rows, config.delimiter, config.column);
    if (config.verbose) {
      const char *from = !format                            ? " in memory"
                         : format == InputFormat::Parquet  ? " from Parquet"
                         : format == InputFormat::ArrowIPC ? " from Arrow IPC"
                                                           : "";
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

  // ---- 4. Distance storage, once every distance setting is in place ----
  // A mapped matrix is <name>.dtwm in the --checkpoint directory, where it is
  // the checkpoint, else in the output directory; a run that writes nothing
  // keeps its matrix in RAM, as does an imported matrix.
  std::optional<fs::path> cache;
  if (!config.output.empty()) {
    const auto path = checkpoint_path(prob, config.checkpoint.empty() ? config.output : config.checkpoint);
    cache = configure_distance_storage(prob, config, method, clara_uses_full_sample, path);
    if (cache && config.verbose) std::cout << "Using memory-mapped distance matrix: " << *cache << "\n";
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
  if (!config.checkpoint.empty() && !cache) { // a mapped matrix is its own checkpoint
    const bool resumed = load_checkpoint(prob, config.checkpoint); // false: no file, start fresh
    if (config.verbose)
      std::cout << (resumed ? "Resumed from checkpoint: " + config.checkpoint + "\n"
                            : "No checkpoint in " + config.checkpoint + ", starting fresh.\n");
  }

  // ---- 5. Cluster; the matrix methods fill through the Problem, on its device ----
  core::ClusteringResult result;
  const index_t k = config.k;
  switch (method) {
  case ClusterMethod::PAM:
    if (config.verbose) std::cout << "Running FastPAM (k=" << k << ") ...\n";
    // Restart r uses seed + r, invocation-local; the strictly lowest cost is
    // kept, so a tie keeps the earlier restart.
    result = fast_pam_seeded(prob, k, config.seed, config.max_iter);
    for (int restart = 1; restart < config.n_init; ++restart) {
      auto candidate =
        fast_pam_seeded(prob, k, config.seed + static_cast<std::uint64_t>(restart), config.max_iter);
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

  // ---- 6. Checkpoints first, then the outputs ----
  prob.set_result(result); // the kept result, in every route

  // Before the results, so a result write that
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

  if (!config.output.empty())
    detail::write_result_files(prob, output, false, config.verbose ? &std::cout : nullptr);
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

void detail::write_result_files(Problem &prob, const fs::path &directory, bool complete, std::ostream *progress)
{
  const auto &labels = prob.labels();
  const auto &medoids = prob.medoids();
  // A RAM-limited Parquet run holds no series: its names are the readers' own `series_<i>`.
  const bool streamed = prob.size() == 0;
  const auto series_name = [&](std::size_t i) {
    return streamed ? "series_" + std::to_string(i) : std::string(prob.series_name(i));
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
                         "silhouettes to save; its labels and medoids are written, with series_<i> names.");
    return;
  }
  // A matrix-free run does not fill an O(N^2) matrix merely to write these files.
  if (!prob.is_distance_matrix_filled()) return;
  const auto matrix_path = file_in_directory("_distance_matrix.csv");
  io::write_csv(std::as_const(prob).distance_matrix(), matrix_path); // the mutable overload clears filled_
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
