/**
 * @file run.cpp
 * @brief dtwc::run: the checks, the method x device resolution, the load, the
 *        distance storage, the dispatch and the outputs of one clustering run.
 *
 * @details Moved from dtwc_cl.cpp's main() and api.cpp's cluster(), which both
 * call it now, so each rule has one copy. Progress lines go to stdout
 * under `verbose`.
 *
 * @author Volkan Kumtepeli
 * @date 24 Sep 2026
 */

#include "run.hpp"

#include "../Problem.hpp"
#include "../algorithms/detail/fast_clara_plan.hpp"
#include "../algorithms/detail/fast_clara_stream.hpp"
#include "../algorithms/fast_clara.hpp"
#include "../algorithms/fast_pam.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp"
#include "../base/timing.hpp"
#include "../checkpoint.hpp"
#include "../core/dtw_dispatch.hpp"
#include "../core/matrix_io.hpp"
#include "../core/thread_scratch.hpp"
#include "../fileOperations.hpp"
#include "../io/arrow_ipc.hpp"
#include "../io/read_data.hpp"
#include "../scores.hpp"
#ifdef DTWC_HAS_ARROW
#include "../io/read_arrow.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#endif

#include <algorithm>
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

std::string method_name(Method method) { return std::string(name_of(method_names, method)); }

/// What --verbose calls a method in its progress lines.
const char *progress_label(Method method)
{
  switch (method) {
  case Method::Auto: break; // resolved before it runs
  case Method::PAM: return "FastPAM";
  case Method::OneBatch: return "OneBatchPAM";
  case Method::CLARA: return "FastCLARA";
  case Method::Hierarchical: return "Hierarchical clustering";
  case Method::Kmedoids: return "kMedoids Lloyd";
  case Method::MIP: return "MIP clustering";
  case Method::LRCore: return "LR-core clustering";
  case Method::TADPole: return "TADPole clustering";
  }
  return "auto";
}

/// The input's format, judged from its name before any file is opened. In a
/// build with Arrow, dtwc_io reads Parquet; the core's reader takes the rest,
/// Arrow IPC among them, and refuses Parquet in a build without Arrow.
InputFormat format_of(const fs::path &input)
{
#ifdef DTWC_HAS_ARROW
  if (const auto format = io::arrow_format(input)) return *format;
#endif
  return input_format(input);
}

/// The series `input` names, read by the reader its format needs, in the precision --dtype names.
Data read_input(const fs::path &input, [[maybe_unused]] InputFormat format, const Config &config)
{
#ifdef DTWC_HAS_ARROW
  if (format != InputFormat::Text)
    return io::read_arrow(input, config.column, config.skip_cols, config.skip_rows, config.dtype);
#endif
  return read_data(input, config.skip_cols, config.skip_rows, config.delimiter, config.column, config.dtype);
}

/// Reject a reader option the input cannot honour: accepting and then ignoring
/// it would be a silent deceit. `format` is empty for series passed in memory.
void require_input_options_apply(const Config &config, std::optional<InputFormat> format)
{
  if (!format && !config.input.empty())
    throw InvalidInput("run: the series are passed in memory, so --input '" + config.input
                       + "' would not be read; leave input empty.");
  require_reader_options(format, config.skip_cols, config.skip_rows, config.delimiter, config.column);
}

/// An owning Float32 copy of Float64 series, owned or a view, each value rounded to float.
Data convert_to_f32(const Data &data_f64)
{
  std::vector<std::vector<float>> series(data_f64.size());
  std::vector<std::string> names(series.size());
  for (std::size_t i = 0; i < series.size(); ++i) {
    const auto source = data_f64.series(i);
    series[i].reserve(source.size());
    for (const double value : source) series[i].push_back(static_cast<float>(value));
    names[i] = data_f64.name(i);
  }
  return Data(std::move(series), std::move(names), data_f64.ndim);
}

/// Whether `method` holds the Problem's matrix (or TADPole its cache of it): OneBatchPAM owns a fixed O(Nm) table and
/// reads no parent matrix; nor does non-full FastCLARA.
bool holds_matrix(Method method, bool clara_uses_full_sample)
{
  return method != Method::OneBatch && (method != Method::CLARA || clara_uses_full_sample);
}

/// What each thread of the in-RAM run of `method` holds of scratch: the DTW kernels' for series of up to `longest` time
/// steps of `precision` and `ndim` features (core::thread_scratch_bytes; 0: the Problem's series'; none where a GPU
/// computes the distances, but for Metal's CPU assignment of FastCLARA, and none without `kernels`: a complete matrix,
/// which no kernel adds to), FasterPAM's swap table where the method runs it, FastCLARA's assignment table, TADPole's
/// count of each point's neighbours, and the silhouettes' sums when the results are written.
std::size_t in_ram_thread_bytes(const Problem &prob, const Config &config, Method method, bool clara_uses_full_sample,
                                std::size_t longest, core::Precision precision, bool kernels = true,
                                std::size_t ndim = 0)
{
  const bool assigns = method == Method::CLARA && !clara_uses_full_sample;
#ifdef DTWC_HAS_CUDA
  const bool cpu_assignment = prob.device().first != Device::GPU;
#else
  const bool cpu_assignment = true; // Metal has no kernel for FastCLARA's assignment
#endif
  std::size_t bytes = 0;
  if ((kernels && prob.device().first != Device::GPU) || (assigns && cpu_assignment)) {
    auto distance = prob.distance();
    if (ndim != 0) distance.ndim = ndim;
    bytes += core::thread_scratch_bytes(distance, longest, precision);
  }
  if (method == Method::PAM || method == Method::CLARA || method == Method::MIP || method == Method::LRCore)
    bytes += detail::fast_pam_thread_bytes(config.k);
  if (assigns && cpu_assignment)
    bytes += algorithms::detail::clara_assign_thread_bytes(static_cast<std::size_t>(config.n_samples) * config.k,
                                                           precision);
  if (method == Method::TADPole) bytes += prob.size() * sizeof(index_t); // tadpole.cpp's row of counts a thread
  if (!config.output.empty()) bytes += scores::detail::silhouette_thread_bytes(config.k);
  return bytes;
}

/// Decide the persistent distance storage once every distance-affecting setting has
/// reached the Problem. Returns the file to map (the caller maps it), or nullopt when
/// the method keeps its own storage or the matrix stays in RAM: while, within an
/// explicit limit, one thread's scratch fits beside the series and the matrix in RAM
/// (`thread_fits`), and the matrix's bytes fit in half of what the series and one
/// thread's scratch (`scratch_bytes`) leave of the memory budget (--ram-limit; 0: what
/// the process may use); the team is then fitted beside the matrix in RAM, and shrinks.
/// Past that a mapped matrix costs one sequential pass and keeps the run out of swap.
/// OneBatchPAM and non-full FastCLARA read no parent matrix (holds_matrix). TADPole
/// reads the matrix when it is complete (a file or a cache from an earlier run) and
/// otherwise computes the pairs it needs.
std::optional<fs::path> configure_distance_storage(const Problem &prob, const Config &config, Method method,
                                                   bool clara_uses_full_sample, std::size_t series_bytes,
                                                   std::size_t scratch_bytes, bool thread_fits,
                                                   const fs::path &cache_path)
{
  const bool checkpoint = !config.checkpoint.empty();
  const bool dist_matrix = !config.dist_matrix.empty();
  if (method == Method::CLARA && !clara_uses_full_sample) {
    if (checkpoint || dist_matrix)
      throw InvalidInput(
        "Non-full FastCLARA does not consume a parent distance matrix; "
        "--checkpoint and --dist-matrix would load or save unused O(N^2) "
        "state. Omit those options, or request a full sample deliberately.");
    return std::nullopt;
  }
  if (!holds_matrix(method, clara_uses_full_sample)) return std::nullopt;
  // The automatic budget is 0, no limit, in a build without llfio: it maps nothing.
  const std::size_t budget = config.ram_limit != 0 ? config.ram_limit : core::available_memory();
  const std::size_t matrix_bytes = core::packed_size(prob.size()) * core::value_bytes(config.matrix_dtype);
  const std::size_t held = series_bytes + scratch_bytes;
  if (budget == 0 || (thread_fits && matrix_bytes <= (budget > held ? (budget - held) / 2 : 0))) return std::nullopt;
  const std::string over =
    thread_fits ? "the distance matrix's " + std::to_string(matrix_bytes)
                    + " bytes exceed half of what the series and one thread's scratch ("
                    + std::to_string(scratch_bytes) + " bytes) leave of --ram-limit=" + std::to_string(config.ram_limit)
                : "beside the series and the distance matrix's " + std::to_string(matrix_bytes)
                    + " bytes in RAM no thread's scratch fits within --ram-limit=" + std::to_string(config.ram_limit);
  // Only an explicit --ram-limit refuses: the automatic one follows the machine's load.
  if (dist_matrix) {
    if (config.ram_limit == 0) return std::nullopt; // the CSV is read into RAM
    throw InvalidInput("--dist-matrix reads a dense CSV matrix into RAM, but " + over
                       + ". Raise --ram-limit, or omit --dist-matrix to compute the distances into a mapped matrix.");
  }
#ifdef DTWC_HAS_MMAP
  return cache_path;
#else
  (void)cache_path;
  throw IOError("method='" + method_name(method) + "' at N=" + std::to_string(prob.size()) + " needs a mapped matrix: "
                + over + ", but this binary was built without mmap support. Rebuild with -DDTWC_ENABLE_LLFIO=ON, "
                "raise --ram-limit only if the packed matrix fits in RAM, or use --method onebatch.");
#endif
}

/// What a run hands to its Result.
struct Outcome
{
  std::shared_ptr<Problem> problem;
  core::ClusteringResult result;
  Method method;
  std::vector<std::string> streamed_names; ///< a RAM-limited run's series names, which its Problem cannot hold
};

/// The run itself; `data` holds in-memory series, else config.input is read.
Outcome execute(const Config &config, std::optional<Data> data)
{
  // ---- 1. Every check that needs no series, before any file is touched ----
  if (!data && config.input.empty())
    throw InvalidInput("--input is required via CLI or config file (TOML or YAML)");
  if (config.k == 0) throw InvalidInput("-k/--n-clusters, the number of clusters, is required.");
  if (config.k < 1)
    throw InvalidInput("-k/--n-clusters must be a positive integer, got " + std::to_string(config.k));
  if (config.n_init < 1) throw InvalidInput("--n-init must be a positive integer");
  if (config.max_iter < 1)
    throw InvalidInput("--max-iter must be a positive integer, got " + std::to_string(config.max_iter));
  if (config.checkpoint_interval != 0 && config.checkpoint.empty())
    throw InvalidInput("--checkpoint-interval requires --checkpoint <dir>.");
  if (config.not_computed != core::NotComputed::NaN && config.dist_matrix.empty())
    throw InvalidInput("--not-computed says how a --dist-matrix file marks a pair, and requires --dist-matrix <path>.");

  auto problem =
    std::make_shared<Problem>(config.name.empty() ? detail::default_name(utf8_to_path(config.input)) : config.name);
  Problem &prob = *problem;
  apply(config, prob);
  validate_gpu_request("run", prob, config.dtype);

  algorithms::CLARAOptions clara;
  clara.n_clusters = config.k;
  clara.sample_size = config.sample_size;
  clara.n_samples = config.n_samples;
  clara.max_iter = config.max_iter;
  clara.random_seed = config.seed;
  if (config.method == Method::CLARA) algorithms::detail::validate_clara_controls(clara, "run");

  const fs::path input = utf8_to_path(config.input);
  std::optional<InputFormat> format; // empty: the series are in memory
  if (!data) format = format_of(input);
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
  Method method = config.method;
  bool stream_payload = false;
  std::size_t n_series = 0;
  bool clara_uses_full_sample = false;
  bool clara_planned = false;
  std::size_t stream_names_bytes = 0; // a streamed run's names, read after the stream beside the labels
  const auto plan_clara = [&] {
    if (method != Method::CLARA || clara_planned) return;
    const auto plan = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
    clara_uses_full_sample = plan.sample_size == plan.n_points;
    if (stream_payload) algorithms::detail::validate_streaming_clara_plan(plan, "run");
    clara_planned = true;
  };
  // A streamed run holds no series: what it holds after the stream, and the options that need the series, are checked
  // before it; `what` is the input's format, `resident` the price that chose the stream.
  const auto check_stream = [&](const char *what, const char *by, std::size_t names_bytes, std::size_t resident,
                                std::size_t limit) {
    // After the stream the run holds every series' name, written beside its label, and the labels: the result's
    // and the Problem's.
    const std::size_t written_bytes = names_bytes + 2 * n_series * sizeof(index_t);
    if (written_bytes > limit)
      throw InvalidInput("--ram-limit = " + std::to_string(limit) + " cannot hold the names and labels of the "
                         + std::to_string(n_series) + " streamed series, which need " + std::to_string(written_bytes)
                         + " bytes; raise the limit.");
    if (!config.checkpoint.empty())
      throw InvalidInput("--checkpoint requires resident series data and cannot be combined with RAM-limited "
                         + std::string(what) + " streaming.");
    if (!config.dist_matrix.empty())
      throw InvalidInput("--dist-matrix requires resident series data and cannot be combined with RAM-limited "
                         + std::string(what) + " streaming.");
    if (config.verbose)
      std::cout << by << " selected streaming: " << n_series << " series, ~" << (resident / (1ULL << 20))
                << " MB resident estimate exceeds the RAM limit [" << clk << "]\n";
  };
  // CLARA's terms beside the series, in RAM as streamed: a sample PAM's matrix, then every sample's labels and
  // nearest distances.
  const auto clara_bytes = [&](const algorithms::detail::ClaraPlan &sample) {
    const std::size_t pam = core::packed_size(static_cast<std::size_t>(sample.sample_size))
                            * core::value_bytes(config.matrix_dtype);
    return std::max(pam, algorithms::detail::clara_pass_bytes(sample, static_cast<std::size_t>(clara.n_samples)));
  };

#ifdef DTWC_HAS_PARQUET
  // A cap changes the load decision, so only the metadata is read here: the
  // readers map the file and its footer; no row group is decoded.
  if (format == InputFormat::Parquet
      && (config.ram_limit > 0 || method == Method::Auto || method == Method::CLARA)) {
    const bool f32 = config.dtype == core::Precision::Float32;
    std::error_code ec;
    const bool folder = fs::is_directory(input, ec);
    auto layout = detail::ParquetLayout::Directory;
    std::size_t resident_bytes = 0;
    std::size_t names_bytes = 0;
    std::size_t values = 0;
    for (const auto &file : parquet_files(input)) {
      const io::ParquetChunkReader metadata(file, config.column, config.skip_cols, config.skip_rows);
      if (!folder)
        layout = metadata.rows_are_series() ? detail::ParquetLayout::ListColumn : detail::ParquetLayout::ScalarColumn;
      n_series += static_cast<std::size_t>(metadata.logical_series_count());
      resident_bytes += metadata.estimated_materialization_peak_bytes(f32);
      names_bytes += metadata.series_names_bytes();
      values += metadata.total_value_count();
    }
    // A CLARA run holds beside the series a sample's PAM matrix, then every sample's labels and nearest distances,
    // in RAM as streamed, and one thread's scratch at the least (in_ram_thread_bytes, for the series' mean length: the
    // longest is known once they are read, where the run's team is fitted on it).
    std::optional<algorithms::detail::ClaraPlan> sample;
    if (n_series > 0 && resolve_method(method, config.device, n_series) == Method::CLARA) {
      require_clusterable(config.k, n_series); // k > N refused as for every input, before CLARA's plan
      sample = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
      resident_bytes += clara_bytes(*sample)
                        + in_ram_thread_bytes(prob, config, Method::CLARA, sample->sample_size == sample->n_points,
                                              values / n_series, config.dtype);
    }
    // An explicit --ram-limit refuses a route that cannot honour it. The automatic
    // one never fails a run: it streams only where FastCLARA can stream a sample
    // smaller than N, of one file of a series per row, without --checkpoint or
    // --dist-matrix.
    std::size_t limit = config.ram_limit;
    if (limit == 0 && sample && layout == detail::ParquetLayout::ListColumn && config.checkpoint.empty()
        && config.dist_matrix.empty() && sample->sample_size < sample->n_points)
      limit = core::available_memory();
    const auto plan = detail::plan_parquet_load(method, config.device, n_series, resident_bytes, limit, layout);
    method = plan.method;
    stream_payload = plan.stream_payload;
    if (stream_payload) {
      clara.ram_limit_bytes = limit;
      clara.parquet_path = input;
      clara.parquet_column = config.column;
      clara.parquet_skip_cols = config.skip_cols;
      clara.parquet_skip_rows = config.skip_rows;
      clara.use_float32 = f32;
      clara.force_parquet_streaming = true;
    }
    plan_clara();
    if (stream_payload) {
      check_stream("Parquet", "Parquet metadata", names_bytes, resident_bytes, limit);
      stream_names_bytes = names_bytes;
    }
  }
#endif

  // Arrow IPC: its walk gives N and every record batch's rows and decoded bytes before any batch is decoded, so what
  // the in-RAM route would hold at once is priced first (io::IpcFile::resident_bytes; a mapped view's values are the
  // page cache's). Beyond the limit FastCLARA reads the file a record batch at a time, where that holds less (decoded
  // batches, the file read without llfio, a copy in the other precision); beyond an explicit limit any other route is
  // refused, beyond the automatic one (no limit without llfio) the series are read into RAM.
  std::optional<io::IpcFile> ipc;
  std::size_t ipc_bytes = 0; // what the in-RAM route holds of the file at once
  if (format == InputFormat::ArrowIPC) {
    ipc.emplace(input);
    n_series = static_cast<std::size_t>(ipc->num_rows());
    ipc_bytes = ipc->resident_bytes(config.dtype, config.device == Device::GPU);
    std::size_t resident = ipc_bytes;
    std::optional<algorithms::detail::ClaraPlan> sample;
    const Method resolved = n_series > 0 ? resolve_method(method, config.device, n_series) : method;
    if (n_series > 0 && resolved == Method::CLARA) {
      require_clusterable(config.k, n_series); // k > N refused as for every input, before CLARA's plan
      sample = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
      resident += clara_bytes(*sample);
    }
    const bool streams = ipc->streams(config.dtype);
    const bool can_stream = sample && sample->sample_size < sample->n_points && streams;
    std::size_t limit = config.ram_limit;
    if (limit == 0 && can_stream && config.checkpoint.empty() && config.dist_matrix.empty())
      limit = core::available_memory();
    // The in-RAM run holds one thread's scratch at the least beside the series (in_ram_thread_bytes), its team fitted
    // on the longest series once they are read: routed by that need, so a limit that routes the series into RAM holds
    // the thread and a smaller one streams them. A series is at least their mean long and at most the walk's bound
    // (IpcFile::longest_bound); where those disagree on the route, the series' lengths are read (IpcFile::longest).
    // The thread is priced for the file's features a time step, which reach the Problem only with its series.
    const auto in_ram_needs = [&](std::size_t steps) {
      return resident + in_ram_thread_bytes(prob, config, Method::CLARA, false, steps, config.dtype, true, ipc->ndim());
    };
    if (limit != 0 && can_stream
        && (in_ram_needs(ipc->num_values() / n_series / ipc->ndim()) > limit
            || (in_ram_needs(ipc->longest_bound()) > limit && in_ram_needs(ipc->longest()) > limit))) {
      stream_payload = true;
      method = resolved;
      clara.ram_limit_bytes = limit;
      plan_clara();
      // The names are read after the stream, beside the labels: their read is priced with them.
      stream_names_bytes = ipc->series_names_bytes() + ipc->names_read_bytes();
      check_stream("Arrow IPC", "Arrow IPC walk", stream_names_bytes, resident, limit);
    } else if (config.ram_limit != 0 && resident > config.ram_limit) {
      // CLARA is named only where its stream could read the file: a file that streams, on the CPU, sampled in fewer
      // than its series (the plan of these options; none where CLARA cannot run on them).
      const auto clara_streams = [&] {
        try {
          const auto plan = sample ? *sample : resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
          return plan.sample_size < plan.n_points;
        } catch (const InvalidInput &) {
          return false;
        }
      };
      const std::string other = config.dtype == core::Precision::Float32 ? "float64" : "float32";
      std::string why;
      if (config.device == Device::GPU)
        why = "on a GPU every route holds the series in RAM at once; raise --ram-limit, or run on the CPU (--device "
              "cpu).";
      else if (!streams)
        why = "its values are mapped where they lie in the file, and a mapped file's values are not streamed: the "
              "views of its " + std::to_string(n_series) + " series are held at once. Raise --ram-limit"
              + (clara_streams() ? ", or read its values as " + other + " (--dtype " + other + ")"
                                     + (resolved == Method::CLARA ? "" : " with --method clara")
                                     + ", whose copy FastCLARA reads a record batch at a time."
                                 : std::string("."));
      else if (resolved != Method::CLARA)
        why = "method '" + method_name(resolved) + "' cannot stream it. "
              + (clara_streams() ? "Use --method clara, which reads the file a record batch at a time, or raise "
                                   "--ram-limit."
                                 : std::string("Raise --ram-limit."));
      else
        why = "FastCLARA streams it only with a sample smaller than its " + std::to_string(n_series)
              + " series; lower --sample-size, or raise --ram-limit.";
      throw InvalidInput("Arrow IPC input needs approximately " + std::to_string(resident)
                         + " bytes of resident series storage, exceeding --ram-limit="
                         + std::to_string(config.ram_limit) + "; " + why);
    }
    if (!stream_payload) ipc.reset(); // read below as every Arrow IPC file is read
  }

  // ---- 3. Load the series into RAM ----
  std::size_t series_bytes = 0;
  if (!stream_payload) {
    Data series = data ? std::move(*data) : read_input(input, format.value_or(InputFormat::Text), config);
    if (config.verbose) {
      const char *from = !format                            ? " in memory"
                         : format == InputFormat::Parquet  ? " from Parquet"
                         : format == InputFormat::ArrowIPC ? " from Arrow IPC"
                                                           : "";
      std::cout << "Data loaded" << from << ": " << series.size() << " series [" << clk << "]\n";
    }
    // A reader stores what --dtype names; Float64 series passed in memory, a view among them, are converted here.
    if (config.dtype == core::Precision::Float32 && !series.is_f32()) {
      series = convert_to_f32(series);
      if (config.verbose) std::cout << "Converted to float32 (2x memory saving)\n";
    }
    std::size_t elements = 0;
    for (std::size_t i = 0; i < series.size(); ++i) elements += series.series_flat_size(i);
    const std::size_t value_bytes = elements * (series.is_f32() ? sizeof(float) : sizeof(data_t));
    if (config.verbose && series.size() > 0)
      std::cout << "  Data memory: ~" << (value_bytes / (1ULL << 20)) << " MB (" << series.size() << " series, "
                << (elements / series.size()) << " avg length, " << name_of(core::precision_names, series.precision)
                << ")\n";
    // What the series hold in RAM: an Arrow IPC file's as its walk priced them (a mapped view's values are the page
    // cache's), others' their values, counted after the read.
    series_bytes = format == InputFormat::ArrowIPC ? ipc_bytes : value_bytes;
    prob.set_data(std::move(series));
    n_series = prob.size();
  }
  if (config.ram_limit > 0 && config.verbose) std::cout << "RAM limit: " << config.ram_limit << " bytes\n";
  require_clusterable(config.k, n_series); // before a cache or checkpoint is touched

  method = resolve_method(method, config.device, n_series);
  if (config.method == Method::Auto && config.verbose)
    std::cout << "Auto-selected method: " << method_name(method) << " (N=" << n_series << ")\n";
  plan_clara();
  prob.set_method(method);
  // What each thread of the in-RAM run holds of scratch (the streamed run prices its own phases): the DTW kernels' for
  // the longest series, in time steps, and the method's tables; without the kernels' where none runs, a complete
  // --dist-matrix (known once it is read).
  std::size_t longest = 0;
  for (std::size_t i = 0; i < prob.data().size(); ++i)
    longest = std::max(longest, prob.data().series_flat_size(i) / prob.data().ndim);
  const auto thread_bytes = [&](bool kernels) {
    return stream_payload ? 0
                          : in_ram_thread_bytes(prob, config, method, clara_uses_full_sample, longest,
                                                prob.data().precision, kernels);
  };
  // Within an explicit --ram-limit the threads of the in-RAM run are those whose scratch fits beside what it holds
  // (core::KernelTeam): the series, in-RAM FastCLARA's sample matrix and labels, the matrix where it is in RAM; every
  // region from here on capped to them. The stream fits each of its phases beside, too, what follows it: the series'
  // names, read beside the labels.
  core::KernelTeam team(config.ram_limit, config.verbose,
                        stream_payload ? 2 * n_series * sizeof(index_t) + stream_names_bytes : 0);
  const std::size_t clara_rest =
    !stream_payload && method == Method::CLARA && !clara_uses_full_sample
      ? clara_bytes(resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run"))
      : 0;
  const std::size_t matrix_bytes = holds_matrix(method, clara_uses_full_sample)
                                     ? core::packed_size(n_series) * core::value_bytes(config.matrix_dtype)
                                     : 0;
  const bool reads_matrix = !config.dist_matrix.empty(); // read into RAM, never mapped
  const auto fit_team = [&](std::size_t matrix, bool kernels) {
    const char *held = clara_rest != 0 ? "the series and FastCLARA's sample matrix and labels"
                       : matrix != 0   ? "the series and the distance matrix"
                                       : "the series";
    return team.fit(progress_label(method), series_bytes + clara_rest + matrix, thread_bytes(kernels), longest, held);
  };

  // ---- 4. Distance storage, once every distance setting is in place ----
  // A mapped matrix is <name>.dtwm in the --checkpoint directory, where it is
  // the checkpoint, else in the output directory; a run that writes nothing
  // keeps its matrix in RAM. In RAM while one thread's scratch (beside a
  // --dist-matrix, which no kernel may add to, without the kernels') fits beside
  // it and the series, and it takes at most half of what the series and that
  // thread leave; the team is then fitted beside it, and shrinks. Mapped where
  // that fails.
  std::optional<fs::path> cache;
  if (!config.output.empty()) {
    const std::size_t each = thread_bytes(!reads_matrix);
    const bool thread_fits = team.fitting(series_bytes + clara_rest + matrix_bytes, each) > 0;
    const auto path = checkpoint_path(prob, config.checkpoint.empty() ? config.output : config.checkpoint);
    cache = configure_distance_storage(prob, config, method, clara_uses_full_sample, series_bytes, each, thread_fits,
                                       path);
  }
  // Refused, where one thread does not fit, before any file is mapped; beside a --dist-matrix once it is read.
  std::optional<TeamCap> cap;
  if (!stream_payload && !reads_matrix) cap.emplace(fit_team(cache ? 0 : matrix_bytes, true));
  if (cache) {
    prob.use_mmap_distance_matrix(*cache);
    if (config.verbose) std::cout << "Using memory-mapped distance matrix: " << *cache << "\n";
  }
  // A matrix the user supplied but that cannot be loaded is an error: going on
  // without it silently recomputed every distance and exited 0.
  if (!config.dist_matrix.empty()) {
    try {
      prob.read_distance_matrix(utf8_to_path(config.dist_matrix), config.not_computed);
    } catch (const std::exception &e) {
      throw IOError("--dist-matrix '" + config.dist_matrix
                    + "' cannot be loaded; fix the file, or omit --dist-matrix to compute the distances. Cause: "
                    + e.what());
    }
    if (config.verbose) std::cout << "Loaded distance matrix from " << config.dist_matrix << "\n";
    if (!stream_payload) cap.emplace(fit_team(matrix_bytes, !prob.is_distance_matrix_filled()));
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
  // A RAM-limited Parquet or Arrow IPC run is the one whose series the Problem does
  // not hold: FastCLARA streams them.
  if (config.verbose) std::cout << "Running " << progress_label(method) << " (k=" << config.k << ") ...\n";
  std::vector<std::string> streamed_names;
  core::ClusteringResult result;
  if (stream_payload && ipc) {
    // The Problem holds no series: its distance takes their features a time step from an empty Data of them.
    prob.set_data(Data(std::vector<std::vector<data_t>>{}, std::vector<std::string>{}, ipc->ndim()));
    const auto plan = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "fast_clara");
    result = config.dtype == core::Precision::Float32
               ? algorithms::detail::fast_clara_chunked<true>(prob, clara, *ipc, plan, team)
               : algorithms::detail::fast_clara_chunked<false>(prob, clara, *ipc, plan, team);
    // Beside the labels and the scratch the stream's threads keep.
    const std::size_t held = 2 * n_series * sizeof(index_t) + team.priced();
    streamed_names = ipc->series_names(clara.ram_limit_bytes > held ? clara.ram_limit_bytes - held : 0);
  } else {
#ifdef DTWC_HAS_PARQUET
    result = stream_payload ? algorithms::detail::fast_clara_parquet(prob, clara, team) : prob.cluster();
    if (stream_payload)
      streamed_names = io::ParquetChunkReader(input, config.column, config.skip_cols, config.skip_rows).series_names();
#else
    result = prob.cluster();
#endif
  }
  if (config.verbose) {
    std::cout << progress_label(method);
    if (method == Method::PAM)
      std::cout << (result.converged ? " converged" : " did not converge") << " in " << result.iterations
                << " iterations";
    else
      std::cout << " finished";
    std::cout << ", cost=" << std::setprecision(6) << result.total_cost << " [" << clk << "]\n";
  }

  // ---- 6. Checkpoints first, then the outputs ----
  // Before the results, so a result write that
  // fails cannot lose the distance matrix. A save that fails is kept and raised
  // once the results are on disk, so it cannot lose them either.
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
    detail::write_result_files(prob, output, false, config.verbose ? &std::cout : nullptr, streamed_names);
  if (checkpoint_failure)
    throw IOError("--checkpoint '" + config.checkpoint + "': the distance checkpoint cannot be saved (the results are "
                  "written to '" + config.output + "'); free space or pass another directory to --checkpoint, or "
                  "omit it. Cause: " + *checkpoint_failure);

  return { std::move(problem), std::move(result), method, std::move(streamed_names) };
}

} // namespace

detail::ParquetPlan detail::plan_parquet_load(Method method, Device device, std::size_t series_count,
                                              std::size_t estimated_resident_bytes, std::size_t ram_limit,
                                              ParquetLayout layout)
{
  if (series_count == 0) throw InvalidInput("Parquet input contains no time series.");
  const ParquetPlan plan{ resolve_method(method, device, series_count), false };
  if (ram_limit == 0 || estimated_resident_bytes <= ram_limit) return plan;
  if (plan.method != Method::CLARA)
    throw InvalidInput("Parquet input needs approximately " + std::to_string(estimated_resident_bytes)
                       + " bytes of resident series storage, exceeding --ram-limit=" + std::to_string(ram_limit)
                       + "; method '" + method_name(plan.method)
                       + "' cannot stream it. Use --method clara with a single list-per-row Parquet file, or "
                         "raise --ram-limit.");
  if (layout == ParquetLayout::ScalarColumn)
    throw InvalidInput(
      "RAM-limited FastCLARA streaming requires a Parquet file of one series "
      "per row (list-per-row, or several Float32/Float64 columns); a single "
      "scalar column is one time series whose rows cannot be clustered as "
      "independent series.");
  if (layout == ParquetLayout::Directory)
    throw InvalidInput(
      "RAM-limited FastCLARA streaming currently requires a single Parquet file; "
      "a Parquet directory exceeds --ram-limit. Convert it to one list-per-row "
      "Parquet file or raise --ram-limit.");
  return { plan.method, true };
}

Result run(const Config &config)
{
  const core::ScratchRelease release; // every thread's scratch given back as the run starts, returns or throws
  auto [problem, result, method, streamed_names] = execute(config, std::nullopt);
  return Result(std::move(problem), result.total_cost, device_text(config), method, result.iterations,
                result.converged, std::move(streamed_names));
}

Result run(const Config &config, Data data)
{
  const core::ScratchRelease release; // every thread's scratch given back as the run starts, returns or throws
  auto [problem, result, method, streamed_names] = execute(config, std::move(data));
  return Result(std::move(problem), result.total_cost, device_text(config), method, result.iterations,
                result.converged, std::move(streamed_names));
}

} // namespace dtwc
