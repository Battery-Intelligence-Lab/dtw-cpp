/**
 * @file run.cpp
 * @brief dtwc::run: the checks, the method x device resolution, the load, the
 *        distance storage, the dispatch and the outputs of one clustering run.
 *
 * @details dtwc_cl.cpp's main() and api.cpp's cluster() both call it, so each
 * rule has one copy. Progress lines go to stdout under `verbose`.
 *
 * @author Volkan Kumtepeli
 * @date 24 Sep 2026
 */

#include "run.hpp"
#include "report.hpp"

#include "../Problem.hpp"
#include "../algorithms/detail/fast_clara_plan.hpp"
#include "../algorithms/detail/fast_clara_stream.hpp"
#include "../algorithms/fast_clara.hpp"
#include "../base/error.hpp"
#include "../base/parallelisation.hpp" // get_max_threads, TeamCap
#include "../base/timing.hpp"
#include "../checkpoint.hpp"
#include "../core/distance_matrix.hpp" // packed_size, value_bytes, group_rows, peak_memory, available_memory
#include "../core/dtw_kernel.hpp"      // dtw_lanes
#include "../core/machine.hpp"         // cpu_seconds
#include "../core/thread_scratch.hpp"
#include "../detail/result_files.hpp" // WriteTimes
#include "../fileOperations.hpp"
#include "../io/arrow_ipc.hpp"
#include "../io/read_data.hpp"
#ifdef DTWC_HAS_ARROW
#include "../io/read_arrow.hpp"
#endif
#ifdef DTWC_HAS_PARQUET
#include "../io/parquet_chunk_reader.hpp"
#endif
#ifdef DTWC_HAS_CUDA
#include "../cuda/launch_prep.hpp" // kMaxPairsPerLaunch
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new> // std::bad_alloc
#include <optional>
#include <string>
#include <string_view>
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

/// Whether `method` holds the Problem's distance matrix (TADPole, its cache of it): OneBatchPAM owns a fixed O(Nm)
/// table and reads no parent matrix; nor does non-full FastCLARA.
bool holds_matrix(Method method, bool clara_uses_full_sample)
{
  return method != Method::OneBatch && (method != Method::CLARA || clara_uses_full_sample);
}

/// What a thread of `method` holds of scratch, which the run's reserve prices for each of its threads: the DTW
/// kernels' for series of up to `length` time steps of `ndim` features in `precision` (core::thread_scratch_bytes;
/// none where a GPU computes the distances, but for FastCLARA's assignment on the CPU, and none beside a --dist-matrix,
/// taken as complete), FasterPAM's swap table where the method runs it (a loss per medoid and candidate of a group),
/// FastCLARA's assignment table (each sample's medoids against a lane block of points), TADPole's count of each of the
/// `n_series` points' neighbours, and the silhouettes' sums when the results are written (a sum per cluster and point
/// of a group).
std::size_t thread_bytes(const Problem &prob, const Config &config, Method method, bool clara_uses_full_sample,
                         std::size_t n_series, std::size_t length, std::size_t ndim, core::Precision precision)
{
  const bool gpu = prob.device().first == Device::GPU;
  const bool assigns = method == Method::CLARA && !clara_uses_full_sample;
#ifdef DTWC_HAS_CUDA
  const bool cpu_assignment = assigns && !gpu;
#else
  const bool cpu_assignment = assigns; // Metal has no kernel for FastCLARA's assignment
#endif
  std::size_t bytes = 0;
  if ((config.dist_matrix.empty() && !gpu) || cpu_assignment) {
    auto distance = prob.distance();
    distance.ndim = ndim; // the input's, which reaches the Problem only with its series
    bytes += core::thread_scratch_bytes(distance, length, precision);
  }
  const std::size_t groups = static_cast<std::size_t>(config.k) * core::group_rows * sizeof(double);
  if (method == Method::PAM || method == Method::CLARA || method == Method::MIP || method == Method::LRCore)
    bytes += groups;
  if (cpu_assignment)
    bytes += static_cast<std::size_t>(config.n_samples) * static_cast<std::size_t>(config.k) * sizeof(double)
             * (precision == core::Precision::Float32 ? core::dtw_lanes<float> : core::dtw_lanes<data_t>);
  if (method == Method::TADPole) bytes += n_series * sizeof(index_t);
  if (!config.output.empty()) bytes += groups;
  return bytes;
}

/// What a GPU holds in host memory beside the series and the distance matrix as it computes: the series it uploads,
/// each padded to the longest of `length` time steps of `ndim` features, in the precision it computes in. CUDA stages
/// them in its launch workspace (pinned where it can be; kept until the run gives its threads' scratch back), 8 bytes a
/// value (4 under --gpu-precision fp32): every series for a fill; for FastCLARA's assignment the medoids of every
/// sample beside a launch's block of series (at most 2^27 values, launch_medoid_kernel) and the block's distances to
/// them, or a sample's fill where that is larger. Metal's fill holds the series and the matrix as floats in shared
/// memory while it runs (its FastCLARA assigns on the CPU). Where the longest series is estimated (twice the mean, a
/// decoded file's), so is this.
std::size_t gpu_staging_bytes([[maybe_unused]] const Config &config, [[maybe_unused]] Method method,
                              [[maybe_unused]] const std::optional<algorithms::detail::ClaraPlan> &sample,
                              [[maybe_unused]] std::size_t n_series, [[maybe_unused]] std::size_t length,
                              [[maybe_unused]] std::size_t ndim)
{
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  if (config.device != Device::GPU) return 0;
  const std::size_t steps = length * ndim;
  const bool assigns = method == Method::CLARA && sample && sample->sample_size < sample->n_points;
  const std::size_t filled = assigns ? static_cast<std::size_t>(sample->sample_size) : n_series; // a fill's series
#  if defined(DTWC_HAS_CUDA)
  const std::size_t value = config.gpu_precision == GpuPrecision::FP32 ? sizeof(float) : sizeof(double);
  std::size_t bytes = filled * steps * value;
  if (assigns) {
    const auto medoids = static_cast<std::size_t>(config.k) * static_cast<std::size_t>(config.n_samples);
    const auto launch = static_cast<std::size_t>(cuda::detail::kMaxPairsPerLaunch);
    const std::size_t block = std::min(n_series, std::max<std::size_t>(1, launch / std::max(medoids, steps)));
    bytes = std::max(bytes, (medoids + block) * steps * value + block * medoids * sizeof(double));
  }
  return bytes;
#  else
  return filled * steps * sizeof(float) + core::packed_size(filled) * sizeof(float);
#  endif
#else
  return 0; // a build without a GPU backend: set_device refused the GPU
#endif
}

/// What `threads` threads of `per_thread` bytes each hold, and a tenth more for what that price does not see: the
/// working set a run reserves.
std::size_t working_set(std::size_t per_thread, int threads)
{
  return static_cast<std::size_t>(threads) * per_thread / 10 * 11;
}

/// The most threads, at most `requested`, whose working set fits beside `need` bytes within `budget`; 0 when one's
/// does not.
int fit_threads(std::size_t budget, std::size_t need, std::size_t per_thread, int requested)
{
  int threads = 0;
  while (threads < requested && need + working_set(per_thread, threads + 1) <= budget) ++threads;
  return threads;
}

/// `ways`, which are not empty, as a sentence's list: "a, b, or c". A refusal and a run out of memory name their ways
/// out alike.
std::string list_or(const std::vector<std::string> &ways)
{
  std::string out = ways.front();
  for (std::size_t i = 1; i < ways.size(); ++i) out += (i + 1 == ways.size() ? ", or " : ", ") + ways[i];
  return out;
}

/// What a run does with its series, decided once (plan_memory): reads them into RAM (the distance matrix in RAM or
/// mapped), streams them through FastCLARA, or refuses.
enum class Route { InRam, Streamed, Refused };

/// What the memory plan is made from: the input's facts and the method's.
struct SourceFacts
{
  std::size_t series = 0;     ///< what the series hold in RAM: the reader's resident bytes, or their values once read
  bool copy = false;          ///< `series` holds the owned copy a GPU uploads of a view's series
  std::size_t staging = 0;    ///< a GPU's host buffers for the series it uploads (gpu_staging_bytes)
  std::size_t clara = 0;      ///< in-RAM FastCLARA's sample matrix, then every sample's labels and nearest distances
  std::size_t fixed = 0;      ///< the method's own terms, in RAM whatever the matrix's storage
  const char *fixed_what = nullptr; ///< what `fixed` is, for a refusal
  bool unpriced = false;      ///< the method's own memory, which the plan does not price, outweighs the matrix (MIP's
                              ///< solver): no way out sized by the plan's terms frees a tenth of what the run needs
  std::size_t matrix = 0;     ///< the distance matrix, where the method holds one
  const char *held = nullptr; ///< why the matrix stays in RAM where it does not fit (nullptr: it is mapped there)
  bool cache = false;         ///< the method keeps the matrix's bytes in RAM, mapped or not (TADPole's cache)
  bool clara_method = false;  ///< the method is FastCLARA, which a refusal does not suggest
  bool can_stream = false;    ///< FastCLARA can stream the series (an Arrow IPC or Parquet file, a sample below N)
  bool streams = false;       ///< the file is one FastCLARA streams (a list-per-row Parquet file, an Arrow IPC file
                              ///< io::IpcFile::streams), whatever the method
  bool halves = false;        ///< --dtype float32 halves `series`: float64 values read from text
  std::size_t streamed = 0;   ///< what a streamed run holds after its stream: every series' name and label
  std::size_t n_series = 0;
  std::size_t length = 0;     ///< the time steps of the longest series expected, which a thread's bytes price
};

/// A run's memory plan (plan_memory): within `budget` (0: none), a reserve for the working set of `threads` of the
/// `requested` threads, `per_thread` bytes each, and the data cap, what the reserve leaves the series and the matrix.
struct MemoryPlan
{
  std::size_t budget = 0, per_thread = 0, length = 0;
  /// What the plan holds in RAM: the series, the method's own terms and the distance matrix where its bytes are in
  /// RAM; a stream, the data cap it reads within.
  std::size_t held = 0;
  int threads = 1, requested = 1;
  Route route = Route::InRam;
  bool mapped = false;  ///< the distance matrix is mapped
  std::string refusal;  ///< why, where `route` is Refused: the InvalidInput's text
  SourceFacts facts;    ///< what the plan is made from, which a run out of memory reads its ways out from

  std::size_t reserve() const { return working_set(per_thread, threads); }
  std::size_t data_cap() const { return budget > reserve() ? budget - reserve() : 0; }
  /// The route, as text() and the run report name it.
  const char *route_name() const
  {
    const char *const routes[] = { "in RAM", "streamed", "refused" };
    return route == Route::InRam && mapped ? "in RAM, the distance matrix mapped" : routes[static_cast<int>(route)];
  }

  /// The plan, one line: what -v prints after "Memory plan: ", and a run out of memory names.
  std::string text() const
  {
    if (budget == 0)
      return "no budget (no --ram-limit, and what the process may use is unknown): in RAM on "
             + std::to_string(requested) + " threads";
    return "budget " + std::to_string(budget) + " bytes, reserve " + std::to_string(reserve()) + " bytes ("
           + std::to_string(threads) + " of " + std::to_string(requested) + " threads, " + std::to_string(per_thread)
           + " bytes each for series of up to " + std::to_string(length) + " time steps), data cap "
           + std::to_string(data_cap()) + " bytes: " + route_name();
  }
};

/// Where a run is, which a run that runs out of memory names (out_of_memory) and its report times: the stage, the
/// input's format, the memory plan once made and the report's figures (unless --no-report). The stages, in order:
/// `check` (the checks that need no series, and apply), `plan` (the input's plan from Parquet metadata or an Arrow IPC
/// file's walk), `read` (the series loaded, and planned where nothing did before: a text file's once read), `storage`
/// (the distance matrix's storage, a --dist-matrix or a checkpoint read), `fill` (the distance matrix, where the method
/// fills it before anything else: PAM, Lloyd's k-medoids, FastCLARA on a sample of every series), `cluster` and
/// `write`.
struct RunStage
{
  const char *stage = "check";
  std::optional<InputFormat> format; ///< empty: the series are in memory
  std::optional<MemoryPlan> plan;
  std::optional<detail::RunReport> report;

  /// Enter stage `name`; the report closes the stage before it, on the plan's threads, and opens `name`'s row.
  void begin(const char *name)
  {
    stage = name;
    if (report) report->begin(name, plan ? plan->threads : 0);
  }

  /// The note of the stage's row in the report, where there is one.
  void note(std::string text)
  {
    if (report) report->stages.back().note = std::move(text);
  }
};

/// The memory plan of a run on `facts` within `budget` (0: none, as without llfio; `explicit_limit`: --ram-limit set),
/// `requested` threads of `per_thread` bytes each. Each decision is one comparison of the budget with terms that do
/// not depend on it, the working set always the requested threads': a budget that runs still runs when raised, and
/// its route never steps back.
/// 1. In RAM on every thread where the series fit within the data cap beside the distance matrix; the matrix is
///    mapped where it does not fit beside them and can be (llfio, an output to map it in, no --dist-matrix), else held
///    in RAM.
/// 2. Else, where FastCLARA can stream the series, streamed on every thread within the data cap, where that holds what
///    the stream keeps after it (every series' name and label); an explicit limit that does not is refused (fewer
///    threads, OMP_NUM_THREADS, leave it more). The stream's own floor (the sampled rows, a sample's PAM, a record
///    batch) is checked inside it, within the data cap.
/// 3. Else in RAM on the most threads whose working set fits beside the series (and a held matrix), fitted once: none
///    is refused under an explicit limit, naming the least limit that runs, one thread under the automatic one, which
///    never refuses.
/// The method's own terms (`fixed`), and TADPole's cache of the matrix, are in RAM whatever the matrix's storage.
MemoryPlan plan_memory(const SourceFacts &facts, std::size_t budget, bool explicit_limit, std::size_t per_thread,
                       int requested)
{
  MemoryPlan plan;
  plan.budget = budget;
  plan.per_thread = per_thread;
  plan.length = facts.length;
  plan.threads = plan.requested = requested;
  plan.facts = facts;
  const std::size_t reserve = working_set(per_thread, requested);
  const std::size_t series = facts.series + facts.staging + facts.clara + facts.fixed;
  // No budget, no plan: in RAM on every thread, nothing mapped or streamed.
  plan.mapped = budget > 0 && facts.matrix > 0 && facts.held == nullptr && series + facts.matrix + reserve > budget;
  const bool matrix = facts.matrix > 0 && (!plan.mapped || facts.cache); // its bytes in RAM
  const std::size_t need = series + (matrix ? facts.matrix : 0);
  plan.held = need;
  if (budget == 0 || need + reserve <= budget) return plan;
  // The refusal: what the run holds beside the working set of `threads`, and the ways out that apply.
  const auto refuse = [&](const std::string &what, int threads, const std::string &needs) {
    std::vector<std::string> ways{ "Raise --ram-limit", "read the series as float32 (--dtype float32)" };
    if (!facts.clara_method) ways.emplace_back("use --method clara");
    if (threads > 1) ways.emplace_back("run fewer threads (OMP_NUM_THREADS)");
    plan.route = Route::Refused;
    plan.threads = threads;
    plan.refusal = "--ram-limit = " + std::to_string(budget) + " cannot hold " + what + " beside the working set of "
                   + std::to_string(working_set(per_thread, threads)) + " bytes (" + std::to_string(threads)
                   + (threads == 1 ? " thread x " : " threads x ") + std::to_string(per_thread)
                   + " bytes: the DTW kernels' scratch for series of up to " + std::to_string(facts.length)
                   + " time steps and the method's tables, x1.1): " + needs + ". " + list_or(ways) + ".";
  };
  if (facts.can_stream) {
    if (facts.streamed + reserve <= budget) {
      plan.route = Route::Streamed;
      plan.held = plan.data_cap();
      return plan;
    }
    if (explicit_limit) {
      refuse("the names and labels of the " + std::to_string(facts.n_series) + " streamed series ("
               + std::to_string(facts.streamed) + " bytes)",
             requested,
             "these alone need " + std::to_string(facts.streamed + reserve)
               + " bytes, and the stream's own floor (the sampled rows, a sample's PAM, a record batch) comes on "
                 "top, checked within the data cap, --ram-limit less the working set");
      return plan;
    }
  }
  plan.threads = fit_threads(budget, need, per_thread, requested);
  if (plan.threads > 0) return plan;
  plan.threads = 1;
  if (!explicit_limit) return plan;
  std::string terms = std::to_string(facts.series) + " for the series"
                      + (facts.copy ? " and the owned copy the GPU uploads of them" : "");
  if (facts.staging > 0) terms += ", " + std::to_string(facts.staging) + " for the GPU's staging of them";
  if (facts.clara > 0) terms += ", " + std::to_string(facts.clara) + " for FastCLARA's sample matrix and labels";
  if (facts.fixed > 0) terms += ", " + std::to_string(facts.fixed) + " for " + facts.fixed_what;
  if (matrix)
    terms += ", " + std::to_string(facts.matrix) + " for the distance matrix, "
             + (facts.held != nullptr ? facts.held : "which TADPole keeps in RAM as its cache");
  refuse((matrix ? "the series and the distance matrix (" : "the series (") + std::to_string(need) + " bytes: " + terms
           + ")",
         1, "it needs at least " + std::to_string(need + working_set(per_thread, 1)) + " bytes");
  return plan;
}

/// Bind persistent distance storage once every distance-affecting setting has reached the Problem, as the memory plan
/// decided it (`mapped`). Returns the mapped file, or nullopt when the method keeps its own storage or the matrix
/// stays in RAM. A mapped matrix costs one sequential pass and keeps the run out of swap. A method that reads no parent
/// matrix (holds_matrix) maps none. TADPole reads the matrix when it is complete (a file or a cache from an earlier
/// run) and otherwise computes the pairs it needs.
std::optional<fs::path> configure_distance_storage(Problem &prob, const Config &config, Method method,
                                                   bool clara_uses_full_sample, bool mapped, const fs::path &cache_path)
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
  if (!mapped || !holds_matrix(method, clara_uses_full_sample)) return std::nullopt;
  prob.use_mmap_distance_matrix(cache_path); // the plan maps nothing in a build without llfio
  return cache_path;
}

/// What a run hands to its Result.
struct Outcome
{
  std::shared_ptr<Problem> problem;
  core::ClusteringResult result;
  Method method;
  std::vector<std::string> streamed_names; ///< a RAM-limited run's series names, which its Problem cannot hold
  std::string report;                      ///< the run report's text; empty under --no-report
};

/// What the run report calls an input's format.
const char *format_name(std::optional<InputFormat> format)
{
  if (!format) return "in memory";
  return *format == InputFormat::Parquet ? "Parquet" : *format == InputFormat::ArrowIPC ? "Arrow IPC" : "text";
}

/// The run itself; `data` holds in-memory series, else config.input is read; `solver`, where given, clusters MIP on
/// HiGHS and LR-core's Kelley root in RAM (detail::run). `stage` follows it: the stage it is in, its memory plan once
/// made, and the report's figures.
Outcome execute(const Config &config, std::optional<Data> data, const ExternalSolver &solver, RunStage &stage)
{
  // ---- 1. Every check that needs no series, before any file is touched (stage check, run_with's) ----
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

  auto problem = std::make_shared<Problem>(run_name(config));
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
  std::optional<InputFormat> &format = stage.format; // empty: the series are in memory
  if (!data) format = format_of(input);
  require_input_options_apply(config, format);

  // A --checkpoint that cannot hold a checkpoint stops the run here, before any
  // work: the save comes after clustering, so a typo would cost the whole run.
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

  // ---- 2. The method, and N and the memory plan where Parquet metadata or an Arrow IPC file's walk give them ----
  stage.begin("plan");
  const Clock clk;
  Method method = config.method;
  bool stream_payload = false;
  std::size_t n_series = 0;
  bool clara_uses_full_sample = false;
  bool clara_planned = false;
  const auto plan_clara = [&] {
    if (method != Method::CLARA || clara_planned) return;
    const auto plan = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
    clara_uses_full_sample = plan.sample_size == plan.n_points;
    if (stream_payload) algorithms::detail::validate_streaming_clara_plan(plan, "run");
    clara_planned = true;
  };
  // A streamed run holds no series: the options that need the series are checked before it (what it holds after the
  // stream, every series' name and label, the plan holds within the data cap); `what` is the input's format,
  // `resident` the price that chose the stream.
  const auto check_stream = [&](const char *what, const char *by, std::size_t resident) {
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
  // nearest distances, and with them the winner's labels that Problem::set_result copies (and the medoid flags it
  // makes), as the stream's `finish` counts them.
  const auto clara_bytes = [&](const algorithms::detail::ClaraPlan &sample) {
    const std::size_t pam = core::packed_size(static_cast<std::size_t>(sample.sample_size))
                            * core::value_bytes(config.matrix_dtype);
    const auto n = static_cast<std::size_t>(sample.n_points);
    const std::size_t pass = algorithms::detail::clara_pass_bytes(sample, static_cast<std::size_t>(clara.n_samples));
    return std::max(pam, pass + n * sizeof(index_t) + n / 8);
  };
  // FastCLARA's plan where `resolved` is clara and there are series (none otherwise). k above the number of series is
  // refused here as for every input, before CLARA's plan.
  const auto clara_sample = [&](Method resolved) -> std::optional<algorithms::detail::ClaraPlan> {
    if (n_series == 0 || resolved != Method::CLARA) return std::nullopt;
    require_clusterable(config.k, n_series);
    return resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
  };
  // Whether FastCLARA can stream a file of one series per row (`rows_are_series`): a sample smaller than N; the automatic
  // budget not beside --checkpoint or --dist-matrix, which an explicit one refuses (check_stream).
  const auto can_stream_file = [&](const std::optional<algorithms::detail::ClaraPlan> &sample, bool rows_are_series) {
    return sample && sample->sample_size < sample->n_points && rows_are_series
           && (config.ram_limit != 0 || (config.checkpoint.empty() && config.dist_matrix.empty()));
  };

  // The memory plan (plan_memory), made once where the input's facts are known: an Arrow IPC file's walk, a Parquet
  // file's metadata where they route it, else the series read. -v prints it; its threads cap every region from there
  // to the end of the run (TeamCap; the release reaches the full team, full_team_threads()).
  const std::size_t budget = core::memory_budget(config.ram_limit);
  const int requested = get_max_threads();
  std::optional<MemoryPlan> &memory = stage.plan;
  std::optional<TeamCap> team;
  // Why a distance matrix that does not fit beside the series stays in RAM (nullptr: it is mapped). It maps only with
  // an output directory, into the --checkpoint directory when one is given.
#ifdef DTWC_HAS_MMAP
  const char *const unmapped = config.output.empty() ? "which a run without an output directory keeps in RAM" : nullptr;
#else
  const char *const unmapped = "which a build without llfio cannot map (DTWC_ENABLE_LLFIO=OFF)";
#endif
  const char *const held = !config.dist_matrix.empty() ? "which --dist-matrix reads into RAM" : unmapped;
  // `auto` by the plan: clara where pam's matrix can be held neither in RAM beside the series and one thread's working
  // set nor mapped.
  const auto resolve = [&](std::size_t series, std::size_t length, std::size_t ndim, core::Precision precision) {
    if (method != Method::Auto) return method;
    const std::size_t matrix = core::packed_size(n_series) * core::value_bytes(config.matrix_dtype);
    const std::size_t pam = thread_bytes(prob, config, Method::PAM, false, n_series, length, ndim, precision);
    const std::size_t staging = gpu_staging_bytes(config, Method::PAM, std::nullopt, n_series, length, ndim);
    return resolve_method(method, config.device, n_series,
                          budget == 0 || held == nullptr || series + staging + matrix + working_set(pam, 1) <= budget);
  };
  // Whether `series`, as plan_of takes it, holds the owned copy a GPU uploads of a view's series.
  bool upload_copy = false;
  // The plan of `resolved` on series that hold `series` bytes, `values` values of up to `length` time steps of `ndim`
  // features in `precision`; FastCLARA's `sample` where it runs; `streams`: a file FastCLARA streams.
  const auto plan_of = [&](Method resolved, const std::optional<algorithms::detail::ClaraPlan> &sample,
                           std::size_t series, std::size_t values, std::size_t length, std::size_t ndim,
                           core::Precision precision, bool can_stream, bool streams, std::size_t streamed) {
    const bool full = sample && sample->sample_size == sample->n_points;
    SourceFacts facts;
    facts.series = series;
    facts.copy = upload_copy;
    facts.staging = gpu_staging_bytes(config, resolved, sample, n_series, length, ndim);
    facts.clara = sample && !full ? clara_bytes(*sample) : 0;
    const std::size_t pairs = core::packed_size(n_series);
    facts.matrix = holds_matrix(resolved, full) ? pairs * core::value_bytes(config.matrix_dtype) : 0;
    // The method's own terms, in RAM whatever the matrix's storage: TADPole's flag a pair (tadpole.cpp's `computed`),
    // its cache of the matrix's pairs (`cache`, unless a complete --dist-matrix is read in its place) and, where it
    // prunes (`can_prune`: bounds_valid's Standard DTW under L1 on univariate float64 series, no missing-data
    // strategy), each series' envelope (`envs`: an upper and a lower vector of its values); the hierarchical
    // clustering's dense copy of N x N doubles (hierarchical.cpp's `work`); OneBatchPAM's batch table, N x m doubles,
    // m as one_batch_pam.cpp's automatic_batch_size or --batch-size takes it, and its per-point vectors, at most 8
    // doubles a point. MIP's and LR-core's solver memory is not planned: MIP's outweighs the matrix (150 series of 10
    // steps: the plan held 102,600 bytes, the run needed about 118 MB), LR-core's does not (2,000 series: float32 or the
    // matrix mapped got runs through where the plain run was refused, 2026-11-03).
    facts.unpriced = resolved == Method::MIP;
    if (resolved == Method::TADPole) {
      const auto distance = prob.distance();
      const bool prunes = distance.variant.variant == core::DTWVariant::Standard
                          && distance.metric == core::MetricType::L1 && ndim == 1
                          && precision != core::Precision::Float32
                          && distance.missing == core::MissingStrategy::Error;
      facts.fixed = pairs + (prunes ? 2 * (values * sizeof(double) + n_series * sizeof(std::vector<double>)) : 0);
      facts.fixed_what = prunes ? "TADPole's flag a pair and the series' envelopes" : "TADPole's flag a pair";
      facts.cache = true;
    } else if (resolved == Method::Hierarchical) {
      facts.fixed = n_series * n_series * sizeof(double);
      facts.fixed_what = "the hierarchical clustering's dense copy of the matrix";
    } else if (resolved == Method::OneBatch) {
      const auto log2_steps = static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(n_series) + 1.0)));
      std::size_t m = config.batch_size > 0 ? static_cast<std::size_t>(config.batch_size)
                                            : std::max<std::size_t>(64, 20 * log2_steps);
      m = std::max(std::min(m, n_series), static_cast<std::size_t>(config.k));
      facts.fixed = n_series * (m + 8) * sizeof(double);
      facts.fixed_what = "OneBatchPAM's batch table and per-point vectors";
    }
    facts.clara_method = resolved == Method::CLARA;
    facts.held = held;
    facts.can_stream = can_stream;
    facts.streams = streams;
    facts.halves = format == InputFormat::Text && precision == core::Precision::Float64;
    facts.streamed = streamed;
    facts.n_series = n_series;
    facts.length = length;
    memory = plan_memory(facts, budget, config.ram_limit != 0,
                         thread_bytes(prob, config, resolved, full, n_series, length, ndim, precision), requested);
    if (config.verbose) std::cout << "Memory plan: " << memory->text() << "\n";
    if (stage.report)
      stage.report->plan = detail::RunReport::Plan{ memory->budget,   memory->reserve(), memory->data_cap(),
                                                    memory->held,     memory->per_thread, memory->length,
                                                    memory->threads,  memory->requested, memory->route_name() };
    if (memory->route != Route::Refused) team.emplace(memory->threads);
    return *memory;
  };
  // The series as the report gives them: `values` values of `ndim` features, `bytes` of them in RAM, of `shortest`
  // to `longest` time steps (0: not known, a stream's).
  const auto report_input = [&](std::size_t series, std::size_t values, std::size_t ndim, std::size_t bytes,
                                std::size_t shortest, std::size_t longest, bool streamed) {
    if (stage.report)
      stage.report->input =
        detail::RunReport::Input{ format_name(format), series, values, ndim, bytes, shortest, longest, streamed };
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
    // The metadata hold no lengths: series of twice the mean expected.
    const std::size_t length = n_series > 0 ? 2 * values / n_series : 0;
    const Method resolved = resolve(resident_bytes, length, 1, config.dtype);
    // A CLARA run holds beside the series a sample's PAM matrix, then every sample's labels and nearest distances,
    // in RAM as streamed.
    const auto sample = clara_sample(resolved);
    const std::size_t resident = resident_bytes + (sample ? clara_bytes(*sample) : 0);
    const bool can_stream = can_stream_file(sample, layout == detail::ParquetLayout::ListColumn);
    const MemoryPlan planned = plan_of(resolved, sample, resident_bytes, values, length, 1, config.dtype, can_stream,
                                       layout == detail::ParquetLayout::ListColumn,
                                       names_bytes + 2 * n_series * sizeof(index_t));
    // The series alone beyond the budget: a route that cannot stream them is refused as it says why (the method or
    // the layout, plan_parquet_load; a sample of every series, FastCLARA's plan). An input of no series is refused.
    const bool over = planned.route == Route::Refused && !can_stream && resident > budget;
    (void)detail::plan_parquet_load(resolved, config.device, n_series, resident, over ? budget : 0, layout);
    if (over && sample) algorithms::detail::validate_streaming_clara_plan(*sample, "run");
    if (planned.route == Route::Refused) throw InvalidInput(planned.refusal);
    method = resolved;
    stream_payload = planned.route == Route::Streamed;
    if (stream_payload) {
      clara.ram_limit_bytes = planned.data_cap();
      clara.parquet_path = input;
      clara.parquet_column = config.column;
      clara.parquet_skip_cols = config.skip_cols;
      clara.parquet_skip_rows = config.skip_rows;
      clara.use_float32 = f32;
      clara.force_parquet_streaming = true;
    }
    plan_clara();
    if (stream_payload) check_stream("Parquet", "Parquet metadata", resident);
    stage.note("Parquet metadata");
    if (stream_payload) report_input(n_series, values, 1, 0, 0, 0, true);
  }
#endif

  // Arrow IPC: its walk gives N and every record batch's rows and decoded bytes before any batch is decoded, so what
  // the in-RAM route would hold at once is priced first (io::IpcFile::resident_bytes; a mapped view's values are the
  // page cache's). Beyond the data cap FastCLARA reads the file a record batch at a time, where that holds less
  // (decoded batches, the file read without llfio, a copy in the other precision); beyond an explicit limit the series
  // alone, any other route is refused as it says why.
  std::optional<io::IpcFile> ipc;
  std::string storage; // how the file's series are held once read (-v and the report's read row name it)
  if (format == InputFormat::ArrowIPC) {
    ipc.emplace(input);
    n_series = static_cast<std::size_t>(ipc->num_rows());
    const std::size_t ipc_bytes = ipc->resident_bytes(config.dtype, config.device == Device::GPU);
    // Series of up to the longest the walk read from a mapped file's offsets, else of twice the mean length.
    std::size_t length = ipc->longest_steps();
    if (length == 0 && n_series > 0) length = 2 * ipc->num_values() / (n_series * ipc->ndim());
    const Method resolved = resolve(ipc_bytes, length, ipc->ndim(), config.dtype);
    const auto sample = clara_sample(resolved);
    const std::size_t resident = ipc_bytes + (sample ? clara_bytes(*sample) : 0);
    const bool streams = ipc->streams(config.dtype);
    const bool can_stream = can_stream_file(sample, streams);
    // The names are read after the stream, beside the labels: their read is priced with them.
    const std::size_t names_bytes = ipc->series_names_bytes() + ipc->names_read_bytes();
    // A GPU uploads an owned copy of the views read_arrow_ipc gives in the file's precision (resident_bytes prices it).
    upload_copy = config.device == Device::GPU && (config.dtype == core::Precision::Float32) == ipc->is_f32();
    const MemoryPlan planned = plan_of(resolved, sample, ipc_bytes, ipc->num_values(), length, ipc->ndim(),
                                       config.dtype, can_stream, streams, names_bytes + 2 * n_series * sizeof(index_t));
    method = resolved;
    if (planned.route == Route::Streamed) {
      stream_payload = true;
      clara.ram_limit_bytes = planned.data_cap();
      plan_clara();
      check_stream("Arrow IPC", "Arrow IPC walk", resident);
    } else if (planned.route == Route::Refused && !can_stream && resident > config.ram_limit) {
      // CLARA is named only where its stream could read the file: a file that streams, sampled in fewer than its series
      // (the plan of these options; none where CLARA cannot run on them).
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
      if (!streams && config.device == Device::GPU)
        why = "its values are mapped where they lie in the file, and a mapped file's values are not streamed: the "
              "views of its " + std::to_string(n_series) + " series and the owned copy the GPU uploads of them are "
              "held at once. Raise --ram-limit, or run on the CPU (--device cpu), which holds no copy.";
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
    if (planned.route == Route::Refused) throw InvalidInput(planned.refusal);
    stage.note("Arrow IPC walk");
    if (stream_payload) report_input(n_series, ipc->num_values(), ipc->ndim(), 0, 0, 0, true);
    if (!stream_payload) {
      storage = ipc->storage(config.dtype); // the read below repeats the walk's decisions on the same file
      ipc.reset();                          // read below as every Arrow IPC file is read
    }
  }

  // ---- 3. Load the series into RAM ----
  stage.begin("read");
  if (stream_payload) stage.note("none: FastCLARA streams them, inside cluster");
  if (!stream_payload) {
    Data series = data ? std::move(*data) : read_input(input, format.value_or(InputFormat::Text), config);
    const std::string held = std::to_string(series.size()) + " series" + (storage.empty() ? "" : ", " + storage);
    if (config.verbose) {
      const char *from = !format                            ? " in memory"
                         : format == InputFormat::Parquet  ? " from Parquet"
                         : format == InputFormat::ArrowIPC ? " from Arrow IPC"
                                                           : "";
      std::cout << "Data loaded" << from << ": " << held << " [" << clk << "]\n";
    }
    // A reader stores what --dtype names; Float64 series passed in memory, a view among them, are converted here.
    if (config.dtype == core::Precision::Float32 && !series.is_f32()) {
      series = detail::owned_copy(series, core::Precision::Float32);
      if (config.verbose) std::cout << "Converted to float32 (2x memory saving)\n";
    }
    std::size_t elements = 0, longest = 0, shortest = series.size() > 0 ? SIZE_MAX : 0;
    for (std::size_t i = 0; i < series.size(); ++i) {
      elements += series.series_flat_size(i);
      longest = std::max(longest, series.series_flat_size(i) / series.ndim);
      shortest = std::min(shortest, series.series_flat_size(i) / series.ndim);
    }
    const std::size_t value_bytes = elements * (series.is_f32() ? sizeof(float) : sizeof(data_t));
    report_input(static_cast<std::size_t>(series.size()), elements, series.ndim, value_bytes, shortest, longest, false);
    stage.note(held);
    if (config.verbose && series.size() > 0)
      std::cout << "  Data memory: ~" << (value_bytes / (1ULL << 20)) << " MB (" << series.size() << " series, "
                << (elements / series.size()) << " avg length, " << name_of(core::precision_names, series.precision)
                << ")\n";
    const std::size_t ndim = series.ndim;
    const core::Precision precision = series.precision;
    const bool view = series.is_view();
    prob.set_data(std::move(series));
    n_series = prob.size();
    // Planned on the series read (CSV, a folder, series in memory, Parquet its metadata did not route): their values,
    // the longest of them, and on a GPU the owned copy it uploads of a view.
    if (!memory) {
      require_clusterable(config.k, n_series); // before CLARA's plan
      method = resolve(value_bytes, longest, ndim, precision);
      std::optional<algorithms::detail::ClaraPlan> sample;
      if (method == Method::CLARA) sample = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "run");
      upload_copy = config.device == Device::GPU && view;
      const std::size_t copy = upload_copy ? detail::owned_series_bytes(elements, n_series, false) : 0;
      if (const MemoryPlan planned =
            plan_of(method, sample, value_bytes + copy, elements, longest, ndim, precision, false, false, 0);
          planned.route == Route::Refused)
        throw InvalidInput(planned.refusal);
    }
  }
  require_clusterable(config.k, n_series); // before a cache or checkpoint is touched

  if (config.method == Method::Auto && config.verbose)
    std::cout << "Auto-selected method: " << method_name(method) << " (N=" << n_series << ")\n";
  plan_clara();
  prob.set_method(method);
  if (stage.report) stage.report->method = method;

  // ---- 4. Distance storage, once every distance setting is in place ----
  // A mapped matrix is <name>.dtwm in the --checkpoint directory, where it is
  // the checkpoint, else in the output directory; a run that writes nothing
  // keeps its matrix in RAM.
  stage.begin("storage");
  std::optional<fs::path> cache;
  if (!config.output.empty()) {
    const auto path = checkpoint_path(prob, config.checkpoint.empty() ? config.output : config.checkpoint);
    cache = configure_distance_storage(prob, config, method, clara_uses_full_sample, memory->mapped, path);
    if (cache && config.verbose) std::cout << "Using memory-mapped distance matrix: " << *cache << "\n";
  }
  // A matrix the user supplied but that cannot be loaded is an error: going on
  // without it would silently recompute every distance and exit 0.
  if (!config.dist_matrix.empty()) {
    try {
      prob.read_distance_matrix(utf8_to_path(config.dist_matrix), config.not_computed);
    } catch (const std::bad_alloc &) {
      throw; // the system refused memory, no fault of the file's: run() reports it with the stage
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
  bool resumed = false;
  if (!config.checkpoint.empty() && !cache) { // a mapped matrix is its own checkpoint
    resumed = load_checkpoint(prob, config.checkpoint); // false: no file, start fresh
    if (config.verbose)
      std::cout << (resumed ? "Resumed from checkpoint: " + config.checkpoint + "\n"
                            : "No checkpoint in " + config.checkpoint + ", starting fresh.\n");
  }
  std::string storage_note = stream_payload || !holds_matrix(method, clara_uses_full_sample)
                               ? std::string("no distance matrix of every series")
                             : cache ? "the distance matrix mapped, " + path_to_utf8(cache->filename())
                                     : std::string("the distance matrix in RAM");
  if (!config.dist_matrix.empty()) storage_note += "; --dist-matrix read";
  if (resumed) storage_note += "; the checkpoint resumed";
  stage.note(std::move(storage_note));

  // ---- 5. Fill, then cluster; the matrix methods fill through the Problem, on its device ----
  // PAM, Lloyd's k-medoids and FastCLARA on a sample of every series fill the whole distance matrix before anything
  // else, past checks this run made already (k, max-iter, n-init, CLARA's plan), so the fill is a stage of its own;
  // the idempotent fill leaves their clustering nothing to compute. Every other method refuses what it cannot run
  // before its own fill (the hierarchical clustering above its size, a MIP without its solver, LR-core's Kelley root
  // without HiGHS), or fills nothing whole: its distances are computed inside the cluster stage.
  // A RAM-limited Parquet or Arrow IPC run is the one whose series the Problem does
  // not hold: FastCLARA streams them.
  const bool fills_first = !stream_payload && (method == Method::PAM || method == Method::Kmedoids
                                               || (method == Method::CLARA && clara_uses_full_sample));
  stage.begin(fills_first ? "fill" : "cluster");
  if (config.verbose) std::cout << "Running " << progress_label(method) << " (k=" << config.k << ") ...\n";
  if (fills_first) {
    stage.note(prob.is_distance_matrix_filled()
                 ? "complete on entry"
                 : std::to_string(n_series) + " series' matrix on " + device_text(config)
                     + (config.device == Device::GPU && prob.data().is_view() ? ", from an owned copy of their views"
                                                                              : ""));
    prob.fill_distance_matrix();
    stage.begin("cluster");
  }
  stage.note(std::string(progress_label(method)) + (fills_first ? "" : ", any distances it computes inside"));
  // In RAM the Problem clusters, but where detail::run was given a solver (a binding that links no HiGHS gives its
  // own) and the run is MIP on HiGHS or LR-core's Kelley root, tested as Python's Problem.cluster override tests them:
  // the solver's clustering is the run's, published on the Problem, which the Result reads.
  const auto cluster_in_ram = [&] {
    if (!solver || !((prob.method() == Method::MIP && prob.solver() == Solver::HiGHS)
                     || (prob.method() == Method::LRCore && prob.mip_settings.lr_root == LRRoot::Kelley)))
      return prob.cluster();
    core::ClusteringResult solved = solver(problem);
    prob.set_result(solved);
    return solved;
  };
  std::vector<std::string> streamed_names;
  core::ClusteringResult result;
  try {
    if (stream_payload && ipc) {
      // The Problem holds no series: its distance takes their features a time step from an empty Data of them.
      prob.set_data(Data(std::vector<std::vector<data_t>>{}, std::vector<std::string>{}, ipc->ndim()));
      const auto plan = resolve_clara_plan(static_cast<std::int64_t>(n_series), clara, "fast_clara");
      result = config.dtype == core::Precision::Float32
                 ? algorithms::detail::fast_clara_chunked<true>(prob, clara, *ipc, plan)
                 : algorithms::detail::fast_clara_chunked<false>(prob, clara, *ipc, plan);
      streamed_names = ipc->series_names(clara.ram_limit_bytes - 2 * n_series * sizeof(index_t)); // beside the labels
    } else {
#ifdef DTWC_HAS_PARQUET
      result = stream_payload ? algorithms::fast_clara_parquet(prob, clara) : cluster_in_ram();
      if (stream_payload)
        streamed_names =
          io::ParquetChunkReader(input, config.column, config.skip_cols, config.skip_rows).series_names();
#else
      result = cluster_in_ram();
#endif
    }
  } catch (const InvalidInput &refused) {
    // A stream's own refusals (its readers', FastCLARA's) check its floor within the data cap, which they name as
    // --ram-limit: say what that is, after their text.
    const std::string text = refused.what();
    if (!stream_payload || text.find("--ram-limit") == std::string::npos) throw;
    throw InvalidInput(text + (text.ends_with('.') ? " " : ". ")
                       + "The --ram-limit it names is the stream's data cap: --ram-limit = "
                       + std::to_string(memory->budget) + " less the reserve of " + std::to_string(memory->reserve())
                       + " bytes for " + std::to_string(memory->threads)
                       + " threads (-v prints the memory plan); raise --ram-limit by what the stream lacks, or run "
                         "fewer threads (OMP_NUM_THREADS) to shrink the reserve.");
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
  if (stage.report) {
    detail::RunReport::Clustering clustering{ method, progress_label(method), result.total_cost, result.iterations,
                                              result.converged, std::vector<std::size_t>(result.medoid_indices.size()),
                                              result.medoid_indices };
    for (const index_t label : result.labels)
      if (label >= 0 && static_cast<std::size_t>(label) < clustering.sizes.size())
        ++clustering.sizes[static_cast<std::size_t>(label)];
    stage.report->clustering = std::move(clustering);
  }

  // ---- 6. Checkpoints first, then the outputs ----
  // The checkpoint goes before the results, so a result write that fails cannot lose the distance matrix. A save that
  // fails is kept and raised once the results are on disk, so it cannot lose them either: what to do, and its cause.
  stage.begin("write");
  stage.note(std::string(config.checkpoint.empty() ? "" : "the checkpoint, then ")
             + (config.output.empty() ? std::string("no result files (no -o)") : "the result files"));
  std::optional<std::string> checkpoint_failure;
  if (!config.checkpoint.empty()) {
    try {
      save_checkpoint(prob, config.checkpoint);
      if (config.verbose) std::cout << "Checkpoint saved to " << config.checkpoint << "\n";
    } catch (const std::bad_alloc &) { // the system refused memory: no disk space frees it
      checkpoint_failure = "out of memory; omit --checkpoint, or give the process more memory";
    } catch (const std::exception &e) {
      checkpoint_failure = "free space or pass another directory to --checkpoint, or omit it. Cause: "
                           + std::string(e.what());
    }
  }

  if (!config.output.empty()) {
    detail::WriteTimes times;
    times.cpu_clock = &core::cpu_seconds;
    detail::write_result_files(prob, output, false, config.verbose ? &std::cout : nullptr, streamed_names,
                               stage.report ? &times : nullptr);
    if (stage.report)
      stage.report->silhouettes = detail::RunReport::Silhouettes{ times.silhouette_wall, times.silhouette_cpu,
                                                                  times.mean_silhouette, times.skipped };
  }
  if (checkpoint_failure)
    throw IOError("--checkpoint '" + config.checkpoint + "': the distance checkpoint cannot be saved (the results are "
                  "written to '" + config.output + "'); " + *checkpoint_failure);

  return { std::move(problem), std::move(result), method, std::move(streamed_names), {} };
}

/// The name a failed run's report gives its error: its type in base/error.hpp.
const char *kind_of(const std::exception &error)
{
  if (dynamic_cast<const UndefinedScore *>(&error)) return "UndefinedScore";
  if (dynamic_cast<const InvalidInput *>(&error)) return "InvalidInput";
  if (dynamic_cast<const SolverError *>(&error)) return "SolverError";
  if (dynamic_cast<const DeviceError *>(&error)) return "DeviceError";
  if (dynamic_cast<const IOError *>(&error)) return "IOError";
  return "std::exception";
}

void write_report(const Config &config, const std::string &text)
{
  const fs::path path = report_path(config);
  auto out = open_output(path);
  out << text;
  close_output(out, path);
}

/// The ways out a run out of memory names, each only where it holds less in this run. Once the method's terms are
/// held (storage, cluster, write): float32 for the matrix (not beside a --checkpoint, whose file keeps its width) or a
/// text file's series, --method clara (not beside a --dist-matrix or --checkpoint, which non-full FastCLARA refuses),
/// the matrix mapped, fewer threads, each where it frees at least a tenth of what the run needs, the plan's own margin
/// (working_set's tenth): what the plan holds and its reserve, and at the storage stage a matrix it maps, whose map is
/// what was refused there; a method whose own memory the plan does not price (`unpriced`): --method clara alone. A
/// stream: a lower --ram-limit, down to its floor; at a read, a file FastCLARA streams: --method clara below a lower
/// one; a --dist-matrix read: the distances computed (its CSV's cells, which the plan does not price, outweigh the
/// matrix four times); a text file's read, before its plan: float32, or, where llfio maps it, Arrow IPC, whose values
/// are mapped where they lie (without llfio the file is read whole, and the one record batch dtwc-convert writes is a
/// stream's floor). A cap on address space counts a mapped file. Where none applies, more memory.
std::string ways_out(const RunStage &stage, const Config &config)
{
  const std::string_view at = stage.stage;
  const bool clara_can = config.dist_matrix.empty() && config.checkpoint.empty();
  const char *const down_to = " (no lower than its floor, which a refusal names)";
#ifdef DTWC_HAS_MMAP
  const char *const counted = " (which a cap on address space, ulimit -v, counts)";
#endif
  std::vector<std::string> ways;
  if (!stage.plan) {
    if (at == "read" && stage.format == InputFormat::Text) {
      if (config.dtype == core::Precision::Float64) ways.emplace_back("--dtype float32");
#ifdef DTWC_HAS_MMAP
      ways.emplace_back(std::string("the file converted to Arrow IPC (dtwc-convert), whose values are mapped where they "
                                    "lie") + counted);
#endif
    }
  } else if (stage.plan->route == Route::Streamed) {
    ways.emplace_back(std::string("a lower --ram-limit, within which the stream reads") + down_to);
  } else if (at == "read") {
    if (stage.plan->facts.streams && clara_can)
      ways.emplace_back(std::string(stage.plan->facts.clara_method ? "" : "--method clara with ")
                        + "a lower --ram-limit, which streams the series" + down_to);
  } else if (at == "storage" && !config.dist_matrix.empty()) {
    ways.emplace_back("the distances computed without --dist-matrix (its CSV is read at 16 bytes a cell beside the "
                      "matrix)");
  } else if (at != "check" && at != "plan" && stage.plan->facts.unpriced) {
    if (clara_can) ways.emplace_back("--method clara");
  } else if (at != "check" && at != "plan") {
    const MemoryPlan &plan = *stage.plan;
    const SourceFacts &facts = plan.facts;
    const std::size_t matrix = !plan.mapped || facts.cache ? facts.matrix : 0; // its bytes in RAM
    const std::size_t map = plan.mapped && at == "storage" ? facts.matrix : 0, need = plan.held + plan.reserve() + map;
    const auto frees = [&](std::size_t bytes) { return bytes > 0 && bytes >= need / 10; };
    if (config.matrix_dtype == core::Precision::Float64 && config.checkpoint.empty() && frees((matrix + map) / 2))
      ways.emplace_back("--matrix-dtype float32");
    if (facts.halves && frees(facts.series / 2)) ways.emplace_back("--dtype float32");
    if (!facts.clara_method && clara_can && frees(matrix + map + facts.fixed)) ways.emplace_back("--method clara");
#ifdef DTWC_HAS_MMAP
    if (!facts.cache && config.dist_matrix.empty() && frees(matrix))
      ways.emplace_back(std::string("a lower --ram-limit") + (config.output.empty() ? " with an output directory (-o)" : "")
                        + ", so the plan maps the distance matrix to a file" + counted);
#endif
    if (frees(plan.reserve() - working_set(plan.per_thread, 1))) ways.emplace_back("fewer threads (OMP_NUM_THREADS)");
  }
  return ways.empty() ? "more memory for the process" : list_or(ways);
}

/// What a run that runs out of memory throws in place of its std::bad_alloc (std::bad_array_new_length among them): the
/// stage, the memory plan and what it holds in RAM, the process's peak resident memory and the ways out. run() builds
/// it in its catch, outside everything the run held (execute's Problem, series and TeamCap, the threads' scratch), so
/// that memory is free by then.
InvalidInput out_of_memory(const RunStage &stage, const Config &config)
{
  const std::string plan = stage.plan ? stage.plan->text() + "; the plan holds " + std::to_string(stage.plan->held)
                                          + " bytes in RAM"
                                      : std::string("none yet");
  return InvalidInput(std::string(stage.stage) + " stage: out of memory; plan: " + plan + "; the process peaked at "
                      + std::to_string(core::peak_memory()) + " bytes resident, mapped files' pages included; ways out: "
                      + ways_out(stage, config) + ".");
}

/// A run of `config` on `data` (else config.input), `solver` as detail::run takes it, its report rendered and, with an
/// output directory, written last.
/// Every thread's scratch is given back as the run starts, returns or throws (ScratchRelease). A std::bad_alloc is
/// caught outside it and outside execute, whose Problem, series and TeamCap are gone by then, and becomes
/// out_of_memory's InvalidInput. A run that fails past its checks writes its report up to the failing stage, with the
/// error, and throws its own exception on: a report that cannot be made or written is dropped, never the error.
Outcome run_with(const Config &config, std::optional<Data> data, const ExternalSolver &solver)
{
  RunStage stage;
  try {
    try {
      if (config.report) stage.report.emplace(config);
      stage.begin("check"); // the scratch given back first, the checks then
      const core::ScratchRelease release;
      Outcome outcome = execute(config, std::move(data), solver, stage);
      if (stage.report) {
        stage.report->close(stage.plan ? stage.plan->threads : 0);
        outcome.report = stage.report->render();
        if (!config.output.empty()) write_report(config, outcome.report);
      }
      return outcome;
    } catch (const std::bad_alloc &) {
      throw out_of_memory(stage, config);
    }
  } catch (const std::exception &error) {
    if (stage.report && !config.output.empty() && std::string_view(stage.stage) != "check") {
      try {
        stage.report->close(stage.plan ? stage.plan->threads : 0);
        stage.report->error_kind = kind_of(error);
        stage.report->error_text = error.what();
        write_report(config, stage.report->render());
      } catch (...) { // the run's own error is what it throws
      }
    }
    throw;
  }
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

Method resolve_method(Method method, Device device, std::size_t n_series, bool pam_matrix_holds)
{
  const Method resolved = resolve_method(method, device, n_series);
  return method == Method::Auto && resolved == Method::PAM && !pam_matrix_holds ? Method::CLARA : resolved;
}

std::size_t core::memory_budget(std::size_t ram_limit)
{
  return ram_limit != 0 ? ram_limit : available_memory() / 10 * 9;
}

std::string run_name(const Config &config)
{
  return config.name.empty() ? detail::default_name(utf8_to_path(config.input)) : config.name;
}

fs::path report_path(const Config &config)
{
  return utf8_to_path(config.output) / utf8_to_path(run_name(config) + "_report.txt");
}

Result detail::run(const Config &config, std::optional<Data> data, const ExternalSolver &solver)
{
  auto [problem, result, method, streamed_names, report] = run_with(config, std::move(data), solver);
  return Result(std::move(problem), result.total_cost, device_text(config), method, result.iterations,
                result.converged, std::move(streamed_names), std::move(report));
}

Result run(const Config &config) { return detail::run(config, std::nullopt, {}); }

Result run(const Config &config, Data data) { return detail::run(config, std::move(data), {}); }

} // namespace dtwc
