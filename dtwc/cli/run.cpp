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
#include "../algorithms/one_batch_pam.hpp"
#include "../algorithms/tadpole.hpp"
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
#include "../cuda/cuda_dtw.cuh"    // cuda_uses_fp32
#include "../cuda/launch_prep.hpp" // kMaxPairsPerLaunch
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
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
/// of a group). cpu+gpu prices the CPU's scratch as `cpu` does (its fill's tasks and its pass's points), and its GPU's
/// staging as `gpu` does (gpu_staging_bytes).
std::size_t thread_bytes(const Problem &prob, const Config &config, Method method, bool clara_uses_full_sample,
                         std::size_t n_series, std::size_t length, std::size_t ndim, core::Precision precision)
{
  const bool gpu = uses_gpu(prob.device().first);
  const bool assigns = method == Method::CLARA && !clara_uses_full_sample;
#ifdef DTWC_HAS_CUDA
  const bool cpu_assignment = assigns && (!gpu || prob.device().first == Device::CPU_GPU); // cpu+gpu's CPU share
#else
  const bool cpu_assignment = assigns; // Metal has no kernel for FastCLARA's assignment
#endif
  std::size_t bytes = 0;
  // cpu+gpu fills as the CPU does, the GPU taking some tile-rows: its threads hold the CPU fill's scratch.
  if ((config.dist_matrix.empty() && (!gpu || prob.device().first == Device::CPU_GPU)) || cpu_assignment) {
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

/// Whether a GPU run reads a view's series where they lie: every CUDA route (its fill pads the series from their
/// spans, FastCLARA's assignment its blocks, and its samples' fills views of them). Metal's fill
/// uploads an owned copy of a view's series.
bool gpu_reads_views([[maybe_unused]] const Config &config)
{
#if defined(DTWC_HAS_CUDA)
  return uses_gpu(config.device);
#else
  return false;
#endif
}

/// What a GPU holds in host memory beside the series and the distance matrix as it computes: the series it uploads,
/// each padded to the longest of `length` time steps of `ndim` features, in the precision it computes in. CUDA stages
/// them in its launch workspace (pinned where it can be; kept until the run gives its threads' scratch back), 8 bytes a
/// value (4 under --gpu-precision fp32): every series for a fill; for FastCLARA's assignment its pipeline
/// (cuda::detail::medoid_blocks: the medoids of every sample and two slots of a block of series, at most 2^27 values a
/// block, with their lengths, and three slots of a block's distances to them; one of each where one block holds every
/// series), its blocks shrunk within `assign_cap` bytes where that is set, or a sample's fill where that is larger.
/// Metal's fill holds the series and the matrix as floats in shared memory while it runs (its FastCLARA assigns on the
/// CPU). Where the longest series is estimated (twice the mean, a decoded file's), so is this.
std::size_t gpu_staging_bytes([[maybe_unused]] const Config &config, [[maybe_unused]] Method method,
                              [[maybe_unused]] const std::optional<algorithms::detail::ClaraPlan> &sample,
                              [[maybe_unused]] std::size_t n_series, [[maybe_unused]] std::size_t length,
                              [[maybe_unused]] std::size_t ndim, [[maybe_unused]] std::size_t assign_cap = 0)
{
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  if (!uses_gpu(config.device)) return 0; // cpu+gpu's GPU share is priced as if it took every series
  const std::size_t steps = length * ndim;
  const bool assigns = method == Method::CLARA && sample && sample->sample_size < sample->n_points;
  const std::size_t filled = assigns ? static_cast<std::size_t>(sample->sample_size) : n_series; // a fill's series
#  if defined(DTWC_HAS_CUDA)
  const std::size_t value = cuda::cuda_uses_fp32(config.gpu_precision, config.device_index) ? sizeof(float) : sizeof(double);
  std::size_t bytes = filled * steps * value;
  if (assigns) {
    const auto medoids = static_cast<std::size_t>(config.k) * static_cast<std::size_t>(config.n_samples);
    bytes = std::max(bytes, cuda::detail::medoid_blocks(n_series, medoids, steps, value, assign_cap)
                              .host_bytes(medoids, steps, value));
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

/// `ways` (one at least) as a sentence's list, "a, b, or c": a refusal and a run out of memory name their ways out alike.
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
  bool halves = false;        ///< --dtype float32 halves `series`: float64 values read from text, on a CPU (a GPU
                              ///< takes Float64 series only)
  Device device = Device::CPU; ///< where the run computes: a GPU has no float32 way out, cpu+gpu streams nothing
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
  /// What the run does once it has passed its checks and its plan, as its series are about to be read: run_with's
  /// announcement of a fallback, which a run refused in either stage never makes.
  std::function<void()> planned;

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
  // No budget, no plan: in RAM on every thread, nothing mapped or streamed. A method that keeps the matrix's bytes in
  // a cache of its own (TADPole) gains nothing from mapping the matrix, which it never writes: it is never mapped.
  plan.mapped = budget > 0 && facts.matrix > 0 && facts.held == nullptr && !facts.cache
                && series + facts.matrix + reserve > budget;
  const bool matrix = facts.matrix > 0 && (!plan.mapped || facts.cache); // its bytes in RAM
  const std::size_t need = series + (matrix ? facts.matrix : 0);
  plan.held = need;
  if (budget == 0 || need + reserve <= budget) return plan;
  // The refusal: what the run holds beside the working set of `threads`, and the ways out that apply.
  const auto refuse = [&](const std::string &what, int threads, const std::string &needs) {
    std::vector<std::string> ways{ "Raise --ram-limit" };
    if (!uses_gpu(facts.device)) ways.emplace_back("read the series as float32 (--dtype float32)");
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
/// matrix (holds_matrix) maps none and refuses a --dist-matrix, which it would load and never read; non-full FastCLARA
/// refuses a --checkpoint too. TADPole reads a complete --dist-matrix (an incomplete one is refused once read: it would
/// read none of it) and otherwise computes the pairs it needs into a cache of its own, so its matrix is never mapped
/// and --checkpoint is refused.
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
  if (dist_matrix && !holds_matrix(method, clara_uses_full_sample))
    throw InvalidInput("--method " + method_name(method)
                       + " does not consume a parent distance matrix; --dist-matrix would load unused O(N^2) state. "
                         "Omit --dist-matrix, or use a method that reads it, such as pam.");
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
  std::string device;                      ///< where the run computed, as device_text writes it
};

/// What the run report calls an input's format.
const char *format_name(std::optional<InputFormat> format)
{
  if (!format) return "in memory";
  return *format == InputFormat::Parquet ? "Parquet" : *format == InputFormat::ArrowIPC ? "Arrow IPC" : "text";
}

const char *unmapped_matrix_reason(const Config &config)
{
  if (!config.dist_matrix.empty()) return "which --dist-matrix reads into RAM";
#if defined(DTWC_HAS_MMAP) && defined(DTWC_HAS_CUDA)
  if (config.output.empty()) return "which a run without an output directory keeps in RAM";
  if (uses_gpu(config.device) && config.gpu_precision == GpuPrecision::Auto)
    return "which a CUDA run with --gpu-precision auto cannot map (select fp32 or fp64)";
  return nullptr;
#elif defined(DTWC_HAS_MMAP)
  return config.output.empty() ? "which a run without an output directory keeps in RAM" : nullptr;
#else
  (void)config;
  return "which a build without llfio cannot map (DTWC_ENABLE_LLFIO=OFF)";
#endif
}

struct RunContext
{
  const Config &config;
  std::optional<Data> &data;
  const ExternalSolver &solver;
  RunStage &stage;

  std::shared_ptr<Problem> problem;
  algorithms::CLARAOptions clara;
  fs::path input;
  std::optional<InputFormat> &format;
  fs::path output;

  Clock clk;
  Method method;
  bool stream_payload = false;
  std::size_t n_series = 0;
  bool clara_uses_full_sample = false;
  bool clara_planned = false;

  std::size_t budget = 0;
  int requested = 1;
  std::optional<MemoryPlan> &memory;
  std::optional<TeamCap> team;
  const char *held = nullptr;
  bool upload_copy = false;

  std::optional<io::IpcFile> ipc;
  std::string storage;
  bool fills_first = false;
  std::vector<std::string> streamed_names;
  core::ClusteringResult result;
};

void plan_clara(RunContext &ctx)
{
  if (ctx.method != Method::CLARA || ctx.clara_planned) return;
  const auto plan = resolve_clara_plan(static_cast<std::int64_t>(ctx.n_series), ctx.clara, "run");
  ctx.clara_uses_full_sample = plan.sample_size == plan.n_points;
  if (ctx.stream_payload) algorithms::detail::validate_streaming_clara_plan(plan, "run");
  ctx.clara_planned = true;
}

void check_stream(const RunContext &ctx, const char *what, const char *by, std::size_t resident)
{
  if (ctx.config.device == Device::CPU_GPU) throw detail::hybrid_streamed_refusal();
  if (!ctx.config.checkpoint.empty())
    throw InvalidInput("--checkpoint requires resident series data and cannot be combined with RAM-limited "
                       + std::string(what) + " streaming.");
  if (!ctx.config.dist_matrix.empty())
    throw InvalidInput("--dist-matrix requires resident series data and cannot be combined with RAM-limited "
                       + std::string(what) + " streaming.");
  if (ctx.config.verbose)
    std::cout << by << " selected streaming: " << ctx.n_series << " series, ~" << (resident / (1ULL << 20))
              << " MB resident estimate exceeds the RAM limit [" << ctx.clk << "]\n";
}

std::size_t clara_bytes(const RunContext &ctx, const algorithms::detail::ClaraPlan &sample)
{
  const std::size_t pam = core::packed_size(static_cast<std::size_t>(sample.sample_size))
                          * core::value_bytes(ctx.config.matrix_dtype);
  const auto n = static_cast<std::size_t>(sample.n_points);
  const std::size_t pass = algorithms::detail::clara_pass_bytes(
    sample, static_cast<std::size_t>(ctx.clara.n_samples), ctx.config.device == Device::CPU_GPU);
  return std::max(pam, pass + n * sizeof(index_t) + n / 8);
}

std::optional<algorithms::detail::ClaraPlan> clara_sample(const RunContext &ctx, Method resolved)
{
  if (ctx.n_series == 0 || resolved != Method::CLARA) return std::nullopt;
  require_clusterable(ctx.config.k, ctx.n_series);
  return resolve_clara_plan(static_cast<std::int64_t>(ctx.n_series), ctx.clara, "run");
}

bool can_stream_file(const RunContext &ctx, const std::optional<algorithms::detail::ClaraPlan> &sample,
                     bool rows_are_series)
{
  return sample && sample->sample_size < sample->n_points && rows_are_series
         && (ctx.config.ram_limit != 0 || (ctx.config.checkpoint.empty() && ctx.config.dist_matrix.empty()));
}

Method resolve_planned_method(const RunContext &ctx, std::size_t series, std::size_t length, std::size_t ndim,
                              core::Precision precision)
{
  if (ctx.method != Method::Auto) return ctx.method;
  const std::size_t matrix = core::packed_size(ctx.n_series) * core::value_bytes(ctx.config.matrix_dtype);
  const std::size_t pam = thread_bytes(*ctx.problem, ctx.config, Method::PAM, false, ctx.n_series, length, ndim, precision);
  const std::size_t staging = gpu_staging_bytes(ctx.config, Method::PAM, std::nullopt, ctx.n_series, length, ndim);
  return resolve_method(ctx.method, ctx.config.device, ctx.n_series,
                        ctx.budget == 0 || ctx.held == nullptr
                          || series + staging + matrix + working_set(pam, 1) <= ctx.budget);
}

MemoryPlan plan_of(RunContext &ctx, Method resolved, const std::optional<algorithms::detail::ClaraPlan> &sample,
                   std::size_t series, std::size_t values, std::size_t length, std::size_t ndim,
                   core::Precision precision, bool can_stream, bool streams, std::size_t streamed)
{
  const bool full = sample && sample->sample_size == sample->n_points;
  SourceFacts facts;
  facts.series = series;
  facts.copy = ctx.upload_copy;
  facts.staging = gpu_staging_bytes(ctx.config, resolved, sample, ctx.n_series, length, ndim);
  facts.clara = sample && !full ? clara_bytes(ctx, *sample) : 0;
  const std::size_t pairs = core::packed_size(ctx.n_series);
  facts.matrix = holds_matrix(resolved, full) ? pairs * core::value_bytes(ctx.config.matrix_dtype) : 0;
  facts.unpriced = resolved == Method::MIP;
  if (resolved == Method::TADPole) {
    const auto distance = ctx.problem->distance();
    const bool prunes = algorithms::detail::bounds_valid(distance.variant.variant, distance.metric, ndim, precision,
                                                         ctx.config.matrix_dtype, distance.missing);
    facts.fixed = pairs + (prunes ? 2 * (values * sizeof(double) + ctx.n_series * sizeof(std::vector<double>)) : 0);
    facts.fixed_what = prunes ? "TADPole's flag a pair and the series' envelopes" : "TADPole's flag a pair";
    facts.cache = true;
  } else if (resolved == Method::Hierarchical) {
    facts.fixed = pairs * sizeof(double);
    facts.fixed_what = "the hierarchical clustering's packed working copy of the matrix";
  } else if (resolved == Method::OneBatch) {
    const std::size_t m = algorithms::detail::resolve_batch_size(ctx.n_series, ctx.config.k, ctx.config.batch_size);
    facts.fixed = ctx.n_series * (m + 8) * sizeof(double);
    facts.fixed_what = "OneBatchPAM's batch table and per-point vectors";
  }
  facts.clara_method = resolved == Method::CLARA;
  facts.held = ctx.held;
  facts.can_stream = can_stream;
  facts.streams = streams;
  facts.halves = ctx.format == InputFormat::Text && precision == core::Precision::Float64 && !uses_gpu(ctx.config.device);
  facts.device = ctx.config.device;
  facts.streamed = streamed;
  facts.n_series = ctx.n_series;
  facts.length = length;
  const std::size_t per_thread = thread_bytes(*ctx.problem, ctx.config, resolved, full, ctx.n_series, length, ndim, precision);
  ctx.memory = plan_memory(facts, ctx.budget, ctx.config.ram_limit != 0, per_thread, ctx.requested);
  if (ctx.memory->route == Route::Refused && ctx.config.ram_limit != 0 && gpu_reads_views(ctx.config)
      && resolved == Method::CLARA && sample && sample->sample_size < sample->n_points) {
    const std::size_t others = facts.series + facts.clara + facts.fixed + working_set(per_thread, 1);
    if (ctx.budget > others) {
      facts.staging = gpu_staging_bytes(ctx.config, resolved, sample, ctx.n_series, length, ndim, ctx.budget - others);
      ctx.clara.gpu_staging_bytes = ctx.budget - others;
      ctx.memory = plan_memory(facts, ctx.budget, true, per_thread, ctx.requested);
      if (ctx.memory->route == Route::Refused) ctx.clara.gpu_staging_bytes = 0;
    }
  }
  if (ctx.config.verbose) std::cout << "Memory plan: " << ctx.memory->text() << "\n";
  if (ctx.stage.report)
    ctx.stage.report->plan = detail::RunReport::Plan{ ctx.memory->budget,   ctx.memory->reserve(),
                                                      ctx.memory->data_cap(), ctx.memory->held,
                                                      ctx.memory->per_thread, ctx.memory->length,
                                                      ctx.memory->threads,   ctx.memory->requested,
                                                      ctx.memory->route_name() };
  if (ctx.memory->route != Route::Refused) ctx.team.emplace(ctx.memory->threads);
  return *ctx.memory;
}

void report_input(RunContext &ctx, std::size_t series, std::size_t values, std::size_t ndim, std::size_t bytes,
                  std::size_t shortest, std::size_t longest, bool streamed, core::Precision precision)
{
  if (ctx.stage.report)
    ctx.stage.report->input = detail::RunReport::Input{
      format_name(ctx.format), series, values, ndim, bytes, shortest, longest, streamed, precision
    };
}

void validate_run_config(RunContext &ctx)
{
  const Config &config = ctx.config;
  if (!ctx.data && config.input.empty())
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
  if (!config.dist_matrix.empty() && !config.checkpoint.empty())
    throw InvalidInput("--dist-matrix and --checkpoint each supply the distance matrix; give one.");
  if (config.method == Method::TADPole && !config.checkpoint.empty())
    throw InvalidInput("--checkpoint saves the distance matrix, and tadpole keeps the pairs it computes in a cache of "
                       "its own: the file would hold none of them. Omit --checkpoint, or give tadpole a complete matrix "
                       "with --dist-matrix.");

  ctx.problem = std::make_shared<Problem>(run_name(config));
  apply(config, *ctx.problem);
  const bool stored_f32 = config.dtype == core::Precision::Float32 || (ctx.data && ctx.data->is_f32());
  validate_gpu_request("run", *ctx.problem, stored_f32 ? core::Precision::Float32 : core::Precision::Float64,
                       ctx.data ? ctx.data->ndim : 1);
  if (uses_gpu(config.device)) require_gpu_device("run", config.device_index);
#ifdef DTWC_HAS_CUDA
  if (!config.checkpoint.empty() && uses_gpu(config.device) && config.gpu_precision == GpuPrecision::Auto)
    throw InvalidInput("--checkpoint on a CUDA GPU needs an explicit --gpu-precision fp32 or fp64: CUDA's auto depends "
                       "on the GPU, so it is not a stable cache identity.");
#endif

  ctx.clara.n_clusters = config.k;
  ctx.clara.sample_size = config.sample_size;
  ctx.clara.n_samples = config.n_samples;
  ctx.clara.max_iter = config.max_iter;
  ctx.clara.random_seed = config.seed;
  ctx.clara.gpu_share = config.gpu_share;
  if (config.method == Method::CLARA) algorithms::detail::validate_clara_controls(ctx.clara, "run");

  ctx.input = utf8_to_path(config.input);
  if (!ctx.data) ctx.format = format_of(ctx.input);
  require_input_options_apply(config, ctx.format);

  const fs::path checkpoint_dir = utf8_to_path(config.checkpoint);
  if (!config.checkpoint.empty()) {
    std::error_code ec;
    fs::create_directories(checkpoint_dir, ec);
    if (ec || !fs::is_directory(checkpoint_dir, ec))
      throw IOError("--checkpoint '" + config.checkpoint + "' is not a usable directory ("
                    + (ec ? ec.message() : std::string("not a directory"))
                    + "); pass a directory path (it is created if missing), or omit --checkpoint.");
  }
  ctx.output = utf8_to_path(config.output);
  if (!config.output.empty()) {
    std::error_code ec;
    fs::create_directories(ctx.output, ec);
    if (ec) throw IOError("--output '" + config.output + "' cannot be created: " + ec.message());
  }
}

#ifdef DTWC_HAS_PARQUET
void plan_parquet_input(RunContext &ctx)
{
  const Config &config = ctx.config;
  if (ctx.format != InputFormat::Parquet
      || (config.ram_limit == 0 && ctx.method != Method::Auto && ctx.method != Method::CLARA))
    return;

  const bool f32 = config.dtype == core::Precision::Float32;
  std::error_code ec;
  const bool folder = fs::is_directory(ctx.input, ec);
  auto layout = detail::ParquetLayout::Directory;
  std::size_t resident_bytes = 0;
  std::size_t names_bytes = 0;
  std::size_t values = 0;
  for (const auto &file : parquet_files(ctx.input)) {
    const io::ParquetChunkReader metadata(file, config.column, config.skip_cols, config.skip_rows);
    if (!folder)
      layout = metadata.rows_are_series() ? detail::ParquetLayout::ListColumn : detail::ParquetLayout::ScalarColumn;
    ctx.n_series += static_cast<std::size_t>(metadata.logical_series_count());
    resident_bytes += metadata.estimated_materialization_peak_bytes(f32);
    names_bytes += metadata.series_names_bytes();
    values += metadata.total_value_count();
  }
  const std::size_t length = ctx.n_series > 0 ? 2 * values / ctx.n_series : 0;
  const Method resolved = resolve_planned_method(ctx, resident_bytes, length, 1, config.dtype);
  const auto sample = clara_sample(ctx, resolved);
  const std::size_t resident = resident_bytes + (sample ? clara_bytes(ctx, *sample) : 0);
  const bool can_stream = can_stream_file(ctx, sample, layout == detail::ParquetLayout::ListColumn);
  const MemoryPlan planned =
    plan_of(ctx, resolved, sample, resident_bytes, values, length, 1, config.dtype, can_stream,
            layout == detail::ParquetLayout::ListColumn, names_bytes + 2 * ctx.n_series * sizeof(index_t));
  const bool over = planned.route == Route::Refused && !can_stream && resident > ctx.budget;
  (void)detail::plan_parquet_load(resolved, config.device, ctx.n_series, resident, over ? ctx.budget : 0, layout);
  if (over && sample) algorithms::detail::validate_streaming_clara_plan(*sample, "run");
  if (planned.route == Route::Refused) throw InvalidInput(planned.refusal);
  ctx.method = resolved;
  ctx.stream_payload = planned.route == Route::Streamed;
  if (ctx.stream_payload) ctx.clara.ram_limit_bytes = planned.data_cap();
  plan_clara(ctx);
  if (ctx.stream_payload) check_stream(ctx, "Parquet", "Parquet metadata", resident);
  ctx.stage.note("Parquet metadata");
  if (ctx.stream_payload) report_input(ctx, ctx.n_series, values, 1, 0, 0, 0, true, config.dtype);
}
#endif

void plan_arrow_ipc_input(RunContext &ctx)
{
  const Config &config = ctx.config;
  if (ctx.format != InputFormat::ArrowIPC) return;

  ctx.ipc.emplace(ctx.input);
  validate_gpu_request("run", *ctx.problem, config.dtype, ctx.ipc->ndim());
  ctx.n_series = static_cast<std::size_t>(ctx.ipc->num_rows());
  const bool reads_views = gpu_reads_views(config);
  const std::size_t ipc_bytes = ctx.ipc->resident_bytes(config.dtype, uses_gpu(config.device) && !reads_views);
  std::size_t length = ctx.ipc->longest_steps();
  if (length == 0 && ctx.n_series > 0) length = 2 * ctx.ipc->num_values() / (ctx.n_series * ctx.ipc->ndim());
  const Method resolved = resolve_planned_method(ctx, ipc_bytes, length, ctx.ipc->ndim(), config.dtype);
  const auto sample = clara_sample(ctx, resolved);
  const std::size_t resident = ipc_bytes + (sample ? clara_bytes(ctx, *sample) : 0);
  const bool streams = ctx.ipc->streams(config.dtype);
  const bool can_stream = can_stream_file(ctx, sample, streams);
  const std::size_t names_bytes = ctx.ipc->series_names_bytes() + ctx.ipc->names_read_bytes();
  ctx.upload_copy = uses_gpu(config.device) && (config.dtype == core::Precision::Float32) == ctx.ipc->is_f32()
                    && !reads_views;
  const MemoryPlan planned =
    plan_of(ctx, resolved, sample, ipc_bytes, ctx.ipc->num_values(), length, ctx.ipc->ndim(), config.dtype, can_stream,
            streams, names_bytes + 2 * ctx.n_series * sizeof(index_t));
  ctx.method = resolved;
  if (planned.route == Route::Streamed) {
    ctx.stream_payload = true;
    ctx.clara.ram_limit_bytes = planned.data_cap();
    plan_clara(ctx);
    check_stream(ctx, "Arrow IPC", "Arrow IPC walk", resident);
  } else if (planned.route == Route::Refused && !can_stream && resident > config.ram_limit) {
    const auto clara_streams = [&] {
      try {
        const auto plan = sample ? *sample : resolve_clara_plan(static_cast<std::int64_t>(ctx.n_series), ctx.clara, "run");
        return plan.sample_size < plan.n_points;
      } catch (const InvalidInput &) {
        return false;
      }
    };
    const std::string other = config.dtype == core::Precision::Float32 ? "float64" : "float32";
    std::string why;
    if (!streams && ctx.upload_copy)
      why = "its values are mapped where they lie in the file, and a mapped file's values are not streamed: the "
            "views of its " + std::to_string(ctx.n_series) + " series and the owned copy the GPU uploads of them are "
            "held at once. Raise --ram-limit, or run on the CPU (--device cpu), which holds no copy.";
    else if (!streams)
      why = "its values are mapped where they lie in the file, and a mapped file's values are not streamed: the "
            "views of its " + std::to_string(ctx.n_series) + " series are held at once. Raise --ram-limit"
            + (clara_streams() ? std::string(uses_gpu(config.device) ? ", or run on the CPU (--device cpu) and " : ", or ")
                                   + "read its values as " + other + " (--dtype " + other + ")"
                                   + (resolved == Method::CLARA ? "" : " with --method clara")
                                   + ", whose copy FastCLARA reads a record batch at a time."
                               : std::string("."));
    else if (resolved != Method::CLARA)
      why = "method '" + method_name(resolved) + "' cannot stream it. "
            + (clara_streams() ? "Use --method clara, which reads the file a record batch at a time, or raise "
                                 "--ram-limit."
                               : std::string("Raise --ram-limit."));
    else
      why = "FastCLARA streams it only with a sample smaller than its " + std::to_string(ctx.n_series)
            + " series; lower --sample-size, or raise --ram-limit.";
    throw InvalidInput("Arrow IPC input needs approximately " + std::to_string(resident)
                       + " bytes of resident series storage, exceeding --ram-limit="
                       + std::to_string(config.ram_limit) + "; " + why);
  }
  if (planned.route == Route::Refused) throw InvalidInput(planned.refusal);
  ctx.stage.note("Arrow IPC walk");
  if (ctx.stream_payload) report_input(ctx, ctx.n_series, ctx.ipc->num_values(), ctx.ipc->ndim(), 0, 0, 0, true, config.dtype);
  if (!ctx.stream_payload) {
    ctx.storage = ctx.ipc->storage(config.dtype);
    ctx.ipc.reset();
  }
}

void load_in_ram_series(RunContext &ctx)
{
  const Config &config = ctx.config;
  if (ctx.stage.planned) ctx.stage.planned();
  ctx.stage.begin("read");
  if (ctx.stream_payload) ctx.stage.note("none: FastCLARA streams them, inside cluster");
  if (!ctx.stream_payload) {
    Data series = ctx.data ? std::move(*ctx.data) : read_input(ctx.input, ctx.format.value_or(InputFormat::Text), config);
    const std::string loaded = std::to_string(series.size()) + " series" + (ctx.storage.empty() ? "" : ", " + ctx.storage);
    if (config.verbose) {
      const char *from = !ctx.format                            ? " in memory"
                         : ctx.format == InputFormat::Parquet  ? " from Parquet"
                         : ctx.format == InputFormat::ArrowIPC ? " from Arrow IPC"
                                                               : "";
      std::cout << "Data loaded" << from << ": " << loaded << " [" << ctx.clk << "]\n";
    }
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
    report_input(ctx, static_cast<std::size_t>(series.size()), elements, series.ndim, value_bytes, shortest, longest, false,
                 series.precision);
    ctx.stage.note(loaded);
    if (config.verbose && series.size() > 0)
      std::cout << "  Data memory: ~" << (value_bytes / (1ULL << 20)) << " MB (" << series.size() << " series, "
                << (elements / series.size()) << " avg length, " << name_of(core::precision_names, series.precision)
                << ")\n";
    const std::size_t ndim = series.ndim;
    const core::Precision precision = series.precision;
    const bool view = series.is_view();
    ctx.problem->set_data(std::move(series));
    ctx.n_series = ctx.problem->size();
    if (!ctx.memory) {
      require_clusterable(config.k, ctx.n_series);
      ctx.method = resolve_planned_method(ctx, value_bytes, longest, ndim, precision);
      std::optional<algorithms::detail::ClaraPlan> sample;
      if (ctx.method == Method::CLARA) sample = resolve_clara_plan(static_cast<std::int64_t>(ctx.n_series), ctx.clara, "run");
      ctx.upload_copy = uses_gpu(config.device) && view && !gpu_reads_views(config);
      const std::size_t copy = ctx.upload_copy ? detail::owned_series_bytes(elements, ctx.n_series, false) : 0;
      if (const MemoryPlan planned =
            plan_of(ctx, ctx.method, sample, value_bytes + copy, elements, longest, ndim, precision, false, false, 0);
          planned.route == Route::Refused)
        throw InvalidInput(planned.refusal);
    }
  }
  require_clusterable(config.k, ctx.n_series);
  if (config.method == Method::Auto && config.verbose)
    std::cout << "Auto-selected method: " << method_name(ctx.method) << " (N=" << ctx.n_series << ")\n";
  plan_clara(ctx);
  ctx.problem->set_method(ctx.method);
  if (ctx.stage.report) ctx.stage.report->method = ctx.method;
}

void setup_distance_storage(RunContext &ctx)
{
  const Config &config = ctx.config;
  Problem &prob = *ctx.problem;
  ctx.stage.begin("storage");
  const auto path = checkpoint_path(prob, config.checkpoint.empty() ? config.output : config.checkpoint);
  const std::optional<fs::path> cache =
    configure_distance_storage(prob, config, ctx.method, ctx.clara_uses_full_sample, ctx.memory->mapped, path);
  if (cache && config.verbose) std::cout << "Using memory-mapped distance matrix: " << *cache << "\n";
  if (!config.dist_matrix.empty()) {
    try {
      prob.read_distance_matrix(utf8_to_path(config.dist_matrix), config.not_computed);
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const std::exception &e) {
      throw IOError("--dist-matrix '" + config.dist_matrix
                    + "' cannot be loaded; fix the file, or omit --dist-matrix to compute the distances. Cause: "
                    + e.what());
    }
    if (ctx.method == Method::TADPole && !prob.is_distance_matrix_filled())
      throw InvalidInput("tadpole reads a --dist-matrix only when it is complete, and '" + config.dist_matrix
                         + "' leaves pairs not computed: tadpole would read none of it and compute every pair again "
                           "beside it. Give a complete matrix (a pam run with -o writes one), or omit --dist-matrix.");
    if (config.verbose) std::cout << "Loaded distance matrix from " << config.dist_matrix << "\n";
  }
  if (config.checkpoint_interval != 0) {
    prob.checkpoint.directory = config.checkpoint;
    prob.checkpoint.save_interval = config.checkpoint_interval;
    prob.checkpoint.enabled = true;
  }
  bool resumed = false;
  if (!config.checkpoint.empty() && !cache && holds_matrix(ctx.method, ctx.clara_uses_full_sample)) {
    resumed = load_checkpoint(prob, config.checkpoint);
    if (config.verbose)
      std::cout << (resumed ? "Resumed from checkpoint: " + config.checkpoint + "\n"
                            : "No checkpoint in " + config.checkpoint + ", starting fresh.\n");
  }
  std::string storage_note = ctx.stream_payload || !holds_matrix(ctx.method, ctx.clara_uses_full_sample)
                               ? std::string("no distance matrix of every series")
                             : cache ? "the distance matrix mapped, " + path_to_utf8(cache->filename())
                                     : std::string("the distance matrix in RAM");
  if (!config.dist_matrix.empty()) storage_note += "; --dist-matrix read";
  if (resumed) storage_note += "; the checkpoint resumed";
  ctx.stage.note(std::move(storage_note));
}

void run_clustering_stage(RunContext &ctx)
{
  const Config &config = ctx.config;
  Problem &prob = *ctx.problem;
  ctx.fills_first = !ctx.stream_payload
                    && (ctx.method == Method::PAM || ctx.method == Method::Kmedoids
                        || (ctx.method == Method::CLARA && ctx.clara_uses_full_sample));
  ctx.stage.begin(ctx.fills_first ? "fill" : "cluster");
  if (config.verbose) std::cout << "Running " << progress_label(ctx.method) << " (k=" << config.k << ") ...\n";
  if (ctx.fills_first) {
    const std::string fill_note =
      prob.is_distance_matrix_filled()
        ? "complete on entry"
        : std::to_string(ctx.n_series) + " series' matrix on " + device_text(config)
            + (uses_gpu(config.device) && prob.data().is_view() && !gpu_reads_views(config)
                 ? ", from an owned copy of their views"
                 : "");
    ctx.stage.note(fill_note);
    prob.fill_distance_matrix();
    if (prob.fill_split()) ctx.stage.note(fill_note + "; " + dtwc::describe(*prob.fill_split()));
    ctx.stage.begin("cluster");
  }
  ctx.stage.note(std::string(progress_label(ctx.method))
                 + (ctx.fills_first ? "" : ", any distances it computes inside"));

  const auto cluster_in_ram = [&] {
    if (prob.method() == Method::CLARA) return algorithms::fast_clara(prob, ctx.clara);
    const bool by_solver = ctx.solver && prob.method() == Method::MIP && prob.solver() == Solver::HiGHS;
    if (!by_solver) return prob.cluster();
    core::ClusteringResult solved = ctx.solver(ctx.problem);
    prob.set_result(solved);
    return solved;
  };
  try {
    if (ctx.stream_payload && ctx.ipc) {
      prob.set_data(Data(std::vector<std::vector<data_t>>{}, std::vector<std::string>{}, ctx.ipc->ndim()));
      const auto plan = resolve_clara_plan(static_cast<std::int64_t>(ctx.n_series), ctx.clara, "fast_clara");
      ctx.result = config.dtype == core::Precision::Float32
                     ? algorithms::detail::fast_clara_chunked<true>(prob, ctx.clara, *ctx.ipc, plan)
                     : algorithms::detail::fast_clara_chunked<false>(prob, ctx.clara, *ctx.ipc, plan);
      ctx.streamed_names = ctx.ipc->series_names(ctx.clara.ram_limit_bytes - 2 * ctx.n_series * sizeof(index_t));
    } else {
#ifdef DTWC_HAS_PARQUET
      if (ctx.stream_payload) {
        const bool f32 = config.dtype == core::Precision::Float32;
        const io::ParquetChunkReader reader(ctx.input, config.column, config.skip_cols, config.skip_rows);
        const auto plan = resolve_clara_plan(reader.logical_series_count(), ctx.clara, "fast_clara");
        if (config.verbose)
          std::cout << "FastCLARA: streaming from Parquet (" << reader.logical_series_count() << " rows, "
                    << reader.num_row_groups() << " row groups, ~" << reader.estimated_resident_bytes(f32) / (1ULL << 20)
                    << " MB resident estimate)\n";
        ctx.result = f32 ? algorithms::detail::fast_clara_chunked<true>(prob, ctx.clara, reader, plan)
                         : algorithms::detail::fast_clara_chunked<false>(prob, ctx.clara, reader, plan);
        ctx.streamed_names = reader.series_names();
      } else {
        ctx.result = cluster_in_ram();
      }
#else
      ctx.result = cluster_in_ram();
#endif
    }
  } catch (const InvalidInput &refused) {
    const std::string text = refused.what();
    if (!ctx.stream_payload || text.find("--ram-limit") == std::string::npos) throw;
    throw InvalidInput(text + (text.ends_with('.') ? " " : ". ")
                       + "The --ram-limit it names is the stream's data cap: --ram-limit = "
                       + std::to_string(ctx.memory->budget) + " less the reserve of "
                       + std::to_string(ctx.memory->reserve()) + " bytes for " + std::to_string(ctx.memory->threads)
                       + " threads (-v prints the memory plan); raise --ram-limit by what the stream lacks, or run "
                         "fewer threads (OMP_NUM_THREADS) to shrink the reserve.");
  }
  if (ctx.result.hybrid)
    ctx.stage.note(std::string(progress_label(ctx.method)) + ", any distances it computes inside; "
                   + algorithms::describe(*ctx.result.hybrid, ctx.result.labels.size()));
  else if (!ctx.fills_first && prob.fill_split())
    ctx.stage.note(std::string(progress_label(ctx.method)) + ", any distances it computes inside; "
                   + dtwc::describe(*prob.fill_split()));
  if (config.verbose) {
    std::cout << progress_label(ctx.method);
    if (ctx.method == Method::PAM)
      std::cout << (ctx.result.converged ? " converged" : " did not converge") << " in " << ctx.result.iterations
                << " iterations";
    else
      std::cout << " finished";
    const auto prev_prec = std::cout.precision();
    std::cout << ", cost=" << std::setprecision(6) << ctx.result.total_cost << " [" << ctx.clk << "]\n";
    std::cout.precision(prev_prec);
  }
  if (ctx.stage.report) {
    detail::RunReport::Clustering clustering{ ctx.method, progress_label(ctx.method), ctx.result.total_cost,
                                              ctx.result.iterations, ctx.result.converged,
                                              std::vector<std::size_t>(ctx.result.medoid_indices.size()),
                                              ctx.result.medoid_indices };
    for (const index_t label : ctx.result.labels)
      if (label >= 0 && static_cast<std::size_t>(label) < clustering.sizes.size())
        ++clustering.sizes[static_cast<std::size_t>(label)];
    ctx.stage.report->clustering = std::move(clustering);
  }
}

void save_outputs_and_checkpoint(RunContext &ctx)
{
  const Config &config = ctx.config;
  Problem &prob = *ctx.problem;
  const bool has_checkpoint = !config.checkpoint.empty();
  const bool saves_checkpoint = has_checkpoint && prob.distance_matrix().size() != 0;
  ctx.stage.begin("write");
  ctx.stage.note(std::string(saves_checkpoint ? "the checkpoint, then "
                             : has_checkpoint ? "no distances were computed: no checkpoint written, then "
                                              : "")
                 + (config.output.empty() ? std::string("no result files (no -o)") : "the result files"));
  std::optional<std::string> checkpoint_failure;
  if (has_checkpoint && !saves_checkpoint && config.verbose)
    std::cout << "No distances were computed: no checkpoint written.\n";
  if (saves_checkpoint) {
    try {
      save_checkpoint(prob, config.checkpoint);
      if (config.verbose) std::cout << "Checkpoint saved to " << config.checkpoint << "\n";
    } catch (const std::bad_alloc &) {
      checkpoint_failure = "out of memory; omit --checkpoint, or give the process more memory";
    } catch (const std::exception &e) {
      checkpoint_failure = "free space or pass another directory to --checkpoint, or omit it. Cause: "
                           + std::string(e.what());
    }
  }

  if (!config.output.empty()) {
    detail::WriteTimes times;
    times.cpu_clock = &core::cpu_seconds;
    detail::write_result_files(prob, ctx.output, false, config.verbose ? &std::cout : nullptr, ctx.streamed_names,
                               ctx.stage.report ? &times : nullptr);
    if (ctx.stage.report)
      ctx.stage.report->silhouettes = detail::RunReport::Silhouettes{
        times.silhouette_wall, times.silhouette_cpu, times.mean_silhouette, times.skipped
      };
  }
  if (checkpoint_failure)
    throw IOError("--checkpoint '" + config.checkpoint + "': the distance checkpoint cannot be saved (the results are "
                  "written to '" + config.output + "'); " + *checkpoint_failure);
}

/// The run itself; `data` holds in-memory series, else config.input is read; `solver`, where given, clusters MIP on
/// HiGHS in RAM (detail::run). `stage` follows it: the stage it is in, its memory plan once made, and the report's
/// figures. A GPU request it cannot honour is a GpuRefusal thrown in its check or plan stage, where `data` is still
/// whole and no series was read: run_with decides what to do with it (`fallback`), and `config.fallback` is cleared.
Outcome execute(const Config &config, std::optional<Data> &data, const ExternalSolver &solver, RunStage &stage)
{
  RunContext ctx{
    config, data, solver, stage,
    nullptr, {}, {}, stage.format, {},
    Clock{}, config.method, false, 0, false, false,
    core::memory_budget(config.ram_limit), get_max_threads(), stage.plan, std::nullopt,
    unmapped_matrix_reason(config), false, std::nullopt, {}, false, {}, {}
  };

  validate_run_config(ctx);
  stage.begin("plan");
#ifdef DTWC_HAS_PARQUET
  plan_parquet_input(ctx);
#endif
  plan_arrow_ipc_input(ctx);
  load_in_ram_series(ctx);
  setup_distance_storage(ctx);
  run_clustering_stage(ctx);
  save_outputs_and_checkpoint(ctx);

  return { std::move(ctx.problem), std::move(ctx.result), ctx.method, std::move(ctx.streamed_names), {}, {} };
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

void write_report(const Config &config, const std::string &text) { write_text(report_path(config), text); }

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
      if (config.dtype == core::Precision::Float64 && !uses_gpu(config.device)) ways.emplace_back("--dtype float32");
#ifdef DTWC_HAS_MMAP
      ways.emplace_back(std::string("the file converted to Arrow IPC (dtwc-convert), whose values are mapped where they "
                                    "lie") + counted);
#endif
    }
  } else if (stage.plan->route == Route::Streamed) {
    ways.emplace_back(std::string("a lower --ram-limit, within which the stream reads") + down_to);
  } else if (at == "read") {
    if (stage.plan->facts.streams && clara_can && stage.plan->facts.device != Device::CPU_GPU) // cpu+gpu streams nothing
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

/// A run of `asked` on `data` (else asked.input), `solver` as detail::run takes it, its report rendered and, with an
/// output directory, written last.
/// Every thread's scratch is given back as the run starts, returns or throws (ScratchRelease). A std::bad_alloc is
/// caught outside it and outside execute, whose Problem, series and TeamCap are gone by then, and becomes
/// out_of_memory's InvalidInput. A run that fails past its checks writes its report up to the failing stage, with the
/// error, and throws its own exception on: a report that cannot be made or written is dropped, never the error.
///
/// `asked.fallback` is this function's: execute refuses a GPU request that cannot be honoured, with a GpuRefusal in its
/// check or plan stage (before a series is read, so `data` is whole), and then the run is made again on the CPU: one
/// line on stderr, said when that run has passed its own check and plan, and the report's configuration row saying why
/// and Result::device() "cpu". Once the run has computed anything, a failure is an error and nothing is run again.
Outcome run_with(const Config &asked, std::optional<Data> data, const ExternalSolver &solver)
{
  Config config = asked; // what execute runs: the request until a refusal, then the CPU's
  config.fallback = false;
  std::string fallback; // why `config` is the CPU's; empty while it is as asked
  for (;;) {
    RunStage stage;
    try {
      try {
        if (asked.report) {
          stage.report.emplace(asked);
          if (!fallback.empty()) stage.report->fell_back(fallback);
        }
        if (!fallback.empty()) // said once the run on the CPU has passed its checks and its plan, not before
          stage.planned = [&] { detail::say_fallback(asked.device, asked.device_index, fallback); };
        stage.begin("check"); // the scratch given back first, the checks then
        const core::ScratchRelease release;
        Outcome outcome = execute(config, data, solver, stage);
        outcome.device = device_text(config);
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
      const std::string_view at = stage.stage;
      if (const auto *refusal = dynamic_cast<const GpuRefusal *>(&error);
          refusal && asked.fallback && fallback.empty() && (at == "check" || at == "plan")) {
        fallback = refusal->reason();
        config.device = Device::CPU;
        config.device_index = 0;
        // cpu+gpu's split has no CPU counterpart; a share forced on any other device stays, and the run refuses it as
        // it does on every build, rather than the fallback hiding it.
        if (asked.device == Device::CPU_GPU) config.gpu_share = -1.0;
        continue;
      }
      if (stage.report && !config.output.empty() && at != "check") {
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
  auto [problem, result, method, streamed_names, report, device] = run_with(config, std::move(data), solver);
  return Result(std::move(problem), result.total_cost, std::move(device), method, result.iterations,
                result.converged, std::move(streamed_names), std::move(report));
}

Result run(const Config &config) { return detail::run(config, std::nullopt, {}); }

Result run(const Config &config, Data data) { return detail::run(config, std::move(data), {}); }

} // namespace dtwc
