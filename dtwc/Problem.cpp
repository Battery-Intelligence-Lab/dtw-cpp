/**
 * @file Problem.cpp
 * @brief Implementation of the DTWC (Dynamic Time Warping Clustering) problem encapsulated in a class.
 *
 * @details The Problem session: its setters, the distance functions they bind, the distance-matrix fill and its
 * cache, the series check, cluster() over the nine methods and Lloyd's k-medoids; also `apply(Config, Problem &)`,
 * `parse_ram_limit` and the GPU queries. The result files are in Problem_IO.cpp, initialisation in initialisation.cpp.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "Problem.hpp"
#include "config.hpp"               // for Config, apply
#include "base/error.hpp"           // for DeviceError
#include "mip.hpp"             // for MIP_clustering_byGurobi, MIP_clustering_byHiGHS, LR_core_clustering
#include "base/parallelisation.hpp" // for run
#include "base/settings.hpp"        // for data_t, index_t, settings::DEFAULT_MIP_SOLVER
#include "core/matrix_io.hpp"  // for operator<<(ostream, DistanceMatrix)
#include "core/medoid_assignment_policy.hpp" // finite assignment contract

#ifdef DTWC_HAS_CUDA
#include "cuda/cuda_dtw.cuh"   // GPU distance matrix computation
#endif
#ifdef DTWC_HAS_METAL
#include "metal/metal_dtw.hpp" // Apple Metal GPU distance matrix computation
#endif
#include "warping.hpp"         // for detail::all_usable, refuse_unusable
#include "core/dtw_kernel.hpp" // for dtw_lanes
#include "core/dtw_dispatch.hpp"           // for resolve_dtw_fn
#include "core/gpu_policy.hpp"             // shared GPU capability checks
#include "core/sha256.hpp"                 // for persistent cache fingerprints
#include "base/missing_utils.hpp"               // for all_missing
#include "algorithms/fast_clara.hpp"       // the methods cluster() dispatches
#include "algorithms/fast_pam.hpp"
#include "algorithms/one_batch_pam.hpp"
#include "algorithms/tadpole.hpp"
#include "io/parse_number.hpp"         // for io::parse_number


#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtwc {

namespace {

#ifdef DTWC_HAS_CUDA
/// Minimum pairs per calibration unit (GPU leading tile-rows and CPU initial columns) in a cpu+gpu matrix fill.
constexpr std::size_t fill_calibration_pairs = std::size_t{ 1 } << 19;
/// Default GPU pair-throughput share before calibration (FP64 and FP32).
constexpr double recorded_fill_gpu_share = 28.5 / (28.5 + 55.0);
constexpr double recorded_fill_gpu_share_fp32 = 13.0 * 28.5 / (13.0 * 28.5 + 55.0);
#endif

using FingerprintHash = core::detail::Sha256;

void hash_u32(FingerprintHash &hash, std::uint32_t value)
{
  std::array<std::uint8_t, 4> encoded{};
  for (std::size_t i = 0; i < encoded.size(); ++i)
    encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
  hash.update(encoded);
}

void hash_u64(FingerprintHash &hash, std::uint64_t value)
{
  std::array<std::uint8_t, 8> encoded{};
  for (std::size_t i = 0; i < encoded.size(); ++i)
    encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
  hash.update(encoded);
}

template <typename Enum>
void hash_enum(FingerprintHash &hash, Enum value)
{
  static_assert(std::is_enum_v<Enum>);
  using Unsigned = std::make_unsigned_t<std::underlying_type_t<Enum>>;
  hash_u64(hash, static_cast<std::uint64_t>(static_cast<Unsigned>(value)));
}

void hash_double(FingerprintHash &hash, double value)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

void hash_float(FingerprintHash &hash, float value)
{
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u32(hash, bits);
}

/// The backend Device::GPU resolves to: CUDA if built, else Metal (a build
/// with neither refuses Device::GPU in set_device).
constexpr std::string_view gpu_backend = core::detail::kGpuBackendName;

/// What Device::GPU computes in for `precision`: Metal has no FP64, so its Auto
/// is FP32; CUDA's Auto stays open until a fill meets a GPU (FP64 where it is fast).
constexpr GpuPrecision resolve_gpu_precision(GpuPrecision precision)
{
  return gpu_backend == "Metal" && precision == GpuPrecision::Auto ? GpuPrecision::FP32 : precision;
}

/// What this build alone refuses of a GPU request: no GPU backend, cpu+gpu without CUDA, and a GPU index other than 0
/// on Metal, which runs on the system default GPU.
void require_gpu_build(std::string_view where, Device device, [[maybe_unused]] int index)
{
  if (device == Device::CPU_GPU && !detail::hybrid_not_built_reason().empty())
    throw detail::hybrid_refusal(where, detail::hybrid_not_built_reason());
  if (device == Device::GPU) {
#if !defined(DTWC_HAS_CUDA) && !defined(DTWC_HAS_METAL)
    throw detail::gpu_not_built_refusal();
#elif !defined(DTWC_HAS_CUDA)
    if (index != 0)
      core::detail::reject_gpu(where,
                               "runs on the system default GPU, but GPU index = " + std::to_string(index)
                                 + " was requested",
                               "Use gpu (index 0).");
#endif
  }
}

/// Whether `method` has a GPU path: onebatch and tadpole compute the pairs they need on the CPU as they go, so there is
/// no fill and no pass for a GPU to take.
constexpr bool has_gpu_path(Method method) { return method != Method::OneBatch && method != Method::TADPole; }

/// Why this build's GPU backend cannot compute on GPU `index` here, "" where it can (require_gpu_device).
std::string gpu_device_refusal([[maybe_unused]] int index)
{
#if defined(DTWC_HAS_CUDA)
  return cuda::cuda_device_refusal(index);
#elif defined(DTWC_HAS_METAL)
  return metal::metal_available() ? std::string()
                                  : "DTWC++ was built with Metal support, but no usable Metal GPU was detected ("
                                      + metal::metal_device_info() + ")";
#else
  return {}; // a build with no GPU backend is refused first, by require_gpu_build
#endif
}

/// Every refusal of `prob`'s GPU request that needs no pair computed, in the order a run meets them: this build, the
/// method (none: a fill), the kernels' rules for the series it holds, the device. A GpuRefusal; a Problem on the CPU
/// passes.
void require_gpu_request(std::string_view where, const Problem &prob, std::optional<Method> method)
{
  const auto [device, index] = prob.device();
  if (!uses_gpu(device)) return;
  require_gpu_build(where, device, index);
  if (method && !has_gpu_path(*method)) throw detail::as_it_goes_refusal(where, name_of(method_names, *method), device);
  validate_gpu_request(where, prob, prob.data().precision, prob.data().ndim);
  require_gpu_device(where, index);
}

/// The device a fill of `prob` computes on: the one it asks for, except that a Problem given `fallback` whose GPU
/// request a fill would refuse computes on the CPU (detail::DeviceScope). The cache's identity names the precision of
/// this device, so a save, a load and a bind outside a fill agree with the saves a fill makes inside its scope.
Device computing_device(const Problem &prob)
{
  const Device device = prob.device().first;
  if (!uses_gpu(device) || !prob.fallback()) return device;
  try {
    require_gpu_request("Problem::distance_checkpoint_identity", prob, std::nullopt);
  } catch (const GpuRefusal &) {
    return Device::CPU;
  }
  return device;
}

/// The pair function a Problem binds: core::resolve_dtw_fn's, and under Soft-DTW the divergence of it, which the
/// clustering uses (core::soft_dtw_divergence), three raw values a pair. `raw`, when given, takes the raw function
/// under Soft-DTW and is emptied under any other variant.
template <typename T>
std::function<double(std::span<const T>, std::span<const T>)>
bind_pair_function(const core::DistanceConfig &config, const Data &data,
                   std::function<double(std::span<const T>, std::span<const T>)> *raw)
{
  auto pair = core::resolve_dtw_fn<T>(config, data);
  const bool divergence = config.variant.variant == core::DTWVariant::SoftDTW;
  if (raw) *raw = divergence ? pair : nullptr;
  if (!divergence) return pair;
  return [pair = std::move(pair)](std::span<const T> x, std::span<const T> y) {
    return core::soft_dtw_divergence(pair(x, y), pair(x, x), pair(y, y));
  };
}

} // namespace

void validate_gpu_request(std::string_view where, const Problem &prob, core::Precision series_precision,
                          std::size_t ndim)
{
  // The GPU routes upload univariate Float64 series and run Standard DTW with no missing-data strategy,
  // in L1 or squared L2; so does cpu+gpu, whose GPU share is the GPU's.
  if (!uses_gpu(prob.device().first)) return;
  core::detail::validate_gpu_distance_config(
    where, series_precision == core::Precision::Float32, ndim, prob.variant_params().variant, prob.missing_strategy());
#if !defined(DTWC_HAS_CUDA) && defined(DTWC_HAS_METAL)
  metal::validate_metal_precision(prob.gpu_precision());
#endif
}

void require_gpu_device(std::string_view where, int index)
{
  if (const std::string why = gpu_device_refusal(index); !why.empty())
    throw GpuRefusal(std::string(where) + ": " + why + ". No CPU fallback was attempted.", why);
}

bool gpu_available()
{
#if defined(DTWC_HAS_CUDA)
  return cuda::cuda_available();
#elif defined(DTWC_HAS_METAL)
  return metal::metal_available();
#else
  return false;
#endif
}

std::string gpu_info()
{
#if defined(DTWC_HAS_CUDA)
  return "CUDA: " + cuda::cuda_device_info(0);
#elif defined(DTWC_HAS_METAL)
  return "Metal: " + metal::metal_device_info();
#else
  return "no GPU backend compiled in (rebuild with -DDTWC_ENABLE_CUDA=ON, or on macOS "
         "-DDTWC_ENABLE_METAL=ON)";
#endif
}

Method resolve_method(Method method, Device device, std::size_t n_series)
{
  // On the CPU; the GPU fills PAM's matrix at any N, and cpu+gpu at least as fast as the GPU alone.
  constexpr std::size_t pam_max_series = 5000;
  if (method != Method::Auto) return method;
  return uses_gpu(device) || n_series <= pam_max_series ? Method::PAM : Method::CLARA;
}

namespace detail {

DeviceScope::DeviceScope(Problem &prob, std::string_view where, std::optional<Method> method)
  : prob_(prob), device_(prob.device_), index_(prob.device_index_), share_(prob.gpu_share_)
{
  if (!uses_gpu(prob.device_)) return;
  if (!prob.fallback_) { // the one refusal made here is a method with no GPU path; the rest are the fill's and the method's own
    if (method && !has_gpu_path(*method)) throw as_it_goes_refusal(where, name_of(method_names, *method), prob.device_);
    return;
  }
  try {
    require_gpu_request(where, prob, method);
  } catch (const GpuRefusal &refusal) {
    say_fallback(prob.device_, prob.device_index_, refusal.reason());
    // Not set_device: it would drop the matrix, and the request is the Problem's to keep.
    prob.device_ = Device::CPU;
    prob.device_index_ = 0;
    prob.gpu_share_ = -1.0; // the CPU has no share to force
    fell_back_ = true;
  }
}

DeviceScope::~DeviceScope()
{
  if (!fell_back_) return;
  prob_.device_ = device_;
  prob_.device_index_ = index_;
  prob_.gpu_share_ = share_;
}

} // namespace detail

std::string describe(const core::FillSplit &fill)
{
  std::ostringstream text;
  text << "cpu+gpu: the GPU filled " << fill.gpu_entries << " of " << fill.entries
       << " entries, the triangle with its diagonal";
  if (fill.gpu_fp32) text << ", its rows in FP32 and the CPU's in FP64";
  text << " (rate share " << std::setprecision(17) << fill.split.gpu_share << ", " << fill.split.source;
  if (fill.split.gpu_rate > 0)
    text << std::setprecision(4) << ": the CPU's team " << fill.split.cpu_rate << " and the GPU "
         << fill.split.gpu_rate << " pairs a second";
  if (fill.early_share >= 0)
    text << "; the slabs before the calibrating one at the recorded " << std::setprecision(17) << fill.early_share;
  text << "), the CPU's team the rest";
  return text.str();
}

void require_clusterable(index_t k, std::size_t n_series)
{
  if (n_series == 0) throw InvalidInput("cluster: dataset is empty.");
  if (static_cast<std::size_t>(k) > n_series) throw InvalidInput("cluster: k must not exceed the number of series.");
}

std::size_t parse_ram_limit(const std::string &text, std::string_view key)
{
  if (text.empty()) return 0;
  const char *const first = text.data();
  const char *const last = first + text.size();
  // A plain byte count reads as an integer, so every count to_config_text()
  // writes reads back exactly.
  std::size_t bytes = 0;
  if (const auto [end, ec] = std::from_chars(first, last, bytes); ec == std::errc{} && end == last) return bytes;

  constexpr std::pair<std::string_view, int> units[]{ // spelling, power of two
    { "", 0 },     { "B", 0 },   { "K", 10 },  { "KB", 10 },  { "KIB", 10 }, { "M", 20 },  { "MB", 20 },
    { "MIB", 20 }, { "G", 30 },  { "GB", 30 }, { "GIB", 30 }, { "T", 40 },   { "TB", 40 }, { "TIB", 40 },
  };
  double value = -1.0;
  const auto [end, ec] = io::parse_number(first, last, value); // not std::from_chars: macOS 26+ only
  std::string unit(end, last);
  for (char &c : unit) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  const auto match = std::find_if(std::begin(units), std::end(units), [&](const auto &u) { return u.first == unit; });
  const double scaled = match == std::end(units) ? -1.0 : std::ldexp(value, match->second);
  // The one range check: a negative, NaN or infinite value and an unknown unit
  // fail it, and so does a count size_t cannot hold, whose cast would be undefined.
  // A conjunction of the passing comparisons, not the negated pair of failing ones: every comparison is false for NaN.
  const bool in_range = scaled >= 0.0 && scaled < std::ldexp(1.0, std::numeric_limits<std::size_t>::digits);
  if (ec != std::errc{} || !in_range)
    throw InvalidInput("Invalid " + std::string(key) + " '" + text
                       + "': expected a size such as 2G, 500M or 1.5GiB, in B, K, M, G or T (KB and KiB alike), "
                         "below 2^64 bytes; 0 is automatic.");
  return static_cast<std::size_t>(std::ceil(scaled));
}

void apply(const Config &config, Problem &prob)
{
  // Each setter checks its value: the distance in one call (parameter domains, variant x missing strategy x metric,
  // the band), so no part of it meets the Problem's old ones, then the cluster count and the iteration limits.
  prob.set_distance(
    { .variant = config.variant, .metric = config.metric, .missing = config.missing, .band = config.band });
  prob.set_n_clusters(config.k);
  prob.set_max_iter(config.max_iter);
  prob.set_n_repetitions(config.n_init);
  prob.set_random_seed(config.seed);
  prob.set_method(config.method);
  prob.set_sample_size(config.sample_size);
  prob.set_n_samples(config.n_samples);
  prob.set_batch_size(config.batch_size);
  prob.set_linkage(config.linkage);
  prob.set_tadpole_dc(config.tadpole_dc); // < 0: auto-select from a DTW subsample
  prob.set_ram_limit(config.ram_limit);
  prob.set_verbose(config.verbose);
  validate_mip_settings(config.mip);
  prob.mip_settings = config.mip;
  // A false set_solver means HiGHS: gurobi on a build without Gurobi must not
  // solve with HiGHS.
  if (!prob.set_solver(config.solver))
    throw SolverError("--solver " + std::string(name_of(solver_names, config.solver))
                      + " is not available: this DTWC++ was built without Gurobi. Use --solver highs, or "
                        "rebuild with -DDTWC_ENABLE_GUROBI=ON and GUROBI_HOME set.");
  prob.set_gpu_precision(config.gpu_precision);
  prob.set_gpu_share(config.gpu_share);
  prob.set_matrix_dtype(config.matrix_dtype);
  // gpu without a GPU backend: the GpuRefusal, unless fallback, which leaves what the GPU cannot honour to the fill or
  // the method that starts (Problem::cluster, a fill) or, in dtwc::run, to the run, which settles it before any file is read.
  prob.set_device(config.device, config.device_index, config.fallback);
  // Whatever the method, the run reaches a fill or a pass the share would not govern on another device (onebatch and
  // tadpole, and a stream, reach neither): refuse it here, as the fill and the pass refuse it where they run.
  if (config.gpu_share != -1.0 && config.device != Device::CPU_GPU)
    throw InvalidInput(
      detail::share_ignored_message("run", "the distance matrix's fill and FastCLARA's pass", config.device));
  if (uses_gpu(config.device) && !config.fallback && !has_gpu_path(config.method))
    throw detail::as_it_goes_refusal("run", name_of(method_names, config.method), config.device);
}

/**
 * @brief Sets the number of clusters for the problem.
 *
 * @param Nc_ The number of clusters to set.
 * @throws InvalidInput for Nc_ < 1. Nc_ > N is accepted: the data may change
 *         after this call, so cluster() checks it against the series it finds.
 */
void Problem::set_n_clusters(index_t Nc_)
{
  if (Nc_ < 1)
    throw InvalidInput("Problem::set_n_clusters: n_clusters must be at least 1; got "
                       + std::to_string(Nc_) + ".");
  Nc = Nc_;
}

void Problem::require_clustered(std::string_view who) const
{
  if (clusters_ind.size() == size() && centroids_ind.size() == static_cast<std::size_t>(Nc)) return;
  throw InvalidInput(std::string(who) + ": this Problem holds no clustering ("
                     + std::to_string(clusters_ind.size()) + " labels for N = " + std::to_string(size())
                     + ", " + std::to_string(centroids_ind.size()) + " medoids for k = "
                     + std::to_string(Nc) + "); cluster it first.");
}

void Problem::require_owned_storage(std::string_view accessor, bool float64_values) const
{
  const std::string at(accessor);
  if (data_.is_view())
    throw InvalidInput(
      at + ": this Problem's series are a non-owning view (set_view_data), so "
           "there is no owned "
      + (float64_values ? "series to return. Use series(i) (data().series_f32(i) "
                          "for Float32), which reads every storage mode."
                        : "name to return. Use series_name(i), which reads every "
                          "storage mode."));
  if (!float64_values) return;
  if (data_.is_f32())
    throw InvalidInput(at + ": this Problem holds Float32 series, and p_vec "
                            "returns the Float64 store. Use data().series_f32(i).");
}

Problem::Problem(Problem &&) = default;

Problem &Problem::operator=(Problem &&) = default;

void Problem::set_max_iter(int n)
{
  if (n < 1)
    throw InvalidInput(
      "Problem::set_max_iter: max_iter must be at least 1; got " + std::to_string(n)
      + ". With no iteration the initial medoids' cost would be reported as the "
        "clustering result.");
  max_iter_ = n;
}

void Problem::set_n_repetitions(int n)
{
  if (n < 1)
    throw InvalidInput(
      "Problem::set_n_repetitions: n_repetitions must be at least 1; got "
      + std::to_string(n) + ". It counts the random restarts, and one run is the "
        "minimum.");
  n_repetitions_ = n;
}

/**
 * @brief Sets the initial centroids for clustering.
 *
 * @param candidate_centroids A vector containing the indices of candidate centroids.
 * @throws InvalidInput if the size of candidate_centroids is not equal to Nc, or an index is outside [0, N).
 */
void Problem::set_clusters(const std::vector<index_t> &candidate_centroids)
{
  if (candidate_centroids.size() != static_cast<size_t>(Nc))
    throw InvalidInput("Set cluster has failed as number of centroids is not same as the number of indices in candidate centroids vector.\n");
  for (const index_t medoid : candidate_centroids)
    if (medoid < 0 || medoid >= size())
      throw InvalidInput("Problem::set_clusters: medoid " + std::to_string(medoid) + " is outside [0, N), N = "
                         + std::to_string(size()) + ".");

  centroids_ind = candidate_centroids;
}

void Problem::set_result(const core::ClusteringResult &result)
{
  const auto &medoids = result.medoid_indices;
  const auto &labels = result.labels;
  // A settings-only Problem (Parquet-streamed CLARA) holds no series; the
  // result's own point count is the reference there.
  const std::size_t n = size() == 0 ? labels.size() : size();
  if (medoids.empty() || medoids.size() > n || labels.size() != n)
    throw InvalidInput("Problem::set_result: expected 1..N medoids and N = " + std::to_string(n)
                       + " labels; got " + std::to_string(medoids.size()) + " medoids and "
                       + std::to_string(labels.size()) + " labels.");
  std::vector<bool> is_medoid(n, false);
  for (const index_t medoid : medoids) {
    if (medoid < 0 || static_cast<std::size_t>(medoid) >= n || is_medoid[static_cast<std::size_t>(medoid)])
      throw InvalidInput("Problem::set_result: medoid " + std::to_string(medoid)
                         + " is outside [0, N) or repeated.");
    is_medoid[static_cast<std::size_t>(medoid)] = true;
  }
  const auto k = static_cast<index_t>(medoids.size());
  for (const index_t label : labels)
    if (label < 0 || label >= k)
      throw InvalidInput("Problem::set_result: label " + std::to_string(label) + " is outside [0, k).");

  set_n_clusters(k);
  centroids_ind = medoids;
  clusters_ind = labels;
}

bool Problem::set_solver(Solver solver_)
{
#ifndef DTWC_ENABLE_GUROBI
  if (solver_ == Solver::Gurobi) {
    mipSolver = settings::DEFAULT_MIP_SOLVER;
    return false;
  }
#endif
  mipSolver = solver_;
  return true;
}

void Problem::print_distance_matrix() const
{
  std::cout << distMat;
}

/**
 * @brief Refreshes the distance matrix.
 * @details Drops the matrix and binds the distance functions again. Does NOT allocate the dense N×N matrix — that is
 * deferred to fill_distance_matrix() so that large-N algorithms (e.g.
 * FastCLARA) can load data without forcing quadratic memory usage. A mapped
 * matrix is detached and its file left as it is, so rebinding it under other
 * settings fails its fingerprint check loudly.
 */
void Problem::refresh_distance_matrix()
{
  distMat = core::DistanceMatrix(0, matrix_dtype_); // empty, of the width every resize keeps
  filled_ = false;
  fill_split_.reset(); // the split of the fill that made the matrix dropped here
  self_terms_ = {};    // Soft-DTW's, of the series and gamma the matrix had
  rebind_dtw_fn(); // series edited in place may have new lengths (the WDTW table)
}

/**
 * @brief Resolve the distance functions from distance_ and the series, once.
 * @details Every function holds copies of the settings it reads, so it stays
 * valid when this Problem moves, and parallel callers only read it. The float32
 * ones are bound here for float32 series, whose settings were validated for
 * float32 when set; for other series dtw_function_f32() binds it after that check.
 */
void Problem::rebind_dtw_fn()
{
  dtw_fn_ = bind_pair_function<data_t>(distance_, data_, &raw_fn_);
  dtw_block_fn_ = core::resolve_dtw_block_fn<data_t>(distance_);
  dtw_fn_f32_ = {};
  raw_fn_f32_ = {};
  dtw_block_fn_f32_ = {};
  if (data_.is_f32()) {
    dtw_fn_f32_ = bind_pair_function<float>(distance_, data_, &raw_fn_f32_);
    dtw_block_fn_f32_ = core::resolve_dtw_block_fn<float>(distance_);
  }
}

void Problem::set_distance(core::DistanceConfig config)
{
  if (config.band < -1)
    throw InvalidInput("Problem::set_distance: band must be -1 (full DTW) or at least 0; got "
                       + std::to_string(config.band) + ".");
  config.ndim = data_.ndim;
  validate_distance(config, data_);
  if (config == distance_) return;
  distance_ = config;
  clusters_ind.clear(); // a clustering describes the distances it was computed with
  centroids_ind.clear();
  refresh_distance_matrix();
}

void Problem::set_band(int b)
{
  if (b < -1)
    throw InvalidInput("Problem::set_band: band must be -1 (full DTW) or at least 0; got "
                       + std::to_string(b) + ".");
  auto config = distance();
  config.band = b;
  set_distance(config);
}

void Problem::set_metric(core::MetricType metric)
{
  auto config = distance();
  config.metric = metric;
  set_distance(config);
}

void Problem::set_missing_strategy(core::MissingStrategy strategy)
{
  auto config = distance();
  config.missing = strategy;
  set_distance(config);
}

void Problem::set_variant(core::DTWVariant v)
{
  auto config = distance();
  config.variant.variant = v;
  set_distance(config);
}

void Problem::set_variant(core::DTWVariantParams params)
{
  auto config = distance();
  config.variant = params;
  set_distance(config);
}

void Problem::set_device(Device device, int index, bool fallback)
{
  if (index < 0)
    throw InvalidInput("Problem::set_device: the GPU index must be >= 0; got "
                       + std::to_string(index) + ".");
  if (!fallback) require_gpu_build("Problem::set_device", device, index);
  fallback_ = fallback; // no distance depends on it: the matrix stays
  const int gpu_index = uses_gpu(device) ? index : 0; // the CPU has no index
  if (device_ == device && device_index_ == gpu_index) return;
  device_ = device;
  device_index_ = gpu_index;
  refresh_distance_matrix();
}

void Problem::set_gpu_share(double share)
{
  if (share != -1.0 && (std::isnan(share) || share < 0.0 || share > 1.0))
    throw InvalidInput("Problem::set_gpu_share: the share must be -1 (measured) or in [0, 1]; got "
                       + std::to_string(share) + ".");
  gpu_share_ = share;
}

void Problem::copy_distance_settings_from(const Problem &other)
{
  set_distance(other.distance());
  set_device(other.device_, other.device_index_, other.fallback_);
  set_gpu_precision(other.gpu_precision_);
  set_matrix_dtype(other.matrix_dtype_);
  verbose_ = other.verbose_;
}

void Problem::validate_distance(core::DistanceConfig config, const Data &data)
{
  config.ndim = data.ndim;
  core::validate(config, data.is_f32());
}

void Problem::adopt_data(Data candidate, std::string_view operation)
{
  candidate.validate_ndim();
  reject_empty_series(candidate, operation);
  validate_distance(distance_, candidate);
  data_ = std::move(candidate);
  distance_.ndim = data_.ndim;
  clusters_ind.clear(); // a clustering describes the series it was computed on
  centroids_ind.clear();
  refresh_distance_matrix();
}

const Problem::dtw_fn_t &Problem::dtw_function()
{
  auto check = series_check("Problem::dtw_function", false);
  check.check(0, static_cast<std::size_t>(data_.size()));
  check.finish();
  fill_self_terms(); // pair_distance's, from the series just checked
  return dtw_function(check);
}

const Problem::dtw_fn_f32_t &Problem::dtw_function_f32()
{
  auto check = series_check("Problem::dtw_function_f32", true);
  check.check(0, static_cast<std::size_t>(data_.size()));
  check.finish();
  fill_self_terms();
  return dtw_function_f32(check);
}

Problem::SeriesCheck Problem::series_check(std::string_view where, bool f32)
{
  if (f32) core::validate(distance_, true); // the series may be Float64: float32 must hold the parameters
  return SeriesCheck(*this, where, f32);
}

const Problem::dtw_fn_t &Problem::dtw_function(const SeriesCheck &check)
{
  if (check.problem_ != this) throw std::logic_error("Problem::dtw_function: the check is another Problem's.");
  return dtw_fn_;
}

const Problem::dtw_fn_f32_t &Problem::dtw_function_f32(const SeriesCheck &check)
{
  if (check.problem_ != this || !check.f32_)
    throw std::logic_error("Problem::dtw_function_f32: the check is another Problem's, or not the float32 function's.");
  if (!dtw_fn_f32_) dtw_fn_f32_ = bind_pair_function<float>(distance_, data_, nullptr);
  return dtw_fn_f32_;
}

void Problem::fill_self_terms()
{
  if (!raw_fn_ || self_terms_.size() == data_.size()) return; // another variant, or computed
  std::vector<double> terms(data_.size());
  auto term = [&](std::size_t i) {
    terms[i] = data_.is_f32() ? raw_fn_f32_(data_.series_f32(i), data_.series_f32(i)) : raw_fn_(series(i), series(i));
  };
  run_openmp(term, terms.size());
  self_terms_ = std::move(terms);
}

double Problem::pair_distance(std::size_t i, std::size_t j) const
{
  const bool f32 = data_.is_f32();
  if (self_terms_.empty()) // another variant, or Soft-DTW's divergence from three raw values
    return f32 ? dtw_fn_f32_(data_.series_f32(i), data_.series_f32(j)) : dtw_fn_(data_.series(i), data_.series(j));
  return core::soft_dtw_divergence(
    f32 ? raw_fn_f32_(data_.series_f32(i), data_.series_f32(j)) : raw_fn_(data_.series(i), data_.series(j)),
    self_terms_[i], self_terms_[j]);
}

core::DistanceMatrix::fingerprint_type
Problem::distance_checkpoint_identity() const
{
  // The precision the distances are computed in: the series' on the CPU, the
  // GPU's on a GPU. Not the device: FP64 on a CPU and on a GPU differ by
  // rounding, so either's cache serves the other (and GPU 0's serves GPU 1),
  // while FP32 distances in an FP64 run are a loss of precision. A Problem
  // given `fallback` is identified as the device a fill computes on.
  const Device device = computing_device(*this);
  const GpuPrecision gpu = resolve_gpu_precision(gpu_precision_);
  if (uses_gpu(device) && gpu == GpuPrecision::Auto)
    throw InvalidInput(
      "A persistent distance cache (a mapped .dtwm or a checkpoint) records the precision "
      "its distances are computed in, but CUDA precision Auto resolves to FP32 or FP64 on the "
      "GPU a fill meets. Select explicit FP32 or FP64 (set_gpu_precision, --gpu-precision) "
      "before binding or saving the cache.");
  const core::Precision computed = device == Device::CPU ? data_.precision
                                 : gpu == GpuPrecision::FP32 ? core::Precision::Float32
                                                             : core::Precision::Float64;

  // The settings the stored distances were computed with, the metric among
  // them: without it a SquaredL2 run writes the same fingerprint as an L1 run
  // over the same data, and a later L1 run then accepts the wrong matrix. All
  // variant parameters are included, even when inactive for the selected
  // variant: a harmless cache miss beats trusting an ambiguous configuration.
  FingerprintHash configuration;
  static constexpr char configuration_domain[] = "dtwc-distance-cache-configuration-v2";
  configuration.update(configuration_domain, sizeof(configuration_domain) - 1);
  hash_enum(configuration, distance_.metric);
  hash_u64(configuration, static_cast<std::uint64_t>(static_cast<std::int64_t>(distance_.band)));
  hash_enum(configuration, distance_.variant.variant);
  hash_double(configuration, distance_.variant.wdtw_g);
  hash_double(configuration, distance_.variant.adtw_penalty);
  hash_double(configuration, distance_.variant.sdtw_gamma);
  hash_double(configuration, distance_.variant.msm_c);
  hash_double(configuration, distance_.variant.twe_nu);
  hash_double(configuration, distance_.variant.twe_lambda);
  hash_enum(configuration, distance_.variant.mv_mode);
  hash_enum(configuration, distance_.missing);
  hash_enum(configuration, computed);
  // A Soft-DTW matrix holds the divergence (rebind_dtw_fn), where an earlier build stored the raw values: a file of
  // those is another matrix. Only Soft-DTW's identity takes the mark, so every other variant's files still load.
  if (distance_.variant.variant == core::DTWVariant::SoftDTW) {
    static constexpr char divergence[] = "soft-dtw-divergence";
    configuration.update(divergence, sizeof(divergence) - 1);
  }

  FingerprintHash hash;
  static constexpr char domain[] = "dtwc-distance-cache-fingerprint-v1";
  hash.update(domain, sizeof(domain) - 1);
  hash.update(configuration.digest());

  // Dataset identity: representation, dimensions, series ordering, each flat
  // length, and every IEEE value bit. Names are intentionally excluded because
  // they cannot affect a distance. Canonical integer encoding keeps the digest
  // independent of std::hash and host word width.
  hash_u64(hash, static_cast<std::uint64_t>(data_.size()));
  hash_u64(hash, static_cast<std::uint64_t>(data_.ndim));
  hash_enum(hash, data_.precision);
  for (std::size_t i = 0; i < data_.size(); ++i) {
    hash_u64(hash, data_.series_flat_size(i));
    if (data_.is_f32()) {
      for (const float value : data_.series_f32(i))
        hash_float(hash, value);
    } else {
      for (const data_t value : series(i))
        hash_double(hash, value);
    }
  }
  return hash.digest();
}

void Problem::use_mmap_distance_matrix(const std::filesystem::path &cache_path)
{
  // Every check, the identity and the mapping come before any change: a bind
  // that fails leaves this Problem's matrix as it was.
  validate_distance(distance_, data_);
  // map() checks an existing file's magic, version, length, N and fingerprint
  // before any of its distances can be read.
  auto mapped = core::DistanceMatrix::map(cache_path, data_.size(),
                                          distance_checkpoint_identity(), matrix_dtype_);
  // A warm start holds every pair.
  const bool complete = mapped.all_computed("Problem::use_mmap_distance_matrix");
  distMat = std::move(mapped);
  filled_ = distMat.size() > 0 && complete;
}

/// Reject automatic-checkpoint settings that fill_distance_matrix cannot honour,
/// before any distance is computed.
void Problem::validate_checkpoint_settings() const
{
  if (!checkpoint.enabled) return;
  if (checkpoint.save_interval < 1)
    throw InvalidInput(
      "Problem::fill_distance_matrix: checkpoint.save_interval must be at least "
      "1 row when checkpoint.enabled; got "
      + std::to_string(checkpoint.save_interval) + ".");
  if (checkpoint.directory.empty())
    throw InvalidInput(
      "Problem::fill_distance_matrix: checkpoint.enabled requires a non-empty "
      "checkpoint.directory.");
  if (distMat.is_mapped() && !distMat.is_mapped_to(checkpoint_path(*this, checkpoint.directory)))
    throw InvalidInput("Problem::fill_distance_matrix: the matrix is mapped to another file than checkpoint.directory's, "
                       "where a fill's saves keep no row; map that checkpoint file itself, or disable checkpointing.");
  // The autosave records the precision its distances were computed in, which
  // CUDA's Auto leaves open: the identity refuses it here, before any pair.
  if (uses_gpu(device_) && resolve_gpu_precision(gpu_precision_) == GpuPrecision::Auto)
    (void)distance_checkpoint_identity();
}

void Problem::validate_fill_request(std::string_view where) const
{
  SeriesCheck check(*this, where, false);
  check.check(0, static_cast<std::size_t>(data_.size()));
  check.finish();
}

Problem::SeriesCheck::SeriesCheck(const Problem &problem, std::string_view where, bool f32)
  : problem_(&problem), where_(where), f32_(f32)
{
  const Data &data = problem.data_;
  // On every device: set_data refuses an empty series, but an edit through
  // p_vec can empty one afterwards, and the kernels answer it with the finite
  // max() sentinel. First, so a band check does not blame the empty series' length.
  reject_empty_series(data, where_);

  // A band narrower than a pair's length difference leaves that pair no warping
  // path: the banded kernels return the finite max() sentinel, which passes every
  // isfinite() guard, so clustering would silently sum 1.8e308. The widest gap is
  // shortest vs longest. Soft-DTW, MSM and TWE ignore the band. Lengths are
  // timesteps, as the multivariate kernels count them.
  const int band = problem.distance_.band;
  if (band >= 0 && data.size() > 1 && core::variant_uses_band(problem.distance_.variant.variant)) {
    std::size_t shortest = 0, longest = 0, low = data.series_length(0), high = low; // the first of each, as found
    for (std::size_t i = 1; i < static_cast<std::size_t>(data.size()); ++i) {
      const std::size_t length = data.series_length(i);
      if (length < low) {
        shortest = i;
        low = length;
      }
      if (length > high) {
        longest = i;
        high = length;
      }
    }
    const std::size_t gap = high - low;
    if (gap > static_cast<std::size_t>(band)) {
      const auto describe = [&](std::size_t i) {
        return "series '" + std::string(data.name(i)) + "' (index "
             + std::to_string(i) + ", length "
             + std::to_string(data.series_length(i)) + ")";
      };
      throw InvalidInput(
        where_ + ": band = " + std::to_string(band) + " is narrower than the length "
        "difference between " + describe(shortest) + " and " + describe(longest)
        + ", so no warping path fits that pair. The smallest feasible band is "
        + std::to_string(gap) + "; pass band >= " + std::to_string(gap)
        + ", or band = -1 for full DTW.");
    }
  }

  const std::size_t budget = problem.ram_limit_ != 0 ? problem.ram_limit_ : core::available_memory(); // 0: unknown
  block_bytes_ = budget == 0 ? std::size_t{ 1 } << 30 : budget / 8;
}

/// Series i's refused value, named by its series and (flat) position: require_finite's message.
void Problem::SeriesCheck::refuse(std::size_t i) const
{
  const Data &data = problem_->data_;
  const std::string name = "series '" + std::string(data.name(i)) + "' (index " + std::to_string(i) + ")";
  const bool nan_is_missing = problem_->distance_.missing != core::MissingStrategy::Error;
  detail::refuse_series_unusable(data, i, name, where_, nan_is_missing);
}

namespace {

/// Ask for the pages of series [first, first + count) of `data` when they lie in a mapped file: a view whose bytes an
/// owner keeps (the Arrow IPC reader's), the series ascending within twice their values' bytes, as a file's rows lie.
/// Owned series are in memory, and a view of scattered storage is left alone.
void prefetch_series(const Data &data, std::size_t first, std::size_t count)
{
  if (count == 0 || !data.is_view() || !data.owner()) return;
  const auto read_ahead = [&](const auto &series_of) {
    std::size_t bytes = 0;
    for (std::size_t i = first; i < first + count; ++i) bytes += series_of(i).size_bytes();
    const auto back = series_of(first + count - 1);
    const auto begin = reinterpret_cast<std::uintptr_t>(series_of(first).data());
    const auto end = reinterpret_cast<std::uintptr_t>(back.data() + back.size());
    if (begin < end && end - begin <= 2 * bytes)
      core::prefetch_pages(series_of(first).data(), end - begin);
  };
  if (data.is_f32())
    read_ahead([&data](std::size_t i) { return data.series_f32(i); });
  else
    read_ahead([&data](std::size_t i) { return data.series(i); });
}

} // namespace

void Problem::SeriesCheck::check(std::size_t first, std::size_t count)
{
  sweep(first, count, std::min(count, static_cast<std::size_t>(problem_->data_.size()) - (first + count)));
}

void Problem::SeriesCheck::sweep(std::size_t first, std::size_t count, std::size_t ahead)
{
  const Data &data = problem_->data_;
  const auto missing = problem_->distance_.missing;
  const std::size_t end = first + count, value_bytes = data.is_f32() ? sizeof(float) : sizeof(data_t);
  for (std::size_t b0 = first; b0 < end;) {
    std::size_t b1 = b0 + 1; // the block [b0, b1)
    for (std::size_t bytes = data.series_flat_size(b0) * value_bytes;
         b1 < end && (bytes += data.series_flat_size(b1) * value_bytes) <= block_bytes_;)
      ++b1;
    // Each series' verdict in its own byte; after the join, the first refused series in the block, the lowest of all
    // as the blocks go in order, throws.
    std::vector<detail::SeriesVerdict> verdicts(b1 - b0);
    auto task = [&](std::size_t i) {
      if (i == 0 && b0 == first) prefetch_series(data, end, ahead);
      verdicts[i] = detail::series_verdict(data, b0 + i, missing);
    };
    run_openmp(task, verdicts.size(), verdicts.size() > 64, 16);
    for (std::size_t i = 0; i < verdicts.size(); ++i) {
      if (verdicts[i] == detail::SeriesVerdict::Refused) refuse(b0 + i);
      if (verdicts[i] == detail::SeriesVerdict::EntirelyNaN) mark_ = std::min(mark_, b0 + i);
    }
    b0 = b1;
  }
}

void Problem::SeriesCheck::check_rows(std::span<const index_t> rows)
{
  const Data &data = problem_->data_;
  const auto missing = problem_->distance_.missing;
  std::vector<detail::SeriesVerdict> verdicts(rows.size());
  auto task = [&](std::size_t i) { verdicts[i] = detail::series_verdict(data, static_cast<std::size_t>(rows[i]), missing); };
  run_openmp(task, rows.size(), rows.size() > 64, 16);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto row = static_cast<std::size_t>(rows[i]);
    if (verdicts[i] == detail::SeriesVerdict::Refused) {
      sweep(0, row, 0); // a lower series refused comes first; nothing past row is read
      refuse(row);
    }
    if (verdicts[i] == detail::SeriesVerdict::EntirelyNaN) mark_ = std::min(mark_, row);
  }
}

void Problem::SeriesCheck::finish() const
{
  const Problem &problem = *problem_;
  if (marked())
    throw InvalidInput(where_ + ": series '" + std::string(problem.series_name(mark_)) + "' (index "
                       + std::to_string(mark_) + ") is entirely NaN, so MissingStrategy::Interpolate has nothing to "
                       "interpolate from. Use ZeroCost or AROW, or drop the series.");

  // The GPU routes also need univariate Float64 series (CUDA reads them where they lie; Metal's fill uploads a view's
  // as an owned copy).
  if (!uses_gpu(problem.device_)) return;
  try {
    validate_gpu_request(where_, problem, problem.data_.precision, problem.data_.ndim);
  } catch (const GpuRefusal &refusal) {
    // The pairs this check guards compute on the CPU whatever the device (dtw_function): with fallback the request the
    // kernels cannot run is said and set aside; a fill is in a DeviceScope, which made the device the CPU's first.
    if (!problem.fallback_) throw;
    detail::say_fallback(problem.device_, problem.device_index_, refusal.reason());
  }
}

/**
 * @brief Fills the distance matrix using brute-force parallel computation.
 * @details Every pair through the bound lane functions or pair_distance (all DTW
 *          variants), block by block (see the body).
 */
namespace {

struct FillLayout
{
  std::size_t n = 0;
  bool f32 = false;
  std::size_t width = 0;
  std::size_t h = 0;
  std::size_t slab_bytes = 0;
  std::size_t entry_bytes = 0;
  std::size_t save_rows = 0;
  std::vector<std::size_t> bounds;
  std::size_t K = 0;
  std::vector<std::size_t> first_column;
  bool ragged = false;
  bool lanes = false;
  std::vector<std::size_t> order;
  bool reordered = false;
  std::size_t written = 0;
  bool fresh = false;
  bool use_slab_panels = false;
  std::vector<std::size_t> slabs;
};

FillLayout build_fill_layout(const Data &data, const core::DistanceMatrix &distMat, std::size_t ram_limit,
                             bool checkpoint_enabled, int checkpoint_save_interval, bool sized, bool has_block_fn)
{
  FillLayout layout;
  layout.n = data.size();
  layout.f32 = data.is_f32();
  const std::size_t N = layout.n;
  const std::size_t W = layout.f32 ? core::dtw_lanes<float> : core::dtw_lanes<data_t>;
  layout.width = W;

  std::size_t values = 0;
  for (std::size_t i = 0; i < N; ++i) values += data.series_flat_size(i);
  const std::size_t series_bytes = values * (layout.f32 ? sizeof(float) : sizeof(data_t));
  layout.entry_bytes = core::value_bytes(distMat.width());
  constexpr std::uint64_t tile_series_bytes = std::uint64_t{ 512 } << 10;
  const auto raw_h = static_cast<std::size_t>(
    std::min<std::uint64_t>(static_cast<std::uint64_t>(N) + W,
                            (tile_series_bytes * static_cast<std::uint64_t>(N)) / std::max<std::uint64_t>(1, series_bytes)));
  std::size_t h = std::max(16 * W, raw_h / W * W);
  layout.slab_bytes = std::numeric_limits<std::size_t>::max();
  if (distMat.is_mapped()) {
    const std::size_t budget = ram_limit != 0 ? ram_limit : core::available_memory(); // 0: unknown, no limit
    // What the series hold of the budget: a mapped view's values are the page cache's, its spans and names its own;
    // any other's values (owned, decoded, read into a buffer) are this process's.
    const std::size_t held = data.is_mapped() ? N * detail::view_series_bytes : series_bytes;
    layout.slab_bytes =
      budget == 0 ? std::size_t{ 1 } << 30 : std::min(budget > held ? (budget - held) / 8 : 0, std::size_t{ 1 } << 30);
    h = std::min(h, std::max(W, layout.slab_bytes / (N * layout.entry_bytes) / W * W)); // the last tile-row fits a slab
  }
  layout.save_rows = checkpoint_enabled ? (static_cast<std::size_t>(checkpoint_save_interval) + W - 1) / W * W : N;
  layout.h = std::min(std::min(h, layout.save_rows), N);

  layout.bounds = { 0 }; // tile-row k: rows [bounds[k], bounds[k + 1])
  for (std::size_t r = N % layout.h; r < N; r += layout.h)
    if (r > 0) layout.bounds.push_back(r);
  layout.bounds.push_back(N);
  layout.K = layout.bounds.size() - 1;
  layout.first_column.assign(layout.K + 1, 0); // tile-row k's columns: the tasks from first_column[k]
  for (std::size_t k = 0; k < layout.K; ++k) layout.first_column[k + 1] = layout.first_column[k] + layout.bounds[k + 1];

  layout.ragged = core::dtw_lanes_packed;
  layout.lanes = has_block_fn;
  layout.order.resize(N);
  std::iota(layout.order.begin(), layout.order.end(), std::size_t{ 0 });
  if (layout.ragged && layout.lanes)
    for (std::size_t k = 0; k < layout.K; ++k)
      std::stable_sort(layout.order.begin() + static_cast<std::ptrdiff_t>(layout.bounds[k]),
                       layout.order.begin() + static_cast<std::ptrdiff_t>(layout.bounds[k + 1]),
                       [&](std::size_t a, std::size_t b) { return data.series_flat_size(a) > data.series_flat_size(b); });
  layout.reordered = !std::is_sorted(layout.order.begin(), layout.order.end());

  layout.written = distMat.rows_written();
  layout.fresh = !sized || (distMat.is_mapped() && layout.written == 0);
  layout.use_slab_panels = layout.fresh && layout.ragged && layout.lanes && ram_limit == 0 && !data.is_mapped();

  layout.slabs = { 0 }; // slab s: tile-rows [slabs[s], slabs[s + 1])
  while (layout.slabs.back() < layout.K) {
    const std::size_t k0 = layout.slabs.back();
    std::size_t end = k0 + 1;
    while (end < layout.K && layout.bounds[end + 1] - layout.bounds[k0] <= layout.save_rows
           && (core::packed_size(layout.bounds[end + 1]) - core::packed_size(layout.bounds[k0])) * layout.entry_bytes
                <= layout.slab_bytes)
      ++end;
    layout.slabs.push_back(end);
  }
  return layout;
}

template <typename T, typename Column>
void pack_slab_panels_for(std::size_t s, const FillLayout &layout, const core::DistanceConfig &distance,
                          const std::vector<std::size_t> &tile_panel_offset, std::size_t total_panels,
                          std::vector<core::PackedLanesPanel<T>> &panels, const Column &column)
{
  constexpr std::size_t Wt = core::dtw_lanes<T>;
  panels.resize(total_panels);
  auto pack_one = [&](std::size_t idx) {
    std::size_t k = layout.slabs[s];
    while (k + 1 < layout.slabs[s + 1] && idx >= tile_panel_offset[k + 1]) ++k;
    const std::size_t p = idx - tile_panel_offset[k];
    const std::size_t R0 = layout.bounds[k], R1 = layout.bounds[k + 1];
    const std::size_t b0 = R0 + p * Wt;
    const std::size_t cnt = std::min(Wt, R1 - b0);
    std::array<std::span<const T>, Wt> ys;
    for (std::size_t w = 0; w < Wt; ++w) ys[w] = column(layout.order[b0 + std::min(w, cnt - 1)]);
    core::pack_dtw_panel<T>(distance, ys.data(), panels[idx]);
  };
  run_openmp(pack_one, total_panels, total_panels > 8);
}

bool build_slab_panels(std::size_t s, const FillLayout &layout, const Data &data, const core::DistanceConfig &distance,
                       std::vector<std::size_t> &tile_panel_offset, std::vector<core::PackedLanesPanel<data_t>> &panels_f64,
                       std::vector<core::PackedLanesPanel<float>> &panels_f32)
{
  if (!layout.use_slab_panels) return false;
  constexpr std::size_t kMaxSlabPanelBytes = std::size_t{ 64 } << 20;
  const std::size_t W = layout.width;
  const std::size_t val_bytes = layout.f32 ? sizeof(float) : sizeof(data_t);
  std::size_t total_panels = 0;
  std::size_t slab_panel_bytes = 0;
  bool fits = true;
  for (std::size_t k = layout.slabs[s]; k < layout.slabs[s + 1]; ++k) {
    tile_panel_offset[k] = total_panels;
    const std::size_t R0 = layout.bounds[k], R1 = layout.bounds[k + 1];
    const std::size_t k_panels = (R1 - R0 + W - 1) / W;
    total_panels += k_panels;
    for (std::size_t p = 0; p < k_panels && fits; ++p) {
      const std::size_t p_max_len = data.series_flat_size(layout.order[R0 + p * W]); // sorted longest-first
      const std::size_t row_bytes = W * val_bytes;
      if (p_max_len > (kMaxSlabPanelBytes - slab_panel_bytes) / row_bytes)
        fits = false;
      else
        slab_panel_bytes += p_max_len * row_bytes;
    }
  }
  if (!fits) {
    panels_f32.clear();
    panels_f64.clear();
    return false;
  }
  if (layout.f32)
    pack_slab_panels_for(s, layout, distance, tile_panel_offset, total_panels, panels_f32,
                         [&](std::size_t i) { return data.series_f32(i); });
  else
    pack_slab_panels_for(s, layout, distance, tile_panel_offset, total_panels, panels_f64,
                         [&](std::size_t i) { return data.series(i); });
  return true;
}

template <typename MatrixT, typename Column, typename BlockFn, typename PackedFn, typename PanelT, typename PairFn>
void fill_tile_column(MatrixT *matrix, std::size_t k, std::size_t R0, std::size_t R1, std::size_t c,
                      const FillLayout &layout, const core::DistanceConfig &distance,
                      const std::vector<std::size_t> &tile_panel_offset, const Column &column,
                      const BlockFn &block, const PackedFn &packed_block,
                      const core::PackedLanesPanel<PanelT> *slab_panels, const PairFn &pair_dist)
{
  using T = typename decltype(column(c))::value_type;
  constexpr std::size_t Wt = core::dtw_lanes<T>;
  const auto computed = [matrix](std::size_t i, std::size_t j) { return !std::isnan(matrix[core::tri_index(i, j)]); };
  const auto store = [matrix](std::size_t i, std::size_t j, double v) { matrix[core::tri_index(i, j)] = static_cast<MatrixT>(v); };
  const auto x = column(c);

  // Fast path for a fresh matrix with pre-packed tile-row panels: skips per-column Y transpose,
  // skips per-column DDTW/Interpolate preprocessing of y, and avoids reading uncomputed NaN slots.
  if (slab_panels && packed_block && (!layout.reordered || c < R0)) {
    if (c >= R0) store(c, c, 0.0);
    std::array<double, Wt> d;
    std::size_t start = (c < R0) ? R0 : (c + 1);
    const std::size_t rem = (start - R0) % Wt;
    if (rem != 0 && start < R1) {
      const std::size_t head_end = std::min(R1, start + (Wt - rem));
      const std::size_t head_cnt = head_end - start;
      std::array<std::span<const T>, Wt> ys;
      for (std::size_t w = 0; w < Wt; ++w) ys[w] = column(start + std::min(w, head_cnt - 1));
      block(x, ys, d);
      for (std::size_t w = 0; w < head_cnt; ++w) store(c, start + w, d[w]);
      start = head_end;
    }
    if (start < R1) {
      const auto x_prep = core::prepare_dtw_query<T>(distance, x);
      const auto *tile_panels = slab_panels + tile_panel_offset[k];
      for (std::size_t b0 = start; b0 < R1; b0 += Wt) {
        const std::size_t p = (b0 - R0) / Wt;
        const std::size_t cnt = std::min(Wt, R1 - b0);
        packed_block(x_prep, tile_panels[p].view(), d);
        for (std::size_t w = 0; w < cnt; ++w) store(c, layout.order[b0 + w], d[w]);
      }
    }
    return;
  }

  if (c >= R0 && (layout.fresh || !computed(c, c))) store(c, c, 0.0);
  if (block) {
    std::array<std::size_t, Wt> rows;
    std::array<std::span<const T>, Wt> ys;
    std::array<double, Wt> d;
    // In index order the rows paired with c start after it.
    for (std::size_t b = layout.reordered ? R0 : std::max(R0, c + 1); b < R1;) {
      std::size_t count = 0;
      for (; count < Wt && b < R1; ++b)
        if (layout.order[b] > c && (layout.fresh || !computed(c, layout.order[b]))) rows[count++] = layout.order[b];
      if (count == 0) break;
      bool equal = true;
      for (std::size_t w = 0; w < Wt; ++w) {
        ys[w] = column(rows[std::min(w, count - 1)]);
        equal = equal && ys[w].size() == x.size();
      }
      if (!(layout.ragged || equal)) continue;
      block(x, ys, d);
      for (std::size_t w = 0; w < count; ++w) store(c, rows[w], d[w]);
    }
  }
  if (!(layout.fresh && block && layout.ragged)) {
    for (std::size_t r = std::max(R0, c + 1); r < R1; ++r)
      if (!computed(c, r)) store(c, r, pair_dist(c, r));
  }
}

#ifdef DTWC_HAS_CUDA
struct HybridFillState
{
  std::size_t clean = 0;
  std::size_t calibrating = 0;
  double share = -1.0;
  double recorded = 0.0;
  core::FillSplit hybrid{};
};

struct SlabUnits
{
  std::size_t kw = 0;
  std::size_t ka = 0;
  std::size_t columns = 0;
  bool holds = false; // the slab's tile-rows hold both calibration units
};

inline std::size_t hybrid_entries(const FillLayout &layout, std::size_t ka, std::size_t kb)
{
  return core::packed_size(layout.bounds[kb]) - core::packed_size(layout.bounds[ka]);
}

inline SlabUnits slab_units(const FillLayout &layout, std::size_t clean, std::size_t k0, std::size_t k1)
{
  SlabUnits u{ k0, k0, 0, false };
  while (u.kw < k1 && core::packed_size(layout.bounds[u.kw]) < clean) ++u.kw;
  if (u.kw == k1) return u;
  u.ka = u.kw;
  while (u.ka + 1 < k1 && hybrid_entries(layout, u.kw, u.ka) < fill_calibration_pairs) ++u.ka;
  const std::size_t last_rows = layout.bounds[k1] - layout.bounds[k1 - 1];
  u.columns = (fill_calibration_pairs + last_rows - 1) / last_rows; // each holds last_rows pairs
  u.holds = hybrid_entries(layout, u.kw, u.ka) >= fill_calibration_pairs && u.columns <= layout.bounds[k1 - 1];
  return u;
}

template <typename MatrixT>
HybridFillState init_hybrid_fill_state(const FillLayout &layout, bool sized, bool is_mapped, bool has_gpu,
                                       bool gpu_fp32, double initial_share, double recorded_fp32, double recorded_fp64,
                                       const MatrixT *matrix)
{
  HybridFillState state;
  state.hybrid.entries = core::packed_size(layout.n);
  state.hybrid.gpu_fp32 = gpu_fp32;
  state.share = initial_share;
  state.hybrid.split.gpu_share = initial_share;
  state.recorded = gpu_fp32 ? recorded_fp32 : recorded_fp64;

  // The GPU writes whole rows, so it takes only tile-rows with no entry computed.
  if (has_gpu && sized) {
    if (is_mapped) {
      state.clean = core::packed_size(layout.written);
    } else {
      for (state.clean = core::packed_size(layout.n); state.clean > 0 && std::isnan(matrix[state.clean - 1]);)
        --state.clean;
    }
  }
  while (has_gpu && state.calibrating + 1 < layout.slabs.size()
         && !slab_units(layout, state.clean, layout.slabs[state.calibrating], layout.slabs[state.calibrating + 1]).holds)
    ++state.calibrating;
  if (has_gpu && state.calibrating + 1 == layout.slabs.size() && state.share < 0) {
    state.share = state.recorded;
    state.hybrid.split.gpu_share = state.share;
    state.hybrid.split.source = "recorded";
  }
  return state;
}

template <typename TaskFn>
void fill_slab_hybrid(std::size_t slab, cuda::DistanceFill &gpu, core::DistanceMatrix &distMat, const FillLayout &layout,
                      HybridFillState &state, const TaskFn &run_task)
{
  const std::size_t k0 = layout.slabs[slab], k1 = layout.slabs[slab + 1];
  const SlabUnits u = slab_units(layout, state.clean, k0, k1);
  const std::size_t kw = u.kw;
  std::size_t from = kw, kg = kw; // the GPU's tile-rows [kw, kg), of which the region fills [from, kg)
  bool calibrated_here = false;
  if (kw < k1) {
    if (slab == state.calibrating && state.share < 0) {
      using clock = std::chrono::steady_clock;
      const auto start = clock::now();
      gpu.rows(layout.bounds[kw], layout.bounds[u.ka], distMat);
      const auto gpu_done = clock::now();
      const std::size_t g0 = layout.first_column[k1 - 1];
      auto unit = [&](std::size_t i) { run_task(g0 + i); };
      run_openmp(unit, u.columns, true, 8);
      const std::chrono::duration<double> gpu_time = gpu_done - start, team_time = clock::now() - gpu_done;
      const int threads = std::max(1, get_max_threads());
      state.hybrid.split.gpu_rate =
        gpu_time.count() > 0.0
          ? static_cast<double>(hybrid_entries(layout, kw, u.ka) - (layout.bounds[u.ka] - layout.bounds[kw]))
              / gpu_time.count()
          : 0.0;
      if (threads <= 1) {
        state.hybrid.split.cpu_rate = 0.0;
        state.share = 1.0;
      } else {
        state.hybrid.split.cpu_rate =
          team_time.count() > 0.0
            ? static_cast<double>(u.columns * (layout.bounds[k1] - layout.bounds[k1 - 1])) / team_time.count()
                * (threads - 1) / threads
            : 0.0;
        const double denom = state.hybrid.split.gpu_rate + state.hybrid.split.cpu_rate;
        state.share = denom > 0.0 ? (state.hybrid.split.gpu_rate / denom) : state.recorded;
      }
      state.hybrid.split.source = "calibrated";
      state.hybrid.split.gpu_share = state.share;
      from = u.ka;
      calibrated_here = true;
    }
    const bool early = state.calibrating + 1 < layout.slabs.size() && slab < state.calibrating;
    const double s_here = early ? state.recorded : state.share;
    if (early) state.hybrid.early_share = state.recorded;
    const auto cost = [&](std::size_t k) {
      const auto on_gpu = static_cast<double>(hybrid_entries(layout, kw, k));
      const auto on_cpu = static_cast<double>(hybrid_entries(layout, k, k1));
      return std::max(on_gpu == 0 ? 0.0 : on_gpu / s_here, on_cpu == 0 ? 0.0 : on_cpu / (1.0 - s_here));
    };
    kg = slab == state.calibrating && s_here > 0 ? u.ka : kw;
    for (std::size_t k = kg + 1; k <= k1; ++k)
      if (cost(k) < cost(kg)) kg = k;
    state.hybrid.gpu_entries += hybrid_entries(layout, kw, kg);
  }

  const std::size_t a0 = layout.first_column[k0], a1 = layout.first_column[kw];
  const std::size_t b0 = layout.first_column[kg], b1 = layout.first_column[k1];
  const bool skip_cal_cols = calibrated_here && kg < k1 && u.columns > 0;
  const std::size_t cal_b = skip_cal_cols ? layout.first_column[k1 - 1] : b1;
  const std::size_t cal_e = skip_cal_cols ? (layout.first_column[k1 - 1] + u.columns) : b1;
  const std::size_t a_len = a1 - a0, b_len = cal_b - b0, c_len = b1 - cal_e;
  const std::size_t cpu_tasks = a_len + b_len + c_len;
  const std::size_t none = layout.first_column[layout.K];
  using FailureSlots = detail::FailureSlots<std::size_t>;
  FailureSlots failures(static_cast<std::size_t>(std::max(1, get_max_threads())), none);
  const auto gpu_rows = [&](FailureSlots::Slot &slot) {
    if (kg <= from) return;
    try {
      gpu.rows(layout.bounds[from], layout.bounds[kg], distMat);
    } catch (...) {
      FailureSlots::record(slot, layout.first_column[from], std::current_exception());
    }
  };
  const auto cpu_task = [&](FailureSlots::Slot &slot, std::size_t i) {
    const std::size_t g = i < a_len ? a0 + i : (i < a_len + b_len ? b0 + (i - a_len) : cal_e + (i - a_len - b_len));
    if (g > slot.failed_at) return;
    try {
      run_task(g);
    } catch (...) {
      FailureSlots::record(slot, g, std::current_exception());
    }
  };
#  ifdef _OPENMP
  const auto end = static_cast<index_t>(cpu_tasks);
  const int threads = static_cast<int>(failures.size());
  const index_t chunk = omp_chunk_size_for(end, 8, threads);
#    pragma omp parallel num_threads(threads)
  {
    auto &slot = failures[static_cast<std::size_t>(omp_get_thread_num())];
    if (omp_get_thread_num() == 0) gpu_rows(slot);
#    pragma omp for schedule(dynamic, chunk) nowait
    for (index_t i = 0; i < end; ++i) cpu_task(slot, static_cast<std::size_t>(i));
  }
#  else
  gpu_rows(failures[0]);
  for (std::size_t i = 0; i < cpu_tasks; ++i) cpu_task(failures[0], i);
#  endif
  failures.rethrow_lowest();
}
#endif

} // namespace

void Problem::fillDistanceMatrix_BruteForce()
{
  const size_t N = data_.size();
  const bool f32 = data_.is_f32();
  const bool sized = distMat.size() == N; // sized before this fill: it may hold computed entries
#ifdef DTWC_HAS_CUDA
  // cpu+gpu: the GPU's session before the matrix is touched, so its refusals come first: the route on the longest
  // series, every series padded by the team and uploaded once (cuda::DistanceFill). The slab loop fills its rows.
  std::optional<cuda::DistanceFill> gpu;
  if (device_ == Device::CPU_GPU) {
    cuda::CUDADistMatOptions opts;
    opts.device_id = device_index_;
    opts.band = distance_.band;
    opts.precision = resolve_gpu_precision(gpu_precision_);
    opts.use_squared_l2 = distance_.metric == core::MetricType::SquaredL2; // L2 is L1 on univariate series
    gpu.emplace(
      N, [this](std::size_t i) { return data_.series(i); }, opts,
      [](std::size_t count, const std::function<void(std::size_t)> &pad) { run_openmp(pad, count, count > 64); },
      distMat);
  }
#endif

  // resize() re-fills every packed slot with NaN, so it must NOT run when the
  // matrix is already the right size (a mapped one always is): an unconditional
  // resize discarded a restored checkpoint and recomputed every pair.
  if (distMat.size() != N) distMat.resize(N);
  if (N == 0) return;

  const bool has_block_fn = f32 ? static_cast<bool>(dtw_block_fn_f32_) : static_cast<bool>(dtw_block_fn_);
  const FillLayout layout =
    build_fill_layout(data_, distMat, ram_limit_, checkpoint.enabled, checkpoint.save_interval, sized, has_block_fn);
  const auto packed_fn_f64 = core::resolve_dtw_packed_block_fn<data_t>(distance_);
  const auto packed_fn_f32 = f32 ? core::resolve_dtw_packed_block_fn<float>(distance_)
                                 : decltype(core::resolve_dtw_packed_block_fn<float>(distance_)){};
  bool slab_panels_ready = false;
  std::vector<size_t> tile_panel_offset(layout.K + 1, 0);
  std::vector<core::PackedLanesPanel<data_t>> panels_f64;
  std::vector<core::PackedLanesPanel<float>> panels_f32;

  const auto pair_dist = [this](std::size_t ci, std::size_t ri) { return pair_distance(ci, ri); };
  auto task = [&](auto *matrix, std::size_t g) {
    const std::size_t k = static_cast<std::size_t>(std::upper_bound(layout.first_column.begin(),
                                                                    layout.first_column.end(), g)
                                                   - layout.first_column.begin())
                          - 1;
    const std::size_t c = g - layout.first_column[k];
    if (f32)
      fill_tile_column(matrix, k, layout.bounds[k], layout.bounds[k + 1], c, layout, distance_, tile_panel_offset,
                       [&](std::size_t i) { return data_.series_f32(i); }, dtw_block_fn_f32_, packed_fn_f32,
                       slab_panels_ready ? panels_f32.data() : nullptr, pair_dist);
    else // Data's accessor, as above: GCC's inline-unit-growth cap leaves the Problem::series wrapper out of line here
      fill_tile_column(matrix, k, layout.bounds[k], layout.bounds[k + 1], c, layout, distance_, tile_panel_offset,
                       [&](std::size_t i) { return data_.series(i); }, dtw_block_fn_, packed_fn_f64,
                       slab_panels_ready ? panels_f64.data() : nullptr, pair_dist);
  };

  // Hashed once: every save of this fill writes the same identity.
  const auto identity = checkpoint.enabled ? distance_checkpoint_identity() : core::DistanceMatrix::fingerprint_type{};
  const path_t file = checkpoint.enabled ? checkpoint_path(*this, checkpoint.directory) : path_t{};
  std::size_t waiting = 0; // the first row of the slabs whose write-back no wait has covered
#ifdef DTWC_HAS_CUDA
  HybridFillState hybrid_state;
#endif
  distMat.visit([&](auto *matrix) {
#ifdef DTWC_HAS_CUDA
    hybrid_state = init_hybrid_fill_state(layout, sized, distMat.is_mapped(), static_cast<bool>(gpu), gpu && gpu->fp32(),
                                          gpu_share_, recorded_fill_gpu_share_fp32, recorded_fill_gpu_share, matrix);
#endif
    distMat.prefetch_rows(0, layout.bounds[layout.slabs[1]]);
    for (std::size_t s = 0; s + 1 < layout.slabs.size(); ++s) {
      const std::size_t r0 = layout.bounds[layout.slabs[s]], r1 = layout.bounds[layout.slabs[s + 1]];
      if (s + 2 < layout.slabs.size()) distMat.prefetch_rows(r1, layout.bounds[layout.slabs[s + 2]]);
      distMat.clear_rows(std::max(r0, layout.written), r1);
      slab_panels_ready = build_slab_panels(s, layout, data_, distance_, tile_panel_offset, panels_f64, panels_f32);

      const std::size_t first = layout.first_column[layout.slabs[s]];
      auto slab_task = [&](std::size_t i) { task(matrix, first + i); };
#ifdef DTWC_HAS_CUDA
      if (gpu)
        fill_slab_hybrid(s, *gpu, distMat, layout, hybrid_state, [&](std::size_t g) { task(matrix, g); });
      else
#endif
        run_openmp(slab_task, layout.first_column[layout.slabs[s + 1]] - first, true, 8);
      refuse_float32_overflow(r0, r1);
      if (checkpoint.enabled)
        distMat.write(file, identity, r0, r1);
      else {
        distMat.flush_rows(r0, r1, false);
        if (layout.written < N) distMat.flush_rows(waiting, r0, true);
      }
      waiting = r0;
    }
  });
  if (layout.written < N) // every row on the device: the file records them all written
    distMat.flush_rows(distMat.rows_written(), N, true);
#ifdef DTWC_HAS_CUDA
  if (gpu) fill_split_ = hybrid_state.hybrid;
#endif
}

void Problem::refuse_float32_overflow(size_t r0, size_t r1)
{
  if (distMat.width() != core::Precision::Float32) return;
  const auto [i, j] = distMat.first_infinite(r0, r1);
  if (i == distMat.size()) return;
  distMat.clear_rows(r0, r1); // no file keeps these rows, nor a later fill a distance a float could not hold
  const double distance = data_.is_f32() ? dtw_fn_f32_(data_.series_f32(j), data_.series_f32(i))
                                         : dtw_fn_(series(j), series(i));
  core::refuse_float32_range("Problem::fill_distance_matrix", i, j, distance);
}

/**
 * @brief Fills the distance matrix by computing distances between all pairs of points.
 * @details The CPU runs the parallel exact fill (all variants); set_device(gpu)
 *          selects this build's GPU backend.
 */
void Problem::fill_distance_matrix()
{
  if (gpu_share_ != -1.0 && device_ != Device::CPU_GPU)
    throw InvalidInput(detail::share_ignored_message("Problem::fill_distance_matrix", "the fill", device_));
  // A fill that will compute resolves its GPU request first, before the checkpoint's precision rule on CUDA and every
  // series check (a filled matrix needs no device, and says nothing).
  std::optional<detail::DeviceScope> device_scope;
  if (!filled_) device_scope.emplace(*this, "Problem::fill_distance_matrix");
  validate_checkpoint_settings();
  // Writes through writable_distance_matrix() commit here: one scan refuses ±inf and a
  // negative distance, which no variant gives, and a matrix they made complete needs no pair.
  if (written_ && !filled_ && data_.size() > 0 && distMat.size() == data_.size()) {
    filled_ = distMat.all_computed("Problem::fill_distance_matrix",
                                   "Leave a pair still to compute NaN, as resize() leaves every pair.");
    written_ = false;
  }
  if (filled_) return;
  fill_split_.reset(); // set again by a cpu+gpu fill; a refused fill leaves none
  validate_fill_request("Problem::fill_distance_matrix");
  fill_self_terms(); // Soft-DTW's, which the CPU's pairs read (a GPU refused Soft-DTW in the check)

  // Each fill below sizes the matrix (deferred from set_data /
  // refresh_distance_matrix) only after its own refusals; a mapped matrix is
  // sized when it is bound.

  if (verbose_)
    std::cout << "Distance matrix is being filled!" << '\n';

  // The missing-data pre-scan is part of validate_fill_request, above: it
  // runs before any pair, here and on every other entry point.

  if (device_ == Device::CPU || device_ == Device::CPU_GPU) {
    fillDistanceMatrix_BruteForce(); // saves a checkpoint after each row block; cpu+gpu's GPU fills some tile-rows
    if (verbose_ && fill_split_) std::cout << "Distance matrix " << describe(*fill_split_) << '\n';
  } else {
    // Device::GPU is this build's backend: CUDA, else Metal. It writes every
    // entry of distMat in place, and never hands a pair to the CPU.
#if defined(DTWC_HAS_CUDA)
    cuda::CUDADistMatOptions opts;
    opts.device_id = device_index_;
#elif defined(DTWC_HAS_METAL)
    metal::MetalDistMatOptions opts; // the system default GPU: set_device refused any other
#else
    throw detail::gpu_not_built_refusal(); // set_device refused it first, or fallback made the CPU's device
#endif
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
    opts.band = distance_.band;
    opts.precision = resolve_gpu_precision(gpu_precision_);
    // L2 is L1 on the univariate series the GPU routes take.
    opts.use_squared_l2 = distance_.metric == core::MetricType::SquaredL2;
    opts.verbose = verbose_;
    const auto parallel = [](std::size_t count, const std::function<void(std::size_t)> &task) {
      run_openmp(task, count, count > 64);
    };
#  if defined(DTWC_HAS_CUDA)
    // CUDA reads the Float64 series (checked above) where they lie, a view's (a mapped Arrow IPC file's, a decoded
    // record batch's) or owned: this thread's team pads them into its staging, each its own rows; no copy is made.
    (void)cuda::compute_distance_matrix_cuda(
      static_cast<std::size_t>(data_.size()), [this](std::size_t i) { return data_.series(i); }, opts, parallel,
      distMat);
#  else
    // Metal reads Float64 series directly via SeriesAt spans and converts/uploads in parallel, avoiding an owned copy.
    (void)metal::compute_distance_matrix_metal(
      static_cast<std::size_t>(data_.size()), [this](std::size_t i) { return data_.series(i); }, opts, parallel,
      distMat);
#  endif
    refuse_float32_overflow(0, distMat.size());
    if (distMat.rows_written() < distMat.size()) // a mapped file's rows are on the device, and it records them
      distMat.flush_rows(0, distMat.size(), true);
    // One call fills every pair, so the only automatic save is here.
    if (checkpoint.enabled) save_checkpoint(*this, checkpoint.directory);
#endif
  }
  filled_ = data_.size() > 0; // an empty Problem has no matrix to call filled

  if (verbose_)
    std::cout << "Distance matrix has been filled!" << '\n';
}
core::ClusteringResult Problem::cluster()
{
  require_clusterable(Nc, size());
  if (gpu_share_ != -1.0 && device_ != Device::CPU_GPU)
    throw InvalidInput(detail::share_ignored_message("Problem::cluster", "the fill and FastCLARA's pass", device_));
  // The GPU request is settled here, before the method's own work: a method that computes on the CPU as it goes
  // (onebatch, tadpole) has no part for a GPU, and TADPole's automatic cutoff would compute its sample before the method
  // refused. `auto` is pam on a GPU and by N on the CPU, so it is resolved again where the request fell back.
  Method method = resolve_method(method_, device_, size());
  const detail::DeviceScope device_scope(*this, "Problem::cluster", method);
  if (device_scope.fell_back()) method = resolve_method(method_, device_, size());
  switch (method) {
  case Method::PAM: {
    // Restart r starts from seed + r; the strictly lowest cost is kept, so a tie
    // keeps the earlier restart.
    auto best = fast_pam(*this, Nc, max_iter_, random_seed_);
    for (int restart = 1; restart < n_repetitions_; ++restart) {
      auto candidate = fast_pam(*this, Nc, max_iter_, random_seed_ + static_cast<std::uint64_t>(restart));
      if (candidate.total_cost < best.total_cost) best = std::move(candidate);
    }
    set_result(best); // each restart published its own
    return best;
  }
  case Method::OneBatch:
    return algorithms::one_batch_pam(
      *this, { .n_clusters = Nc, .batch_size = batch_size_, .max_iter = max_iter_, .random_seed = random_seed_ });
  case Method::CLARA:
    return algorithms::fast_clara(*this, { .n_clusters = Nc, .sample_size = sample_size_, .n_samples = n_samples_,
                                           .max_iter = max_iter_, .random_seed = random_seed_, .gpu_share = gpu_share_ });
  case Method::Hierarchical:
    return algorithms::cut_dendrogram(algorithms::build_dendrogram(*this, { .linkage = linkage_ }), *this, Nc);
  case Method::Kmedoids: cluster_by_kmedoids_lloyd(); break;
  case Method::MIP: cluster_by_mip(); break;
  case Method::LRCore: LR_core_clustering(*this); break;
  case Method::TADPole:
    return algorithms::tadpole(*this, Nc, tadpole_dc_ > 0.0 ? tadpole_dc_ : algorithms::tadpole_auto_dc(*this));
  case Method::Auto: break; // resolved above
  }
  core::ClusteringResult result;
  result.labels = clusters_ind;
  result.medoid_indices = centroids_ind;
  result.total_cost = find_total_cost();
  // Lloyd counts its iterations; the exact methods and TADPole run to the end.
  result.iterations = method == Method::Kmedoids ? last_iterations_ : 0;
  result.converged = method != Method::Kmedoids || last_converged_;
  return result;
}

void Problem::cluster_by_mip()
{
  // A negative `mip_gap` would otherwise reach HiGHS as an out-of-domain option
  // value, reported as a solver failure rather than as bad input.
  validate_mip_settings(mip_settings);

  // The compact model on the selected solver at every N, as in v1.0.0; large N
  // belongs to Method::LRCore, never to a silent reroute.
  switch (mipSolver) {
  case Solver::Gurobi:
    MIP_clustering_byGurobi(*this);
    break;
  case Solver::HiGHS:
    MIP_clustering_byHiGHS(*this);
    break;
  }
}

void Problem::assign_clusters()
{
  fill_distance_matrix(); // every point against every medoid: a read each
  std::vector<index_t> labels(data_.size());
  distMat.visit([&](const auto *matrix) {
    auto assignClustersTask = [this, &labels, matrix](size_t i_p) //!< i_p and i_c in [0, Np)
    {
      const auto ip = static_cast<index_t>(i_p);
      double best_distance = std::numeric_limits<double>::max();
      index_t best_slot = 0;
      bool has_best = false;
      for (std::size_t slot = 0; slot < centroids_ind.size(); ++slot) {
        const index_t medoid = centroids_ind[slot];
        // Not the matrix's intake check: a fill of finite series can overflow
        // (values near DBL_MAX give +inf), and Lloyd refuses that before it
        // publishes labels.
        const double distance = core::detail::require_finite_medoid_distance(
          static_cast<double>(matrix[core::tri_index(static_cast<std::size_t>(ip), static_cast<std::size_t>(medoid))]),
          "kmedoids_lloyd", i_p, static_cast<index_t>(slot), medoid);
        // A medoid tied with another medoid (a duplicate series) serves itself,
        // or its own cluster would be published empty.
        if (!has_best || distance < best_distance || (distance == best_distance && medoid == ip)) {
          best_distance = distance;
          best_slot = static_cast<index_t>(slot);
          has_best = true;
        }
      }
      labels[i_p] = best_slot;
    };
    if (data_.size() * centroids_ind.size() >= 32768)
      run(assignClustersTask, data_.size()); // read-only lookups, one label slot per point
    else
      for (std::size_t i_p = 0; i_p < data_.size(); ++i_p) assignClustersTask(i_p);
  });
  clusters_ind = std::move(labels);
}

void Problem::calculate_medoids()
{
  require_clustered("calculate_medoids");
  fill_distance_matrix(); // every pair within a cluster
  const auto n_series = static_cast<std::size_t>(size());
  const auto k = static_cast<std::size_t>(n_clusters());
  std::vector<std::vector<std::size_t>> members(k);
  for (std::size_t i = 0; i < n_series; ++i)
    members[static_cast<std::size_t>(clusters_ind[i])].push_back(i);

  std::vector<double> pointCosts(n_series);
  distMat.visit([&](const auto *matrix) {
    auto findBetterMedoidTask = [&](size_t i_p) // i_p is point index.
    {
      // Keep sequential accumulation order across compilers so equal-cost medoid ties are deterministic.
      const auto *const *const values = &matrix;
      const auto &cluster_members = members[static_cast<std::size_t>(clusters_ind[i_p])];
      double sum{ 0 };
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
      for (const std::size_t i : cluster_members)
        sum += static_cast<double>((*values)[core::tri_index(i_p, i)]);

      pointCosts[i_p] = sum;
    };
    const std::size_t work = (n_series * n_series) / std::max<std::size_t>(k, 1);
    if (work >= 32768)
      run(findBetterMedoidTask, n_series);
    else
      for (std::size_t i_p = 0; i_p < n_series; ++i_p) findBetterMedoidTask(i_p);
  });

  std::vector<double> clusterCosts(n_clusters(), std::numeric_limits<double>::max());
  for (std::size_t i = 0; i < pointCosts.size(); ++i)
    if (pointCosts[i] < clusterCosts[clusters_ind[i]]) {
      clusterCosts[clusters_ind[i]] = pointCosts[i];
      centroids_ind[clusters_ind[i]] = static_cast<index_t>(i);
    }
}

/**
 * @brief Performs the clustering using the Lloyd k-medoids algorithm.
 * @details Executes the Lloyd k-medoids algorithm (alternating assign + update medoids within clusters)
 * with multiple repetitions, each time initializing medoids randomly.
 * The repetition yielding the lowest total cost is chosen as the best solution.
 */
void Problem::cluster_by_kmedoids_lloyd()
{
  require_clusterable(Nc, size());
  const int repetitions = n_repetitions();

  fill_distance_matrix(); // Ensure all distances computed before parallel clustering.

  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();
  int best_iterations = 0;
  std::vector<index_t> best_medoids;
  std::vector<index_t> best_labels;

  for (int i_rand = 0; i_rand < repetitions; i_rand++) {
    if (verbose_) std::cout << "Medoid initialisation is started.\n";
    // Unsigned: a seed near 2^64 wraps to another valid seed.
    init_fun(*this, random_seed_ + static_cast<std::uint64_t>(i_rand));

    if (verbose_)
      std::cout << "Medoid initialisation is finished. "
                << Nc << " medoids are initialised.\n"
                << "Start clustering:\n";

    // A caller's init_fun may leave the medoids unset; it is then called a second time.
    if (centroids_ind.empty())
      init_fun(*this, random_seed_ + static_cast<std::uint64_t>(i_rand));

    auto oldmedoids = centroids_ind;
    bool converged = false;
    int actual_iters = 0;
    const int iteration_limit = max_iter();
    for (int i = 0; i < iteration_limit; i++) {
      actual_iters = i + 1;
      if (verbose_) {
        std::cout << "Medoids: ";
        for (auto medoid : centroids_ind)
          std::cout << series_name(medoid) << ' ';
      }

      assign_clusters();
      if (verbose_) {
        const auto prev_prec = std::cout.precision();
        std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
                  << find_total_cost() << ".\n";
        std::cout.precision(prev_prec);
        print_clusters();
      }
      calculate_medoids(); // Changes centroids_ind
      if (oldmedoids == centroids_ind) {
        converged = true;
        break;
      }
      oldmedoids = centroids_ind;
    }

    // A converged iteration leaves the medoids unchanged, so its assignment is
    // already current. A capped iteration may have just updated the medoids;
    // realign once before costing and keeping this repetition's state.
    if (!converged)
      assign_clusters();

    const double total_cost = find_total_cost();
    if (verbose_)
      std::cout << "Procedure is completed with cost: " << total_cost << '\n';

    last_iterations_ = actual_iters;
    if (verbose_) {
      std::cout << (converged ? "Medoids are same for last two iterations, algorithm is converged!\n"
                              : "Maximum iteration is reached before medoids are converged!\n");
    }

    if (i_rand == 0 || total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
      best_iterations = actual_iters;
      last_converged_ = converged;
      best_medoids = centroids_ind;
      best_labels = clusters_ind;
    }
    if (verbose_)
      std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';
  }

  centroids_ind = std::move(best_medoids);
  clusters_ind = std::move(best_labels);
  last_iterations_ = best_iterations;
  if (verbose_)
    std::cout << "Best repetition: " << best_rep << '\n';
}

double Problem::find_total_cost()
{
  require_clustered("find_total_cost");
  // A matrix-free clustering (TADPole) leaves the matrix unfilled: the N
  // point-to-medoid distances are then computed by pair_distance, which
  // is what a fill would store, instead of filling N(N-1)/2 pairs.
  std::vector<double> distances(size());
  if (filled_) {
    for (std::size_t i = 0; i < size(); ++i)
      distances[i] = dist_by_ind(static_cast<index_t>(i), centroid_of(static_cast<index_t>(i)));
  } else {
    validate_fill_request("Problem::find_total_cost");
    fill_self_terms();
    auto point_distance = [&](std::size_t i) {
      const auto medoid = static_cast<std::size_t>(centroid_of(static_cast<index_t>(i)));
      distances[i] = i == medoid ? 0.0 : pair_distance(i, medoid);
    };
    run_openmp(point_distance, size());
  }
  // k-medoids objective: sum of raw DTW distances (not squared, unlike k-means).
  return core::detail::ordered_medoid_objective(distances, "kmedoids_lloyd");
}

} // namespace dtwc
