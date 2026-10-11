/**
 * @file Problem.cpp
 * @brief Implementation of the DTWC (Dynamic Time Warping Clustering) problem encapsulated in a class.
 *
 * @details This file includes the implementation of the Problem class, which contains methods for clustering,
 * initializing clusters, calculating distances, and other functionalities related to the DTWC problem.
 *
 * @date 06 Nov 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#include "Problem.hpp"
#include "config.hpp"               // for Config, apply
#include "base/error.hpp"           // for DeviceError
#include "mip.hpp"             // for MIP_clustering_byGurobi, MIP_clustering_byHiGHS
#include "base/parallelisation.hpp" // for run
#include "scores.hpp"          // for silhouette
#include "base/settings.hpp"        // for data_t, randGenerator, band, isDebug
#include "core/matrix_io.hpp"  // for operator<<(ostream, DistanceMatrix)
#include "core/medoid_assignment_policy.hpp" // finite assignment contract

#ifdef DTWC_HAS_CUDA
#include "cuda/cuda_dtw.cuh"   // GPU distance matrix computation
#endif
#ifdef DTWC_HAS_METAL
#include "metal/metal_dtw.hpp" // Apple Metal GPU distance matrix computation
#endif
#include "warping.hpp"         // for detail::require_finite
#include "core/dtw_kernel.hpp" // for dtw_lanes
#include "warping_wdtw.hpp"    // for wdtw_weights (cache population)
#include "types/Range.hpp"     // for Range
#include "initialisation.hpp"  // For initialisation functions
#include "core/dtw_dispatch.hpp"           // for resolve_dtw_fn
#include "core/sha256.hpp"                 // for persistent cache fingerprints
#include "base/missing_utils.hpp"               // for all_missing
#include "algorithms/fast_clara.hpp"       // the methods cluster() dispatches
#include "algorithms/fast_pam.hpp"
#include "algorithms/one_batch_pam.hpp"
#include "algorithms/tadpole.hpp"
#include "io/parse_number.hpp"         // for parse_ram_limit


#include <algorithm> // for min
#include <array>     // for array
#include <cctype>    // for toupper
#include <charconv>  // for from_chars
#include <cmath>     // for ldexp, ceil
#include <cstdint>   // for uint32_t, uint64_t
#include <cstring>   // for memcpy
#include <iomanip>   // for operator<<, setprecision
#include <iostream>  // for cout
#include <limits>    // for numeric_limits
#include <numeric>   // for iota
#include <cassert>
#include <string>    // for allocator, char_traits, operator+
#include <type_traits> // for underlying_type_t
#include <utility>   // for pair
#include <vector>    // for vector, operator==

namespace dtwc {

namespace {

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

// Enum spellings for validate_fill_request's messages: the C++ and Python
// enumerator names.
const char *variant_name(core::DTWVariant v)
{
  switch (v) {
  case core::DTWVariant::Standard: return "Standard";
  case core::DTWVariant::DDTW: return "DDTW";
  case core::DTWVariant::WDTW: return "WDTW";
  case core::DTWVariant::ADTW: return "ADTW";
  case core::DTWVariant::SoftDTW: return "SoftDTW";
  case core::DTWVariant::MSM: return "MSM";
  case core::DTWVariant::TWE: return "TWE";
  }
  return "?";
}

const char *missing_strategy_name(core::MissingStrategy m)
{
  switch (m) {
  case core::MissingStrategy::Error: return "Error";
  case core::MissingStrategy::ZeroCost: return "ZeroCost";
  case core::MissingStrategy::AROW: return "AROW";
  case core::MissingStrategy::Interpolate: return "Interpolate";
  }
  return "?";
}

/// The backend Device::GPU resolves to: CUDA if built, else Metal (a build
/// with neither refuses Device::GPU in set_device).
#ifdef DTWC_HAS_CUDA
constexpr std::string_view gpu_backend = "CUDA";
#else
constexpr std::string_view gpu_backend = "Metal";
#endif

/// What Device::GPU computes in for `precision`: Metal has no FP64, so its Auto
/// is FP32; CUDA's Auto stays open until a fill meets a GPU (FP64 where it is fast).
constexpr GpuPrecision resolve_gpu_precision(GpuPrecision precision)
{
  return gpu_backend == "Metal" && precision == GpuPrecision::Auto ? GpuPrecision::FP32 : precision;
}

/// A GPU request its backend cannot honour: a DeviceError, never a CPU fallback.
[[noreturn]] void reject_gpu_request(std::string_view where, const std::string &request, const std::string &fix)
{
  throw DeviceError(std::string(where) + ": " + std::string(gpu_backend) + " " + request
                    + "; no backend call or CPU fallback was attempted. " + fix);
}

} // namespace

void validate_gpu_request(std::string_view where, const Problem &prob, core::Precision series_precision)
{
  // The GPU routes upload owned Float64 series (data_.p_vec) and run Standard
  // DTW with no missing-data strategy, in L1 or squared L2.
  if (prob.device().first != Device::GPU) return;
  if (series_precision == core::Precision::Float32)
    reject_gpu_request(where, "computes from Float64 series, but precision = Float32 was requested",
                       "Load the series as Float64, or use device cpu.");
  const auto variant = prob.variant_params().variant;
  if (variant != core::DTWVariant::Standard)
    reject_gpu_request(where,
                       std::string("implements Standard DTW only, but variant = ") + variant_name(variant)
                         + " was requested",
                       "Use device cpu for this variant.");
  if (prob.missing_strategy() != core::MissingStrategy::Error)
    reject_gpu_request(where,
                       std::string("has no missing-data strategy, but missing_strategy = ")
                         + missing_strategy_name(prob.missing_strategy()) + " was requested",
                       "Use device cpu, or missing_strategy Error for data without NaN.");
#if !defined(DTWC_HAS_CUDA) && defined(DTWC_HAS_METAL)
  metal::validate_metal_precision(prob.gpu_precision());
#endif
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
  constexpr std::size_t pam_max_series = 5000; // on the CPU; the GPU fills PAM's matrix at any N
  if (method != Method::Auto) return method;
  return device == Device::GPU || n_series <= pam_max_series ? Method::PAM : Method::CLARA;
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
  if (ec != std::errc{} || !(scaled >= 0.0 && scaled < std::ldexp(1.0, std::numeric_limits<std::size_t>::digits)))
    throw InvalidInput("Invalid " + std::string(key) + " '" + text
                       + "': expected a size such as 2G, 500M or 1.5GiB, in B, K, M, G or T (KB and KiB alike), "
                         "below 2^64 bytes; 0 is automatic.");
  return static_cast<std::size_t>(std::ceil(scaled));
}

void apply(const Config &config, Problem &prob)
{
  // Each setter checks its value: parameter domains, variant x missing
  // strategy x metric, the cluster count and the iteration limits.
  prob.set_variant(config.variant);
  prob.set_missing_strategy(config.missing);
  prob.set_metric(config.metric);
  prob.set_band(config.band);
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
  prob.set_matrix_dtype(config.matrix_dtype);
  prob.set_device(config.device, config.device_index); // gpu without a GPU backend: the DeviceError
  if (config.device == Device::GPU && (config.method == Method::OneBatch || config.method == Method::TADPole))
    throw DeviceError(
      "run: method '" + std::string(name_of(method_names, config.method))
      + "' computes its distances on the CPU as it goes, so device 'gpu' would sit idle; the GPU fills the "
        "distance matrix that pam, kmedoids, mip, lrcore and hierarchical use, and clara's sample matrices. "
        "Choose one of those, or device 'cpu'. No CPU fallback was attempted.");
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
  maxIter = n;
}

int Problem::max_iter() const
{
  return maxIter;
}

void Problem::set_n_repetitions(int n)
{
  if (n < 1)
    throw InvalidInput(
      "Problem::set_n_repetitions: n_repetitions must be at least 1; got "
      + std::to_string(n) + ". It counts the random restarts, and one run is the "
        "minimum.");
  N_repetition = n;
}

int Problem::n_repetitions() const
{
  return N_repetition;
}

/**
 * @brief Sets the initial centroids for clustering.
 *
 * @param candidate_centroids A vector containing the indices of candidate centroids.
 * @throws InvalidInput if the size of candidate_centroids is not equal to Nc.
 */
void Problem::set_clusters(const std::vector<index_t> &candidate_centroids)
{
  if (candidate_centroids.size() != static_cast<size_t>(Nc))
    throw InvalidInput("Set cluster has failed as number of centroids is not same as the number of indices in candidate centroids vector.\n");

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

/**
 * @brief Sets the solver to be used for clustering.
 *
 * @param solver_ The solver to use.
 * @return True if the solver is set successfully,
 * False otherwise (e.g., if Gurobi is not available and a default solver is used instead).
 */
bool Problem::set_solver(Solver solver_)
{
  if (solver_ == Solver::Gurobi) {
#ifdef DTWC_ENABLE_GUROBI
    mipSolver = Solver::Gurobi;
    return true;
#else
    mipSolver = settings::DEFAULT_MIP_SOLVER;
    return false;
#endif
  }

  mipSolver = solver_;
  return true;
}

/**
 * @brief Prints the current distance matrix to the standard output.
 * @details Outputs the distance matrix in a human-readable format, useful for debugging and verification.
 */
void Problem::print_distance_matrix() const
{
  std::cout << distMat;
}

/**
 * @brief Refreshes the distance matrix.
 * @details Drops the matrix and binds the distance functions again, a direct
 * write to `band` included. Does NOT allocate the dense N×N matrix — that is
 * deferred to fill_distance_matrix() so that large-N algorithms (e.g.
 * FastCLARA) can load data without forcing quadratic memory usage. A mapped
 * matrix is detached and its file left as it is, so rebinding it under other
 * settings fails its fingerprint check loudly.
 */
void Problem::refresh_distance_matrix()
{
  if (band != distance_.band) {
    set_band(band); // drops the matrix too
    return;
  }
  distMat = core::DistanceMatrix(0, matrix_dtype_); // empty, of the width every resize keeps
  filled_ = false;
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
  dtw_fn_ = core::resolve_dtw_fn<data_t>(distance_, data_);
  dtw_block_fn_ = core::resolve_dtw_block_fn<data_t>(distance_);
  dtw_fn_f32_ = {};
  dtw_block_fn_f32_ = {};
  if (data_.is_f32()) {
    dtw_fn_f32_ = core::resolve_dtw_fn<float>(distance_, data_);
    dtw_block_fn_f32_ = core::resolve_dtw_block_fn<float>(distance_);
  }
}

void Problem::sync_band()
{
  if (band != distance_.band) set_band(band);
}

void Problem::set_distance(core::DistanceConfig config)
{
  if (config.band < -1)
    throw InvalidInput("Problem::set_distance: band must be -1 (full DTW) or at least 0; got "
                       + std::to_string(config.band) + ".");
  config.ndim = data_.ndim;
  validate_distance(config, data_);
  if (config == distance_ && band == config.band) return;
  distance_ = config;
  band = config.band;
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

void Problem::set_device(Device device, int index)
{
  if (index < 0)
    throw InvalidInput("Problem::set_device: the GPU index must be >= 0; got "
                       + std::to_string(index) + ".");
  if (device == Device::GPU) {
#if !defined(DTWC_HAS_CUDA) && !defined(DTWC_HAS_METAL)
    throw DeviceError(detail::gpu_not_built_message());
#elif !defined(DTWC_HAS_CUDA)
    if (index != 0)
      reject_gpu_request("Problem::set_device",
                         "runs on the system default GPU, but GPU index = " + std::to_string(index)
                           + " was requested",
                         "Use gpu (index 0).");
#endif
  }
  const int gpu_index = device == Device::GPU ? index : 0; // the CPU has no index
  if (device_ == device && device_index_ == gpu_index) return;
  device_ = device;
  device_index_ = gpu_index;
  refresh_distance_matrix();
}

void Problem::copy_distance_settings_from(const Problem &other)
{
  set_distance(other.distance());
  set_device(other.device_, other.device_index_);
  set_gpu_precision(other.gpu_precision_);
  set_matrix_dtype(other.matrix_dtype_);
  verbose_ = other.verbose_;
}

void Problem::validate_distance(core::DistanceConfig config, const Data &data)
{
  config.ndim = data.ndim;
  core::validate(config, data.is_f32());
}

const Problem::dtw_fn_t &Problem::dtw_function()
{
  sync_band();
  validate_fill_request("Problem::dtw_function");
  return dtw_fn_;
}

const Problem::dtw_fn_f32_t &Problem::dtw_function_f32()
{
  sync_band();
  core::validate(distance_, true); // the series may be Float64: float32 must hold the parameters
  validate_fill_request("Problem::dtw_function_f32");
  if (!dtw_fn_f32_) dtw_fn_f32_ = core::resolve_dtw_fn<float>(distance_, data_);
  return dtw_fn_f32_;
}

core::DistanceMatrix::fingerprint_type
Problem::distance_checkpoint_identity() const
{
  return distance_checkpoint_identity(distance_.metric);
}

core::DistanceMatrix::fingerprint_type
Problem::distance_checkpoint_identity(core::MetricType metric) const
{
  // The precision the distances are computed in: the series' on the CPU, the
  // GPU's on a GPU. Not the device: FP64 on a CPU and on a GPU differ by
  // rounding, so either's cache serves the other (and GPU 0's serves GPU 1),
  // while FP32 distances in an FP64 run are a loss of precision.
  const GpuPrecision gpu = resolve_gpu_precision(gpu_precision_);
  if (device_ == Device::GPU && gpu == GpuPrecision::Auto)
    throw InvalidInput(
      "A persistent distance cache (a mapped .dtwm or a checkpoint) records the precision "
      "its distances are computed in, but CUDA precision Auto resolves to FP32 or FP64 on the "
      "GPU a fill meets. Select explicit FP32 or FP64 (set_gpu_precision, --gpu-precision) "
      "before binding or saving the cache.");
  const core::Precision computed = device_ == Device::CPU ? data_.precision
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
  hash_enum(configuration, metric);
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
    hash_u64(hash, static_cast<std::uint64_t>(data_.series_flat_size(i)));
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
  use_mmap_distance_matrix(cache_path, distance_.metric);
}

void Problem::use_mmap_distance_matrix(
  const std::filesystem::path &cache_path, core::MetricType metric)
{
  sync_band();
  // Every check, the identity and the mapping come before any change: a bind
  // that fails leaves this Problem's metric and matrix as they were.
  auto candidate = distance_;
  candidate.metric = metric;
  validate_distance(candidate, data_);
  // map() checks an existing file's magic, version, length, N and fingerprint
  // before any of its distances can be read.
  auto mapped = core::DistanceMatrix::map(cache_path, data_.size(),
                                          distance_checkpoint_identity(metric), matrix_dtype_);
  // A warm start holds every pair.
  const bool complete = mapped.all_computed("Problem::use_mmap_distance_matrix",
                                            distance_.variant.variant == core::DTWVariant::SoftDTW);
  if (distance_.metric != metric) { // new semantics, as in set_metric
    distance_.metric = metric;
    clusters_ind.clear();
    centroids_ind.clear();
    rebind_dtw_fn();
  }
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
  // The autosave records the precision its distances were computed in, which
  // CUDA's Auto leaves open: the identity refuses it here, before any pair.
  if (device_ == Device::GPU && resolve_gpu_precision(gpu_precision_) == GpuPrecision::Auto)
    (void)distance_checkpoint_identity();
}

void Problem::validate_fill_request(std::string_view where) const
{
  const std::string at(where);

  // On every device: set_data refuses an empty series, but an edit through
  // p_vec can empty one afterwards, and the kernels answer it with the finite
  // max() sentinel. First, so a band check does not blame the empty series' length.
  reject_empty_series(data_, at);

  // A band narrower than a pair's length difference leaves that pair no warping
  // path: the banded kernels return the finite max() sentinel, which passes every
  // isfinite() guard, so clustering would silently sum 1.8e308. The widest gap is
  // shortest vs longest. Soft-DTW, MSM and TWE ignore the band. Lengths are
  // timesteps, as the multivariate kernels count them.
  const auto variant = distance_.variant.variant;
  const int band = distance_.band;
  if (band >= 0 && data_.size() > 1 && variant != core::DTWVariant::SoftDTW
      && variant != core::DTWVariant::MSM && variant != core::DTWVariant::TWE) {
    std::size_t shortest = 0, longest = 0;
    for (std::size_t i = 1; i < data_.size(); ++i) {
      if (data_.series_length(i) < data_.series_length(shortest)) shortest = i;
      if (data_.series_length(i) > data_.series_length(longest)) longest = i;
    }
    const std::size_t gap =
      data_.series_length(longest) - data_.series_length(shortest);
    if (gap > static_cast<std::size_t>(band)) {
      const auto describe = [&](std::size_t i) {
        return "series '" + std::string(series_name(i)) + "' (index "
             + std::to_string(i) + ", length "
             + std::to_string(data_.series_length(i)) + ")";
      };
      throw InvalidInput(
        at + ": band = " + std::to_string(band) + " is narrower than the length "
        "difference between " + describe(shortest) + " and " + describe(longest)
        + ", so no warping path fits that pair. The smallest feasible band is "
        + std::to_string(gap) + "; pass band >= " + std::to_string(gap)
        + ", or band = -1 for full DTW.");
    }
  }

  // The kernels take NaN and ±inf as numbers. ±inf gives inf, or
  // inf − inf = NaN, under any strategy; NaN is a missing value only to a
  // missing-data strategy, and otherwise poisons the recurrence (NaN also marks
  // an uncomputed matrix entry). One check through the raw entry points' own
  // boundary test, serial so the message names the series and its (flat)
  // position.
  const bool nan_is_missing = distance_.missing != core::MissingStrategy::Error;
  std::string name;
  for (std::size_t i = 0; i < data_.size(); ++i) {
    name.assign("series '").append(series_name(i)).append("' (index ")
      .append(std::to_string(i)).append(")");
    if (data_.is_f32())
      detail::require_finite(data_.series_f32(i), name, at, nan_is_missing);
    else
      detail::require_finite(series(i), name, at, nan_is_missing);
  }
  if (distance_.missing == core::MissingStrategy::Interpolate) {
    // interpolate_linear_into() has no observed value to interpolate from when a
    // series is entirely NaN, and used to throw from inside the per-pair lambda.
    for (std::size_t i = 0; i < data_.size(); ++i) {
      const bool all_nan = data_.is_f32() ? all_missing(data_.series_f32(i))
                                          : all_missing(series(i));
      if (all_nan)
        throw InvalidInput(
          at + ": series '" + std::string(series_name(i)) + "' (index "
          + std::to_string(i) + ") is entirely NaN, so MissingStrategy::Interpolate "
            "has nothing to interpolate from. Use ZeroCost or AROW, or drop the "
            "series.");
    }
  }

  // The GPU routes also need owned, resident, univariate series.
  if (device_ != Device::GPU) return;
  if (data_.is_view())
    reject_gpu_request(at,
                       "needs owned series in RAM, but this Problem's series are a non-owning "
                       "view (set_view_data)",
                       "Install owning series with set_data, or use device cpu.");
  validate_gpu_request(at, *this, data_.precision);
  if (data_.ndim > 1)
    reject_gpu_request(at, "is univariate only, but ndim = " + std::to_string(data_.ndim) + " was requested",
                       "Use device cpu for multivariate series.");
}

/**
 * @brief Fills the distance matrix using brute-force parallel computation.
 * @details Every pair through the bound lane and per-pair functions (all DTW
 *          variants), block by block (see the body).
 */
void Problem::fillDistanceMatrix_BruteForce()
{
  const size_t N = data_.size();
  const bool f32 = data_.is_f32();
  const dtw_fn_f32_t *f32_function = f32 ? &dtw_fn_f32_ : nullptr;

  // resize() re-fills every packed slot with NaN, so it must NOT run when the
  // matrix is already the right size (a mapped one always is): an unconditional
  // resize discarded a restored checkpoint and recomputed every pair.
  if (distMat.size() != N) distMat.resize(N);
  if (N == 0) return;

  // Blocked as a matrix multiplication is. Tile-rows: packed rows [R0, R1), each
  // one contiguous byte range of the matrix, cut at N - h, N - 2h, ... (only the
  // one holding row 0 is shorter). A task is a column c of a tile-row: the pairs
  // (c, r), r in R, c < r, and (c, c) if c is in R, so every pair has one task and
  // the tasks share no write and no lock. A thread's chunk of consecutive columns
  // writes one slot in each of R's packed rows, then the slot beside it: a cache
  // line comes from memory once for its eight entries, and R's series stay in L2
  // from one column to the next (h keeps them within 512 KiB, half an E-core's
  // share of its cluster's 4 MB L2, and at least 16 lane blocks). A slab, whole
  // tile-rows, is one parallel loop: the whole matrix in RAM; the rows between two
  // checkpoint saves; for a mapped matrix an eighth of what the series leave of
  // the memory budget (three slabs are in flight: filled, prefetched, written
  // back), so each page is written once, while its slab runs.
  //
  // In a column x = series c, the smaller index, as every per-pair call takes it;
  // R's series not yet paired with it go to the lane function W at a time (a short
  // last block repeats its last series in the lanes past it, whose results are
  // dropped). A lane's value depends on x, its own series and the band only, so a
  // pair gets the same bits in any block: the per-pair one, bit for bit, unless a
  // compiler contracts a multiply-add in one kernel and not the other. Without a
  // band, where the lanes are packed (core::dtw_lanes_packed), they take series of
  // any lengths, each lane running to its block's longest, so R's rows go longest
  // first (equal lengths in index order); otherwise a block goes to the lanes only
  // if its series are as long as x. The per-pair loop fills the rest.
  const size_t W = f32 ? core::dtw_lanes<float> : core::dtw_lanes<data_t>;
  size_t values = 0;
  for (size_t i = 0; i < N; ++i) values += data_.series_flat_size(i);
  const size_t series_bytes = values * (f32 ? sizeof(float) : sizeof(data_t));
  const size_t entry_bytes = core::value_bytes(distMat.width());
  constexpr size_t tile_series_bytes = size_t{ 512 } << 10;
  size_t h = std::max(16 * W, tile_series_bytes * N / series_bytes / W * W);
  size_t slab_bytes = std::numeric_limits<size_t>::max();
  if (distMat.is_mapped()) {
    const size_t budget = ram_limit_ != 0 ? ram_limit_ : core::available_memory(); // 0: unknown, no limit
    slab_bytes = budget == 0 ? size_t{ 1 } << 30
                             : std::min(budget > series_bytes ? (budget - series_bytes) / 8 : 0, size_t{ 1 } << 30);
    h = std::min(h, std::max(W, slab_bytes / (N * entry_bytes) / W * W)); // the last tile-row fits a slab
  }
  const size_t save_rows =
    checkpoint.enabled ? (static_cast<size_t>(checkpoint.save_interval) + W - 1) / W * W : N;
  h = std::min(std::min(h, save_rows), N);

  std::vector<size_t> bounds{ 0 }; // tile-row k: rows [bounds[k], bounds[k + 1])
  for (size_t r = N % h; r < N; r += h)
    if (r > 0) bounds.push_back(r);
  bounds.push_back(N);
  const size_t K = bounds.size() - 1;
  std::vector<size_t> first_column(K + 1, 0); // tile-row k's columns: the tasks from first_column[k]
  for (size_t k = 0; k < K; ++k) first_column[k + 1] = first_column[k] + bounds[k + 1];

  const bool ragged = core::dtw_lanes_packed && distance_.band < 0;
  const bool lanes = f32 ? static_cast<bool>(dtw_block_fn_f32_) : static_cast<bool>(dtw_block_fn_);
  std::vector<size_t> order(N);
  std::iota(order.begin(), order.end(), size_t{ 0 });
  if (ragged && lanes)
    for (size_t k = 0; k < K; ++k)
      std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(bounds[k]),
                       order.begin() + static_cast<std::ptrdiff_t>(bounds[k + 1]),
                       [&](size_t a, size_t b) { return data_.series_flat_size(a) > data_.series_flat_size(b); });
  const bool reordered = !std::is_sorted(order.begin(), order.end());

  // `matrix` holds the values at the matrix's width: a pair is NaN until computed, then its distance rounded to it.
  auto fill_column = [&](auto *matrix, size_t R0, size_t R1, size_t c, auto column, const auto &block,
                         const auto &one_pair) {
    using T = typename decltype(column(c))::value_type;
    using V = std::remove_pointer_t<decltype(matrix)>;
    constexpr size_t Wt = core::dtw_lanes<T>;
    const auto computed = [matrix](size_t i, size_t j) { return !std::isnan(matrix[core::tri_index(i, j)]); };
    const auto store = [matrix](size_t i, size_t j, double v) { matrix[core::tri_index(i, j)] = static_cast<V>(v); };
    const auto x = column(c);
    if (c >= R0 && !computed(c, c)) store(c, c, 0.0);
    if (block) {
      std::array<size_t, Wt> rows;
      std::array<std::span<const T>, Wt> ys;
      std::array<double, Wt> d;
      // In index order the rows paired with c start after it.
      for (size_t b = reordered ? R0 : std::max(R0, c + 1); b < R1;) {
        size_t count = 0;
        for (; count < Wt && b < R1; ++b)
          if (order[b] > c && !computed(c, order[b])) rows[count++] = order[b];
        if (count == 0) break;
        bool equal = true;
        for (size_t w = 0; w < Wt; ++w) {
          ys[w] = column(rows[std::min(w, count - 1)]);
          equal = equal && ys[w].size() == x.size();
        }
        if (!(ragged || equal)) continue;
        block(x, ys, d);
        for (size_t w = 0; w < count; ++w) store(c, rows[w], d[w]);
      }
    }
    for (size_t r = std::max(R0, c + 1); r < R1; ++r)
      if (!computed(c, r)) store(c, r, one_pair(x, column(r)));
  };
  auto task = [&](auto *matrix, size_t g) {
    const size_t k = static_cast<size_t>(std::upper_bound(first_column.begin(), first_column.end(), g)
                                         - first_column.begin()) - 1;
    const size_t c = g - first_column[k];
    if (f32)
      fill_column(matrix, bounds[k], bounds[k + 1], c, [&](size_t i) { return data_.series_f32(i); },
                  dtw_block_fn_f32_, *f32_function);
    else
      fill_column(matrix, bounds[k], bounds[k + 1], c, [&](size_t i) { return series(i); }, dtw_block_fn_, dtw_fn_);
  };

  std::vector<size_t> slabs{ 0 }; // slab s: tile-rows [slabs[s], slabs[s + 1])
  while (slabs.back() < K) {
    const size_t k0 = slabs.back();
    size_t end = k0 + 1;
    while (end < K && bounds[end + 1] - bounds[k0] <= save_rows
           && (core::packed_size(bounds[end + 1]) - core::packed_size(bounds[k0])) * entry_bytes <= slab_bytes)
      ++end;
    slabs.push_back(end);
  }
  // Hashed once: every save of this fill writes the same identity.
  const auto identity = checkpoint.enabled ? distance_checkpoint_identity() : core::DistanceMatrix::fingerprint_type{};
  const path_t file = checkpoint.enabled ? checkpoint_path(*this, checkpoint.directory) : path_t{};
  // A mapped file's rows from `written` on were never written (no NaN prefill): each slab makes them NaN just
  // before it computes them, and a wait that leaves rows on the device records them written (flush_rows).
  const size_t written = distMat.rows_written();
  size_t waiting = 0; // the first row of the slabs whose write-back no wait has covered
  distMat.visit([&](auto *matrix) {
    distMat.prefetch_rows(0, bounds[slabs[1]]);
    for (size_t s = 0; s + 1 < slabs.size(); ++s) {
      const size_t r0 = bounds[slabs[s]], r1 = bounds[slabs[s + 1]];
      if (s + 2 < slabs.size()) distMat.prefetch_rows(r1, bounds[slabs[s + 2]]);
      distMat.clear_rows(std::max(r0, written), r1);
      // run_openmp catches inside the parallel region and rethrows the lowest
      // column's failure after the join; a failed pair stays uncomputed.
      const size_t first = first_column[slabs[s]];
      auto slab_task = [&](size_t i) { task(matrix, first + i); };
      run_openmp(slab_task, first_column[slabs[s + 1]] - first, true, 8);
      refuse_float32_overflow(r0, r1);
      // On this thread after the join, so a save never sees a slab half written;
      // tagged with metric(), the metric this fill computed (both autosave sites).
      if (checkpoint.enabled)
        distMat.write(file, identity, r0, r1);
      else {
        distMat.flush_rows(r0, r1, false);                       // this slab's write-back starts
        if (written < N) distMat.flush_rows(waiting, r0, true); // the slabs before it are on the device
      }
      waiting = r0;
    }
  });
  if (written < N) // every row on the device: the file records them all written
    distMat.flush_rows(distMat.rows_written(), N, true);
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
  validate_checkpoint_settings();
  sync_band();
  // Writes through writable_distance_matrix() commit here: one scan refuses ±inf and a
  // negative distance no variant but Soft-DTW gives, and a matrix they made complete needs no pair.
  if (written_ && !filled_ && data_.size() > 0 && distMat.size() == data_.size()) {
    filled_ = distMat.all_computed("Problem::fill_distance_matrix",
                                   distance_.variant.variant == core::DTWVariant::SoftDTW,
                                   "Leave a pair still to compute NaN, as resize() leaves every pair.");
    written_ = false;
  }
  if (filled_) return;
  validate_fill_request("Problem::fill_distance_matrix");

  // Each fill below sizes the matrix (deferred from set_data /
  // refresh_distance_matrix) only after its own refusals; a mapped matrix is
  // sized when it is bound.

  if (verbose_)
    std::cout << "Distance matrix is being filled!" << '\n';

  // The serial missing-data pre-scan is part of validate_fill_request,
  // above: it runs before any pair, here and on every other entry point.

  if (device_ == Device::CPU) {
    fillDistanceMatrix_BruteForce(); // saves a checkpoint after each row block
  } else {
    // Device::GPU is this build's backend: CUDA, else Metal. It writes every
    // entry of distMat in place, and never hands a pair to the CPU.
#if defined(DTWC_HAS_CUDA)
    cuda::CUDADistMatOptions opts;
    opts.device_id = device_index_;
#elif defined(DTWC_HAS_METAL)
    metal::MetalDistMatOptions opts; // the system default GPU: set_device refused any other
#else
    throw DeviceError(detail::gpu_not_built_message()); // set_device refused it first
#endif
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
    opts.band = distance_.band;
    opts.precision = resolve_gpu_precision(gpu_precision_);
    // L2 is L1 on the univariate series the GPU routes take.
    opts.use_squared_l2 = distance_.metric == core::MetricType::SquaredL2;
    opts.verbose = verbose_;
#  if defined(DTWC_HAS_CUDA)
    (void)cuda::compute_distance_matrix_cuda(data_.p_vec, opts, distMat);
#  else
    (void)metal::compute_distance_matrix_metal(data_.p_vec, opts, distMat);
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
/**
 * @brief Clusters the series by method().
 * @details PAM, OneBatchPAM, CLARA and the hierarchical cut return their own result.
 * Lloyd, MIP, LR-core and TADPole leave their clustering in the Problem; its cost
 * is then find_total_cost().
 */
core::ClusteringResult Problem::cluster()
{
  require_clusterable(Nc, size());
  const Method method = resolve_method(method_, device_, size());
  switch (method) {
  case Method::PAM: {
    // Restart r starts from seed + r; the strictly lowest cost is kept, so a tie
    // keeps the earlier restart.
    auto best = fast_pam(*this, Nc, maxIter, random_seed_);
    for (int restart = 1; restart < N_repetition; ++restart) {
      auto candidate = fast_pam(*this, Nc, maxIter, random_seed_ + static_cast<std::uint64_t>(restart));
      if (candidate.total_cost < best.total_cost) best = std::move(candidate);
    }
    set_result(best); // each restart published its own
    return best;
  }
  case Method::OneBatch:
    return algorithms::one_batch_pam(
      *this, { .n_clusters = Nc, .batch_size = batch_size_, .max_iter = maxIter, .random_seed = random_seed_ });
  case Method::CLARA:
    return algorithms::fast_clara(*this, { .n_clusters = Nc, .sample_size = sample_size_, .n_samples = n_samples_,
                                           .max_iter = maxIter, .random_seed = random_seed_ });
  case Method::Hierarchical:
    return algorithms::cut_dendrogram(algorithms::build_dendrogram(*this, { .linkage = linkage_ }), *this, Nc);
  case Method::Kmedoids: cluster_by_kmedoids_lloyd(); break;
  case Method::MIP: cluster_by_mip(); break;
  case Method::LRCore: LR_core_clustering(*this); break;
  case Method::TADPole:
    algorithms::tadpole(*this, Nc, tadpole_dc_ > 0.0 ? tadpole_dc_ : algorithms::tadpole_auto_dc(*this));
    break;
  case Method::Auto: break; // resolved above
  }
  core::ClusteringResult result;
  result.labels = clusters_ind;
  result.medoid_indices = centroids_ind;
  result.total_cost = find_total_cost();
  // Lloyd counts its iterations; the exact methods and TADPole run to the end.
  result.iterations = method == Method::Kmedoids ? last_iterations_ : 0;
  result.converged = method != Method::Kmedoids || last_iterations_ < maxIter;
  return result;
}

/**
 * @brief Executes the clustering process and additional post-processing tasks.
 * @details Performs clustering, then prints and writes the cluster results, including silhouettes, to files.
 */
void Problem::cluster_and_process()
{
  // cluster() is side-effect free by contract; the heuristic run artifacts
  // (per-repetition medoids, best-repetition record) belong to this entry point.
  struct ArtifactScope
  {
    bool &flag;
    explicit ArtifactScope(bool &f) : flag{ f } { flag = true; }
    ~ArtifactScope() { flag = false; }
  } artifact_scope{ persist_run_artifacts_ };
  cluster();
  print_clusters(); // Prints to screen.
  write_distance_matrix();
  write_clusters(); // Prints to file.
  write_silhouettes();
}

/**
 *@brief Clusters the data using Mixed Integer Programming (MIP) based on the chosen solver.
 *@details Uses either Gurobi or HiGHS solver for MIP clustering, depending on the solver set in the Problem instance.
 */
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


/**
 * @brief Assigns each data point to the nearest cluster centroid.
 * @details Iterates over each data point, calculating its distance to each centroid, and assigns it to the nearest one.
 */
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
          matrix[core::tri_index(static_cast<std::size_t>(ip), static_cast<std::size_t>(medoid))], "kmedoids_lloyd",
          i_p, static_cast<index_t>(slot), medoid);
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
    run(assignClustersTask, data_.size()); // read-only lookups, one label slot per point
  });
  clusters_ind = std::move(labels);
}

/**
 * @brief Calculates and updates the medoids of each cluster.
 * @details This function iterates through each data point and calculates the total cost of designating that point
 * as the medoid of its cluster. The point with the minimum total cost is set as the new medoid for that cluster.
 */
void Problem::calculate_medoids()
{
  require_clustered("calculate_medoids");
  fill_distance_matrix(); // every pair within a cluster
  std::vector<double> pointCosts(size());

  distMat.visit([&](const auto *matrix) {
    auto findBetterMedoidTask = [&](size_t i_p) // i_p is point index.
    {
      // The values' pointer is read at each pair summed, as dist_by_ind read it. Held in a register (a capture by
      // value), clang 21 vectorised the sum: another order. Read through the capture, GCC 13 and cl 19.50 loaded
      // the capture too at each pair (2026-10-11).
      const auto *const *const values = &matrix;
      double sum{ 0 };
      for (const auto i : Range(size()))
        if (clusters_ind[i] == clusters_ind[i_p]) // If they are in the same cluster
          sum += (*values)[core::tri_index(i_p, i)];

      pointCosts[i_p] = sum;
    };
    run(findBetterMedoidTask, size());
  });

  std::vector<double> clusterCosts(n_clusters(), std::numeric_limits<double>::max());
  for (const auto i : Range(size()))
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
  const bool persist_artifacts = persist_run_artifacts_;
  // The setters reject both; the public fields maxIter and N_repetition bypass them.
  const int repetitions = n_repetitions();
  if (repetitions <= 0)
    throw InvalidInput("Lloyd k-medoids requires n_repetitions >= 1.");
  if (max_iter() <= 0)
    throw InvalidInput("Lloyd k-medoids requires max_iter >= 1.");

  fill_distance_matrix(); // Ensure all distances computed before parallel clustering.

  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();
  int best_iterations = 0;
  std::vector<index_t> best_medoids;
  std::vector<index_t> best_labels;

  for (int i_rand = 0; i_rand < repetitions; i_rand++) {
    if (verbose_) std::cout << "Metoid initialisation is started.\n";
    // Unsigned: a seed near 2^64 wraps to another valid seed.
    init_with_seed(random_seed_ + static_cast<std::uint64_t>(i_rand));

    if (verbose_)
      std::cout << "Metoid initialisation is finished. "
                << Nc << " medoids are initialised.\n"
                << "Start clustering:\n";

    auto [status, total_cost, iters] =
      cluster_by_kMedoidsLloyd_single(i_rand, persist_artifacts);
    last_iterations_ = iters;

    if (verbose_) {
      if (status == 0)
        std::cout << "Medoids are same for last two iterations, algorithm is converged!\n";
      else if (status == -1)
        std::cout << "Maximum iteration is reached before medoids are converged!\n";
    }

    if (i_rand == 0 || total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
      best_iterations = iters;
      best_medoids = centroids_ind;
      best_labels = clusters_ind;
    }
    if (verbose_)
      std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';
  }

  centroids_ind = std::move(best_medoids);
  clusters_ind = std::move(best_labels);
  last_iterations_ = best_iterations;
  if (persist_artifacts)
    writeBestRep(best_rep);
  else if (verbose_)
    std::cout << "Best repetition: " << best_rep << '\n';
}

/**
 * @brief Executes a single run of the Lloyd k-medoids clustering.
 * @details This function performs a single run of the Lloyd k-medoids algorithm, updating the medoids and clusters,
 * and calculating the total cost for this run.
 * @param rep The current repetition number.
 * @return A pair containing the status (whether the algorithm converged or not) and the total cost of clustering for this repetition.
 */
std::tuple<int, double, int> Problem::cluster_by_kMedoidsLloyd_single(
  int rep, bool persist_artifacts)
{
  if (centroids_ind.empty())
    init_with_seed(random_seed_ + static_cast<std::uint64_t>(rep));

  auto oldmedoids = centroids_ind;

  int status = -1;
  std::vector<std::vector<index_t>> centroids_all;

  int actual_iters = 0;
  const int iteration_limit = max_iter();
  for (int i = 0; i < iteration_limit; i++) {
    actual_iters = i + 1;

    if (verbose_) {
      std::cout << "Medoids: ";
      for (auto medoid : centroids_ind)
        std::cout << series_name(medoid) << ' ';
    }

    centroids_all.push_back(centroids_ind);

    assign_clusters();

    if (verbose_) {
      std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
                << find_total_cost() << ".\n"; // Uses clusters_ind to find cost.

      print_clusters();
    }
    calculate_medoids(); // Changes centroids_ind

    if (oldmedoids == centroids_ind) {
      status = 0;
      break;
    }

    oldmedoids = centroids_ind;
  }

  // A converged iteration leaves the medoids unchanged, so its assignment is
  // already current. A capped iteration may have just updated the medoids;
  // realign once before costing and snapshotting the returned state.
  if (status == -1)
    assign_clusters();

  const double total_cost = find_total_cost();
  if (verbose_)
    std::cout << "Procedure is completed with cost: " << total_cost << '\n';
  if (persist_artifacts)
    writeMedoids(centroids_all, rep, total_cost);
  return {status, total_cost, actual_iters};
}

/**
 * @brief Calculates the total cost of the current clustering solution.
 * @details Computes the sum of the distances between each point and its closest medoid.
 * This serves as a measure of the quality of the current clustering solution.
 * @return The total cost of the clustering.
 */
double Problem::find_total_cost()
{
  require_clustered("find_total_cost");
  // A matrix-free clustering (TADPole) leaves the matrix unfilled: the N
  // point-to-medoid distances are then computed with the bound function, which
  // is what a fill would store, instead of filling N(N-1)/2 pairs.
  std::vector<double> distances(size());
  if (filled_) {
    for (std::size_t i = 0; i < size(); ++i)
      distances[i] = dist_by_ind(static_cast<index_t>(i), centroid_of(static_cast<index_t>(i)));
  } else {
    validate_fill_request("Problem::find_total_cost");
    auto point_distance = [&](std::size_t i) {
      const auto medoid = static_cast<std::size_t>(centroid_of(static_cast<index_t>(i)));
      distances[i] = i == medoid ? 0.0
                   : data_.is_f32() ? dtw_fn_f32_(data_.series_f32(i), data_.series_f32(medoid))
                                    : dtw_fn_(series(i), series(medoid));
    };
    run_openmp(point_distance, size());
  }
  // k-medoids objective: sum of raw DTW distances (not squared, unlike k-means).
  return core::detail::ordered_medoid_objective(distances, "kmedoids_lloyd");
}

} // namespace dtwc
