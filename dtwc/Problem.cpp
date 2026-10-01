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
#include "core/distance_semantics.hpp"      // validate_problem_distance_semantics
#include "core/variant_validation.hpp"     // validate_variant_params
#include "core/sha256.hpp"                 // for persistent cache fingerprints
#include "base/missing_utils.hpp"               // for all_missing
#include "algorithms/tadpole.hpp"          // for Method::TADPole dispatch


#include <algorithm> // for min
#include <array>     // for array
#include <cstdint>   // for uint32_t, uint64_t
#include <cstring>   // for memcpy
#include <iomanip>   // for operator<<, setprecision
#include <iostream>  // for cout
#include <limits>    // for numeric_limits
#include <cassert>
#include <stdexcept> // for logic_error
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

const char *metric_name(core::MetricType m)
{
  switch (m) {
  case core::MetricType::L1: return "L1";
  case core::MetricType::L2: return "L2";
  case core::MetricType::SquaredL2: return "SquaredL2";
  }
  return "?";
}

#ifdef DTWC_HAS_METAL
/// The Metal selector for `precision`; validate_metal_precision() then rejects
/// FP64, which Metal cannot run.
metal::MetalPrecision metal_precision(GpuPrecision precision)
{
  return precision == GpuPrecision::FP32 ? metal::MetalPrecision::FP32
       : precision == GpuPrecision::FP64 ? metal::MetalPrecision::FP64
                                         : metal::MetalPrecision::Auto;
}
#endif

/// A GPU request its backend cannot honour: every FX-1 rule words it so.
[[noreturn]] void reject_gpu_request(std::string_view where, bool cuda, const std::string &request,
                                     const std::string &fix)
{
  throw DeviceError(std::string(where) + (cuda ? ": CUDA " : ": Metal ") + request
                    + "; no backend call or CPU fallback was attempted. " + fix);
}

} // namespace

void validate_gpu_request(std::string_view where, DistanceMatrixStrategy strategy,
                          const core::DTWVariantParams &variant, core::MissingStrategy missing,
                          core::Precision precision, const CUDASettings &gpu)
{
  // The GPU routes upload owned Float64 series (data_.p_vec) and run Standard
  // DTW with no missing-data strategy, in L1 or squared L2.
  const bool cuda = strategy == DistanceMatrixStrategy::CUDA;
  if (!cuda && strategy != DistanceMatrixStrategy::Metal) return;
  if (precision == core::Precision::Float32)
    reject_gpu_request(where, cuda, "computes from Float64 series, but precision = Float32 was requested",
                       "Load the series as Float64, or use device cpu.");
  if (variant.variant != core::DTWVariant::Standard)
    reject_gpu_request(where, cuda,
                       std::string("implements Standard DTW only, but variant = ")
                         + variant_name(variant.variant) + " was requested",
                       "Use device cpu for this variant.");
  if (missing != core::MissingStrategy::Error)
    reject_gpu_request(where, cuda,
                       std::string("has no missing-data strategy, but missing_strategy = ")
                         + missing_strategy_name(missing) + " was requested",
                       "Use device cpu, or missing_strategy Error for data without NaN.");
  if (!cuda && gpu.device_id != 0)
    reject_gpu_request(where, cuda,
                       "runs on the system default GPU, but GPU index = " + std::to_string(gpu.device_id)
                         + " was requested",
                       "Use gpu (index 0).");
#ifdef DTWC_HAS_METAL
  if (!cuda) metal::validate_metal_precision(metal_precision(gpu.precision));
#endif
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
    std::cout << "Solver Gurobi is not available; therefore using default solver\n";
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
  distMat = core::DistanceMatrix{};
  filled_ = false;
  rebind_dtw_fn(); // series edited in place may have new lengths (the WDTW table)
}

/**
 * @brief Resolve the distance functions from distance_ and the series, once.
 * @details Every function holds copies of the settings it reads, so it stays
 * valid when this Problem moves, and parallel callers only read it.
 */
void Problem::rebind_dtw_fn()
{
  dtw_fn_ = core::resolve_dtw_fn<data_t>(distance_, data_);
  dtw_block_fn_ = core::resolve_dtw_block_fn<data_t>(distance_);
  if (core::active_variant_params_representable_f32(distance_.variant)) {
    dtw_fn_f32_ = core::resolve_dtw_fn<float>(distance_, data_);
    dtw_block_fn_f32_ = core::resolve_dtw_block_fn<float>(distance_);
  } else {
    dtw_fn_f32_ = {};
    dtw_block_fn_f32_ = {};
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
  switch (device) {
  case Device::CPU:
    // Auto is the CPU brute-force fill, never a GPU.
    if (distance_strategy_ == DistanceMatrixStrategy::CUDA
        || distance_strategy_ == DistanceMatrixStrategy::Metal)
      set_distance_strategy(DistanceMatrixStrategy::Auto);
    return;
  case Device::GPU: {
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
    auto settings = cuda_settings_;
    settings.device_id = index;
    set_cuda_settings(settings);
#  if defined(DTWC_HAS_CUDA)
    set_distance_strategy(DistanceMatrixStrategy::CUDA);
#  else
    set_distance_strategy(DistanceMatrixStrategy::Metal);
#  endif
    return;
#else
    throw DeviceError(detail::gpu_not_built_message());
#endif
  }
  }
}

void Problem::validate_distance(
  const core::DistanceConfig &config, const Data &data, bool force_float32)
{
  const auto &params = config.variant;
  const auto missing = config.missing;
  const auto metric = config.metric;
  core::validate_problem_distance_semantics(
    params, missing, data.ndim, force_float32 || data.is_f32());
  // resolve_dtw_fn passes the metric to the Standard kernels only; every other
  // variant and missing-data strategy would silently compute L1.
  if (metric != core::MetricType::L1
      && (params.variant != core::DTWVariant::Standard
          || missing != core::MissingStrategy::Error))
    throw InvalidInput(
      std::string("Problem: metric ") + metric_name(metric)
      + " is implemented for Standard DTW with MissingStrategy::Error only, but "
        "variant = " + variant_name(params.variant) + " and missing_strategy = "
      + missing_strategy_name(missing) + " were requested. Use metric L1 for "
        "this configuration.");
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
  // Bound whenever the active parameters are representable (rebind_dtw_fn).
  core::validate_active_variant_params_f32(distance_.variant);
  validate_fill_request("Problem::dtw_function_f32");
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
  if (distance_strategy_ == DistanceMatrixStrategy::CUDA
      && cuda_settings_.precision == GpuPrecision::Auto) {
    throw InvalidInput(
      "use_mmap_distance_matrix: CUDA precision=Auto is not safe for persistent "
      "warm-start caches because its resolved FP32/FP64 semantics depend on the "
      "runtime GPU. Select explicit FP32 or FP64 before binding the cache.");
  }

  // The settings the stored distances were computed with, the metric among
  // them: without it a SquaredL2 run writes the same fingerprint as an L1 run
  // over the same data, and a later L1 run then accepts the wrong matrix. All
  // variant parameters are included, even when inactive for the selected
  // variant: a harmless cache miss beats trusting an ambiguous configuration.
  FingerprintHash configuration;
  static constexpr char configuration_domain[] = "dtwc-distance-cache-configuration-v1";
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
  // Backend/precision can change the stored numeric result even when the
  // mathematical recurrence is the same (notably GPU FP32 versus CPU FP64).
  hash_enum(configuration, distance_strategy_);
  hash_u64(configuration, static_cast<std::uint64_t>(static_cast<std::int64_t>(cuda_settings_.device_id)));
  hash_u64(configuration, static_cast<std::uint64_t>(static_cast<std::int64_t>(cuda_settings_.precision)));

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
                                          distance_checkpoint_identity(metric));
  // A warm start holds every pair.
  const bool complete = mapped.all_computed("Problem::use_mmap_distance_matrix");
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
}

void Problem::validate_fill_request(std::string_view where) const
{
  const std::string at(where);

  // A band narrower than a pair's length difference leaves that pair no warping
  // path: the banded kernels return the finite max() sentinel, which passes every
  // isfinite() guard, so clustering would silently sum 1.8e308 (design §9,
  // D-12). The widest gap is shortest vs longest. Soft-DTW, MSM and TWE ignore
  // the band. Lengths are timesteps, as the multivariate kernels count them.
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

  // FX-15: the kernels take NaN and ±inf as numbers. ±inf gives inf, or
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
    // interpolate_linear() has no observed value to interpolate from when a
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
  const bool cuda = distance_strategy_ == DistanceMatrixStrategy::CUDA;
  if (!cuda && distance_strategy_ != DistanceMatrixStrategy::Metal) return;
  if (data_.is_view())
    reject_gpu_request(at, cuda,
                       "needs owned series in RAM, but this Problem's series are a non-owning "
                       "view (set_view_data, as FastCLARA's in-memory subsamples are)",
                       "Install owning series with set_data, or use device cpu.");
  validate_gpu_request(at, distance_strategy_, distance_.variant, distance_.missing, data_.precision,
                       cuda_settings_);
  if (data_.ndim > 1)
    reject_gpu_request(at, cuda, "is univariate only, but ndim = " + std::to_string(data_.ndim) + " was requested",
                       "Use device cpu for multivariate series.");
}

/**
 * @brief Fills the distance matrix using brute-force parallel computation.
 * @details Original implementation: parallel loop over all upper-triangle pairs
 *          using the bound dtw_fn_ (supports all DTW variants).
 */
void Problem::fillDistanceMatrix_BruteForce()
{
  const size_t N = data_.size();
  const dtw_fn_f32_t *f32_function = data_.is_f32() ? &dtw_fn_f32_ : nullptr;

  // resize() re-fills every packed slot with NaN, so it must NOT run when the
  // matrix is already the right size (a mapped one always is): an unconditional
  // resize discarded a restored checkpoint and recomputed every pair.
  if (distMat.size() != N) distMat.resize(N);

  for (size_t i = 0; i < N; ++i)
    if (!distMat.is_computed(i, i)) distMat.set(i, i, 0.0);

  // Lock-free by design: each worker owns a disjoint row. run_openmp catches
  // inside the structured block and deterministically rethrows the lowest-row
  // failure after the join; a failed pair remains uncomputed.
  //
  // Row i first takes its columns W at a time through the lane function, where
  // the block's series are as long as series i and not all its pairs are known;
  // each of those distances is the per-pair one (bit for bit, unless the compiler
  // contracts a multiply-add in one kernel and not the other). The block at the
  // row's end repeats its last column in the lanes past it, whose results are
  // dropped.
  // The per-pair loop then fills the rest: mixed-length blocks, every pair
  // without a lane function.
  auto fill_lanes = [&](size_t i, auto x, auto column, const auto &block) {
    using T = typename decltype(x)::value_type;
    constexpr size_t W = core::dtw_lanes<T>;
    if (!block) return;
    std::array<std::span<const T>, W> ys;
    std::array<double, W> d;
    for (size_t j = i + 1; j < N; j += W) {
      const size_t count = std::min(W, N - j);
      bool equal = true, known = true;
      for (size_t w = 0; w < W; ++w) {
        const size_t c = j + std::min(w, count - 1);
        ys[w] = column(c);
        equal = equal && ys[w].size() == x.size();
        known = known && distMat.is_computed(i, c);
      }
      if (!equal || known) continue;
      block(x, ys, d);
      for (size_t w = 0; w < count; ++w)
        if (!distMat.is_computed(i, j + w)) distMat.set(i, j + w, d[w]);
    }
  };
  auto fill_row = [&](size_t i) {
    if (data_.is_f32()) {
      const auto si = data_.series_f32(i);
      fill_lanes(i, si, [&](size_t j) { return data_.series_f32(j); }, dtw_block_fn_f32_);
      for (size_t j = i + 1; j < N; ++j)
        if (!distMat.is_computed(i, j)) distMat.set(i, j, (*f32_function)(si, data_.series_f32(j)));
    } else {
      const auto si = series(i);
      fill_lanes(i, si, [&](size_t j) { return series(j); }, dtw_block_fn_);
      for (size_t j = i + 1; j < N; ++j)
        if (!distMat.is_computed(i, j)) distMat.set(i, j, dtw_fn_(si, series(j)));
    }
  };
  if (!checkpoint.enabled) {
    run_openmp(fill_row, N, true, 8);
    return;
  }

  // Invariant: each block is a disjoint row range [block_begin, end); the save
  // runs on the main thread after run_openmp has joined, so it never observes a
  // partially written row.
  const size_t stride = static_cast<size_t>(checkpoint.save_interval);
  size_t block_begin = 0;
  auto fill_block = [&](size_t k) { fill_row(block_begin + k); };
  for (; block_begin < N; block_begin += stride) {
    const size_t end = std::min(block_begin + stride, N);
    run_openmp(fill_block, end - block_begin, true, 8);
    // Tagged with metric(), the metric this fill computed (both autosave sites).
    save_checkpoint(*this, checkpoint.directory);
  }
}

/**
 * @brief Fills the distance matrix by computing distances between all pairs of points.
 * @details Auto and BruteForce run the parallel exact CPU fill (all variants);
 *          CUDA and Metal are selected by set_device(gpu) or explicitly.
 */
void Problem::fill_distance_matrix()
{
  validate_checkpoint_settings();
  sync_band();
  // Writes through distance_matrix() commit here: one scan refuses ±inf, and a
  // matrix they made complete needs no pair.
  if (written_ && !filled_ && data_.size() > 0 && distMat.size() == data_.size()) {
    filled_ = distMat.all_computed("Problem::fill_distance_matrix");
    written_ = false;
  }
  if (filled_) return;
  validate_fill_request("Problem::fill_distance_matrix");

  // Each fill below sizes the matrix (deferred from set_data /
  // refresh_distance_matrix) only after its own refusals; a mapped matrix is
  // sized when it is bound.

  if (verbose_)
    std::cout << "Distance matrix is being filled!" << '\n';

  // The serial missing-data pre-scan is part of validate_fill_request (FX-15),
  // above: it runs before any pair, here and on every other entry point.

  DistanceMatrixStrategy effective = distance_strategy_;
  if (effective == DistanceMatrixStrategy::Auto)
    effective = DistanceMatrixStrategy::BruteForce;

  // A GPU backend writes every entry of distMat in place; an explicitly
  // requested backend never changes to CPU.
  switch (effective) {
  case DistanceMatrixStrategy::CUDA:
#ifdef DTWC_HAS_CUDA
  {
    dtwc::cuda::CUDADistMatOptions cuda_opts;
    cuda_opts.band = distance_.band;
    cuda_opts.device_id = cuda_settings_.device_id;
    if (cuda_settings_.precision == GpuPrecision::FP32)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    else if (cuda_settings_.precision == GpuPrecision::FP64)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
    // L2 is L1 on the univariate series the GPU routes take.
    cuda_opts.use_squared_l2 = distance_.metric == core::MetricType::SquaredL2;
    cuda_opts.verbose = verbose_;

    (void)dtwc::cuda::compute_distance_matrix_cuda(data_.p_vec, cuda_opts, distMat);
    break;
  }
#else
    throw DeviceError(
      "CUDA distance strategy requested but CUDA is not compiled in. "
      "Rebuild with -DDTWC_ENABLE_CUDA=ON. No CPU fallback was attempted.");
#endif
  case DistanceMatrixStrategy::Metal:
#ifdef DTWC_HAS_METAL
  {
    dtwc::metal::MetalDistMatOptions metal_opts;
    metal_opts.band = distance_.band;
    metal_opts.precision = metal_precision(cuda_settings_.precision);
    metal_opts.use_squared_l2 = distance_.metric == core::MetricType::SquaredL2;
    metal_opts.verbose = verbose_;

    (void)dtwc::metal::compute_distance_matrix_metal(data_.p_vec, metal_opts, distMat);
    break;
  }
#else
    throw DeviceError(
      "Metal distance strategy requested but Metal is not compiled in. "
      "Rebuild on macOS with -DDTWC_ENABLE_METAL=ON. No CPU fallback was attempted.");
#endif
  case DistanceMatrixStrategy::BruteForce:
    fillDistanceMatrix_BruteForce();
    break;
  case DistanceMatrixStrategy::Auto:
    throw std::logic_error(
      "Problem::fill_distance_matrix: unresolved Auto strategy");
  }

  // BruteForce already saved after its last row block; every other backend fills
  // in one call, so its only automatic save is here.
  if (checkpoint.enabled && effective != DistanceMatrixStrategy::BruteForce)
    save_checkpoint(*this, checkpoint.directory);
  filled_ = data_.size() > 0; // an empty Problem has no matrix to call filled

  if (verbose_)
    std::cout << "Distance matrix has been filled!" << '\n';
}
/**
 * @brief Performs clustering based on the specified method.
 * @details Chooses between different clustering methods (K-medoids or MIP) and performs the clustering accordingly.
 */
void Problem::cluster()
{
  switch (method_) {
  case Method::Kmedoids:
    cluster_by_kmedoids_lloyd();
    break;
  case Method::MIP:
    cluster_by_mip();
    break;
  case Method::LRCore:
    LR_core_clustering(*this);
    break;
  case Method::TADPole: {
    const double dc = (tadpole_dc_ > 0.0)
                        ? tadpole_dc_
                        : algorithms::tadpole_auto_dc(*this);
    algorithms::tadpole(*this, Nc, dc);
    break;
  }
  }
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
  auto assignClustersTask = [this, &labels](size_t i_p) //!< i_p and i_c in [0, Np)
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
        dist_by_ind(ip, medoid), "kmedoids_lloyd", i_p,
        static_cast<index_t>(slot), medoid);
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

  auto findBetterMedoidTask = [&](size_t i_p) // i_p is point index.
  {
    double sum{ 0 };
    for (const auto i : Range(size()))
      if (clusters_ind[i] == clusters_ind[i_p]) // If they are in the same cluster
        sum += dist_by_ind(static_cast<index_t>(i_p), static_cast<index_t>(i));

    pointCosts[i_p] = sum;
  };

  run(findBetterMedoidTask, size());

  std::vector<double> clusterCosts(n_clusters(), std::numeric_limits<double>::max());
  for (const auto i : Range(size()))
    if (pointCosts[i] < clusterCosts[clusters_ind[i]]) {
      clusterCosts[clusters_ind[i]] = pointCosts[i];
      centroids_ind[clusters_ind[i]] = static_cast<index_t>(i);
    }
}

void Problem::init_with_seed(std::uint64_t seed)
{
  using initializer_t = void (*)(Problem &);
  const auto target = init_fun.target<initializer_t>();
  if (target != nullptr && *target == &init::random) {
    init::random_seeded(*this, seed);
    return;
  }
  if (target != nullptr && *target == &init::Kmeanspp) {
    init::Kmeanspp_seeded(*this, seed);
    return;
  }

  // `init_fun` is a public extension point. An arbitrary callback has no seed
  // parameter, so retain its exact legacy invocation semantics rather than
  // silently replacing it with the default initializer.
  init();
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
