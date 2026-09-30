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

bool variant_params_equal(
  const core::DTWVariantParams &a, const core::DTWVariantParams &b) noexcept
{
  return a.variant == b.variant
      && a.wdtw_g == b.wdtw_g
      && a.adtw_penalty == b.adtw_penalty
      && a.sdtw_gamma == b.sdtw_gamma
      && a.msm_c == b.msm_c
      && a.twe_nu == b.twe_nu
      && a.twe_lambda == b.twe_lambda
      && a.mv_mode == b.mv_mode;
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
 * @brief Resizes data structures based on the current number of clusters.
 *
 * @details Adjusts the size of cluster_members, centroids_ind, and clusters_ind arrays based on the current
 * value of Nc (number of clusters).
 */
void Problem::resize()
{
  clusters_ind.resize(size());
  centroids_ind.resize(n_clusters());
}

/**
 * @brief Sets the number of clusters for the problem.
 *
 * @param Nc_ The number of clusters to set.
 */
void Problem::set_n_clusters(int Nc_)
{
  Nc = Nc_;
  resize();
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
void Problem::set_clusters(std::vector<int> &candidate_centroids)
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
  for (const int medoid : medoids) {
    if (medoid < 0 || static_cast<std::size_t>(medoid) >= n || is_medoid[static_cast<std::size_t>(medoid)])
      throw InvalidInput("Problem::set_result: medoid " + std::to_string(medoid)
                         + " is outside [0, N) or repeated.");
    is_medoid[static_cast<std::size_t>(medoid)] = true;
  }
  const int k = static_cast<int>(medoids.size());
  for (const int label : labels)
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
  validate_mmap_cache_identity();
  validate_dense_cache_configuration();
  std::cout << distMat;
}

/**
 * @brief Refreshes the distance matrix.
 * @details Resets state and rebinds the DTW function. Does NOT allocate the
 * dense N×N matrix — that is deferred to fill_distance_matrix() so that
 * large-N algorithms (e.g. FastCLARA) can load data without forcing
 * quadratic memory usage.
 *
 * If the matrix was previously allocated (e.g. from a prior fill_distance_matrix()
 * call), it is reset to size 0 so that stale entries are not reused after a
 * variant or data change.
 */
void Problem::refresh_distance_matrix()
{
  // Every known semantic error must precede cache release or mmap detachment.
  // Raw public-field edits remain caller-owned and recoverable: correcting the
  // edit exposes the last valid cache/callable state again.
  preflight_current_distance_semantics();
  // A semantic mutation (set_data/set_band/set_variant) must never keep
  // distances computed under the prior configuration. The heap matrix is
  // released (re-allocated by fill_distance_matrix()); a mapped one is detached
  // and its file left as it is, so rebinding it under the changed semantics
  // fails its fingerprint check loudly.
  distMat = core::DistanceMatrix{};
  clear_mmap_cache_identity();
  fill_request_validated_ = false; // new data or semantics: a new request
  rebind_dtw_fn();
}

void Problem::refresh_variant_caches() const
{
  wdtw_weights_cache_.clear();

  if (variant_params.variant != core::DTWVariant::WDTW || data_.size() == 0)
    return;

  const auto g = static_cast<data_t>(variant_params.wdtw_g);

  // Precompute WDTW weights for every unique max_dev that can arise.
  // max_dev = max(len_x, len_y) for univariate, max(steps_x, steps_y)-1 for MV.
  // We precompute for every unique series length so the parallel DTW lambda
  // never mutates the cache. Thread-safe by design: no insertion after this point.
  if (data_.ndim > 1) {
    for (size_t i = 0; i < data_.size(); ++i) {
      const size_t steps = data_.series_flat_size(i) / data_.ndim;
      if (steps == 0) continue;
      const size_t max_dev = steps - 1;
      wdtw_weights_cache_.try_emplace(max_dev, wdtw_weights<data_t>(static_cast<int>(max_dev), g));
    }
    return;
  }

  for (size_t i = 0; i < data_.size(); ++i) {
    const size_t len = data_.series_flat_size(i);
    if (len == 0) continue;
    // max_dev = len - 1 to match the canonical wdtwBanded(x, y, band, g)
    // convention (Jeong et al. 2011). Dispatch lambda uses the same key.
    const size_t max_dev = len - 1;
    wdtw_weights_cache_.try_emplace(max_dev, wdtw_weights<data_t>(static_cast<int>(max_dev), g));
  }
}

/**
 * @brief Rebind the DTW distance function based on current variant_params and band.
 */
void Problem::rebind_dtw_fn() const
{
  // Resolve dispatch once, here, at rebind time. The returned std::function
  // reads mutable members (band, variant_params, missing_strategy, ndim,
  // wdtw_weights_cache_) at call time via a reference to `*this`. Problem moves
  // transfer the function object, so the public compute/access gateways repair
  // that reference before invoking it.
  //
  // Historical note: previously this function was a ~130-line nested switch
  // that also silently bound dtw_fn_f32_ to Standard DTW regardless of the
  // configured variant/missing strategy (fast_clara's chunked-Parquet path
  // hit this). Both f64 and f32 now share core::resolve_dtw_fn.
  preflight_current_distance_semantics();
  refresh_variant_caches();
  dtw_fn_ = core::resolve_dtw_fn<data_t>(*this);
  dtw_block_fn_ = core::resolve_dtw_block_fn<data_t>(*this);
  if (core::active_variant_params_representable_f32(variant_params)) {
    dtw_fn_f32_ = core::resolve_dtw_fn<float>(*this);
    dtw_block_fn_f32_ = core::resolve_dtw_block_fn<float>(*this);
  } else {
    dtw_fn_f32_ = {};
    dtw_block_fn_f32_ = {};
  }
  dense_cache_configuration_ = distance_cache_configuration(metric_);
  dense_cache_configuration_bound_ = true;
  dtw_binding_owner_ = this;
}

void Problem::set_variant(core::DTWVariant v)
{
  auto candidate = variant_params;
  candidate.variant = v;
  preflight_distance_semantics(
    candidate, missing_strategy, metric_, data_);
  if (variant_params.variant == v) return;
  variant_params.variant = v;
  refresh_distance_matrix(); // calls rebind_dtw_fn() internally
}

void Problem::set_variant(core::DTWVariantParams params)
{
  preflight_distance_semantics(
    params, missing_strategy, metric_, data_);
  if (variant_params_equal(variant_params, params)) return;
  variant_params = params;
  refresh_distance_matrix(); // calls rebind_dtw_fn() internally
}

void Problem::set_device(Device device, int index)
{
  if (index < 0)
    throw InvalidInput("Problem::set_device: the GPU index must be >= 0; got "
                       + std::to_string(index) + ".");
  switch (device) {
  case Device::CPU:
    // Auto is the CPU brute-force fill, never a GPU.
    if (distance_strategy == DistanceMatrixStrategy::CUDA
        || distance_strategy == DistanceMatrixStrategy::Metal)
      set_distance_strategy(DistanceMatrixStrategy::Auto);
    return;
  case Device::GPU: {
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
    auto settings = cuda_settings;
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

Problem::cache_fingerprint_t
Problem::distance_cache_configuration_fingerprint(core::MetricType metric) const
{
  FingerprintHash hash;
  static constexpr char domain[] = "dtwc-distance-cache-configuration-v1";
  hash.update(domain, sizeof(domain) - 1);

  // Distance semantics. All variant parameters are included, even when
  // inactive for the selected variant. This deliberately prefers a harmless
  // cache miss over trusting distances after an ambiguous configuration edit.
  hash_enum(hash, metric);
  hash_u64(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(band)));
  hash_enum(hash, variant_params.variant);
  hash_double(hash, variant_params.wdtw_g);
  hash_double(hash, variant_params.adtw_penalty);
  hash_double(hash, variant_params.sdtw_gamma);
  hash_double(hash, variant_params.msm_c);
  hash_double(hash, variant_params.twe_nu);
  hash_double(hash, variant_params.twe_lambda);
  hash_enum(hash, variant_params.mv_mode);
  hash_enum(hash, missing_strategy);

  // Backend/precision can change the stored numeric result even when the
  // mathematical recurrence is the same (notably GPU FP32 versus CPU FP64).
  hash_enum(hash, distance_strategy);
  hash_u64(hash, static_cast<std::uint64_t>(
                   static_cast<std::int64_t>(cuda_settings.device_id)));
  hash_u64(hash, static_cast<std::uint64_t>(
                   static_cast<std::int64_t>(cuda_settings.precision)));

  return hash.digest();
}

Problem::DistanceCacheConfiguration
Problem::distance_cache_configuration(core::MetricType metric) const
{
  return {
    metric,
    band,
    variant_params,
    missing_strategy,
    distance_strategy,
    cuda_settings.device_id,
    cuda_settings.precision
  };
}

bool Problem::distance_cache_configuration_matches(
  const DistanceCacheConfiguration &expected) const
{
  return expected.metric == metric_
      && expected.band == band
      && variant_params_equal(expected.variant_params, variant_params)
      && expected.missing_strategy == missing_strategy
      && expected.distance_strategy == distance_strategy
      && expected.cuda_device_id == cuda_settings.device_id
      && expected.cuda_precision == cuda_settings.precision;
}

bool Problem::dense_cache_configuration_is_current() const
{
  return dense_cache_configuration_bound_
      && distance_cache_configuration_matches(dense_cache_configuration_);
}

void Problem::preflight_distance_semantics(
  const core::DTWVariantParams &params,
  core::MissingStrategy missing,
  core::MetricType metric,
  const Data &candidate_data,
  bool force_float32)
{
  core::validate_problem_distance_semantics(
    params, missing, candidate_data.ndim,
    force_float32 || candidate_data.is_f32());
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

void Problem::preflight_current_distance_semantics() const
{
  preflight_distance_semantics(
    variant_params, missing_strategy, metric_, data_);
}

void Problem::preflight_float32_distance_semantics() const
{
  preflight_distance_semantics(
    variant_params, missing_strategy, metric_, data_,
    true);
}

const Problem::dtw_fn_f32_t &Problem::validated_dtw_function_f32() const
{
  preflight_float32_distance_semantics();
  if (!dtw_fn_f32_) {
    throw std::logic_error(
      "Problem: float32 DTW function is unavailable despite representable "
      "active variant parameters.");
  }
  return dtw_fn_f32_;
}

void Problem::repair_dtw_binding_after_relocation()
{
  if (dtw_binding_owner_ == this) return;

  // A default move transfers every state field without a maintenance list, but
  // resolve_dtw_fn's closures still name the source address. Preserve a valid
  // moved distance matrix when its semantic snapshot is current. If raw public
  // configuration already drifted before the move, retain the existing cache
  // invalidation contract instead of blessing stale bits with a fresh snapshot.
  if (dense_cache_configuration_is_current())
    rebind_dtw_fn();
  else
    refresh_distance_matrix();
}

void Problem::ensure_dense_cache_configuration_current()
{
  preflight_current_distance_semantics();
  ensure_dense_cache_configuration_current_preflighted();
}

void Problem::ensure_dense_cache_configuration_current_preflighted()
{
  repair_dtw_binding_after_relocation();
  if (distMat.is_mapped() || dense_cache_configuration_is_current())
    return;

  // Public fields remain source-compatible, and nested language-binding
  // objects can be mutated without invoking a whole-property setter. Treat
  // detected drift exactly like an explicit semantic setter.
  refresh_distance_matrix();
}

void Problem::validate_dense_cache_configuration() const
{
  preflight_current_distance_semantics();
  if (distMat.is_mapped() || dense_cache_configuration_is_current())
    return;

  throw InvalidInput(
    "Distance matrix: cached distance configuration changed through a raw "
    "or nested mutation. Use a semantic setter or a non-const compute path to "
    "refresh the matrix before reading cached values.");
}

void Problem::ensure_dtw_function_configuration_current()
{
  preflight_current_distance_semantics();
  repair_dtw_binding_after_relocation();
  if (dense_cache_configuration_is_current()) return;

  // The fixed-size M25 snapshot records every input used when the dispatcher
  // was last bound. Reconcile legacy public-field edits exactly like a semantic
  // setter: discard any distance cache whose values now describe old semantics
  // (including a mapped cache), refresh variant-specific state, and bind both
  // precisions to the current configuration.
  refresh_distance_matrix();
}

void Problem::validate_dtw_function_configuration() const
{
  preflight_current_distance_semantics();

  // Relocation changes only the address captured by derived dispatch state.
  // Repairing that state is logically const and preserves a semantically
  // current distance cache. True raw-configuration drift remains a loud const
  // error below.
  if (dtw_binding_owner_ != this
      && dense_cache_configuration_is_current()) {
    rebind_dtw_fn();
  }
  if (dense_cache_configuration_is_current()) return;

  throw InvalidInput(
    "Problem: bound DTW function configuration changed through a raw or nested "
    "mutation. Use a semantic setter or a mutable dtw_function accessor to "
    "refresh the dispatcher before const access.");
}

Problem::DistanceCacheIdentity
Problem::distance_cache_identity(core::MetricType metric) const
{
  if (distance_strategy == DistanceMatrixStrategy::CUDA
      && cuda_settings.precision == GpuPrecision::Auto) {
    throw InvalidInput(
      "use_mmap_distance_matrix: CUDA precision=Auto is not safe for persistent "
      "warm-start caches because its resolved FP32/FP64 semantics depend on the "
      "runtime GPU. Select explicit FP32 or FP64 before binding the cache.");
  }

  DistanceCacheIdentity identity;
  identity.configuration_values = distance_cache_configuration(metric);
  identity.precision = data_.precision;
  identity.n = data_.size();
  identity.ndim = data_.ndim;
  identity.configuration = distance_cache_configuration_fingerprint(metric);

  FingerprintHash hash;
  static constexpr char domain[] = "dtwc-distance-cache-fingerprint-v1";
  hash.update(domain, sizeof(domain) - 1);
  hash.update(identity.configuration);

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

  identity.full = hash.digest();
  return identity;
}

core::DistanceMatrix::fingerprint_type
Problem::distance_checkpoint_identity() const
{
  return distance_checkpoint_identity(metric_);
}

core::DistanceMatrix::fingerprint_type
Problem::distance_checkpoint_identity(core::MetricType metric) const
{
  preflight_current_distance_semantics();
  // The metric is part of the identity: without it a SquaredL2 run writes the
  // same fingerprint as an L1 run over the same data, and a later L1 run then
  // accepts the wrong matrix.
  return distance_cache_identity(metric).full;
}

void Problem::clear_mmap_cache_identity()
{
  mmap_cache_identity_bound_ = false;
  mmap_cache_data_validated_ = false;
  mmap_cache_identity_ = DistanceCacheIdentity{};
}

void Problem::validate_mmap_cache_identity() const
{
  preflight_current_distance_semantics();
  if (!distMat.is_mapped()) return;
  if (!mmap_cache_identity_bound_) {
    throw InvalidInput(
      "Mapped distance matrix: mapped storage has no bound Problem cache identity");
  }

  // The file's fingerprint is mmap_cache_identity_.full: map() checked it when
  // use_mmap_distance_matrix bound the two. What can drift since is the data
  // and the distance settings.
  if (data_.size() != mmap_cache_identity_.n
      || data_.ndim != mmap_cache_identity_.ndim
      || data_.precision != mmap_cache_identity_.precision
      || !distance_cache_configuration_matches(
        mmap_cache_identity_.configuration_values)) {
    throw InvalidInput(
      "Mapped distance matrix: bound cache fingerprint mismatch after Problem data "
      "or distance configuration changed. Call refresh_distance_matrix(), then "
      "bind a cache created for the new semantics.");
  }

  // Exactly one full data hash starts a bound-cache use session. Subsequent
  // cached lookups retain their O(1) contract and compare only the fixed-size
  // configuration snapshot above. Semantic setters detach the cache. A caller
  // can still mutate backing storage referenced by view-mode Data; such
  // external edits after this point are unsupported and require an explicit
  // refresh_distance_matrix() before the edit.
  if (!mmap_cache_data_validated_) {
    const DistanceCacheIdentity current = distance_cache_identity(
      mmap_cache_identity_.configuration_values.metric);
    if (current.full != mmap_cache_identity_.full) {
      throw InvalidInput(
        "Mapped distance matrix: bound cache fingerprint mismatch after Problem data "
        "changed before first use. Call refresh_distance_matrix(), then bind a "
        "cache created for the new data.");
    }
    mmap_cache_data_validated_ = true;
  }
}

void Problem::use_mmap_distance_matrix(const std::filesystem::path &cache_path)
{
  use_mmap_distance_matrix(cache_path, metric_);
}

void Problem::use_mmap_distance_matrix(
  const std::filesystem::path &cache_path, core::MetricType metric)
{
  // Every check, the identity and the mapping come before any change: a bind
  // that fails leaves this Problem's metric and matrix as they were.
  preflight_distance_semantics(
    variant_params, missing_strategy, metric, data_);
  // Reconcile dispatcher semantics before publishing a new mapped identity.
  // Without this generic guard, replacing an already-bound mmap after a raw
  // configuration mutation could label Standard-DTW writes with an ADTW (or
  // missing-policy) fingerprint.
  ensure_dtw_function_configuration_current();
  DistanceCacheIdentity identity = distance_cache_identity(metric);
  // map() checks an existing file's magic, version, length, N and fingerprint
  // before any of its distances can be read.
  auto mapped = core::DistanceMatrix::map(cache_path, data_.size(), identity.full);
  if (metric_ != metric) { // new semantics, as in set_metric
    metric_ = metric;
    fill_request_validated_ = false;
    rebind_dtw_fn();
  }
  distMat = std::move(mapped);
  mmap_cache_identity_ = identity;
  mmap_cache_identity_bound_ = true;
  mmap_cache_data_validated_ = false;
}

/**
 *@brief Retrieves or calculates the distance between two points by their indices.
 *@param i Index of the first point.
 *@param j Index of the second point.
 *@return The distance between the two points.
 *
 *@note Thread safety: the lazy-alloc + compute path is NOT thread-safe.
 *      Call fill_distance_matrix() before entering any parallel region.
 *      After that, all calls are read-only lookups (no race by design).
 *      A bound mmap cache's first-use data validation also initializes its
 *      session flag; perform fill_distance_matrix() or
 *      is_distance_matrix_filled() once serially before parallel lookups.
 */
double Problem::dist_by_ind(int i, int j)
{
  // Exactly ONE preflight per call: the SWAP kernel issues N^2 of these per
  // iteration. Order (preflight → mmap identity → dense-cache) is load-bearing.
  preflight_current_distance_semantics();
  validate_mmap_cache_identity();
  ensure_dense_cache_configuration_current_preflighted();
  if (i == j) return 0.0;

  // The lazy path starts at the first off-diagonal call after a
  // (re)configuration, cached pair or not, so a caller's serial prime (see
  // below) validates before any parallel region. Once per configuration, never
  // per pair; a fill or dtw_function accessor that validated sets the flag. A
  // dense matrix with every pair known (a loaded file or checkpoint, one
  // written through the matrix accessors) computes nothing, so nothing in it
  // can be infeasible; installing or editing a matrix re-arms the check. A
  // mapped cache is always checked: its identity binds the band, so only this
  // library's kernels under the same request filled it, and proving it
  // complete would read the whole mapped file.
  if (!fill_request_validated_) {
    const bool complete =
      !distMat.is_mapped() && distMat.size() == data_.size() && distMat.all_computed();
    if (!complete) validate_fill_request("Problem::dist_by_ind");
    fill_request_validated_ = true;
  }

  const size_t N = data_.size();

  // Lazily allocate the heap matrix on first individual distance request (a
  // mapped matrix is sized when it is bound). The critical section prevents
  // duplicate allocation. Callers that enter a parallel region must still prime
  // one non-diagonal distance serially first (or call fill_distance_matrix),
  // because rebind_dtw_fn mutates shared state.
  if (distMat.size() != N) {
#ifdef _OPENMP
    #pragma omp critical(distByInd_init)
#endif
    {
      if (distMat.size() != N) {
        distMat.resize(N);
        rebind_dtw_fn();
      }
    }
  }

  if (distMat.is_computed(i, j)) return distMat.get(i, j);

  const double d = data_.is_f32()
                     ? validated_dtw_function_f32()(data_.series_f32(i), data_.series_f32(j))
                     : dtw_fn_(series(i), series(j));
  distMat.set(i, j, d);
  return d;
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
  const auto variant = variant_params.variant;
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
  const bool nan_is_missing = missing_strategy != core::MissingStrategy::Error;
  std::string name;
  for (std::size_t i = 0; i < data_.size(); ++i) {
    name.assign("series '").append(series_name(i)).append("' (index ")
      .append(std::to_string(i)).append(")");
    if (data_.is_f32())
      detail::require_finite(data_.series_f32(i), name, at, nan_is_missing);
    else
      detail::require_finite(series(i), name, at, nan_is_missing);
  }
  if (missing_strategy == core::MissingStrategy::Interpolate) {
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
  const bool cuda = distance_strategy == DistanceMatrixStrategy::CUDA;
  if (!cuda && distance_strategy != DistanceMatrixStrategy::Metal) return;
  if (data_.is_view())
    reject_gpu_request(at, cuda,
                       "needs owned series in RAM, but this Problem's series are a non-owning "
                       "view (set_view_data, as FastCLARA's in-memory subsamples are)",
                       "Install owning series with set_data, or use device cpu.");
  validate_gpu_request(at, distance_strategy, variant_params, missing_strategy, data_.precision,
                       cuda_settings);
  if (data_.ndim > 1)
    reject_gpu_request(at, cuda, "is univariate only, but ndim = " + std::to_string(data_.ndim) + " was requested",
                       "Use device cpu for multivariate series.");
}

void Problem::validate_fill_request_once(std::string_view where) const
{
  if (fill_request_validated_) return;
  validate_fill_request(where);
  fill_request_validated_ = true;
}

/**
 * @brief Fills the distance matrix using brute-force parallel computation.
 * @details Original implementation: parallel loop over all upper-triangle pairs
 *          using the bound dtw_fn_ (supports all DTW variants).
 */
void Problem::fillDistanceMatrix_BruteForce()
{
  const size_t N = data_.size();
  const dtw_fn_f32_t *f32_function = data_.is_f32()
                                       ? &validated_dtw_function_f32()
                                       : nullptr;

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
  // each of those distances is bitwise the per-pair one. The block at the row's
  // end repeats its last column in the lanes past it, whose results are dropped.
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
  preflight_current_distance_semantics();
  validate_mmap_cache_identity();
  ensure_dense_cache_configuration_current();
  if (is_distance_matrix_filled()) return;
  validate_fill_request("Problem::fill_distance_matrix");
  fill_request_validated_ = true; // the same request needs no lazy re-check

  // Allocate the heap N×N matrix on first call (deferred from set_data /
  // refresh_distance_matrix); a mapped matrix is sized when it is bound.
  if (distMat.size() != data_.size()) distMat.resize(data_.size());

  // Re-bind the DTW function in case missing_strategy was changed after construction
  // (e.g., user sets prob.missing_strategy = ZeroCost after prob.set_data(...)).
  rebind_dtw_fn();

  if (verbose_)
    std::cout << "Distance matrix is being filled!" << '\n';

  // The serial missing-data pre-scan is part of validate_fill_request (FX-15),
  // above: it runs before any pair, here and on every other entry point.

  DistanceMatrixStrategy effective = distance_strategy;
  if (effective == DistanceMatrixStrategy::Auto)
    effective = DistanceMatrixStrategy::BruteForce;

  // Shared post-GPU handler. Templated on the backend's result type (both
  // CUDADistMatResult and MetalDistMatResult derive from gpu::DistMatResultBase,
  // so any base accessor works). Returns true on success; false signals the
  // caller to continue. An explicitly requested backend never changes to CPU.
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  auto dispatch_gpu_backend = [&](const auto &result, const char *backend) -> bool {
    if (result.pairs_computed == 0 && data_.size() > 1) {
      throw DeviceError(std::string(backend)
                        + " returned no distance pairs. No CPU fallback was attempted.");
    }
    if (distMat.size() != result.n) distMat.resize(result.n);
    for (size_t i = 0; i < result.n; ++i)
      for (size_t j = i; j < result.n; ++j)
        distMat.set(i, j, result.matrix[i * result.n + j]);
    return true;
  };
#endif

  switch (effective) {
  case DistanceMatrixStrategy::CUDA:
#ifdef DTWC_HAS_CUDA
  {
    dtwc::cuda::CUDADistMatOptions cuda_opts;
    cuda_opts.band = band;
    cuda_opts.device_id = cuda_settings.device_id;
    if (cuda_settings.precision == GpuPrecision::FP32)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    else if (cuda_settings.precision == GpuPrecision::FP64)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
    // L2 is L1 on the univariate series the GPU routes take.
    cuda_opts.use_squared_l2 = metric_ == core::MetricType::SquaredL2;
    cuda_opts.verbose = verbose_;

    auto cuda_result = dtwc::cuda::compute_distance_matrix_cuda(
      data_.p_vec, cuda_opts);
    (void)dispatch_gpu_backend(cuda_result, "CUDA");
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
    metal_opts.band = band;
    metal_opts.precision = metal_precision(cuda_settings.precision);
    metal_opts.use_squared_l2 = metric_ == core::MetricType::SquaredL2;
    metal_opts.verbose = verbose_;

    auto metal_result = dtwc::metal::compute_distance_matrix_metal(
      data_.p_vec, metal_opts);
    (void)dispatch_gpu_backend(metal_result, "Metal");
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
  std::vector<int> labels(data_.size());
  auto assignClustersTask = [this, &labels](size_t i_p) //!< i_p and i_c in [0, Np)
  {
    const int ip = static_cast<int>(i_p);
    double best_distance = std::numeric_limits<double>::max();
    int best_slot = 0;
    bool has_best = false;
    for (std::size_t slot = 0; slot < centroids_ind.size(); ++slot) {
      const int medoid = centroids_ind[slot];
      const double distance = core::detail::require_finite_medoid_distance(
        dist_by_ind(ip, medoid), "kmedoids_lloyd", i_p,
        static_cast<int>(slot), medoid);
      // A medoid tied with another medoid (a duplicate series) serves itself,
      // or its own cluster would be published empty.
      if (!has_best || distance < best_distance || (distance == best_distance && medoid == ip)) {
        best_distance = distance;
        best_slot = static_cast<int>(slot);
        has_best = true;
      }
    }
    labels[i_p] = best_slot;
  };

  // If the full matrix is not materialised yet, dist_by_ind() may lazily compute
  // symmetric entries on demand. Different points can request the same packed
  // (i,j)/(j,i) slot concurrently, so the lazy-compute path is not safe to run
  // in parallel. Once fill_distance_matrix() has completed, all lookups are
  // read-only and the parallel path is safe again.
  const size_t workers = is_distance_matrix_filled() ? 32u : 1u;
  run(assignClustersTask, data_.size(), workers);
  clusters_ind = std::move(labels);
}

/**
 * @brief Calculates the pairwise distances within each cluster.
 * @details Iterates through each data point, determining its cluster and calculating the distance to other points
 * within the same cluster. This method populates the distance matrix with these intra-cluster distances.
 */
void Problem::distanceInClusters()
{
  auto distanceInClustersTask = [&, N = size()](size_t i_p) {
    const int clusterNo{ clusters_ind[i_p] };
    for (size_t i{ i_p }; i < N; i++)
      if (clusters_ind[i] == clusterNo) // If they are in the same cluster
        dist_by_ind(static_cast<int>(i_p), static_cast<int>(i));
  };

  run(distanceInClustersTask, size());
}

/**
 * @brief Calculates and updates the medoids of each cluster.
 * @details This function iterates through each data point and calculates the total cost of designating that point
 * as the medoid of its cluster. The point with the minimum total cost is set as the new medoid for that cluster.
 */
void Problem::calculate_medoids()
{
  std::vector<double> pointCosts(size());

  auto findBetterMedoidTask = [&](size_t i_p) // i_p is point index.
  {
    double sum{ 0 };
    for (const auto i : Range(size()))
      if (clusters_ind[i] == clusters_ind[i_p]) // If they are in the same cluster
        sum += dist_by_ind(static_cast<int>(i_p), static_cast<int>(i));

    pointCosts[i_p] = sum;
  };

  run(findBetterMedoidTask, size());

  std::vector<double> clusterCosts(n_clusters(), std::numeric_limits<double>::max());
  for (const auto i : Range(size()))
    if (pointCosts[i] < clusterCosts[clusters_ind[i]]) {
      clusterCosts[clusters_ind[i]] = pointCosts[i];
      centroids_ind[clusters_ind[i]] = static_cast<int>(i);
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
  std::vector<int> best_medoids;
  std::vector<int> best_labels;

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
  std::vector<std::vector<int>> centroids_all;

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
    distanceInClusters(); // Just populates distance matrix ahead.
    calculate_medoids();   // Changes centroids_ind

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
  core::detail::OrderedMedoidObjective total("kmedoids_lloyd");
  for (const auto idx : Range(size())) {
    const int i = static_cast<int>(idx);
    const int medoid_slot = clusters_ind[i];
    const int medoid_index = centroids_ind[medoid_slot];
    const double distance = core::detail::require_finite_medoid_distance(
      dist_by_ind(i, medoid_index), "kmedoids_lloyd",
      idx, medoid_slot, medoid_index);
    if constexpr (settings::isDebug)
      std::cout << "Distance between " << i << " and closest cluster " << clusters_ind[i]
                << " which is: " << distance << "\n";

    // k-medoids objective: sum of raw DTW distances (not squared, unlike k-means).
    total.add(distance);
  }

  return total.value();
}

} // namespace dtwc
