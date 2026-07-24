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
#include "error.hpp"           // for DeviceError
#include "mip.hpp"             // for MIP_clustering_byGurobi, MIP_clustering_byBenders
#include "parallelisation.hpp" // for run
#include "scores.hpp"          // for silhouette
#include "settings.hpp"        // for data_t, randGenerator, band, isDebug
#include "core/matrix_io.hpp"  // for operator<<(ostream, DenseDistanceMatrix)
#include "core/medoid_assignment_policy.hpp" // finite assignment contract

#ifdef DTWC_HAS_CUDA
#include "cuda/cuda_dtw.cuh"   // GPU distance matrix computation
#endif
#ifdef DTWC_HAS_METAL
#include "metal/metal_dtw.hpp" // Apple Metal GPU distance matrix computation
#endif
#include "warping_wdtw.hpp"    // for wdtw_weights (cache population)
#include "types/Range.hpp"     // for Range
#include "initialisation.hpp"  // For initialisation functions
#include "core/dtw_dispatch.hpp"           // for resolve_dtw_fn
#include "core/distance_semantics.hpp"      // validate_problem_distance_semantics
#include "core/variant_validation.hpp"     // validate_variant_params
#include "core/pruned_distance_matrix.hpp" // for fill_distance_matrix_pruned
#include "core/sha256.hpp"                 // for persistent cache fingerprints
#include "missing_utils.hpp"               // for has_missing
#include "algorithms/tadpole.hpp"          // for Method::TADPole dispatch


#include <algorithm> // for max_element, min, min_element, sample
#include <array>     // for array
#include <cmath>     // for sqrt, floor
#include <cstdint>   // for uint32_t, uint64_t
#include <cstring>   // for memcpy
#include <iomanip>   // for operator<<, setprecision
#include <iostream>  // for cout
#include <iterator>  // for back_insert_iterator, back_inserter
#include <limits>    // for numeric_limits
#include <random>    // for mt19937, discrete_distribution, unifo...
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

} // namespace

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
 * @throws std::runtime_error if the size of candidate_centroids is not equal to Nc.
 */
void Problem::set_n_clusters(int Nc_)
{
  Nc = Nc_;
  resize();
}

/**
 * @brief Sets the initial centroids for clustering.
 *
 * @param candidate_centroids A vector containing the indices of candidate centroids.
 */
void Problem::set_clusters(std::vector<int> &candidate_centroids)
{
  if (candidate_centroids.size() != static_cast<size_t>(Nc))
    throw std::runtime_error("Set cluster has failed as number of centroids is not same as the number of indices in candidate centroids vector.\n");

  centroids_ind = candidate_centroids;
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
  validate_solver(solver_);
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
  visit_distmat([](const auto &m) { std::cout << m; });
}

/**
 * @brief Refreshes the distance matrix.
 * @details Resets state and rebinds the DTW function. Does NOT allocate the
 * dense N×N matrix — that is deferred to fillDistanceMatrix() so that
 * large-N algorithms (e.g. FastCLARA) can load data without forcing
 * quadratic memory usage.
 *
 * If the matrix was previously allocated (e.g. from a prior fillDistanceMatrix()
 * call), it is reset to size 0 so that stale entries are not reused after a
 * variant or data change.
 */
void Problem::refresh_distance_matrix()
{
  // Every known semantic error must precede cache release or mmap detachment.
  // Raw public-field edits remain caller-owned and recoverable: correcting the
  // edit exposes the last valid cache/callable state again.
  preflight_current_distance_semantics();
  if (std::holds_alternative<core::MmapDistanceMatrix>(distMat)) {
    // A semantic mutation (set_data/set_band/set_variant) must never keep a
    // mapped matrix whose computed bits describe the prior configuration.
    // Detach without deleting or rewriting the persistent file; rebinding it
    // under changed semantics will then fail its fingerprint check loudly.
    distMat = core::DenseDistanceMatrix{};
    clear_mmap_cache_identity();
  } else {
    auto &m = std::get<core::DenseDistanceMatrix>(distMat);
    if (m.size() != 0)
      m.resize(0); // Release old data; re-allocation deferred to fillDistanceMatrix().
  }
  rebind_dtw_fn();
}

void Problem::refresh_variant_caches()
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
void Problem::rebind_dtw_fn()
{
  // Resolve dispatch once, here, at rebind time. The returned std::function
  // reads mutable members (band, variant_params, missing_strategy, ndim,
  // wdtw_weights_cache_) at call time via a stable reference to `*this`, so
  // changing e.g. `prob.band = 50` after construction takes effect without a
  // second rebind.
  //
  // Historical note: previously this function was a ~130-line nested switch
  // that also silently bound dtw_fn_f32_ to Standard DTW regardless of the
  // configured variant/missing strategy (fast_clara's chunked-Parquet path
  // hit this). Both f64 and f32 now share core::resolve_dtw_fn.
  preflight_current_distance_semantics();
  refresh_variant_caches();
  dtw_fn_ = core::resolve_dtw_fn<data_t>(*this);
  if (core::active_variant_params_representable_f32(variant_params))
    dtw_fn_f32_ = core::resolve_dtw_fn<float>(*this);
  else
    dtw_fn_f32_ = {};
  dense_cache_configuration_ = distance_cache_configuration(core::MetricType::L1);
  dense_cache_configuration_bound_ = true;
}

void Problem::set_variant(core::DTWVariant v)
{
  auto candidate = variant_params;
  candidate.variant = v;
  preflight_distance_semantics(
    candidate, missing_strategy, data_, distance_strategy, cuda_settings);
  if (variant_params.variant == v) return;
  variant_params.variant = v;
  refresh_distance_matrix(); // calls rebind_dtw_fn() internally
}

void Problem::set_variant(core::DTWVariantParams params)
{
  preflight_distance_semantics(
    params, missing_strategy, data_, distance_strategy, cuda_settings);
  if (variant_params_equal(variant_params, params)) return;
  variant_params = params;
  refresh_distance_matrix(); // calls rebind_dtw_fn() internally
}

Problem::cache_fingerprint_t
Problem::distance_cache_configuration_fingerprint(core::MetricType metric) const
{
  core::validate_metric_type(metric);
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
  core::validate_metric_type(metric);
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
  return expected.band == band
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
  const Data &candidate_data,
  DistanceMatrixStrategy candidate_distance_strategy,
  const CUDASettings &candidate_cuda_settings,
  bool force_float32)
{
  core::validate_precision(candidate_data.precision);
  validate_distance_matrix_strategy(candidate_distance_strategy);
  validate_cuda_settings_precision(candidate_cuda_settings.precision);
  core::validate_problem_distance_semantics(
    params, missing, candidate_data.ndim,
    force_float32 || candidate_data.is_f32());
}

void Problem::preflight_current_distance_semantics() const
{
  preflight_distance_semantics(
    variant_params, missing_strategy, data_, distance_strategy, cuda_settings);
}

void Problem::preflight_float32_distance_semantics() const
{
  preflight_distance_semantics(
    variant_params, missing_strategy, data_, distance_strategy, cuda_settings, true);
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

void Problem::ensure_dense_cache_configuration_current()
{
  preflight_current_distance_semantics();
  if (!std::holds_alternative<core::DenseDistanceMatrix>(distMat)
      || dense_cache_configuration_is_current())
    return;

  // Public fields remain source-compatible, and nested language-binding
  // objects can be mutated without invoking a whole-property setter. Treat
  // detected drift exactly like an explicit semantic setter.
  refresh_distance_matrix();
}

void Problem::validate_dense_cache_configuration() const
{
  preflight_current_distance_semantics();
  if (!std::holds_alternative<core::DenseDistanceMatrix>(distMat)
      || dense_cache_configuration_is_current())
    return;

  throw std::runtime_error(
    "DenseDistanceMatrix: cached distance configuration changed through a raw "
    "or nested mutation. Use a semantic setter or a non-const compute path to "
    "refresh the matrix before reading cached values.");
}

void Problem::ensure_dtw_function_configuration_current()
{
  preflight_current_distance_semantics();
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
  if (dense_cache_configuration_is_current()) return;

  throw std::runtime_error(
    "Problem: bound DTW function configuration changed through a raw or nested "
    "mutation. Use a semantic setter or a mutable dtw_function accessor to "
    "refresh the dispatcher before const access.");
}

Problem::DistanceCacheIdentity
Problem::distance_cache_identity(core::MetricType metric) const
{
  core::validate_metric_type(metric);
  if (data_.is_metadata_only()) {
    throw std::runtime_error(
      "use_mmap_distance_matrix: cannot fingerprint metadata-only data; "
      "time-series values must be resident before a distance cache can be bound");
  }
  if (distance_strategy == DistanceMatrixStrategy::CUDA
      && cuda_settings.precision == 0) {
    throw std::runtime_error(
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

core::MmapDistanceMatrix::fingerprint_type
Problem::distance_checkpoint_identity() const
{
  preflight_current_distance_semantics();
  return distance_cache_identity(core::MetricType::L1).full;
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
  if (!std::holds_alternative<core::MmapDistanceMatrix>(distMat)) return;
  if (!mmap_cache_identity_bound_) {
    throw std::runtime_error(
      "MmapDistanceMatrix: mapped storage has no bound Problem cache identity");
  }

  const auto &matrix = std::get<core::MmapDistanceMatrix>(distMat);
  if (matrix.fingerprint() != mmap_cache_identity_.full
      || data_.size() != mmap_cache_identity_.n
      || data_.ndim != mmap_cache_identity_.ndim
      || data_.precision != mmap_cache_identity_.precision
      || !distance_cache_configuration_matches(
        mmap_cache_identity_.configuration_values)) {
    throw std::runtime_error(
      "MmapDistanceMatrix: bound cache fingerprint mismatch after Problem data "
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
      throw std::runtime_error(
        "MmapDistanceMatrix: bound cache fingerprint mismatch after Problem data "
        "changed before first use. Call refresh_distance_matrix(), then bind a "
        "cache created for the new data.");
    }
    mmap_cache_data_validated_ = true;
  }
}

void Problem::use_mmap_distance_matrix(
  const std::filesystem::path &cache_path, core::MetricType metric)
{
  core::validate_metric_type(metric);
  preflight_current_distance_semantics();
  // Reconcile dispatcher semantics before publishing a new mapped identity.
  // Without this generic guard, replacing an already-bound mmap after a raw
  // configuration mutation could label Standard-DTW writes with an ADTW (or
  // missing-policy) fingerprint.
  ensure_dtw_function_configuration_current();
  const size_t N = data_.size();
  DistanceCacheIdentity identity = distance_cache_identity(metric);
  if (std::filesystem::exists(cache_path)) {
    // open(path, expected) validates version, header integrity, length, and the
    // full semantic fingerprint before exposing the mapped computed-bit region.
    distMat = core::MmapDistanceMatrix::open(cache_path, identity.full);
    auto &m = std::get<core::MmapDistanceMatrix>(distMat);
    if (m.size() != N)
      throw std::runtime_error("Mmap cache N=" + std::to_string(m.size())
                               + " != data N=" + std::to_string(N));
  } else {
    distMat = core::MmapDistanceMatrix(cache_path, N, identity.full);
  }
  mmap_cache_identity_ = std::move(identity);
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
 *      Call fillDistanceMatrix() before entering any parallel region.
 *      After that, all calls are read-only lookups (no race by design).
 *      A bound mmap cache's first-use data validation also initializes its
 *      session flag; perform fill_distance_matrix() or
 *      is_distance_matrix_filled() once serially before parallel lookups.
 */
double Problem::dist_by_ind(int i, int j)
{
  preflight_current_distance_semantics();
  validate_mmap_cache_identity();
  ensure_dense_cache_configuration_current();
  if (i == j) return 0.0;

  const size_t N = data_.size();

  // Lazily allocate the dense matrix on first individual distance request.
  // MmapDistanceMatrix is pre-allocated at creation, so only Dense needs this.
  // The critical section prevents duplicate allocation. Callers that enter a
  // parallel region must still prime one non-diagonal distance serially first
  // (or call fillDistanceMatrix), because rebind_dtw_fn mutates shared state.
  bool needs_init = visit_distmat([&](const auto &m) { return m.size() != N; });
  if (needs_init) {
#ifdef _OPENMP
    #pragma omp critical(distByInd_init)
#endif
    {
      bool initialised_here = false;
      visit_distmat([&](auto &m) {
        if (m.size() != N) {
          if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
            m.resize(N);
            initialised_here = true;
          }
        }
      });
      if (initialised_here) rebind_dtw_fn();
    }
  }

  bool computed = visit_distmat([&](const auto &m) { return m.is_computed(i, j); });
  if (computed)
    return visit_distmat([&](const auto &m) { return m.get(i, j); });

  if (std::holds_alternative<core::MmapDistanceMatrix>(distMat)
      && mmap_cache_identity_.configuration_values.metric != core::MetricType::L1) {
    throw std::runtime_error(
      "MmapDistanceMatrix: a non-L1 cache is external-fill-only; lazy CPU "
      "dist_by_ind computes L1 and cannot populate this cache. Fill it through "
      "the matching GPU/backend producer before reading the pair.");
  }

  const double d = data_.is_f32()
                     ? validated_dtw_function_f32()(data_.series_f32(i), data_.series_f32(j))
                     : dtw_fn_(series(i), series(j));
  visit_distmat([&](auto &m) { m.set(i, j, d); });
  return d;
}

/**
 * @brief Determines whether the pruned distance matrix strategy is applicable.
 * @details The pruned strategy requires Standard/ADTW with MissingStrategy::Error
 *          (raw lower-bound kernels cannot implement a missing-data dispatcher).
 * @return true if pruned strategy can be used.
 */
static bool pruned_strategy_applicable(const Problem &prob, bool has_dense_storage)
{
  // LB_Keogh is a valid lower bound for Standard DTW and ADTW: ADTW penalties
  // only increase cost, so LB_Keogh(x,y) <= DTW(x,y) <= ADTW(x,y,penalty).
  const bool supported_variant = prob.variant_params.variant == core::DTWVariant::Standard
                               || prob.variant_params.variant == core::DTWVariant::ADTW;
  return supported_variant
      && prob.missing_strategy == core::MissingStrategy::Error
      && has_dense_storage
      && prob.band >= 0
      && prob.size() >= 64;
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

  // Resize (Dense only — mmap is pre-allocated at creation).
  visit_distmat([&](auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      m.resize(N);
    }
  });

  // Set diagonal to 0
  visit_distmat([&](auto &m) {
    for (size_t i = 0; i < N; ++i)
      if (!m.is_computed(i, i))
        m.set(i, i, 0.0);
  });

  // Lock-free by design: each worker owns a disjoint row. run_openmp catches
  // inside the structured block and deterministically rethrows the lowest-row
  // failure after the join; a failed pair remains uncomputed.
  auto fill_row = [&](size_t i) {
    if (data_.is_f32()) {
      const auto si = data_.series_f32(i);
      for (size_t j = i + 1; j < N; ++j) {
        bool computed = visit_distmat([&](const auto &m) { return m.is_computed(i, j); });
        if (!computed)
          visit_distmat([&](auto &m) {
            m.set(i, j, (*f32_function)(si, data_.series_f32(j)));
          });
      }
    } else {
      const auto si = series(i);
      for (size_t j = i + 1; j < N; ++j) {
        bool computed = visit_distmat([&](const auto &m) { return m.is_computed(i, j); });
        if (!computed)
          visit_distmat([&](auto &m) { m.set(i, j, dtw_fn_(si, series(j))); });
      }
    }
  };
  run_openmp(fill_row, N, true, 8);
}

/**
 * @brief Fills the distance matrix by computing distances between all pairs of points.
 * @details Uses a strategy-based approach:
 *   - Auto: selects Pruned for Standard DTW variant, BruteForce otherwise.
 *   - BruteForce: parallel brute-force (all variants).
 *   - Pruned: parallel with LB_Kim + LB_Keogh early-abandon (Standard DTW only).
 * - CUDA: selected externally for NVIDIA GPU dispatch (e.g., via CLI).
 * - Metal: selected externally for Apple GPU dispatch.
 */
void Problem::fill_distance_matrix()
{
  preflight_current_distance_semantics();
  validate_lower_bound_strategy(lb_strategy_);
  validate_mmap_cache_identity();
  ensure_dense_cache_configuration_current();
  if (is_distance_matrix_filled()) return;

  if (std::holds_alternative<core::MmapDistanceMatrix>(distMat)
      && mmap_cache_identity_.configuration_values.metric != core::MetricType::L1) {
    throw std::runtime_error(
      "MmapDistanceMatrix: a non-L1 cache is external-fill-only; the Problem CPU "
      "fill path computes L1. Fill it through the matching GPU/backend producer.");
  }

  // Allocate the dense N×N matrix on first call (deferred from set_data / refreshDistanceMatrix).
  // MmapDistanceMatrix is pre-allocated at creation, so only Dense needs this.
  visit_distmat([&](auto &m) {
    if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
      if (m.size() != data_.size())
        m.resize(data_.size());
    }
  });

  // Re-bind the DTW function in case missing_strategy was changed after construction
  // (e.g., user sets prob.missing_strategy = ZeroCost after prob.set_data(...)).
  rebind_dtw_fn();

  if (verbose_)
    std::cout << "Distance matrix is being filled!" << '\n';

  // Pre-scan for NaN if strategy is Error
  if (missing_strategy == core::MissingStrategy::Error) {
    for (size_t i = 0; i < data_.size(); ++i) {
      const bool has_nan = data_.is_f32()
                             ? has_missing(data_.series_f32(i))
                             : has_missing(series(i));
      if (has_nan) {
        throw std::runtime_error(
          "fill_distance_matrix: NaN detected in series '" + std::string(series_name(i))
          + "' (index " + std::to_string(i)
          + "). Set missing_strategy to ZeroCost, AROW, or Interpolate to handle missing data.");
      }
    }
  }

  // Resolve Auto strategy
  const bool has_mmap_storage =
    std::holds_alternative<core::MmapDistanceMatrix>(distMat);
  DistanceMatrixStrategy effective = distance_strategy;
  if (effective == DistanceMatrixStrategy::Auto) {
    if (pruned_strategy_applicable(*this, !has_mmap_storage))
      effective = DistanceMatrixStrategy::Pruned;
    else
      effective = DistanceMatrixStrategy::BruteForce;
  }

  if (has_mmap_series_storage()
      && (effective == DistanceMatrixStrategy::CUDA
          || effective == DistanceMatrixStrategy::Metal)) {
    const char *backend =
      effective == DistanceMatrixStrategy::CUDA ? "CUDA" : "Metal";
    throw DeviceError(
      std::string("Problem::fill_distance_matrix: ") + backend
      + " does not support mmap-backed series data; no backend call or CPU "
        "fallback was attempted. Select StoragePolicy::Heap before set_data.");
  }

  // The pruned builder calls raw Standard/ADTW kernels; it cannot implement a
  // configured missing-data dispatcher. DistanceMatrixStrategy::Pruned is an
  // exact optimization hint (like lb_strategy=None below), so preserve the
  // requested distance semantics by routing the ordinary bound dispatcher.
  if (effective == DistanceMatrixStrategy::Pruned
      && missing_strategy != core::MissingStrategy::Error) {
    if (verbose_) {
      std::cout << "Pruned lower bounds support missing_strategy=Error only; "
                   "using exact BruteForce to preserve the configured missing-data policy.\n";
    }
    effective = DistanceMatrixStrategy::BruteForce;
  }

  // Lower-bound pruning currently writes DenseDistanceMatrix directly. Mapped
  // storage is still fully supported through the exact generic row fill.
  if (effective == DistanceMatrixStrategy::Pruned && has_mmap_storage) {
    if (verbose_) {
      std::cout << "Pruned strategy requires dense distance storage; using exact "
                   "BruteForce to fill the configured mmap distance matrix.\n";
    }
    effective = DistanceMatrixStrategy::BruteForce;
  }

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
    visit_distmat([&](auto &m) {
      if constexpr (std::is_same_v<std::decay_t<decltype(m)>, core::DenseDistanceMatrix>) {
        m.resize(result.n);
      }
      for (size_t i = 0; i < result.n; ++i)
        for (size_t j = i; j < result.n; ++j)
          m.set(i, j, result.matrix[i * result.n + j]);
    });
    if (verbose_) {
      std::cout << backend << " distance matrix: " << result.pairs_computed
                << " pairs in " << std::setprecision(3)
                << result.gpu_time_sec * 1000 << " ms";
      if (result.lb_time_sec > 0) {
        std::cout << " (LB_Keogh: " << result.lb_time_sec * 1000 << " ms"
                  << ", pruned " << result.pairs_pruned << ")";
      }
      std::cout << "\n";
    }
    return true;
  };
#endif

  switch (effective) {
  case DistanceMatrixStrategy::Pruned: {
    // LowerBoundStrategy::None inside Pruned path would compute all DTWs
    // without any pruning — that's exactly BruteForce, so short-circuit.
    if (lb_strategy_ == LowerBoundStrategy::None) {
      if (verbose_) std::cout << "lb_strategy=None; using BruteForce.\n";
      fillDistanceMatrix_BruteForce();
      break;
    }
    auto stats = core::fill_distance_matrix_pruned(*this, band, lb_strategy_);
    if (verbose_) {
      std::cout << "Pruned strategy: " << stats.total_pairs << " pairs, "
                << stats.early_abandoned << " early-abandoned, "
                << "pruning ratio: " << stats.pruning_ratio() << '\n';
    }
    break;
  }
  case DistanceMatrixStrategy::CUDA:
#ifdef DTWC_HAS_CUDA
  {
    if (!dtwc::cuda::cuda_available()) {
      throw DeviceError(
        "CUDA distance strategy requested but no CUDA GPU was detected. "
        "No CPU fallback was attempted.");
    }

    dtwc::cuda::CUDADistMatOptions cuda_opts;
    cuda_opts.band = band;
    cuda_opts.device_id = cuda_settings.device_id;
    if (cuda_settings.precision == 1)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP32;
    else if (cuda_settings.precision == 2)
      cuda_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
    cuda_opts.use_squared_l2 = false;
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
    if (!dtwc::metal::metal_available()) {
      throw DeviceError(
        "Metal distance strategy requested but no Metal GPU was detected. "
        "No CPU fallback was attempted.");
    }

    dtwc::metal::MetalDistMatOptions metal_opts;
    metal_opts.band = band;
    metal_opts.use_squared_l2 = false;
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
  default:
    validate_distance_matrix_strategy(effective);
    throw std::logic_error(
      "Problem::fill_distance_matrix: unreachable distance strategy");
  }

  if (verbose_)
    std::cout << "Distance matrix has been filled!" << '\n';
}
/**
 * @brief Performs clustering based on the specified method.
 * @details Chooses between different clustering methods (K-medoids or MIP) and performs the clustering accordingly.
 */
void Problem::cluster()
{
  validate_method(method_);
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
  default:
    throw std::logic_error("Problem::cluster: unreachable Method");
  }
}

/**
 * @brief Executes the clustering process and additional post-processing tasks.
 * @details Performs clustering, then prints and writes the cluster results, including silhouettes, to files.
 */
void Problem::cluster_and_process()
{
  cluster();
  printClusters(); // Prints to screen.
  writeDistanceMatrix();
  writeClusters(); // Prints to file.
  writeSilhouettes();
}

/**
 *@brief Clusters the data using Mixed Integer Programming (MIP) based on the chosen solver.
 *@details Uses either Gurobi or HiGHS solver for MIP clustering, depending on the solver set in the Problem instance.
 */
void Problem::cluster_by_mip()
{
  // Validate before Benders policy: an invalid stored selector must not bypass
  // membership checks merely because the large-N route ignores mipSolver.
  validate_solver(mipSolver);
  // Auto-dispatch to Benders decomposition for large N
  const bool use_benders = (mip_settings.benders == "on") || (mip_settings.benders == "auto" && data_.size() > 200);

  if (use_benders) {
    MIP_clustering_byBenders(*this);
    return;
  }

  switch (mipSolver) {
  case Solver::Gurobi:
    MIP_clustering_byGurobi(*this);
    break;
  case Solver::HiGHS:
    MIP_clustering_byHiGHS(*this);
    break;
  default:
    throw std::logic_error("Problem::cluster_by_mip: unreachable Solver");
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
      if (!has_best || distance < best_distance) {
        best_distance = distance;
        best_slot = static_cast<int>(slot);
        has_best = true;
      }
    }
    labels[i_p] = best_slot;
  };

  // If the full matrix is not materialised yet, distByInd() may lazily compute
  // symmetric entries on demand. Different points can request the same packed
  // (i,j)/(j,i) slot concurrently, so the lazy-compute path is not safe to run
  // in parallel. Once fillDistanceMatrix() has completed, all lookups are
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
  cluster_by_kmedoids_lloyd_impl(true);
}

void Problem::cluster_by_kmedoids_lloyd_impl(bool persist_artifacts)
{
  if (N_repetition <= 0)
    throw InvalidInput("Lloyd k-medoids requires n_repetitions >= 1.");
  const auto restart_offset = static_cast<std::uint64_t>(N_repetition - 1);
  if (restart_offset
      > std::numeric_limits<std::uint64_t>::max() - random_seed_)
    throw InvalidInput("Lloyd k-medoids random_seed + repetition index overflows uint64.");

  fill_distance_matrix(); // Ensure all distances computed before parallel clustering.

  int best_rep = 0;
  double best_cost = std::numeric_limits<data_t>::max();
  int best_iterations = 0;
  std::vector<int> best_medoids;
  std::vector<int> best_labels;

  for (int i_rand = 0; i_rand < N_repetition; i_rand++) {
    std::cout << "Metoid initialisation is started.\n";
    init_with_seed(random_seed_ + static_cast<std::uint64_t>(i_rand));

    std::cout << "Metoid initialisation is finished. "
              << Nc << " medoids are initialised.\n"
              << "Start clustering:\n";

    auto [status, total_cost, iters] =
      cluster_by_kMedoidsLloyd_single(i_rand, persist_artifacts);
    last_iterations_ = iters;

    if (status == 0)
      std::cout << "Medoids are same for last two iterations, algorithm is converged!\n";
    else if (status == -1)
      std::cout << "Maximum iteration is reached before medoids are converged!\n";

    if (i_rand == 0 || total_cost < best_cost) {
      best_cost = total_cost;
      best_rep = i_rand;
      best_iterations = iters;
      best_medoids = centroids_ind;
      best_labels = clusters_ind;
    }
    std::cout << "Tot cost: " << total_cost << " best cost: " << best_cost << " i rand: " << i_rand << '\n';
  }

  centroids_ind = std::move(best_medoids);
  clusters_ind = std::move(best_labels);
  last_iterations_ = best_iterations;
  if (persist_artifacts)
    writeBestRep(best_rep);
  else
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
  for (int i = 0; i < maxIter; i++) {
    actual_iters = i + 1;

    std::cout << "Medoids: ";
    for (auto medoid : centroids_ind)
      std::cout << get_name(medoid) << ' ';

    centroids_all.push_back(centroids_ind);

    assign_clusters();

    std::cout << " Iteration: " << i << " completed with cost: " << std::setprecision(10)
              << find_total_cost() << ".\n"; // Uses clusters_ind to find cost.

    printClusters();
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
