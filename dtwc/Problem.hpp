/**
 * @file Problem.hpp
 * @brief Encapsulates the DTWC (Dynamic Time Warping Clustering) problem in a class.
 *
 * @details This file contains the definition of the Problem class used in DTWC applications.
 * It includes various methods for manipulating and analyzing clusters.
 *
 * @date 19 Oct 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "Data.hpp"           // for Data
#include "DataLoader.hpp"     // for DataLoader
#include "base/settings.hpp"       // for data_t, DEFAULT_BAND
#include "base/env.hpp"            // for Device
#include "base/error.hpp"          // for InvalidInput
#include "base/names.hpp"          // for Name
#include "enums/enums.hpp"    // for using Enum types.
#include "initialisation.hpp" // for init functions
#include "core/dtw_options.hpp" // for DTWVariant
#include "core/storage.hpp"     // for Precision
#include "core/distance_matrix.hpp" // for DistanceMatrix
#include "core/clustering_result.hpp" // for set_result

#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t, int64_t
#include <filesystem>  // for operator/, path
#include <string>      // for char_traits, operator+, operator<<
#include <string_view> // for string_view
#include <utility>     // for pair
#include <vector>      // for vector, allocator
#include <functional>  // std::function
#include <unordered_map> // std::unordered_map
#include <span>        // std::span
#include <memory>
#include <atomic>      // for std::atomic (RelaxedFlag)
#include <stdexcept>

#include "checkpoint.hpp" // for CheckpointOptions, load_checkpoint

namespace dtwc {

/// GPU compute precision, on every GPU backend. `Auto` is FP32 on consumer CUDA
/// GPUs and FP64 on HPC ones; Metal computes in FP32 and rejects FP64. The values
/// are hashed into the distance-matrix identity, so they never change.
enum class GpuPrecision { Auto = 0, FP32 = 1, FP64 = 2 };

/// The spellings of `--gpu-precision`.
inline constexpr Name<GpuPrecision> gpu_precision_names[]{
  { "auto", GpuPrecision::Auto },
  { "fp32", GpuPrecision::FP32 }, { "float32", GpuPrecision::FP32 }, { "f32", GpuPrecision::FP32 },
  { "float", GpuPrecision::FP32 },
  { "fp64", GpuPrecision::FP64 }, { "float64", GpuPrecision::FP64 }, { "f64", GpuPrecision::FP64 },
  { "double", GpuPrecision::FP64 },
};

/// GPU compute settings, read by the CUDA and Metal routes. Metal runs on the
/// system default device in FP32: a device_id other than 0, or precision FP64,
/// is rejected on Metal rather than ignored.
struct CUDASettings {
  int device_id = 0;  ///< GPU index (Problem::set_device(Device::GPU, index)).
  GpuPrecision precision = GpuPrecision::Auto; ///< Compute precision.
};

/// MIP solver tuning parameters.
struct MIPSettings {
  double mip_gap = 1e-5;          ///< Relative MIP gap tolerance.
  int time_limit_sec = -1;        ///< Solver time limit in seconds (-1 = unlimited).
  bool warm_start = true;          ///< Run FastPAM first and feed as MIP start.
  int numeric_focus = 1;           ///< Gurobi NumericFocus (0-3).
  int mip_focus = 2;               ///< Gurobi MIPFocus (0=balanced, 1=feasible, 2=optimal, 3=bound).
  bool verbose_solver = false;     ///< Show solver log output.
  std::int64_t lr_max_nodes = 2000000; ///< Method::LRCore branch-and-bound node cap. Fixed width: `long` is 32-bit on Windows and 64-bit on Linux, so the public range would be platform-dependent.
};

/// Reject MIP settings a solver would otherwise turn into a solver-worded error.
/// `mip_gap < 0` (or NaN) is outside HiGHS's `mip_rel_gap` domain, which the
/// option guard reports as "HiGHS rejected option" rather than as bad input.
inline void validate_mip_settings(const MIPSettings &s)
{
  const auto reject = [](std::string_view field, std::string_view rule, const std::string &got) {
    throw InvalidInput("MIPSettings::" + std::string(field) + " must be " + std::string(rule)
                       + "; got " + got + ".");
  };
  if (!(s.mip_gap >= 0.0)) reject("mip_gap", ">= 0", std::to_string(s.mip_gap));
  if (s.lr_max_nodes < 1) reject("lr_max_nodes", ">= 1", std::to_string(s.lr_max_nodes));
}

/// Strategy for computing the pairwise distance matrix.
enum class DistanceMatrixStrategy {
  Auto,       ///< BruteForce; set_device(gpu) selects CUDA or Metal instead
  BruteForce, ///< Parallel exact fill on the CPU
  CUDA,       ///< NVIDIA CUDA GPU (requires DTWC_HAS_CUDA)
  Metal       ///< Apple Metal GPU (requires DTWC_HAS_METAL)
};

inline void validate_distance_matrix_strategy(DistanceMatrixStrategy value)
{
  switch (value) {
  case DistanceMatrixStrategy::Auto:
  case DistanceMatrixStrategy::BruteForce:
  case DistanceMatrixStrategy::CUDA:
  case DistanceMatrixStrategy::Metal:
    return;
  }
  throw InvalidInput("Invalid DistanceMatrixStrategy value.");
}

/// FX-1's GPU rules that need no series: Float32 values, a variant or a
/// missing-data strategy the GPU kernels do not implement, and a GPU index or
/// precision Metal cannot honour. A fill applies them to its Problem
/// (validate_fill_request); dtwc::run applies them to a configuration before
/// it reads a series. A CPU strategy passes.
/// @throws DeviceError naming `where`, the backend and the axis.
void validate_gpu_request(std::string_view where, DistanceMatrixStrategy strategy,
                          const core::DTWVariantParams &variant, core::MissingStrategy missing,
                          core::Precision precision, const CUDASettings &gpu);

/**
 * @class Problem
 * @brief Class representing a problem in DTWC.
 *
 * @details This class encapsulates all the functionalities and data structures required to solve
 * a dynamic time warping clustering problem. It includes methods for initialising clusters,
 * calculating distances, clustering, and writing results.
 */
class Problem
{
public:
  using path_t = std::filesystem::path;

  /// DTW distance function types: float64 and float32 variants.
  /// Both return double (distance precision is always double).
  using dtw_fn_t = std::function<data_t(std::span<const data_t>, std::span<const data_t>)>;
  using dtw_fn_f32_t = std::function<double(std::span<const float>, std::span<const float>)>;

private:
  int Nc{ 1 };                                      /*!< Number of clusters. */
  core::DistanceMatrix distMat;                     /*!< Distance matrix, on the heap or mapped (use_mmap_distance_matrix). */
  Solver mipSolver{ settings::DEFAULT_MIP_SOLVER }; /*!< Solver for MIP. */
  mutable dtw_fn_t dtw_fn_;                         /*!< Derived DTW dispatcher for float64. */
  mutable dtw_fn_f32_t dtw_fn_f32_;                 /*!< Derived DTW dispatcher for float32. */
  /// The fill's lane functions (core::resolve_dtw_block_fn); empty where none applies.
  mutable std::function<void(std::span<const data_t>, std::span<const std::span<const data_t>>,
                             std::span<double>)> dtw_block_fn_;
  mutable std::function<void(std::span<const float>, std::span<const std::span<const float>>,
                             std::span<double>)> dtw_block_fn_f32_;
  mutable const Problem *dtw_binding_owner_{ nullptr }; /*!< Address captured by the dispatchers. */
  mutable std::unordered_map<size_t, std::vector<data_t>> wdtw_weights_cache_; /*!< Derived WDTW weights keyed by max_dev. */

  using cache_fingerprint_t = core::DistanceMatrix::fingerprint_type;
  struct DistanceCacheConfiguration {
    core::MetricType metric{ core::MetricType::L1 };
    int band{ settings::DEFAULT_BAND };
    core::DTWVariantParams variant_params{};
    core::MissingStrategy missing_strategy{ core::MissingStrategy::Error };
    DistanceMatrixStrategy distance_strategy{ DistanceMatrixStrategy::Auto };
    int cuda_device_id{ 0 };
    GpuPrecision cuda_precision{ GpuPrecision::Auto };
  };
  struct DistanceCacheIdentity {
    cache_fingerprint_t full{};
    cache_fingerprint_t configuration{};
    DistanceCacheConfiguration configuration_values{};
    core::Precision precision{ core::Precision::Float64 };
    size_t n{ 0 };
    size_t ndim{ 1 };
  };
  /// Boolean flag written from a const method, where two threads may hold one
  /// const Problem& (Python releases the GIL around the const writers). Relaxed
  /// ordering suffices: the flag guards a pure recomputation, not a publication.
  /// std::atomic is neither copyable nor movable, so the value-moving members
  /// are what keep Problem's `= default` move operations well-formed.
  class RelaxedFlag
  {
    std::atomic<bool> value_{ false };
    bool get() const noexcept { return value_.load(std::memory_order_relaxed); }

  public:
    RelaxedFlag() = default;
    RelaxedFlag(RelaxedFlag &&other) noexcept : value_{ other.get() } {}
    RelaxedFlag &operator=(RelaxedFlag &&other) noexcept { return *this = other.get(); }
    RelaxedFlag &operator=(bool v) noexcept { value_.store(v, std::memory_order_relaxed); return *this; }
    explicit operator bool() const noexcept { return get(); }
  };

  DistanceCacheIdentity mmap_cache_identity_{};
  bool mmap_cache_identity_bound_{ false };
  mutable RelaxedFlag mmap_cache_data_validated_{};
  /// Set once validate_fill_request() has passed for the current data and
  /// semantics (by a fill, the lazy dist_by_ind path or a dtw_function
  /// accessor), or once the lazy path found a dense matrix holding every pair.
  /// refresh_distance_matrix() clears it, as does installing or editing a
  /// matrix. Mutable: the const accessors validate too.
  mutable RelaxedFlag fill_request_validated_{};
  mutable DistanceCacheConfiguration dense_cache_configuration_{};
  mutable bool dense_cache_configuration_bound_{ false };

  Method method_{ Method::Kmedoids };
  core::MetricType metric_{ core::MetricType::L1 }; //!< Pointwise cost of every distance (set_metric).
  std::uint64_t random_seed_{ settings::DEFAULT_RANDOM_SEED };
  int last_iterations_{ 0 };
  double tadpole_dc_{ -1.0 };
  bool verbose_{ false };
  /// Run-artifact files (per-repetition medoids, best-repetition record) belong
  /// to cluster_and_process(); cluster() itself is side-effect free.
  bool persist_run_artifacts_{ false };
  path_t output_folder_{ "./results/" }; //!< Relative to the working directory; set_output_folder.
  std::string name_{};
  Data data_;

  void rebind_dtw_fn() const; ///< Rebind derived dispatch state to this address/configuration.
  void refresh_variant_caches() const; ///< Refresh precomputed variant-specific caches.
  cache_fingerprint_t distance_cache_configuration_fingerprint(
    core::MetricType metric) const;
  DistanceCacheConfiguration distance_cache_configuration(
    core::MetricType metric) const;
  bool distance_cache_configuration_matches(
    const DistanceCacheConfiguration &expected) const;
  bool dense_cache_configuration_is_current() const;
  static void preflight_distance_semantics(
    const core::DTWVariantParams &params,
    core::MissingStrategy missing,
    core::MetricType metric,
    const Data &candidate_data,
    DistanceMatrixStrategy candidate_distance_strategy,
    const CUDASettings &candidate_cuda_settings,
    bool force_float32 = false);
  void preflight_current_distance_semantics() const;
  void preflight_float32_distance_semantics() const;
  const dtw_fn_f32_t &validated_dtw_function_f32() const;
  void repair_dtw_binding_after_relocation();
  void ensure_dense_cache_configuration_current();
  /// As ensure_dense_cache_configuration_current(), minus the leading preflight,
  /// for callers that have already run preflight_current_distance_semantics().
  /// The SWAP kernel issues N^2 dist_by_ind() calls per iteration, so paying for
  /// that preflight twice per element is not free.
  void ensure_dense_cache_configuration_current_preflighted();
  void validate_dense_cache_configuration() const;
  void ensure_dtw_function_configuration_current();
  void validate_dtw_function_configuration() const;
  DistanceCacheIdentity distance_cache_identity(core::MetricType metric) const;
  void validate_mmap_cache_identity() const;
  void validate_checkpoint_settings() const;
  /// FX-1: the one check of a distance request — every device axis, band
  /// feasibility and (FX-15) the series values the missing-data strategy
  /// cannot take — before any pair is computed. O(N·L); called through
  /// validate_fill_request_once(), never per pair.
  void validate_fill_request(std::string_view where) const;
  /// validate_fill_request() unless it already passed for this configuration.
  /// The dtw_function accessors, which OneBatchPAM and FastCLARA's assignment
  /// compute through serially before their parallel loops, call it; they hand
  /// out the kernel itself, so no filled matrix can serve their pairs.
  void validate_fill_request_once(std::string_view where) const;
  /// get_name / p_vec return references into owned heap storage. A view
  /// (set_view_data) has none, nor has a Float32 store Float64 values:
  /// indexing would read past an empty vector in a Release build (F25).
  void require_owned_storage(std::string_view accessor, bool float64_values) const;
  void clear_mmap_cache_identity();
  void fillDistanceMatrix_BruteForce(); ///< Brute-force parallel distance matrix fill.

  // Private functions:
  friend bool load_checkpoint(Problem &prob, const std::string &path,
                              core::MetricType metric);
  std::tuple<int, double, int> cluster_by_kMedoidsLloyd_single(
    int rep, bool persist_artifacts);
  void init_with_seed(std::uint64_t seed);

  void writeBestRep(int best_rep);
  void writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost);
  void distanceInClusters();

  /// An empty series has no finite DTW distance to anything, so it clustered
  /// with DBL_MAX distances and exit 0 (it used to arrive from a blank line).
  static void reject_empty_series(const Data &data, std::string_view operation)
  {
    for (std::size_t i = 0; i < data.size(); ++i)
      if (data.series_flat_size(i) == 0)
        throw InvalidInput(
          std::string(operation) + ": series " + std::to_string(i) + " ('"
          + std::string(data.name(i)) + "') is empty; every series needs at "
            "least one value.");
  }

public:
  int maxIter{ 100 };                        /*!< Maximum number of iteration for iterative-methods. */
  int N_repetition{ 1 };                     /*!< Repetition for iterative-methods. */
  int band{ settings::DEFAULT_BAND };        /*!< Band length for Sakoe-Chiba band, -1 for full DTW. */
  /// DTW variant selection and parameters.
  /// Prefer set_variant(), which invalidates and rebinds eagerly. Legacy direct
  /// writes remain source-compatible and are detected by the fixed-size dense
  /// configuration snapshot before cached values can be reused.
  core::DTWVariantParams variant_params;
  core::MissingStrategy missing_strategy = core::MissingStrategy::Error; /*!< Strategy for handling NaN values in series. */
  DistanceMatrixStrategy distance_strategy{ DistanceMatrixStrategy::Auto }; /*!< Distance matrix strategy. */
  CUDASettings cuda_settings;                /*!< GPU options (used when distance_strategy == GPU). */
  MIPSettings mip_settings;                  /*!< MIP solver tuning parameters. */
  CheckpointOptions checkpoint;              /*!< Automatic mid-fill checkpointing (see checkpoint.hpp). */

  std::function<void(Problem &)> init_fun{ init::random }; /*!< Initialisation function. */

  /// Empty until a clustering writes them (set_result, set_clusters for the
  /// medoids, the algorithms); set_n_clusters and set_data size neither.
  /// require_clustered() tells a clustering from a sizing.
  std::vector<int> clusters_ind;  //!< Indices of which point belongs to which cluster. [0,Nc)
  std::vector<int> centroids_ind; //!< indices of cluster centroids. [0, Np)

  // Constructors:
  Problem() { rebind_dtw_fn(); }
  Problem(std::string_view problem_name) : name_{ problem_name }
  {
    rebind_dtw_fn();
  }
  Problem(std::string_view problem_name, DataLoader &loader)
    : name_{ problem_name }, data_{ loader.load() }
  {
    reject_empty_series(data_, "Problem(name, DataLoader)");
    refresh_distance_matrix(); // also calls rebind_dtw_fn()
  }
  Problem(const Problem &) = delete;
  Problem &operator=(const Problem &) = delete;
  Problem(Problem &&);
  /// Not noexcept: moving the members (the dispatchers, the WDTW weight map)
  /// may throw, and a noexcept promise would turn that into std::terminate (S-13).
  Problem &operator=(Problem &&);

  auto size() const { return data_.size(); }
  /// Number of clusters (canonical 2.0 read accessor; was `cluster_size()`).
  auto n_clusters() const { return Nc; }
  [[deprecated("use n_clusters")]] auto cluster_size() const { return n_clusters(); }

  /// Mutable name access (owned names only). series_name(i) reads a name in
  /// every storage mode.
  /// @throws InvalidInput on a view (set_view_data, an mmap series store).
  auto &get_name(size_t i)
  {
    require_owned_storage("Problem::get_name", false);
    return data_.p_names[i];
  }
  auto const &get_name(size_t i) const
  {
    require_owned_storage("Problem::get_name", false);
    return data_.p_names[i];
  }

  /// Mutable vector access (owned Float64 series only). series(i) reads a
  /// series in every storage mode.
  /// Call refresh_distance_matrix() before mutating values when any distance
  /// cache has been used; raw in-place edits during a bound-cache session are
  /// unsupported because warm lookups intentionally remain O(1).
  /// @throws InvalidInput on a view, Float32 or metadata-only store.
  auto &p_vec(size_t i)
  {
    require_owned_storage("Problem::p_vec", true);
    return data_.p_vec[i];
  }
  auto const &p_vec(size_t i) const
  {
    require_owned_storage("Problem::p_vec", true);
    return data_.p_vec[i];
  }

  /// Zero-copy span view of series i (works for heap, mmap, and view modes).
  std::span<const data_t> series(size_t i) const { return data_.series(i); }

  /// Name of series i as a string_view.
  std::string_view series_name(size_t i) const { return data_.name(i); }

  /// Canonical read accessors (API contract §2.2): the raw fields
  /// `clusters_ind`/`centroids_ind` stay public, but `labels()`/`medoids()` are
  /// the cross-language read path (parity with `Result::labels`/`Result::medoids`).
  const std::vector<int> &labels() const { return clusters_ind; }
  const std::vector<int> &medoids() const { return centroids_ind; }

  void refresh_distance_matrix();
  [[deprecated("use refresh_distance_matrix")]] void refreshDistanceMatrix() { refresh_distance_matrix(); }

  // Getters and setters:
  /// The centroid of the cluster of i_p, i_p in [0, N). Unchecked like dist_by_ind:
  /// write_clusters and the bindings call it per series, so a caller that cannot
  /// vouch for the clustering calls require_clustered() once first.
  int centroid_of(int i_p) const { return centroids_ind[clusters_ind[i_p]]; }

  void read_distance_matrix(const fs::path &distMat_path);
  [[deprecated("use read_distance_matrix")]]
  void readDistanceMatrix(const fs::path &p) { read_distance_matrix(p); }

  void set_n_clusters(int Nc_);
  [[deprecated("use set_n_clusters")]] void set_numberOfClusters(int Nc_) { set_n_clusters(Nc_); }

  /// A Problem is clustered when it holds one label per series and one medoid per
  /// cluster. Every call that reads the whole clustering checks that once.
  /// @throws InvalidInput ("<who>: ... cluster it first") when it is not.
  void require_clustered(std::string_view who) const;
  void set_clusters(std::vector<int> &candidate_centroids);
  /// Publish a clustering: k = the number of medoids, which are distinct
  /// indices in [0, N), and one label in [0, k) per series. Anything else is
  /// InvalidInput and leaves the Problem unchanged.
  void set_result(const core::ClusteringResult &result);
  /// @return false when Gurobi is requested on a build without it: the solver
  /// is then HiGHS, which a caller that asked for Gurobi must not ignore.
  [[nodiscard]] bool set_solver(dtwc::Solver solver_);

  // Canonical configuration reads for the ten encapsulated fields.
  // last_iterations() and data() are read-only; mutable configuration uses
  // the corresponding setters below.
  Method method() const { return method_; }
  std::uint64_t random_seed() const { return random_seed_; }
  int last_iterations() const { return last_iterations_; }
  double tadpole_dc() const { return tadpole_dc_; }
  bool verbose() const { return verbose_; }
  const path_t &output_folder() const { return output_folder_; }
  const std::string &name() const { return name_; }
  const Data &data() const { return data_; }

  void set_method(Method m)
  {
    validate_method(m);
    method_ = m;
  }
  /// @throws InvalidInput for b < -1: -1 is full DTW and b >= 0 a Sakoe-Chiba
  ///         half-width; nothing lies between.
  void set_band(int b)
  {
    if (b < -1)
      throw InvalidInput("Problem::set_band: band must be -1 (full DTW) or at least 0; got "
                         + std::to_string(b) + ".");
    preflight_current_distance_semantics();
    if (band == b) return;
    band = b;
    refresh_distance_matrix();
  }
  /// @throws InvalidInput for n < 1: no iteration would report the initial
  ///         medoids' cost as a clustering result (O-06).
  void set_max_iter(int n);
  int max_iter() const;
  /// @throws InvalidInput for n < 1: at least one run is needed (O-06).
  void set_n_repetitions(int n);
  int n_repetitions() const;
  void set_random_seed(std::uint64_t seed) { random_seed_ = seed; }
  void set_tadpole_dc(double dc) { tadpole_dc_ = dc; }
  void set_missing_strategy(core::MissingStrategy strategy)
  {
    preflight_distance_semantics(
      variant_params, strategy, metric_, data_, distance_strategy, cuda_settings);
    if (missing_strategy == strategy) return;
    missing_strategy = strategy;
    refresh_distance_matrix();
  }
  /// Pointwise cost of every distance this Problem computes: the CPU fill and
  /// lazy lookups, the GPU routes, the mmap cache and checkpoint identities.
  /// L1 by default. A metric other than L1 is implemented for Standard DTW with
  /// MissingStrategy::Error (univariate or multivariate): the Problem passes the
  /// metric to the Standard kernels only.
  /// @throws InvalidInput for an invalid value, or a metric other than L1 with
  ///         a variant other than Standard or a missing-data strategy.
  void set_metric(core::MetricType metric)
  {
    preflight_distance_semantics(
      variant_params, missing_strategy, metric, data_, distance_strategy, cuda_settings);
    if (metric_ == metric) return;
    metric_ = metric;
    refresh_distance_matrix();
  }
  core::MetricType metric() const noexcept { return metric_; }
  void set_distance_strategy(DistanceMatrixStrategy strategy)
  {
    preflight_distance_semantics(
      variant_params, missing_strategy, metric_, data_, strategy, cuda_settings);
    if (distance_strategy == strategy) return;
    distance_strategy = strategy;
    refresh_distance_matrix();
  }
  /// Where this Problem computes distances. `cpu` keeps a CPU strategy you
  /// chose (BruteForce) and moves a GPU one to Auto; `gpu` selects
  /// this build's GPU backend (CUDA, else Metal) and records `index`, the GPU
  /// ordinal. A Problem never reads the process-wide default (dtwc::device());
  /// until told otherwise it computes on the CPU.
  /// @throws DeviceError for `gpu` on a build with no GPU backend;
  ///         InvalidInput for a negative index.
  void set_device(Device device, int index = 0);
  void set_cuda_settings(CUDASettings settings)
  {
    preflight_distance_semantics(
      variant_params, missing_strategy, metric_, data_, distance_strategy, settings);
    if (cuda_settings.device_id == settings.device_id
        && cuda_settings.precision == settings.precision)
      return;
    cuda_settings = settings;
    refresh_distance_matrix();
  }

  void set_verbose(bool value) { verbose_ = value; }
  void set_output_folder(path_t folder)
  {
    output_folder_ = std::move(folder);
  }
  void set_name(std::string problem_name)
  {
    name_ = std::move(problem_name);
  }

  void set_data(dtwc::Data candidate)
  {
    core::validate_precision(candidate.precision);
    candidate.validate_ndim();
    reject_empty_series(candidate, "Problem::set_data");
    preflight_distance_semantics(
      variant_params, missing_strategy, metric_, candidate, distance_strategy, cuda_settings);
    data_ = std::move(candidate);
    refresh_distance_matrix();
  }

  /// Set view-mode data (non-owning spans). Sizes distance matrix but skips mmap cache.
  void set_view_data(dtwc::Data candidate)
  {
    core::validate_precision(candidate.precision);
    candidate.validate_ndim();
    reject_empty_series(candidate, "Problem::set_view_data");
    preflight_distance_semantics(
      variant_params, missing_strategy, metric_, candidate, distance_strategy, cuda_settings);
    data_ = std::move(candidate);
    refresh_distance_matrix();
  }

  /// Set DTW variant and rebind the distance function.
  void set_variant(core::DTWVariant v);
  void set_variant(core::DTWVariantParams params);

  data_t max_distance() const
  {
    validate_mmap_cache_identity();
    validate_dense_cache_configuration();
    return distMat.max();
  }
  [[deprecated("use max_distance")]] data_t maxDistance() const { return max_distance(); }

  data_t dist_by_ind(int i, int j);
  [[deprecated("use dist_by_ind")]] data_t distByInd(int i, int j) { return dist_by_ind(i, j); }

  /// Access the bound DTW distance function (float64). Mutable access repairs
  /// legacy raw configuration mutations before returning the dispatcher;
  /// const access rejects stale semantics instead of silently using them.
  /// The first accessor after a Problem move or a (re)configuration repairs
  /// logically-const derived dispatch state and validates the request against
  /// this Problem's series (validate_fill_request), so it must run serially
  /// before parallel use.
  /// @throws InvalidInput for a band no pair's warping path fits, a ±inf
  ///         series value, or a NaN under MissingStrategy::Error; DeviceError
  ///         for a GPU strategy the kernels cannot honour.
  const dtw_fn_t &dtw_function()
  {
    ensure_dtw_function_configuration_current();
    validate_fill_request_once("Problem::dtw_function");
    return dtw_fn_;
  }
  const dtw_fn_t &dtw_function() const
  {
    validate_dtw_function_configuration();
    validate_fill_request_once("Problem::dtw_function");
    return dtw_fn_;
  }

  /// Float32 counterpart of dtw_function(), with the same semantic guard.
  const dtw_fn_f32_t &dtw_function_f32()
  {
    preflight_float32_distance_semantics();
    ensure_dtw_function_configuration_current();
    validate_fill_request_once("Problem::dtw_function_f32");
    return validated_dtw_function_f32();
  }
  const dtw_fn_f32_t &dtw_function_f32() const
  {
    preflight_float32_distance_semantics();
    validate_dtw_function_configuration();
    validate_fill_request_once("Problem::dtw_function_f32");
    return validated_dtw_function_f32();
  }

  /// Read-only access to the WDTW weights cache (consumed by core::resolve_dtw_fn).
  /// Cache is populated serially by refresh_variant_caches() before parallel fill
  /// and is lock-free for parallel readers.
  const std::unordered_map<std::size_t, std::vector<data_t>> &wdtw_weights_cache() const
  {
    return wdtw_weights_cache_;
  }
  bool is_distance_matrix_filled() const
  {
    validate_mmap_cache_identity();
    if (!distMat.is_mapped() && !dense_cache_configuration_is_current())
      return false;
    return distMat.size() > 0 && distMat.all_computed();
  }
  [[deprecated("use is_distance_matrix_filled")]] bool isDistanceMatrixFilled() const { return is_distance_matrix_filled(); }

  /// The distance matrix, on the heap or mapped (const).
  const core::DistanceMatrix &distance_matrix() const
  {
    validate_mmap_cache_identity();
    validate_dense_cache_configuration();
    return distMat;
  }
  /// The distance matrix (mutable). The caller may change which pairs are
  /// known, so the next lazy lookup re-checks the request.
  core::DistanceMatrix &distance_matrix()
  {
    validate_mmap_cache_identity();
    ensure_dense_cache_configuration_current();
    fill_request_validated_ = false;
    return distMat;
  }

  /// Full data-plus-distance-semantics identity used by durable checkpoints,
  /// for this Problem's metric(). Names and clustering outputs are
  /// intentionally excluded because they do not affect any stored distance.
  core::DistanceMatrix::fingerprint_type distance_checkpoint_identity() const;
  /// The same identity for distances computed with `metric`, which may differ
  /// from metric() only for a matrix a producer outside this Problem filled.
  core::DistanceMatrix::fingerprint_type distance_checkpoint_identity(
    core::MetricType metric) const;
  /// Map the distance matrix to the `.dtwm` file `cache_path`, bound to this
  /// Problem's exact data and distance settings, metric() included. An existing
  /// file is reopened with the distances it holds (InvalidInput if they are for
  /// other series or settings, IOError if it is not a whole `.dtwm` file); an
  /// absent one is created. IOError on a build without llfio.
  /// The full data fingerprint is verified at bind and once again on first use;
  /// subsequent warm lookups compare a fixed-size configuration snapshot to
  /// preserve O(1) access. After first use, replace data through set_data and
  /// distance-semantic setters, or call refresh_distance_matrix() before any
  /// raw in-place Data edit; mutating raw storage during a bound session is
  /// unsupported.
  void use_mmap_distance_matrix(const std::filesystem::path &cache_path);
  /// Bind a cache for `metric`, which becomes this Problem's metric as with
  /// set_metric: the cache and every distance computed into it share it. A
  /// bind that throws leaves the metric and the matrix unchanged.
  void use_mmap_distance_matrix(
    const std::filesystem::path &cache_path, core::MetricType metric);

  void fill_distance_matrix();
  [[deprecated("use fill_distance_matrix")]] void fillDistanceMatrix() { fill_distance_matrix(); }

  void print_distance_matrix() const;
  [[deprecated("use print_distance_matrix")]] void printDistanceMatrix() const { print_distance_matrix(); }

  // Canonical I/O owns behavior; retained 1.x names are deprecated forwarders.
  void write_distance_matrix(const std::string &name_) const;
  void write_distance_matrix() const
  {
    write_distance_matrix(name_ + "_distanceMatrix.csv");
  }
  [[deprecated("use write_distance_matrix")]]
  void writeDistanceMatrix(const std::string &name_) const
  {
    write_distance_matrix(name_);
  }
  [[deprecated("use write_distance_matrix")]]
  void writeDistanceMatrix() const { write_distance_matrix(); }

  void print_clusters() const;
  [[deprecated("use print_clusters")]]
  void printClusters() const { print_clusters(); }
  void write_clusters();
  [[deprecated("use write_clusters")]]
  void writeClusters() { write_clusters(); }

  void write_medoid_members(int iter, int rep = 0) const;
  [[deprecated("use write_medoid_members")]]
  void writeMedoidMembers(int iter, int rep = 0) const
  {
    write_medoid_members(iter, rep);
  }
  void write_silhouettes();
  [[deprecated("use write_silhouettes")]]
  void writeSilhouettes() { write_silhouettes(); }

  // Initialisation of clusters:
  void init() { init_fun(*this); }

  // Clustering functions:
  void cluster();
  void cluster_by_mip();
  [[deprecated("use cluster_by_mip")]] void cluster_by_MIP() { cluster_by_mip(); }
  void cluster_by_kmedoids_lloyd();
  [[deprecated("use cluster_by_kmedoids_lloyd")]] void cluster_by_kMedoidsPAM() { cluster_by_kmedoids_lloyd(); }

  void cluster_and_process();

  // Auxillary
  double find_total_cost();
  [[deprecated("use find_total_cost")]] double findTotalCost() { return find_total_cost(); }
  void assign_clusters();
  [[deprecated("use assign_clusters")]] void assignClusters() { assign_clusters(); }

  void calculate_medoids();
  [[deprecated("use calculate_medoids")]] void calculateMedoids() { calculate_medoids(); }
};


} // namespace dtwc
