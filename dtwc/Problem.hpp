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
#include "settings.hpp"       // for data_t, resultsPath
#include "error.hpp"          // for InvalidInput
#include "enums/enums.hpp"    // for using Enum types.
#include "initialisation.hpp" // for init functions
#include "core/dtw_options.hpp" // for DTWVariant
#include "core/storage.hpp"     // for StoragePolicy

#include "core/mmap_distance_matrix.hpp"
#include <variant>

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

#include "core/distance_matrix.hpp"
#include "checkpoint.hpp" // for CheckpointOptions, load_checkpoint

namespace dtwc {

/// CUDA-specific compute settings. Only used when distance_strategy == CUDA.
/// Metal has no equivalent — it auto-picks the system default device and FP32.
struct CUDASettings {
  int device_id = 0;  ///< CUDA device index.
  /// Compute precision. `Auto` → FP32 on consumer GPUs, FP64 on HPC GPUs.
  /// Declared as a plain int here (rather than `dtwc::cuda::CUDAPrecision`)
  /// so this header stays parsable when DTWC_HAS_CUDA is undefined.
  /// Values: 0 = Auto, 1 = FP32, 2 = FP64. See settings::Precision constants.
  int precision = 0;
};

inline void validate_cuda_settings_precision(int value)
{
  if (value < 0 || value > 2)
    throw InvalidInput("Invalid CUDA precision value.");
}

/// MIP solver tuning parameters.
struct MIPSettings {
  double mip_gap = 1e-5;          ///< Relative MIP gap tolerance.
  int time_limit_sec = -1;        ///< Solver time limit in seconds (-1 = unlimited).
  bool warm_start = true;          ///< Run FastPAM first and feed as MIP start.
  int numeric_focus = 1;           ///< Gurobi NumericFocus (0-3).
  int mip_focus = 2;               ///< Gurobi MIPFocus (0=balanced, 1=feasible, 2=optimal, 3=bound).
  bool verbose_solver = false;     ///< Show solver log output.
  int max_benders_iter = 200;      ///< Maximum Benders iterations (cap exhausted ⇒ SolverError).
  std::string benders = "auto";    ///< Benders mode: "auto" (N>200), "on", "off".
  std::int64_t lr_max_nodes = 2000000; ///< Method::LRCore branch-and-bound node cap (mip::LagrangianParams::max_nodes). Fixed width: `long` is 32-bit on Windows and 64-bit on Linux, so the public range would be platform-dependent.
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
  if (s.max_benders_iter <= 0) reject("max_benders_iter", ">= 1", std::to_string(s.max_benders_iter));
  if (s.lr_max_nodes < 1) reject("lr_max_nodes", ">= 1", std::to_string(s.lr_max_nodes));
  if (s.benders != "auto" && s.benders != "on" && s.benders != "off")
    reject("benders", "'auto', 'on' or 'off'", "'" + s.benders + "'");
}

/// Strategy for computing the pairwise distance matrix.
enum class DistanceMatrixStrategy {
  Auto,       ///< Choose best strategy automatically
  BruteForce, ///< Parallel brute-force (no lower-bound pruning)
  Pruned,     ///< Parallel with lower-bound pruning (LB_Kim / LB_Keogh / cascade)
  CUDA,       ///< NVIDIA CUDA GPU (requires DTWC_HAS_CUDA)
  Metal       ///< Apple Metal GPU (requires DTWC_HAS_METAL)
};

inline void validate_distance_matrix_strategy(DistanceMatrixStrategy value)
{
  switch (value) {
  case DistanceMatrixStrategy::Auto:
  case DistanceMatrixStrategy::BruteForce:
  case DistanceMatrixStrategy::Pruned:
  case DistanceMatrixStrategy::CUDA:
  case DistanceMatrixStrategy::Metal:
    return;
  }
  throw InvalidInput("Invalid DistanceMatrixStrategy value.");
}

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
  using distMat_t = std::variant<core::DenseDistanceMatrix, core::MmapDistanceMatrix>;
  using path_t = std::filesystem::path;

  /// DTW distance function types: float64 and float32 variants.
  /// Both return double (distance precision is always double).
  using dtw_fn_t = std::function<data_t(std::span<const data_t>, std::span<const data_t>)>;
  using dtw_fn_f32_t = std::function<double(std::span<const float>, std::span<const float>)>;

private:
  int Nc{ 1 };                                      /*!< Number of clusters. */
  distMat_t distMat;                                /*!< Distance matrix. */
  Solver mipSolver{ settings::DEFAULT_MIP_SOLVER }; /*!< Solver for MIP. */
  mutable dtw_fn_t dtw_fn_;                         /*!< Derived DTW dispatcher for float64. */
  mutable dtw_fn_f32_t dtw_fn_f32_;                 /*!< Derived DTW dispatcher for float32. */
  mutable const Problem *dtw_binding_owner_{ nullptr }; /*!< Address captured by the dispatchers. */
  mutable std::unordered_map<size_t, std::vector<data_t>> wdtw_weights_cache_; /*!< Derived WDTW weights keyed by max_dev. */

  using cache_fingerprint_t = core::MmapDistanceMatrix::fingerprint_type;
  struct DistanceCacheConfiguration {
    core::MetricType metric{ core::MetricType::L1 };
    int band{ settings::DEFAULT_BAND };
    core::DTWVariantParams variant_params{};
    core::MissingStrategy missing_strategy{ core::MissingStrategy::Error };
    DistanceMatrixStrategy distance_strategy{ DistanceMatrixStrategy::Auto };
    int cuda_device_id{ 0 };
    int cuda_precision{ 0 };
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
  mutable DistanceCacheConfiguration dense_cache_configuration_{};
  mutable bool dense_cache_configuration_bound_{ false };

  Method method_{ Method::Kmedoids };
  std::uint64_t random_seed_{ settings::DEFAULT_RANDOM_SEED };
  int last_iterations_{ 0 };
  double tadpole_dc_{ -1.0 };
  LowerBoundStrategy lb_strategy_{ LowerBoundStrategy::Auto };
  core::StoragePolicy storage_policy_{ core::StoragePolicy::Auto };
  std::size_t ram_limit_bytes_{ 0 }; //!< set_data() footprint threshold override (bytes); 0 = default (50% free RAM).
  bool verbose_{ false };
  /// Run-artifact files (per-repetition medoids, best-repetition record) belong
  /// to cluster_and_process(); cluster() itself is side-effect free.
  bool persist_run_artifacts_{ false };
  path_t output_folder_{ settings::paths::results };
  std::string name_{};
  std::unique_ptr<LoadedData> series_storage_owner_;
  Data data_;

  /// Dispatch through variant via std::visit.
  template <typename F>
  decltype(auto) visit_distmat(F &&f)
  {
    return std::visit(std::forward<F>(f), distMat);
  }

  template <typename F>
  decltype(auto) visit_distmat(F &&f) const
  {
    return std::visit(std::forward<F>(f), distMat);
  }

  void rebind_dtw_fn() const; ///< Rebind derived dispatch state to this address/configuration.
  void refresh_variant_caches() const; ///< Refresh precomputed variant-specific caches.
  cache_fingerprint_t distance_cache_configuration_fingerprint(
    core::MetricType metric) const;
  DistanceCacheConfiguration distance_cache_configuration(
    core::MetricType metric) const;
  bool distance_cache_configuration_matches(
    const DistanceCacheConfiguration &expected) const;
  bool dense_cache_configuration_is_current() const;
  /// Metric the dense cache was bound with, i.e. the metric the CPU fill
  /// computes. Automatic checkpoint saves tag their generation with it.
  core::MetricType dense_cache_metric() const noexcept { return dense_cache_configuration_.metric; }
  static void preflight_distance_semantics(
    const core::DTWVariantParams &params,
    core::MissingStrategy missing,
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
  void clear_mmap_cache_identity();
  void fillDistanceMatrix_BruteForce(); ///< Brute-force parallel distance matrix fill.
  void resize();                        ///< Resize cluster/centroid buffers to size()/Nc. Private invariant maintenance.

  // Private functions:
  friend struct ProblemStoragePolicyTestAccess;
  friend bool load_checkpoint(Problem &prob, const std::string &path,
                              core::MetricType metric);
  friend void MIP_clustering_byBenders(Problem &prob);
  // Benders disables the nested heuristic's artifact files unconditionally; the
  // public Lloyd forwards persist_run_artifacts_.
  void cluster_by_kmedoids_lloyd_impl(bool persist_artifacts);
  std::tuple<int, double, int> cluster_by_kMedoidsLloyd_single(
    int rep, bool persist_artifacts);
  void init_with_seed(std::uint64_t seed);

  void writeBestRep(int best_rep);
  void writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost);
  void distanceInClusters();

  void adopt_loaded_data(LoadedData loaded)
  {
    if (loaded.is_mmap()) {
      auto owner =
        std::make_unique<LoadedData>(std::move(loaded));
      Data view = owner->data;
#ifdef DTWC_HAS_MMAP
      if (!view.is_view()
          || owner->names.size() != view.size()) {
        throw std::logic_error(
          "Problem::adopt_loaded_data: mmap name ownership invariant failed.");
      }
      for (std::size_t i = 0; i < view.size(); ++i) {
        if (view.name(i).data() != owner->names[i].data()
            || view.name(i).size() != owner->names[i].size()) {
          throw std::logic_error(
            "Problem::adopt_loaded_data: mmap name ownership invariant failed.");
        }
      }
#endif
      data_ = std::move(view);
      series_storage_owner_ = std::move(owner);
      return;
    }
    data_ = std::move(loaded.data);
    series_storage_owner_.reset();
  }

  bool has_mmap_series_storage() const
  {
    return series_storage_owner_
        && series_storage_owner_->is_mmap();
  }

public:
  [[deprecated("use set_max_iter/max_iter")]]
  int maxIter{ 100 };                        /*!< Maximum number of iteration for iterative-methods. */
  [[deprecated("use set_n_repetitions/n_repetitions")]]
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

  std::vector<int> clusters_ind;  //!< Indices of which point belongs to which cluster. [0,Nc)
  std::vector<int> centroids_ind; //!< indices of cluster centroids. [0, Np)

  // Constructors:
  // GCC emits -Wdeprecated-declarations for in-class initializers of the
  // deprecated maxIter / N_repetition fields at every constructor definition.
  // Canonical construction must stay silent (F22); caller access of those
  // fields must still diagnose. Same push/pop as Problem.cpp accessors.
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4996)
#endif
  Problem() { rebind_dtw_fn(); }
  Problem(std::string_view problem_name) : name_{ problem_name }
  {
    rebind_dtw_fn();
  }
  Problem(std::string_view problem_name, DataLoader &loader)
    : storage_policy_{ loader.storage_policy() },
      ram_limit_bytes_{ loader.ram_limit() }, name_{ problem_name }
  {
    adopt_loaded_data(loader.load_stored());
    refresh_distance_matrix(); // also calls rebind_dtw_fn()
  }
#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif
  Problem(const Problem &) = delete;
  Problem &operator=(const Problem &) = delete;
  Problem(Problem &&);
  Problem &operator=(Problem &&) noexcept;

  auto size() const { return data_.size(); }
  /// Number of clusters (canonical 2.0 read accessor; was `cluster_size()`).
  auto n_clusters() const { return Nc; }
  [[deprecated("use n_clusters")]] auto cluster_size() const { return n_clusters(); }

  /// Mutable name access (heap-mode only — asserts if view-mode).
  auto &get_name(size_t i)
  {
    assert(!data_.is_view());
    return data_.p_names[i];
  }
  auto const &get_name(size_t i) const
  {
    assert(!data_.is_view());
    return data_.p_names[i];
  }

  /// Mutable vector access (heap-mode only — asserts if view-mode).
  /// Call refresh_distance_matrix() before mutating values when any distance
  /// cache has been used; raw in-place edits during a bound-cache session are
  /// unsupported because warm lookups intentionally remain O(1).
  auto &p_vec(size_t i)
  {
    assert(!data_.is_view());
    return data_.p_vec[i];
  }
  auto const &p_vec(size_t i) const
  {
    assert(!data_.is_view());
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
  int centroid_of(int i_p) const { return centroids_ind[clusters_ind[i_p]]; } // [0, Np) Get the centroid of the cluster of i_p

  void read_distance_matrix(const fs::path &distMat_path);
  [[deprecated("use read_distance_matrix")]]
  void readDistanceMatrix(const fs::path &p) { read_distance_matrix(p); }

  void set_n_clusters(int Nc_);
  [[deprecated("use set_n_clusters")]] void set_numberOfClusters(int Nc_) { set_n_clusters(Nc_); }

  void set_clusters(std::vector<int> &candidate_centroids);
  bool set_solver(dtwc::Solver solver_);

  // Canonical configuration reads for the ten encapsulated fields.
  // last_iterations() and data() are read-only; mutable configuration uses
  // the corresponding setters below.
  Method method() const { return method_; }
  std::uint64_t random_seed() const { return random_seed_; }
  int last_iterations() const { return last_iterations_; }
  double tadpole_dc() const { return tadpole_dc_; }
  LowerBoundStrategy lb_strategy() const { return lb_strategy_; }
  core::StoragePolicy storage_policy() const { return storage_policy_; }
  std::size_t ram_limit() const { return ram_limit_bytes_; }
  bool verbose() const { return verbose_; }
  const path_t &output_folder() const { return output_folder_; }
  const std::string &name() const { return name_; }
  const Data &data() const { return data_; }

  void set_method(Method m)
  {
    validate_method(m);
    method_ = m;
  }
  void set_band(int b)
  {
    preflight_current_distance_semantics();
    if (band == b) return;
    band = b;
    refresh_distance_matrix();
  }
  void set_max_iter(int n);
  int max_iter() const;
  void set_n_repetitions(int n);
  int n_repetitions() const;
  void set_random_seed(std::uint64_t seed) { random_seed_ = seed; }
  void set_tadpole_dc(double dc) { tadpole_dc_ = dc; }
  void set_missing_strategy(core::MissingStrategy strategy)
  {
    preflight_distance_semantics(
      variant_params, strategy, data_, distance_strategy, cuda_settings);
    if (missing_strategy == strategy) return;
    missing_strategy = strategy;
    refresh_distance_matrix();
  }
  void set_distance_strategy(DistanceMatrixStrategy strategy)
  {
    preflight_distance_semantics(
      variant_params, missing_strategy, data_, strategy, cuda_settings);
    if (distance_strategy == strategy) return;
    distance_strategy = strategy;
    refresh_distance_matrix();
  }
  void set_lb_strategy(LowerBoundStrategy strategy)
  {
    validate_lower_bound_strategy(strategy);
    if (lb_strategy_ == strategy) return;
    // Lower bounds are exact optimization hints and do not change distances.
    lb_strategy_ = strategy;
  }
  void set_storage_policy(core::StoragePolicy policy)
  {
    core::validate_storage_policy(policy);
    if (storage_policy_ == policy) return;
    // Governs future owning set_data calls; installed data is not moved.
    storage_policy_ = policy;
  }
  /// Footprint threshold override for the next owning set_data (bytes; 0 = default).
  void set_ram_limit(std::size_t bytes) { ram_limit_bytes_ = bytes; }
  void set_cuda_settings(CUDASettings settings)
  {
    preflight_distance_semantics(
      variant_params, missing_strategy, data_, distance_strategy, settings);
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
    preflight_distance_semantics(
      variant_params, missing_strategy, candidate, distance_strategy, cuda_settings);
    auto loaded = detail::route_series_storage(
      std::move(candidate),
      storage_policy_,
      ram_limit_bytes_,
      {},
      "Problem::set_data");
    adopt_loaded_data(std::move(loaded));
    refresh_distance_matrix();
  }

  /// Set view-mode data (non-owning spans). Sizes distance matrix but skips mmap cache.
  void set_view_data(dtwc::Data candidate)
  {
    core::validate_precision(candidate.precision);
    candidate.validate_ndim();
    preflight_distance_semantics(
      variant_params, missing_strategy, candidate, distance_strategy, cuda_settings);
    data_ = std::move(candidate);
    series_storage_owner_.reset();
    refresh_distance_matrix();
    resize(); // sizes distance matrix for new N
  }

  /// Set DTW variant and rebind the distance function.
  void set_variant(core::DTWVariant v);
  void set_variant(core::DTWVariantParams params);

  data_t max_distance() const
  {
    validate_mmap_cache_identity();
    validate_dense_cache_configuration();
    return visit_distmat([](const auto &m) { return m.max(); });
  }
  [[deprecated("use max_distance")]] data_t maxDistance() const { return max_distance(); }

  data_t dist_by_ind(int i, int j);
  [[deprecated("use dist_by_ind")]] data_t distByInd(int i, int j) { return dist_by_ind(i, j); }

  /// Access the bound DTW distance function (float64). Mutable access repairs
  /// legacy raw configuration mutations before returning the dispatcher;
  /// const access rejects stale semantics instead of silently using them.
  /// The first accessor after a Problem move repairs logically-const derived
  /// dispatch state and must run before parallel use.
  const dtw_fn_t &dtw_function()
  {
    ensure_dtw_function_configuration_current();
    return dtw_fn_;
  }
  const dtw_fn_t &dtw_function() const
  {
    validate_dtw_function_configuration();
    return dtw_fn_;
  }

  /// Float32 counterpart of dtw_function(), with the same semantic guard.
  const dtw_fn_f32_t &dtw_function_f32()
  {
    preflight_float32_distance_semantics();
    ensure_dtw_function_configuration_current();
    return validated_dtw_function_f32();
  }
  const dtw_fn_f32_t &dtw_function_f32() const
  {
    preflight_float32_distance_semantics();
    validate_dtw_function_configuration();
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
    if (std::holds_alternative<core::DenseDistanceMatrix>(distMat)
        && !dense_cache_configuration_is_current())
      return false;
    return visit_distmat([](const auto &m) { return m.size() > 0 && m.all_computed(); });
  }
  [[deprecated("use is_distance_matrix_filled")]] bool isDistanceMatrixFilled() const { return is_distance_matrix_filled(); }

  /// Access the underlying distance matrix (const).
  const distMat_t &distance_matrix() const
  {
    validate_mmap_cache_identity();
    validate_dense_cache_configuration();
    return distMat;
  }
  /// Access the underlying distance matrix (mutable).
  distMat_t &distance_matrix()
  {
    validate_mmap_cache_identity();
    ensure_dense_cache_configuration_current();
    return distMat;
  }

  /// Access the Dense distance matrix. Throws std::bad_variant_access if mmap is active.
  const core::DenseDistanceMatrix &dense_distance_matrix() const
  {
    validate_dense_cache_configuration();
    return std::get<core::DenseDistanceMatrix>(distMat);
  }
  core::DenseDistanceMatrix &dense_distance_matrix()
  {
    ensure_dense_cache_configuration_current();
    return std::get<core::DenseDistanceMatrix>(distMat);
  }
  /// Full data-plus-distance-semantics identity used by durable checkpoints.
  /// Names and clustering outputs are intentionally excluded because they do
  /// not affect any stored distance.
  core::MmapDistanceMatrix::fingerprint_type distance_checkpoint_identity(
    core::MetricType metric = core::MetricType::L1) const;
  /// Bind persistent storage to this Problem's exact data/configuration.
  /// Non-L1 identities are for matching external/GPU producers only; the CPU
  /// lazy/fill paths reject them before writing because their local cost is L1.
  /// The full data fingerprint is verified at bind and once again on first use;
  /// subsequent warm lookups compare a fixed-size configuration snapshot to
  /// preserve O(1) access. After first use, replace data through set_data and
  /// distance-semantic setters, or call refresh_distance_matrix() before any
  /// raw in-place Data edit; mutating raw storage during a bound session is
  /// unsupported.
  void use_mmap_distance_matrix(
    const std::filesystem::path &cache_path,
    core::MetricType metric = core::MetricType::L1);

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
  [[deprecated("use cluster_by_kmedoids_lloyd")]] void cluster_by_kMedoidsLloyd() { cluster_by_kmedoids_lloyd(); }

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
