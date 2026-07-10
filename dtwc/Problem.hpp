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
#include "fileOperations.hpp" // for load_batch_file, readFile
#include "settings.hpp"       // for data_t, resultsPath
#include "enums/enums.hpp"    // for using Enum types.
#include "initialisation.hpp" // for init functions
#include "core/dtw_options.hpp" // for DTWVariant
#include "core/storage.hpp"     // for StoragePolicy

#include "core/mmap_distance_matrix.hpp"
#include <variant>

#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t
#include <filesystem>  // for operator/, path
#include <ostream>     // for operator<<, basic_ostream, ofstream
#include <string>      // for char_traits, operator+, operator<<
#include <string_view> // for string_view
#include <utility>     // for pair
#include <vector>      // for vector, allocator
#include <type_traits> // std::decay_t
#include <functional>  // std::function
#include <unordered_map> // std::unordered_map
#include <span>        // std::span
#include <iostream>

#include "core/distance_matrix.hpp"

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

/// MIP solver tuning parameters.
struct MIPSettings {
  double mip_gap = 1e-5;          ///< Relative MIP gap tolerance.
  int time_limit_sec = -1;        ///< Solver time limit in seconds (-1 = unlimited).
  bool warm_start = true;          ///< Run FastPAM first and feed as MIP start.
  int numeric_focus = 1;           ///< Gurobi NumericFocus (0-3).
  int mip_focus = 2;               ///< Gurobi MIPFocus (0=balanced, 1=feasible, 2=optimal, 3=bound).
  bool verbose_solver = false;     ///< Show solver log output.
  int max_benders_iter = 200;      ///< Maximum Benders iterations.
  std::string benders = "auto";    ///< Benders mode: "auto" (N>200), "on", "off".
};

/// Strategy for computing the pairwise distance matrix.
enum class DistanceMatrixStrategy {
  Auto,       ///< Choose best strategy automatically
  BruteForce, ///< Parallel brute-force (no lower-bound pruning)
  Pruned,     ///< Parallel with lower-bound pruning (LB_Kim / LB_Keogh / cascade)
  CUDA,       ///< NVIDIA CUDA GPU (requires DTWC_HAS_CUDA)
  Metal       ///< Apple Metal GPU (requires DTWC_HAS_METAL)
};

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
  dtw_fn_t dtw_fn_;                                 /*!< DTW distance function for float64. */
  dtw_fn_f32_t dtw_fn_f32_;                         /*!< DTW distance function for float32. */
  std::unordered_map<size_t, std::vector<data_t>> wdtw_weights_cache_; /*!< Precomputed WDTW weights keyed by max_dev. */

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
  DistanceCacheIdentity mmap_cache_identity_{};
  bool mmap_cache_identity_bound_{ false };
  mutable bool mmap_cache_data_validated_{ false };

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

  void rebind_dtw_fn(); ///< Rebind dtw_fn_ based on current variant_params and band.
  void refresh_variant_caches(); ///< Refresh precomputed variant-specific caches.
  cache_fingerprint_t distance_cache_configuration_fingerprint(
    core::MetricType metric) const;
  DistanceCacheConfiguration distance_cache_configuration(
    core::MetricType metric) const;
  bool distance_cache_configuration_matches(
    const DistanceCacheConfiguration &expected) const;
  DistanceCacheIdentity distance_cache_identity(core::MetricType metric) const;
  void validate_mmap_cache_identity() const;
  void clear_mmap_cache_identity();
  void fillDistanceMatrix_BruteForce(); ///< Brute-force parallel distance matrix fill.
  void resize(); ///< Resize cluster/centroid buffers to size()/Nc. Internal invariant maintenance (Task 1.6: private).

  // Private functions:
  friend void MIP_clustering_byBenders(Problem &prob);
  // Benders disables only heuristic artifact files; public Lloyd always passes true.
  void cluster_by_kmedoids_lloyd_impl(bool persist_artifacts);
  std::tuple<int, double, int> cluster_by_kMedoidsLloyd_single(
    int rep, bool persist_artifacts);
  void init_with_seed(std::uint64_t seed);

  void writeBestRep(int best_rep);
  void writeMedoids(std::vector<std::vector<int>> &centroids_all, int rep, double total_cost);
  void distanceInClusters();

public:
  Method method{ Method::Kmedoids };         /*!< Clustering method. */
  int maxIter{ 100 };                        /*!< Maximum number of iteration for iterative-methods. */
  int N_repetition{ 1 };                     /*!< Repetition for iterative-methods. */
  std::uint64_t random_seed{ settings::DEFAULT_RANDOM_SEED }; /*!< Invocation-local seed for Tier-1 stochastic work. */
  int last_iterations{ 0 };                  /*!< Actual iteration count from last clustering run. */
  int band{ settings::DEFAULT_BAND }; /*!< Band length for Sakoe-Chiba band, -1 for full DTW. */
  double tadpole_dc{ -1.0 };          /*!< TADPole density cutoff dc (Method::TADPole). <0 ⇒ auto-select from a DTW subsample. */
  /// DTW variant selection and parameters.
  /// INVARIANT: a direct write to this field does NOT rebind `dtw_fn_` — always
  /// use `set_variant(...)` (which rebinds) to change the variant safely. The
  /// field stays public only because the Python bindings bind it by address
  /// (`_dtwcpp_core.cpp:426`); it becomes private behind `set_variant` in Phase 2.
  core::DTWVariantParams variant_params;
  core::MissingStrategy missing_strategy = core::MissingStrategy::Error; /*!< Strategy for handling NaN values in series. */
  DistanceMatrixStrategy distance_strategy{ DistanceMatrixStrategy::Auto }; /*!< Distance matrix strategy. */
  LowerBoundStrategy lb_strategy{ LowerBoundStrategy::Auto }; /*!< Lower-bound selection for the Pruned CPU path. */
  core::StoragePolicy storage_policy{ core::StoragePolicy::Auto }; /*!< How series data is stored. */
  CUDASettings cuda_settings;                /*!< GPU options (used when distance_strategy == GPU). */
  MIPSettings mip_settings;                  /*!< MIP solver tuning parameters. */
  bool verbose{ false };                     /*!< Print progress messages for long-running operations. */

  std::function<void(Problem &)> init_fun{ init::random }; /*!< Initialisation function. */

  path_t output_folder{ settings::paths::results }; /*!< Output folder for results. */
  std::string name{};                            /*!< Problem name. */
  Data data;                                     /*!< Data associated with the problem. */

  std::vector<int> clusters_ind;  //!< Indices of which point belongs to which cluster. [0,Nc)
  std::vector<int> centroids_ind; //!< indices of cluster centroids. [0, Np)

  // Constructors:
  Problem() { rebind_dtw_fn(); }
  Problem(std::string_view name_) : name{ name_ } { rebind_dtw_fn(); }
  Problem(std::string_view name_, DataLoader &loader_)
    : name{ name_ }, data{ loader_.load() }
  {
    refresh_distance_matrix(); // also calls rebind_dtw_fn()
  }

  auto size() const { return data.size(); }
  /// Number of clusters (canonical 2.0 read accessor; was `cluster_size()`).
  auto n_clusters() const { return Nc; }
  [[deprecated("use n_clusters")]] auto cluster_size() const { return n_clusters(); }

  /// Mutable name access (heap-mode only — asserts if view-mode).
  auto &get_name(size_t i) { assert(!data.is_view()); return data.p_names[i]; }
  auto const &get_name(size_t i) const { assert(!data.is_view()); return data.p_names[i]; }

  /// Mutable vector access (heap-mode only — asserts if view-mode).
  /// Call refresh_distance_matrix() before mutating values when any distance
  /// cache has been used; raw in-place edits during a bound-cache session are
  /// unsupported because warm lookups intentionally remain O(1).
  auto &p_vec(size_t i) { assert(!data.is_view()); return data.p_vec[i]; }
  auto const &p_vec(size_t i) const { assert(!data.is_view()); return data.p_vec[i]; }

  /// Zero-copy span view of series i (works for heap, mmap, and view modes).
  std::span<const data_t> series(size_t i) const { return data.series(i); }

  /// Name of series i as a string_view.
  std::string_view series_name(size_t i) const { return data.name(i); }

  /// Canonical read accessors (API contract §2.2): the raw fields
  /// `clusters_ind`/`centroids_ind` stay public, but `labels()`/`medoids()` are
  /// the cross-language read path (parity with `Result::labels`/`Result::medoids`).
  const std::vector<int> &labels() const { return clusters_ind; }
  const std::vector<int> &medoids() const { return centroids_ind; }

  void refresh_distance_matrix();
  [[deprecated("use refresh_distance_matrix")]] void refreshDistanceMatrix() { refresh_distance_matrix(); }

  // Getters and setters:
  int centroid_of(int i_p) const { return centroids_ind[clusters_ind[i_p]]; } // [0, Np) Get the centroid of the cluster of i_p

  // read_distance_matrix / write_* are defined out-of-line in Problem_IO.cpp under
  // their camelCase names; the snake_case canonical names are additive forwarders
  // this phase (the camelCase originals retire in the Phase 2 IO pass).
  void readDistanceMatrix(const fs::path &distMat_path);
  void read_distance_matrix(const fs::path &p) { readDistanceMatrix(p); }

  void set_n_clusters(int Nc_);
  [[deprecated("use set_n_clusters")]] void set_numberOfClusters(int Nc_) { set_n_clusters(Nc_); }

  void set_clusters(std::vector<int> &candidate_centroids);
  bool set_solver(dtwc::Solver solver_);

  // Canonical 2.0 config setters/accessors. The underlying fields (`method`,
  // `band`, `maxIter`, `N_repetition`) stay public this phase for binding
  // compatibility — the Python bindings take `&Problem::maxIter` /
  // `&Problem::N_repetition` by address (`_dtwcpp_core.cpp:423-424`); full field
  // privatisation lands with the Phase 2 binding rewrite. Prefer these setters:
  // `set_band` also invalidates any populated dense/mmap matrix. Legacy naked
  // writes remain source-compatible; a bound mmap cache detects them before
  // returning a computed distance.
  void set_method(Method m) { method = m; }
  void set_band(int b)
  {
    if (band == b) return;
    band = b;
    refresh_distance_matrix();
  }
  void set_max_iter(int n) { maxIter = n; }
  int max_iter() const { return maxIter; }
  void set_n_repetitions(int n) { N_repetition = n; }
  int n_repetitions() const { return N_repetition; }
  void set_random_seed(std::uint64_t seed) { random_seed = seed; }

  void set_data(dtwc::Data data_)
  {
    data_.validate_ndim();
    data = std::move(data_);
    refresh_distance_matrix();
  }

  /// Set view-mode data (non-owning spans). Sizes distance matrix but skips mmap cache.
  void set_view_data(dtwc::Data data_)
  {
    data = std::move(data_);
    // validate_ndim() already called by Data's view-mode constructor
    refresh_distance_matrix();
    resize(); // sizes distance matrix for new N
  }

  /// Set DTW variant and rebind the distance function.
  void set_variant(core::DTWVariant v);
  void set_variant(core::DTWVariantParams params);

  data_t max_distance() const
  {
    validate_mmap_cache_identity();
    return visit_distmat([](const auto &m) { return m.max(); });
  }
  [[deprecated("use max_distance")]] data_t maxDistance() const { return max_distance(); }

  data_t dist_by_ind(int i, int j);
  [[deprecated("use dist_by_ind")]] data_t distByInd(int i, int j) { return dist_by_ind(i, j); }

  /// Access the bound DTW distance function (float64).
  const dtw_fn_t &dtw_function() const { return dtw_fn_; }

  /// Access the bound DTW distance function (float32).
  const dtw_fn_f32_t &dtw_function_f32() const { return dtw_fn_f32_; }

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
    return visit_distmat([](const auto &m) { return m.size() > 0 && m.all_computed(); });
  }
  [[deprecated("use is_distance_matrix_filled")]] bool isDistanceMatrixFilled() const { return is_distance_matrix_filled(); }

  /// Access the underlying distance matrix (const).
  const distMat_t &distance_matrix() const
  {
    validate_mmap_cache_identity();
    return distMat;
  }
  /// Access the underlying distance matrix (mutable).
  distMat_t &distance_matrix()
  {
    validate_mmap_cache_identity();
    return distMat;
  }

  /// Access the Dense distance matrix. Throws std::bad_variant_access if mmap is active.
  const core::DenseDistanceMatrix &dense_distance_matrix() const
  {
    return std::get<core::DenseDistanceMatrix>(distMat);
  }
  core::DenseDistanceMatrix &dense_distance_matrix()
  {
    return std::get<core::DenseDistanceMatrix>(distMat);
  }
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

  // I/O writers (definitions in Problem_IO.cpp). snake_case names are additive
  // canonical forwarders; camelCase originals retire in the Phase 2 IO pass.
  void writeDistanceMatrix(const std::string &name_) const;
  void writeDistanceMatrix() const { writeDistanceMatrix(name + "_distanceMatrix.csv"); }
  void write_distance_matrix(const std::string &name_) const { writeDistanceMatrix(name_); }
  void write_distance_matrix() const { writeDistanceMatrix(); }

  void printClusters() const;
  void print_clusters() const { printClusters(); }
  void writeClusters();
  void write_clusters() { writeClusters(); }

  void writeMedoidMembers(int iter, int rep = 0) const;
  void write_medoid_members(int iter, int rep = 0) const { writeMedoidMembers(iter, rep); }
  void writeSilhouettes();
  void write_silhouettes() { writeSilhouettes(); }

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
