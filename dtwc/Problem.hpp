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
#include <span>        // std::span
#include <memory>
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
  index_t Nc{ 1 };                                  /*!< Number of clusters. */
  core::DistanceMatrix distMat;                     /*!< Distance matrix, on the heap or mapped (use_mmap_distance_matrix). */
  Solver mipSolver{ settings::DEFAULT_MIP_SOLVER }; /*!< Solver for MIP. */
  /// What every distance of this Problem means. Only the setters change it;
  /// `band` is the v1 field's value when it was bound, and `ndim` the series'.
  core::DistanceConfig distance_{};
  DistanceMatrixStrategy distance_strategy_{ DistanceMatrixStrategy::Auto };
  CUDASettings cuda_settings_{};
  /// Bound from distance_ (and, for WDTW, the series lengths) whenever either
  /// changes; each holds copies of the settings it reads. The float32 one is
  /// bound then for float32 series, else by dtw_function_f32() (rebind_dtw_fn).
  dtw_fn_t dtw_fn_;
  dtw_fn_f32_t dtw_fn_f32_;
  /// The fill's lane functions (core::resolve_dtw_block_fn); empty where none applies.
  std::function<void(std::span<const data_t>, std::span<const std::span<const data_t>>,
                     std::span<double>)> dtw_block_fn_;
  std::function<void(std::span<const float>, std::span<const std::span<const float>>,
                     std::span<double>)> dtw_block_fn_f32_;
  /// distMat holds every pair under distance_: set by a fill and by a complete
  /// load or bind; cleared by any change of series or distance settings and by
  /// writable_distance_matrix().
  bool filled_{ false };
  /// A caller may have written into distMat through
  /// writable_distance_matrix(): the next fill scans it, where those writes commit.
  /// Every other way a matrix enters scans it there.
  bool written_{ false };

  Method method_{ Method::Kmedoids };
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

  void rebind_dtw_fn(); ///< Resolve the distance functions from distance_ and the series.
  /// A direct write to the v1 field `band` takes effect here, as set_band(band).
  void sync_band();
  /// core::validate(config) for `data`'s channels and precision, before a setter
  /// or set_data changes anything.
  static void validate_distance(core::DistanceConfig config, const Data &data);
  void validate_checkpoint_settings() const;
  /// FX-1: the one check of a distance request — every device axis, band
  /// feasibility and (FX-15) the series values the missing-data strategy
  /// cannot take — before any pair is computed. O(N·L): run once per fill and
  /// per dtw_function() call, never per pair.
  void validate_fill_request(std::string_view where) const;
  /// get_name / p_vec return references into owned heap storage. A view
  /// (set_view_data) has none, nor has a Float32 store Float64 values:
  /// indexing would read past an empty vector in a Release build (F25).
  void require_owned_storage(std::string_view accessor, bool float64_values) const;
  void fillDistanceMatrix_BruteForce(); ///< Brute-force parallel distance matrix fill.

  // Private functions:
  friend bool load_checkpoint(Problem &prob, const std::string &path,
                              core::MetricType metric);
  std::tuple<int, double, int> cluster_by_kMedoidsLloyd_single(
    int rep, bool persist_artifacts);
  void init_with_seed(std::uint64_t seed);

  void writeBestRep(int best_rep);
  void writeMedoids(std::vector<std::vector<index_t>> &centroids_all, int rep, double total_cost);

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
  /// Band length for Sakoe-Chiba band, -1 for full DTW (the v1 field). Prefer
  /// set_band(); a direct write takes effect at the next fill.
  int band{ settings::DEFAULT_BAND };
  MIPSettings mip_settings;                  /*!< MIP solver tuning parameters. */
  CheckpointOptions checkpoint;              /*!< Automatic mid-fill checkpointing (see checkpoint.hpp). */

  std::function<void(Problem &)> init_fun{ init::random }; /*!< Initialisation function. */

  /// Empty until a clustering writes them (set_result, set_clusters for the
  /// medoids, the algorithms); set_n_clusters sizes neither, and set_data and
  /// set_view_data empty both (the labels describe the old series).
  /// require_clustered() tells a clustering from a sizing.
  std::vector<index_t> clusters_ind;  //!< Indices of which point belongs to which cluster. [0,Nc)
  std::vector<index_t> centroids_ind; //!< indices of cluster centroids. [0, Np)

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
    distance_.ndim = data_.ndim;
    refresh_distance_matrix(); // also calls rebind_dtw_fn()
  }
  Problem(const Problem &) = delete;
  Problem &operator=(const Problem &) = delete;
  Problem(Problem &&);
  /// Not noexcept: moving the members (the dispatchers, the WDTW weights they
  /// hold) may throw, and a noexcept promise would turn that into std::terminate (S-13).
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
  /// Call refresh_distance_matrix() after mutating values: the distance matrix,
  /// on the heap or mapped, holds the distances of the series as they were.
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
  const std::vector<index_t> &labels() const { return clusters_ind; }
  const std::vector<index_t> &medoids() const { return centroids_ind; }

  void refresh_distance_matrix();
  [[deprecated("use refresh_distance_matrix")]] void refreshDistanceMatrix() { refresh_distance_matrix(); }

  // Getters and setters:
  /// The centroid of the cluster of i_p, i_p in [0, N). Unchecked like dist_by_ind:
  /// write_clusters and the bindings call it per series, so a caller that cannot
  /// vouch for the clustering calls require_clustered() once first.
  index_t centroid_of(index_t i_p) const { return centroids_ind[clusters_ind[i_p]]; }

  void read_distance_matrix(const fs::path &distMat_path);
  [[deprecated("use read_distance_matrix")]]
  void readDistanceMatrix(const fs::path &p) { read_distance_matrix(p); }

  void set_n_clusters(index_t Nc_);
  [[deprecated("use set_n_clusters")]] void set_numberOfClusters(int Nc_) { set_n_clusters(Nc_); }

  /// A Problem is clustered when it holds one label per series and one medoid per
  /// cluster. Every call that reads the whole clustering checks that once.
  /// @throws InvalidInput ("<who>: ... cluster it first") when it is not.
  void require_clustered(std::string_view who) const;
  void set_clusters(const std::vector<index_t> &candidate_centroids);
  /// The v1.0.0 signature, kept so v1 code compiles; a braced list takes the
  /// index_t overload.
  [[deprecated("use set_clusters(const std::vector<index_t> &)")]]
  void set_clusters(std::vector<int> &candidate_centroids)
  {
    set_clusters(std::vector<index_t>(candidate_centroids.begin(), candidate_centroids.end()));
  }
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
    method_ = m;
  }
  /// The distance settings: variant, metric, missing-data strategy, band (the
  /// v1 field, a direct write included) and the series' ndim.
  core::DistanceConfig distance() const noexcept
  {
    auto config = distance_;
    config.band = band;
    return config;
  }
  const core::DTWVariantParams &variant_params() const noexcept { return distance_.variant; }
  core::MissingStrategy missing_strategy() const noexcept { return distance_.missing; }
  core::MetricType metric() const noexcept { return distance_.metric; }

  /// Change the distance settings; `config.ndim` is ignored, the series decide
  /// it. A change drops the distance matrix and the clustering, which describe
  /// the old distances; the same settings change nothing.
  /// @throws InvalidInput for a band below -1, or a combination no kernel
  ///         implements (see set_metric); the Problem is then unchanged.
  void set_distance(core::DistanceConfig config);
  /// @throws InvalidInput for b < -1: -1 is full DTW and b >= 0 a Sakoe-Chiba
  ///         half-width; nothing lies between.
  void set_band(int b);
  /// @throws InvalidInput for n < 1: no iteration would report the initial
  ///         medoids' cost as a clustering result (O-06).
  void set_max_iter(int n);
  int max_iter() const;
  /// @throws InvalidInput for n < 1: at least one run is needed (O-06).
  void set_n_repetitions(int n);
  int n_repetitions() const;
  void set_random_seed(std::uint64_t seed) { random_seed_ = seed; }
  void set_tadpole_dc(double dc) { tadpole_dc_ = dc; }
  void set_missing_strategy(core::MissingStrategy strategy);
  /// Pointwise cost of every distance this Problem computes: the CPU fill and
  /// lazy lookups, the GPU routes, the mmap cache and checkpoint identities.
  /// L1 by default. A metric other than L1 is implemented for Standard DTW, with
  /// or without a missing-data strategy, and for DDTW (univariate or
  /// multivariate; L2 with AROW univariate only).
  /// @throws InvalidInput for a metric other than L1 with WDTW, ADTW, Soft-DTW,
  ///         MSM or TWE, whose kernels compute L1 (core::validate).
  void set_metric(core::MetricType metric);
  DistanceMatrixStrategy distance_strategy() const noexcept { return distance_strategy_; }
  void set_distance_strategy(DistanceMatrixStrategy strategy)
  {
    if (distance_strategy_ == strategy) return;
    distance_strategy_ = strategy;
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
  /// GPU options (the device index and precision), read by the CUDA and Metal routes.
  const CUDASettings &cuda_settings() const noexcept { return cuda_settings_; }
  /// @throws InvalidInput for a negative device_id, as set_device refuses the same index.
  void set_cuda_settings(CUDASettings settings)
  {
    if (settings.device_id < 0)
      throw InvalidInput("Problem::set_cuda_settings: device_id must be >= 0; got "
                         + std::to_string(settings.device_id) + ".");
    if (cuda_settings_.device_id == settings.device_id
        && cuda_settings_.precision == settings.precision)
      return;
    cuda_settings_ = settings;
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
    candidate.validate_ndim();
    reject_empty_series(candidate, "Problem::set_data");
    validate_distance(distance_, candidate);
    data_ = std::move(candidate);
    distance_.ndim = data_.ndim;
    clusters_ind.clear(); // a clustering describes the series it was computed on
    centroids_ind.clear();
    refresh_distance_matrix();
  }

  /// Set view-mode data (non-owning spans). Sizes distance matrix but skips mmap cache.
  void set_view_data(dtwc::Data candidate)
  {
    candidate.validate_ndim();
    reject_empty_series(candidate, "Problem::set_view_data");
    validate_distance(distance_, candidate);
    data_ = std::move(candidate);
    distance_.ndim = data_.ndim;
    clusters_ind.clear();
    centroids_ind.clear();
    refresh_distance_matrix();
  }

  /// Set DTW variant and rebind the distance function.
  void set_variant(core::DTWVariant v);
  void set_variant(core::DTWVariantParams params);

  data_t max_distance() const { return distMat.max(); }
  [[deprecated("use max_distance")]] data_t maxDistance() const { return max_distance(); }

  /// The distance between series i and j, read from the matrix: O(1), no check
  /// and no computation, so parallel loops read it freely. Unchecked like
  /// centroid_of: i and j are in [0, N) and the matrix holds the pair — call
  /// fill_distance_matrix() first, as every method that reads the matrix does.
  data_t dist_by_ind(index_t i, index_t j) const
  {
    return distMat.get(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
  }
  /// The v1.0.0 lookup, which computed a pair on demand: it fills the matrix on
  /// its first call, then reads it.
  [[deprecated("use dist_by_ind")]] data_t distByInd(int i, int j)
  {
    fill_distance_matrix();
    return dist_by_ind(i, j);
  }

  /// The bound DTW distance function (float64), for methods that compute the
  /// pairs they need instead of filling the matrix. Each call first takes a
  /// direct write to `band` into account and checks the request against the
  /// series (validate_fill_request, O(N·L)), so call it once, serially, before a
  /// parallel loop; the function itself only reads what it holds.
  /// @throws InvalidInput for a band no pair's warping path fits, a ±inf
  ///         series value, or a NaN under MissingStrategy::Error; DeviceError
  ///         for a GPU strategy the kernels cannot honour.
  const dtw_fn_t &dtw_function();
  /// Float32 counterpart of dtw_function(), with the same checks.
  /// @throws InvalidInput also for an active variant parameter float32 cannot represent.
  const dtw_fn_f32_t &dtw_function_f32();

  /// True when the matrix holds every pair under the current settings: after a
  /// fill, a complete read or load, or binding a complete mapped file. O(1).
  bool is_distance_matrix_filled() const { return filled_ && band == distance_.band; }
  [[deprecated("use is_distance_matrix_filled")]] bool isDistanceMatrixFilled() const { return is_distance_matrix_filled(); }

  /// The distance matrix, on the heap or mapped, to read.
  const core::DistanceMatrix &distance_matrix() const { return distMat; }
  /// The distance matrix, opened for writing. The caller may change which
  /// pairs are known, so the Problem no longer calls it filled: the next
  /// fill_distance_matrix() refuses a pair set to ±inf (InvalidInput) and
  /// computes the pairs left NaN, none when all are set. A reader calls the
  /// const distance_matrix(), which changes nothing.
  core::DistanceMatrix &writable_distance_matrix()
  {
    sync_band();
    filled_ = false;
    written_ = true;
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
  /// other series or settings or one is ±inf, IOError if it is not a whole
  /// `.dtwm` file); an absent one is created. IOError on a build without llfio.
  /// The fingerprint of the data and the settings is checked here, once; every
  /// setter and set_data detach the file. Call refresh_distance_matrix() before
  /// editing series values in place.
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
