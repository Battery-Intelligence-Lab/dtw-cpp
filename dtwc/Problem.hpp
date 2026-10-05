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
#include "base/env.hpp"            // for Device, GpuPrecision
#include "base/error.hpp"          // for InvalidInput
#include "enums/enums.hpp"    // for using Enum types.
#include "initialisation.hpp" // for init functions
#include "core/dtw_options.hpp" // for DTWVariant
#include "core/storage.hpp"     // for Precision
#include "core/distance_matrix.hpp" // for DistanceMatrix
#include "core/clustering_result.hpp" // for set_result
#include "algorithms/hierarchical.hpp" // for Linkage

#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t, int64_t
#include <filesystem>  // for operator/, path
#include <string>      // for char_traits, operator+, operator<<
#include <string_view> // for string_view
#include <utility>     // for pair
#include <vector>      // for vector, allocator
#include <functional>  // std::function
#include <iosfwd>      // std::ostream
#include <span>        // std::span
#include <memory>
#include <stdexcept>

#include "checkpoint.hpp" // for CheckpointOptions, load_checkpoint

namespace dtwc {

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

class Problem;

/// The GPU rules that need no series: `prob`'s variant or missing-data
/// strategy the GPU kernels do not implement, Float32 series
/// (`series_precision`) and a precision Metal cannot honour. A fill applies them
/// with its series' precision (validate_fill_request); dtwc::run applies them
/// with the configured one before it reads a series. A CPU Problem passes.
/// @throws DeviceError naming `where`, the backend and the axis.
void validate_gpu_request(std::string_view where, const Problem &prob, core::Precision series_precision);

/// True when this build's GPU backend (CUDA, else Metal) finds a GPU, so
/// Device::GPU can compute here.
bool gpu_available();
/// One line naming this build's GPU backend and the GPU that Device::GPU (index
/// 0) computes on — "CUDA: <device>", "Metal: <device>" — or why there is none.
std::string gpu_info();

/// The method `auto` stands for: pam on a GPU, and on the CPU pam for up to 5000
/// series, clara above. Any other method is itself.
Method resolve_method(Method method, Device device, std::size_t n_series);

namespace detail {

/**
 * @brief Write a clustered Problem's result files into `directory`, which is created if missing.
 *
 * `<name>_labels.csv` and `<name>_medoids.csv`, then, when the distance matrix is filled,
 * `<name>_distance_matrix.csv` and, for k > 1, `<name>_silhouettes.csv`. An undefined silhouette is a
 * warning on stderr; any other failure propagates. A Problem without series (a RAM-limited Parquet run)
 * names its series `series_<i>` and has no matrix or silhouettes to write.
 *
 * @param complete  true (Result::save): fill the matrix first, so all four files are written, and throw
 *                  InvalidInput after the labels and medoids when there is no matrix to fill. false (the
 *                  CLI): write the matrix and silhouettes only if the matrix is already filled, so a
 *                  matrix-free run does not fill O(N^2) for them.
 * @param progress  when set, one line per file written.
 * @throws IOError for a file that cannot be written in full.
 */
void write_result_files(Problem &prob, const std::filesystem::path &directory, bool complete,
                        std::ostream *progress = nullptr);

} // namespace detail

/// What every method needs: at least one series, and no more clusters than series.
/// @throws InvalidInput "cluster: dataset is empty." or "cluster: k must not exceed the number of series."
void require_clusterable(index_t k, std::size_t n_series);

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
  Device device_{ Device::CPU };
  int device_index_{ 0 }; ///< The GPU ordinal of Device::GPU.
  GpuPrecision gpu_precision_{ GpuPrecision::Auto };
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
  index_t sample_size_{ -1 }; ///< CLARA's subsample size; -1: its automatic size
  int n_samples_{ 5 };        ///< CLARA's subsamples
  index_t batch_size_{ -1 };  ///< OneBatchPAM's batch size; -1: its automatic size
  algorithms::Linkage linkage_{ algorithms::Linkage::Average };
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
  /// The one check of a distance request — every device axis, band
  /// feasibility and the series values the missing-data strategy
  /// cannot take — before any pair is computed. O(N·L): run once per fill and
  /// per dtw_function() call, never per pair.
  void validate_fill_request(std::string_view where) const;
  /// get_name / p_vec return references into owned heap storage. A view
  /// (set_view_data) has none, nor has a Float32 store Float64 values:
  /// indexing would read past an empty vector in a Release build.
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
  /// hold) may throw, and a noexcept promise would turn that into std::terminate.
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

  /// Canonical read accessors: the raw fields
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
  index_t sample_size() const { return sample_size_; }
  int n_samples() const { return n_samples_; }
  index_t batch_size() const { return batch_size_; }
  algorithms::Linkage linkage() const { return linkage_; }
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
  ///         medoids' cost as a clustering result.
  void set_max_iter(int n);
  int max_iter() const;
  /// @throws InvalidInput for n < 1: at least one run is needed.
  void set_n_repetitions(int n);
  int n_repetitions() const;
  void set_random_seed(std::uint64_t seed) { random_seed_ = seed; }
  void set_tadpole_dc(double dc) { tadpole_dc_ = dc; }
  /// CLARA's subsample size and count, OneBatchPAM's batch size and the hierarchical
  /// linkage. cluster() checks them, as the method it runs takes them.
  void set_sample_size(index_t n) { sample_size_ = n; }
  void set_n_samples(int n) { n_samples_ = n; }
  void set_batch_size(index_t n) { batch_size_ = n; }
  void set_linkage(algorithms::Linkage linkage) { linkage_ = linkage; }
  void set_missing_strategy(core::MissingStrategy strategy);
  /// Pointwise cost of every distance this Problem computes: the CPU fill and
  /// lazy lookups, the GPU routes, the mmap cache and checkpoint identities.
  /// L1 by default. A metric other than L1 is implemented for Standard DTW, with
  /// or without a missing-data strategy, and for DDTW (univariate or
  /// multivariate; L2 with AROW univariate only).
  /// @throws InvalidInput for a metric other than L1 with WDTW, ADTW, Soft-DTW,
  ///         MSM or TWE, whose kernels compute L1 (core::validate).
  void set_metric(core::MetricType metric);
  /// Where this Problem computes distances: the CPU, or GPU `index` of this
  /// build's GPU backend (CUDA, else Metal), which the fill resolves. A Problem
  /// never reads the process-wide default (dtwc::device()); until told
  /// otherwise it computes on the CPU. A change drops the distance matrix.
  /// @throws DeviceError for `gpu` on a build with no GPU backend, and for a
  ///         GPU index other than 0 on Metal, which runs on the system default
  ///         GPU; InvalidInput for a negative index.
  void set_device(Device device, int index = 0);
  /// The device and GPU index set_device recorded.
  std::pair<Device, int> device() const noexcept { return { device_, device_index_ }; }
  /// What a GPU computes in (the CPU computes in the series' precision). A
  /// change drops the distance matrix.
  void set_gpu_precision(GpuPrecision precision)
  {
    if (gpu_precision_ == precision) return;
    gpu_precision_ = precision;
    refresh_distance_matrix();
  }
  GpuPrecision gpu_precision() const noexcept { return gpu_precision_; }

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
  ///         for a GPU device the kernels cannot honour.
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
  /// Cluster the series into n_clusters() by method() (`auto` resolved for this
  /// Problem's device and series). The labels and medoids are published as by
  /// set_result(); they are returned with the cost, the iterations and whether the
  /// method converged within max_iter().
  core::ClusteringResult cluster();
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
