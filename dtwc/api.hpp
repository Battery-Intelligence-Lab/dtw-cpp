/**
 * @file api.hpp
 * @brief Tier-1 DTWC++ 2.0 API: device -> load -> cluster -> Result.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "Data.hpp"
#include "base/env.hpp"               // device(), device(name)
#include "core/clustering_result.hpp" // ExternalSolver's result
#include "core/storage.hpp"           // core::Precision

#include <concepts>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace dtwc {

class Problem;
class Result;
struct Config;
enum class Method; // enums/Method.hpp
namespace core {
class DistanceMatrix; // core/distance_matrix.hpp
} // namespace core

// cli/run.hpp documents these; Result is built only by detail::run, which both run() overloads call.
Result run(const Config &config);
Result run(const Config &config, Data data);
using ExternalSolver = std::function<core::ClusteringResult(const std::shared_ptr<Problem> &)>;
namespace detail {
Result run(const Config &config, std::optional<Data> data, const ExternalSolver &solver);
/// The run's distance matrix as it is stored, packed, its values float64 or float32 (`matrix_dtype`), filled first
/// where the method left it unfilled: what a binding expands at that width (io::to_full_matrix<T>, as Python's
/// Problem.distance_matrix does), a float32 matrix as float32; Result::distance_matrix() is its expansion to doubles.
/// The Result's own, valid while the Result lives. @throws InvalidInput for a streamed run, as Result::score.
const core::DistanceMatrix &filled_matrix(const Result &result);
} // namespace detail

/** A lazy path-or-memory dataset handle.  Construct with dtwc::load(). */
class Dataset
{
public:
  using series_type = std::vector<std::vector<data_t>>;

  bool is_path() const noexcept;
  const std::filesystem::path &path() const;
  const std::string &name() const noexcept { return name_; }
  index_t skip_cols() const noexcept { return skip_cols_; }
  index_t skip_rows() const noexcept { return skip_rows_; }
  char delimiter() const noexcept { return delimiter_; }
  /// The precision cluster() stores the series in, as `--dtype` does.
  core::Precision dtype() const noexcept { return dtype_; }

private:
  friend Dataset load(const std::filesystem::path &, index_t, index_t, char, std::string_view, core::Precision);
  friend Dataset load(series_type, index_t, index_t, char, std::string_view, core::Precision);
  friend Result cluster(Dataset &&, index_t, std::string_view, int, std::string_view, int);

  explicit Dataset(std::filesystem::path source, index_t skip_cols, index_t skip_rows,
                   char delimiter, std::string name, core::Precision dtype);
  explicit Dataset(series_type source, index_t skip_cols, index_t skip_rows,
                   char delimiter, std::string name, core::Precision dtype);

  /// The in-memory series, moved out, with skip_rows / skip_cols applied (a path
  /// is run()'s to read).
  Data materialize_local() &&;

  std::variant<std::filesystem::path, series_type> source_;
  index_t skip_cols_ = 0;
  index_t skip_rows_ = 0;
  char delimiter_ = 0;
  std::string name_ = "dataset";
  core::Precision dtype_ = core::Precision::Float64;
};

/** Wrap a path lazily.  No file is opened until cluster() is called.
 *  `skip_rows` drops that many leading LINES of the file, as `--skip-rows` does.
 *  `dtype` Float32 stores the series as float, half the memory: the reader rounds
 *  each value, never holding a double copy. */
Dataset load(const std::filesystem::path &source, index_t skip_cols = 0,
             index_t skip_rows = 0, char delimiter = 0, std::string_view name = "",
             core::Precision dtype = core::Precision::Float64);

/** Wrap an in-memory row-per-series array.  `skip_rows` drops that many leading
 *  SERIES — one memory row is one file line. `dtype` Float32 rounds each value to
 *  float when cluster() runs. */
Dataset load(Dataset::series_type source, index_t skip_cols = 0, index_t skip_rows = 0,
             char delimiter = 0, std::string_view name = "",
             core::Precision dtype = core::Precision::Float64);

/** `load(src, skip_cols, delimiter)` does not compile: without these,
 *  `load(p, 0, ',')` would bind the char to `skip_rows` (',' == 44). Only a
 *  char matches: an int literal converts to index_t and to char alike, so a
 *  plain char parameter would make `load(p, 0, 1)` ambiguous. */
Dataset load(const std::filesystem::path &, index_t, std::same_as<char> auto,
             std::string_view = "") = delete;
Dataset load(Dataset::series_type, index_t, std::same_as<char> auto,
             std::string_view = "") = delete;

/** The owning result of a Tier-1 clustering call. */
class Result
{
public:
  const std::vector<index_t> &labels() const noexcept;
  const std::vector<index_t> &medoids() const noexcept;
  /** @throws InvalidInput for a streamed run (its series over --ram-limit, which FastCLARA read a record batch at a
   *  time): its Problem holds no series, so no distances to score. */
  double score(std::string_view name) const;
  /** Dense row-major N*N pairwise DTW distances, matching Python's
   *  `Result.distance_matrix`.  Fills the matrix first if it is not yet
   *  materialised, exactly as score() does: detail::filled_matrix, as doubles.
   *  @throws InvalidInput for a streamed run, as score(). */
  std::vector<double> distance_matrix() const;
  /** The four result files dtwc_cl writes, byte for byte (labels, medoids, the matrix and the silhouettes, filled and
   *  computed here where the run left them out), then the run report (report()), last, where the run made one. A
   *  streamed run writes what dtwc_cl writes for it: the labels and medoids, with the reader's names, then the report;
   *  it holds no series, so no matrix or silhouettes. */
  void save(const std::filesystem::path &directory) const;
  /** The run report: the run's machine, memory plan, stages, algorithm and result, in plain text, as run() writes it
   *  to `<name>_report.txt`; empty where the run was told `report = false` (dtwc_cl --no-report). */
  const std::string &report() const noexcept { return report_; }
  double cost() const noexcept { return cost_; }
  const std::string &device() const noexcept { return device_; }
  /** The method that ran, `auto` resolved (name_of(method_names, m) spells it). */
  Method method() const noexcept { return method_; }
  /** The method's iteration count, and whether it converged within max_iter. */
  int iterations() const noexcept { return iterations_; }
  bool converged() const noexcept { return converged_; }

private:
  friend Result detail::run(const Config &, std::optional<Data>, const ExternalSolver &);
  friend const core::DistanceMatrix &detail::filled_matrix(const Result &);

  Result(std::shared_ptr<Problem> problem, double cost, std::string device_name,
         Method method, int iterations, bool converged, std::vector<std::string> streamed_names,
         std::string report);

  std::shared_ptr<Problem> problem_;
  double cost_ = 0.0;
  std::string device_ = "cpu";
  Method method_{};
  int iterations_ = 0;
  bool converged_ = false;
  std::vector<std::string> streamed_names_; ///< a streamed run's series names, which its Problem cannot hold; empty in RAM
  std::string report_;                      ///< the run report's text
};

/**
 * Cluster a lazy Dataset: run() with dtwc_cl's defaults for every other setting,
 * writing nothing. `device` "" means the process device (dtwc::device()).
 *
 * Methods: auto (the default: pam on a GPU and for up to 5000 series on the CPU,
 * clara above), pam, onebatch, clara, kmedoids, mip, lrcore, tadpole,
 * hierarchical, with dtwc_cl's aliases (hclust, obp, lr).  Unknown names fail loudly.
 * An in-memory Dataset passed as an rvalue hands its series to the run; an lvalue
 * one is copied.
 */
Result cluster(const Dataset &data, index_t k, std::string_view method = "auto",
               int band = -1, std::string_view device = "", int max_iter = 100);
Result cluster(Dataset &&data, index_t k, std::string_view method = "auto",
               int band = -1, std::string_view device = "", int max_iter = 100);

} // namespace dtwc
