/**
 * @file api.hpp
 * @brief Tier-1 DTWC++ 2.0 API: device -> load -> cluster -> Result.
 */

#pragma once

#include "Data.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace dtwc {

class Problem;
class Result;
struct Config;
enum class ClusterMethod; // cli/config.hpp

// cli/run.hpp documents these; Result is built only by them.
Result run(const Config &config);
Result run(const Config &config, Data data);

/** A lazy path-or-memory dataset handle.  Construct with dtwc::load(). */
class Dataset
{
public:
  using series_type = std::vector<std::vector<data_t>>;

  bool is_path() const noexcept;
  const std::filesystem::path &path() const;
  const std::string &name() const noexcept { return name_; }
  int skip_cols() const noexcept { return skip_cols_; }
  int skip_rows() const noexcept { return skip_rows_; }
  char delimiter() const noexcept { return delimiter_; }

private:
  friend Dataset load(const std::filesystem::path &, int, int, char, std::string_view);
  friend Dataset load(series_type, int, int, char, std::string_view);
  friend Result cluster(Dataset &&, int, std::string_view, int, std::string_view, int);

  explicit Dataset(std::filesystem::path source, int skip_cols, int skip_rows,
                   char delimiter, std::string name);
  explicit Dataset(series_type source, int skip_cols, int skip_rows,
                   char delimiter, std::string name);

  /// The in-memory series, moved out, with skip_rows / skip_cols applied (a path
  /// is run()'s to read).
  Data materialize_local() &&;

  std::variant<std::filesystem::path, series_type> source_;
  int skip_cols_ = 0;
  int skip_rows_ = 0;
  char delimiter_ = 0;
  std::string name_ = "dataset";
};

/** Wrap a path lazily.  No file is opened until cluster() is called.
 *  `skip_rows` drops that many leading LINES of the file, as `--skip-rows` does. */
Dataset load(const std::filesystem::path &source, int skip_cols = 0,
             int skip_rows = 0, char delimiter = 0, std::string_view name = "");

/** Wrap an in-memory row-per-series array.  `skip_rows` drops that many leading
 *  SERIES — one memory row is one file line. */
Dataset load(Dataset::series_type source, int skip_cols = 0, int skip_rows = 0,
             char delimiter = 0, std::string_view name = "");

/** `load(src, skip_cols, delimiter)` does not compile: without these,
 *  `load(p, 0, ',')` would bind the char to `skip_rows` (',' == 44). */
Dataset load(const std::filesystem::path &, int, char, std::string_view = "") = delete;
Dataset load(Dataset::series_type, int, char, std::string_view = "") = delete;

/** Set the process-wide device and return its canonical name. */
std::string device(std::string_view name);

/** Return the canonical process-wide device name. */
std::string device();

/** The owning result of a Tier-1 clustering call. */
class Result
{
public:
  const std::vector<index_t> &labels() const noexcept;
  const std::vector<index_t> &medoids() const noexcept;
  double score(std::string_view name) const;
  /** Dense row-major N*N pairwise DTW distances, matching Python's
   *  `Result.distance_matrix`.  Fills the matrix first if it is not yet
   *  materialised, exactly as score() does. */
  std::vector<double> distance_matrix() const;
  void save(const std::filesystem::path &directory) const;
  double cost() const noexcept { return cost_; }
  const std::string &device() const noexcept { return device_; }
  /** The method that ran, `auto` resolved (name_of(cluster_method_names, m) spells it). */
  ClusterMethod method() const noexcept { return method_; }
  /** The method's iteration count, and whether it converged within max_iter. */
  int iterations() const noexcept { return iterations_; }
  bool converged() const noexcept { return converged_; }

private:
  friend Result run(const Config &);
  friend Result run(const Config &, Data);

  Result(std::shared_ptr<Problem> problem, double cost, std::string device_name,
         ClusterMethod method, int iterations, bool converged);

  std::shared_ptr<Problem> problem_;
  double cost_ = 0.0;
  std::string device_ = "cpu";
  ClusterMethod method_{};
  int iterations_ = 0;
  bool converged_ = false;
};

/**
 * Cluster a lazy Dataset: run() with dtwc_cl's defaults for every other setting,
 * writing nothing. `device` "" means the process device (dtwc::device()).
 *
 * Methods: auto, pam, onebatch, clara, kmedoids, mip, lrcore, tadpole,
 * hierarchical, with dtwc_cl's aliases (hclust, obp, lr).  Unknown names fail loudly.
 * An in-memory Dataset passed as an rvalue hands its series to the run; an lvalue
 * one is copied.
 */
Result cluster(const Dataset &data, int k, std::string_view method = "pam",
               int band = -1, std::string_view device = "", int max_iter = 100);
Result cluster(Dataset &&data, int k, std::string_view method = "pam",
               int band = -1, std::string_view device = "", int max_iter = 100);

} // namespace dtwc
