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

/** A lazy path-or-memory dataset handle.  Construct with dtwc::load(). */
class Dataset
{
public:
  using series_type = std::vector<std::vector<data_t>>;

  bool is_path() const noexcept;
  const std::filesystem::path &path() const;
  const std::string &name() const noexcept { return name_; }
  int skip_cols() const noexcept { return skip_cols_; }
  char delimiter() const noexcept { return delimiter_; }

private:
  friend Dataset load(const std::filesystem::path &, int, char, std::string_view);
  friend Dataset load(series_type, int, char, std::string_view);
  friend class Result;
  friend Result cluster(const Dataset &, int, std::string_view, int,
                        std::string_view, int);

  explicit Dataset(std::filesystem::path source, int skip_cols, char delimiter,
                   std::string name);
  explicit Dataset(series_type source, int skip_cols, char delimiter,
                   std::string name);

  Data materialize_local() const;

  std::variant<std::filesystem::path, series_type> source_;
  int skip_cols_ = 0;
  char delimiter_ = 0;
  std::string name_ = "dataset";
};

/** Wrap a path lazily.  No file is opened until cluster() is called. */
Dataset load(const std::filesystem::path &source, int skip_cols = 0,
             char delimiter = 0, std::string_view name = "");

/** Wrap an in-memory row-per-series array. */
Dataset load(Dataset::series_type source, int skip_cols = 0,
             char delimiter = 0, std::string_view name = "");

/** Set the process-wide device and return its canonical name. */
std::string device(std::string_view name);

/** Return the canonical process-wide device name. */
std::string device();

/** The owning result of a Tier-1 clustering call. */
class Result
{
public:
  const std::vector<int> &labels() const noexcept;
  const std::vector<int> &medoids() const noexcept;
  double score(std::string_view name) const;
  void save(const std::filesystem::path &directory) const;
  double cost() const noexcept { return cost_; }
  const std::string &device() const noexcept { return device_; }

private:
  friend Result cluster(const Dataset &, int, std::string_view, int,
                        std::string_view, int);

  Result(std::shared_ptr<Problem> problem, double cost, std::string device_name);

  std::shared_ptr<Problem> problem_;
  double cost_ = 0.0;
  std::string device_ = "cpu";
};

/**
 * Cluster a lazy Dataset.
 *
 * Methods: auto, pam, onebatch, clara, kmedoids, mip, lrcore, tadpole,
 * hierarchical (alias hclust).  Unknown names fail loudly.
 */
Result cluster(const Dataset &data, int k, std::string_view method = "pam",
               int band = -1, std::string_view device = "", int max_iter = 100);

} // namespace dtwc
