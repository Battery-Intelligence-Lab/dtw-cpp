/**
 * @file DataLoader.hpp
 * @brief Encapsulating DTWC data loading configurations in a class.
 * Uses method chaining for easier input taking.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 04 Dec 2022
 */

#pragma once

#include "Data.hpp"
#include "fileOperations.hpp"
#include "base/settings.hpp"
#include "core/storage.hpp"

#include <filesystem>

namespace dtwc {

class DataLoader
{
  index_t start_col_{ 0 };
  index_t start_row_{ 0 };
  index_t Ndata{ -1 };
  int verbose{ 1 };
  char delim{ ',' };
  bool delim_explicit_{ false }; ///< True once delimiter() was called; path() must not override it.
  core::Precision dtype_{ core::Precision::Float64 };
  std::filesystem::path data_path{ "." };

public:
  DataLoader() = default;
  DataLoader(const fs::path &path_) { path(path_); }
  DataLoader(const fs::path &path_, index_t Ndata_)
  {
    path(path_);
    n_data(Ndata_);
  }

  auto startColumn() const noexcept { return start_col_; }
  auto startRow() const noexcept { return start_row_; }
  auto n_data() const noexcept { return Ndata; }
  auto delimiter() const noexcept { return delim; }
  const std::filesystem::path &path() const noexcept { return data_path; }
  auto verbosity() const noexcept { return verbose; }
  auto dtype() const noexcept { return dtype_; }

  DataLoader &start_column(index_t N)
  {
    start_col_ = N;
    return *this;
  }

  DataLoader &start_row(index_t N)
  {
    start_row_ = N;
    return *this;
  }

  /// Set number of series to read (-1 = all). Rejects N < -1.
  DataLoader &n_data(index_t N)
  {
    validate_ndata(N, "DataLoader::n_data");
    Ndata = N;
    return *this;
  }

  DataLoader &delimiter(char delim_)
  {
    delim = delim_;
    delim_explicit_ = true;
    return *this;
  }

  /// Set data path and infer delimiter from extension unless delimiter() was called explicitly.
  DataLoader &path(const std::filesystem::path &data_path_)
  {
    data_path = data_path_;
    if (delim_explicit_) return *this;
    // The extension match is case-insensitive, so ".TSV" is not silently left on ','. The extension is read as
    // UTF-8: path::string() converts to the Windows ANSI code page and throws for an extension it cannot map.
    const auto ext =
      text_io_detail::lower_ascii(path_to_utf8(data_path_.extension()));
    if (ext == ".csv")
      delim = ',';
    else if (ext == ".tsv" || ext == ".txt")
      delim = '\t';
    return *this;
  }

  DataLoader &verbosity(int N)
  {
    verbose = N;
    return *this;
  }

  /// Set the precision the series are stored in: Float64 (the default), or Float32 at half the memory, each value
  /// the double parse rounded to float.
  DataLoader &dtype(core::Precision precision)
  {
    dtype_ = precision;
    return *this;
  }

  /// Load data from a batch file or folder in precision dtype().
  Data load()
  {
    const LoadOptions opts{ Ndata, verbose, start_row_, start_col_, delim };
    const auto read = [&](auto stored) {
      using T = decltype(stored);
      auto [series, names] = fs::is_directory(data_path) ? load_folder<T>(data_path, opts)
                                                         : load_batch_file<T>(data_path, opts);
      return Data(std::move(series), std::move(names));
    };
    return dtype_ == core::Precision::Float32 ? read(float{}) : read(data_t{});
  }
};

} // namespace dtwc
