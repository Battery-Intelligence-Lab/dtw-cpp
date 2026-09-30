/**
 * @file DataLoader.hpp
 * @brief Encapsulating DTWC data loading configurations in a class.
 * Uses method chaining for easier input taking.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 04 Dec 2022
 */

#pragma once

#include "Data.hpp"           //!< For Data class
#include "fileOperations.hpp" //!< For load_batch_file(), load_folder()
#include "base/settings.hpp"  //!< For data_t type

#include <filesystem> //!< For filesystem objects like path
#include <tuple>      //!< For std::tie(), std::tuple

namespace dtwc {

/**
 * @brief Data loader class
 */
class DataLoader
{
  index_t start_col_{ 0 };                //!< Starting column for data extraction
  index_t start_row_{ 0 };                //!< Starting row for data extraction
  index_t Ndata{ -1 };                    //!< Number of data rows to load
  int verbose{ 1 };                       //!< Verbosity level
  char delim{ ',' };                      //!< Column delimiter character
  bool delim_explicit_{ false };          //!< True once delimiter() was called; path() must not override it.
  std::filesystem::path data_path{ "." }; //!< Path to data file or folder

public:
  // Constructors
  DataLoader() = default;                                  //!< Default constructor.
  DataLoader(const fs::path &path_) { this->path(path_); } //!< Constructor with path initialization.
  DataLoader(const fs::path &path_, index_t Ndata_)
  {
    this->path(path_);
    this->n_data(Ndata_);
  }

  // Accessor methods
  auto startColumn() { return start_col_; } //!< Get the starting column for data loading.
  auto startRow() { return start_row_; }    //!< Get the starting row for data loading.
  auto n_data() { return Ndata; }          //!< Get the number of data points to load.
  auto delimiter() { return delim; }       //!< Get the delimiter used in data files.
  auto path() { return data_path; }        //!< Get the path of the data file or directory.
  auto verbosity() { return verbose; }     //!< Get the verbosity level for data loading.

  // Setters with chaining

  /**
   * @brief Set start column
   * @param N Starting column
   * @return Reference to self for chaining
   */
  DataLoader &start_column(index_t N)
  {
    start_col_ = N;
    return *this;
  }
  [[deprecated("use start_column")]]
  DataLoader &startColumn(int N) { return start_column(N); }

  //!< Set start row
  DataLoader &start_row(index_t N)
  {
    start_row_ = N;
    return *this;
  }
  [[deprecated("use start_row")]]
  DataLoader &startRow(int N) { return start_row(N); }

  //!< Set number of series to read (-1 = all). Rejects N < -1.
  DataLoader &n_data(index_t N)
  {
    validate_ndata(N, "DataLoader::n_data");
    Ndata = N;
    return *this;
  }
  //!< Set delimiter
  DataLoader &delimiter(char delim_)
  {
    delim = delim_;
    delim_explicit_ = true;
    return *this;
  }

  /**
   * @brief Set data path
   *
   * Sets delimiter based on file extension
   *
   * @param data_path_ Path to data
   * @return Reference to self for chaining
   */
  DataLoader &path(const std::filesystem::path &data_path_)
  {
    data_path = data_path_;
    // An explicit delimiter() wins over extension inference; the extension
    // match is case-insensitive so ".TSV" is not silently left on ','.
    if (delim_explicit_) return *this;
    const auto ext =
      text_io_detail::lower_ascii(data_path_.extension().string());
    if (ext == ".csv")
      delim = ',';
    else if (ext == ".tsv" || ext == ".txt")
      delim = '\t';
    return *this;
  }

  //!< Set verbosity level
  DataLoader &verbosity(int N)
  {
    verbose = N;
    return *this;
  }

  /**
   * @brief Load data into RAM.
   * @details Calls the appropriate loader based on the path being a file or folder.
   * @return Loaded data.
   */
  Data load()
  {
    Data d;
    const LoadOptions opts{ Ndata, verbose, start_row_, start_col_, delim };
    if (fs::is_directory(data_path))
      std::tie(d.p_vec, d.p_names) = load_folder<data_t>(data_path, opts);
    else
      std::tie(d.p_vec, d.p_names) = load_batch_file<data_t>(data_path, opts);
    return d;
  }
};

} // namespace dtwc
