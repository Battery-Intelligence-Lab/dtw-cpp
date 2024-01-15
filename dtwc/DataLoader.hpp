/**
 * @file DataLoader.hpp
 * @brief Encapsulating DTWC data loading configurations in a class.
 * Uses method chaining for easier input taking.
 * @author Volkan Kumtepeli, Becky Perriment
 * @date 04 Dec 2022
 */

#pragma once

#include "Data.hpp"           //!< For Data class
#include "fileOperations.hpp" //!< For load_batch_file(), load_folder()
#include "settings.hpp"       //!< For data_t type

#include <cstddef>    //!< For size_t
#include <filesystem> //!< For filesystem objects like path
#include <tuple>      //!< For std::tie(), std::tuple
#include <vector>     //!< For std::vector

namespace dtwc {
/**
 * @brief Data loader class
 */
class DataLoader
{
  int start_col{ 0 };                     //!< Starting column for data extraction
  int start_row{ 0 };                     //!< Starting row for data extraction
  int Ndata{ -1 };                        //!< Number of data rows to load
  int verbose{ 1 };                       //!< Verbosity level
  char delim{ ',' };                      //!< Column delimiter character
  std::filesystem::path data_path{ "." }; //!< Path to data file or folder

public:
  // Constructors
  DataLoader() = default;                                  //!< Default constructor.
  DataLoader(const fs::path &path_) { this->path(path_); } //!< Constructor with path initialization.
  DataLoader(const fs::path &path_, int Ndata_)
  {
    this->path(path_);
    this->n_data(Ndata_);
  }

  // Accessor methods
  auto startColumn() { return start_col; } //!< Get the starting column for data loading.
  auto startRow() { return start_row; }    //!< Get the starting row for data loading.
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
  DataLoader &startColumn(int N)
  {
    start_col = N;
    return *this;
  }

  //!< Set start row
  DataLoader &startRow(int N)
  {
    start_row = N;
    return *this;
  }

  //!< Set number of data rows
  DataLoader &n_data(int N)
  {
    Ndata = N;
    return *this;
  }
  //!< Set delimiter
  DataLoader &delimiter(char delim_)
  {
    delim = delim_;
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
    if (data_path_.extension() == ".csv")
      delim = ',';
    else if (data_path_.extension() == ".tsv")
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
   * @brief Load data
   * @details Calls appropriate loader based on path being file or folder.
   * @return Loaded data
   */
  Data load()
  {
    Data d;
    if (fs::is_directory(data_path))
      std::tie(d.p_vec, d.p_names) = load_folder<data_t>(data_path, Ndata, verbose, start_row, start_col, delim);
    else
      std::tie(d.p_vec, d.p_names) = load_batch_file<data_t>(data_path, Ndata, verbose, start_row, start_col, delim);

    return d;
  }
};

} // namespace dtwc