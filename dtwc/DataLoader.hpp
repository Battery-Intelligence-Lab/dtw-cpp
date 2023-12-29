/*
 * DataLoader.hpp
 *
 * Encapsulating DTWC data loading configurations in a class.
 * Uses method chaining for easier input taking.
 *
 * Created on: 04 Dec 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "Data.hpp"           // for Data
#include "fileOperations.hpp" // for load_batch_file, load_folder
#include "settings.hpp"       // for data_t

#include <cstddef>    // for size_t
#include <filesystem> // for path, is_directory, operator==, direct...
#include <tuple>      // for tie, tuple
#include <vector>     // for vector

namespace dtwc {

class DataLoader
{
  int start_col{ 0 }, start_row{ 0 }, Ndata{ -1 }, verbose{ 1 };
  char delim{ ',' };
  fs::path data_path{ "." };

public:
  DataLoader() = default;
  DataLoader(const fs::path &path_) { this->path(path_); }
  DataLoader(const fs::path &path_, int Ndata_)
  {
    this->path(path_);
    this->n_data(Ndata_);
  }

  // Get methods:
  auto startColumn() { return start_col; }
  auto startRow() { return start_row; }
  auto n_data() { return Ndata; }
  auto delimiter() { return delim; }
  auto path() { return data_path; }
  auto verbosity() { return verbose; }

  // Some methods for chaining:
  DataLoader &startColumn(int N)
  {
    start_col = N;
    return *this;
  }

  DataLoader &startRow(int N)
  {
    start_row = N;
    return *this;
  }

  DataLoader &n_data(int N)
  {
    Ndata = N;
    return *this;
  }

  DataLoader &delimiter(char delim_)
  {
    delim = delim_;
    return *this;
  }

  DataLoader &path(const fs::path &data_path_)
  {
    data_path = data_path_;

    if (data_path_.extension() == ".csv")
      delim = ',';
    else if (data_path_.extension() == ".tsv")
      delim = '\t';

    return *this;
  }

  DataLoader &verbosity(int N)
  {
    verbose = N;
    return *this;
  }

  Data load()
  {
    Data d;
    if (fs::is_directory(data_path))
      std::tie(d.p_vec, d.p_names) = load_folder<data_t>(data_path, Ndata, verbose, start_row, start_col, delim);
    else
      std::tie(d.p_vec, d.p_names) = load_batch_file<data_t>(data_path, Ndata, verbose, start_row, start_col, delim);

    d.Nb = static_cast<int>(d.p_vec.size());
    return d;
  }
};

} // namespace dtwc