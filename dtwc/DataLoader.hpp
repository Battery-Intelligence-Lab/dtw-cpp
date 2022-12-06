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

#include "settings.hpp"
#include "utility.hpp"
#include "fileOperations.hpp"
#include "Data.hpp"

#include <vector>

namespace dtwc {

class DataLoader
{
  int start_col{ 0 }, start_row{ 0 }, Ndata{ -1 }, verbose{ 0 };
  char delimiter{ ',' };
  fs::path data_path;

public:
  DataLoader() = default;
  DataLoader(const fs::path &path_) { this->path(path_); }

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

  DataLoader &path(const fs::path &data_path_)
  {
    data_path = data_path_;

    if (data_path_.extension() == ".tsv")
      delimiter = '\t';

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
      std::tie(d.p_vec, d.p_names) = load_folder<data_t>(data_path, Ndata, verbose > 0, start_row, start_col, delimiter);
    else
      std::tie(d.p_vec, d.p_names) = load_batch_file<data_t>(data_path, Ndata, verbose > 0, start_row, start_col, delimiter);

    d.Nb = d.p_vec.size();
    return d;
  }
};

} // namespace dtwc