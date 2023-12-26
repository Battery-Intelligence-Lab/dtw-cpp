/*
 * fileOperations.hpp
 *
 * Functions for file operations
 *
 *  Created on: 21 Jan 2022
 *  Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "settings.hpp" // for resultsPath

#include <cassert>    // for assert
#include <chrono>     // for filesystem
#include <cstdlib>    // for size_t
#include <filesystem> // for operator<<, path, operator/, directory_iterator
#include <iostream>   // for operator<<, ifstream, basic_ostream, operator>>
#include <string>     // for string, getline, to_string
#include <utility>    // for pair
#include <vector>     // for vector
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept> // for std::runtime_error

#include <Eigen/Dense>

namespace dtwc {

namespace fs = std::filesystem;

inline void ignoreBOM(std::istream &in)
{
  char BOMchars[] = { '\xEF', '\xBB', '\xBF' };
  int seek = 0;
  char c = '.';
  while (in >> c) {
    if (BOMchars[seek] != c) {
      in.putback(c);
      break;
    }
    seek++;
  }
  in.clear(); // Clear EOF flag if end of file was reached
}


template <typename data_t>
auto readFile(const fs::path &name, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    std::cerr << "Error in readFile. File " << name << " could not be opened.\n";
    std::runtime_error("");
  }

  ignoreBOM(in);

  // https://stackoverflow.com/questions/70497719/read-from-comma-separated-file-into-vector-of-objects
  std::string line{};
  char c = '.';

  for (int i = 0; i < start_row; i++) // Skip first start_row rows to start from start_row.
    std::getline(in, line);


  data_t temp, p_i;
  std::vector<data_t> p;
  p.reserve(10000);

  while (std::getline(in, line)) {
    std::istringstream iss(line);

    for (int i = 0; i < start_col; i++) // Skip first start_col columns to start from start_col.
    {
      iss >> temp;
      if (delimiter != ' ' && delimiter != '\t') // These we do not need to remove from stream.
        iss >> c;
    }

    iss >> p_i; // Finally we got it!  % #TODO does not work for many-column arrays.
    p.push_back(p_i);
  }

  p.shrink_to_fit();
  return p;
}


template <typename data_t, typename Tpath>
auto load_folder(Tpath &folder_path, int Ndata = -1, bool print = false, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::cout << "Reading data:" << std::endl;

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  int i_data = 0;
  for (const auto &entry : fs::directory_iterator(folder_path)) {

    auto p = readFile<data_t>(entry.path(), start_row, start_col, delimiter);

    if (print || p.empty())
      std::cout << entry.path() << "\tSize: " << p.size() << '\n';

    assert(p.size() > 2);

    p_vec.push_back(std::move(p));
    p_names.push_back(entry.path().stem().string());

    i_data++;
    if (i_data == Ndata) break;
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}

template <typename data_t>
auto load_batch_file(fs::path &file_path, int Ndata = -1, bool print = false, int start_row = 0, int start_col = 0, char delimiter = ',')
{
  std::cout << "Reading data:" << std::endl;

  std::vector<std::vector<data_t>> p_vec;
  std::vector<std::string> p_names;

  auto myAbsPath = fs::absolute(file_path);

  std::ifstream in(file_path, std::ios_base::in);
  if (!in.good()) // check if we could open the file
  {
    std::cerr << "Error in readFile. File " << file_path << " could not be opened.\n";
    throw 2;
  }

  std::string line;
  int n_rows{ 0 };
  while ((Ndata == -1 || n_rows < Ndata) && std::getline(in, line)) //!< Read file.
  {
    if (n_rows < start_row) // Skip first rows.
      continue;

    n_rows++;

    std::vector<data_t> p;
    p.reserve(10000);
    std::istringstream in_line(line);
    data_t temp, p_i;
    char c;

    for (int i = 0; i < start_col; i++) // Skip first start_col columns to start from start_col.
    {
      in_line >> temp;
      if (delimiter != ' ' && delimiter != '\t') // These we do not need to remove from stream.
        in_line >> c;
    }

    while (in_line >> p_i) {
      p.push_back(p_i);
      if (delimiter != ' ' && delimiter != '\t') // These we do not need to remove from stream.
        in_line >> c;
    }

    p.shrink_to_fit();

    if (print || p.empty()) // It should say if p is empty!
      std::cout << file_path << '\t' << "data: " << n_rows << " Size: " << p.size() << '\n';

    p_vec.push_back(std::move(p));
    p_names.push_back(std::to_string(n_rows));
  }

  std::cout << p_vec.size() << " time-series data are read.\n";

  return std::pair(p_vec, p_names);
}


template <typename matrix_t, typename path_t>
void writeMatrix(const matrix_t &matrix, const path_t &path)
{
  std::ofstream myFile(path, std::ios_base::out);

  if (!myFile.good()) // check if we could open the file
  {
    std::cerr << "Error in writeMatrix. File " << path
              << " could not be opened. Please ensure that you have the folder "
              << " and file is not open in any other program.\n";
    std::runtime_error("");
  }

  Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
  myFile << matrix.format(CSVFormat);
  myFile.close();
}

template <typename data_t>
void readMatrix(Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> &matrix, const fs::path &name)
{
  // Adapted from: https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix/39146048#39146048
  using Eigen::Dynamic;
  std::ifstream in(name, std::ios_base::in);
  if (!in.good()) // check if we could open the file
    std::cout << "File " << name << " is not found. Matrix will not be read." << std::endl;

  std::vector<data_t> data;
  std::string line;
  data_t x{};

  size_t rows = 0;
  while (std::getline(in, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ','))
      data.push_back(std::stod(cell));

    ++rows;
  }

  const size_t cols = data.empty() ? 0 : (data.size() / rows);

  matrix = Eigen::Map<Eigen::Array<data_t, Dynamic, Dynamic, Eigen::RowMajor>>(data.data(), rows, cols);
}


} // namespace dtwc