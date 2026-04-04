/**
 * @file matrix_io.hpp
 * @brief CSV I/O and Eigen expansion for DenseDistanceMatrix.
 *
 * @details Free functions in dtwc::io operating on DenseDistanceMatrix
 *          by (const) reference. Separated from distance_matrix.hpp to satisfy
 *          SRP: the core data structure carries no I/O knowledge.
 *
 *          CSV format: full N×N, comma-separated, 15-digit precision.
 *          Uncomputed entries are written as an empty field (not "nan"),
 *          so the file is clean and easy to inspect in spreadsheet tools.
 *          An empty field on read is treated as uncomputed.
 *
 *          operator<< lives in dtwc::core (not dtwc::io) so that
 *          ADL resolves it for DenseDistanceMatrix arguments.
 *
 * @author Volkan Kumtepeli
 * @date 04 Apr 2026
 */

#pragma once

#include "distance_matrix.hpp"

#include <Eigen/Core>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtwc::io {

/// Write the full N×N matrix to a CSV file.
/// Uncomputed entries are written as an empty field.
inline void write_csv(const core::DenseDistanceMatrix &dm, const std::filesystem::path &path)
{
  std::ofstream file(path);
  if (!file.good())
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  file << std::setprecision(15);
  const size_t n = dm.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (j > 0) file << ',';
      if (dm.is_computed(i, j))
        file << dm.get(i, j);
      // uncomputed → empty field
    }
    file << '\n';
  }
  if (!file.good())
    throw std::runtime_error("Write error on file: " + path.string());
}

/// Read a full N×N CSV file into the matrix.
/// Empty fields are treated as uncomputed; numeric fields are set.
inline void read_csv(core::DenseDistanceMatrix &dm, const std::filesystem::path &path)
{
  std::ifstream file(path);
  if (!file.good())
    throw std::runtime_error("Cannot open file for reading: " + path.string());

  struct Cell { double value; bool valid; };
  std::vector<std::vector<Cell>> rows;

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<Cell> row;
    std::istringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      if (cell.empty()) {
        row.push_back({0.0, false});   // empty field → uncomputed
      } else {
        row.push_back({std::stod(cell), true});
      }
    }
    rows.push_back(std::move(row));
  }

  if (!rows.empty()) {
    const size_t N = rows.size();
    dm.resize(N);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N && j < rows[i].size(); ++j)
        if (rows[i][j].valid)
          dm.set(i, j, rows[i][j].value);
  }
}

/// Expand packed triangular storage to a full N×N Eigen matrix.
/// Useful for numpy/MATLAB export from the binding layer.
inline Eigen::MatrixXd to_full_matrix(const core::DenseDistanceMatrix &dm)
{
  const size_t n = dm.size();
  Eigen::MatrixXd full = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j <= i; ++j) {
      const double v = dm.get(i, j);
      full(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = v;
      full(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = v;
    }
  return full;
}

} // namespace dtwc::io

namespace dtwc::core {

/// Stream output: prints CSV format (same as io::write_csv) to any ostream.
/// Defined in dtwc::core so ADL resolves it for DenseDistanceMatrix arguments.
inline std::ostream &operator<<(std::ostream &os, const DenseDistanceMatrix &dm)
{
  const auto old_precision = os.precision(15);
  const size_t n = dm.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (j > 0) os << ',';
      if (dm.is_computed(i, j))
        os << dm.get(i, j);
      // uncomputed → empty field
    }
    os << '\n';
  }
  os.precision(old_precision);
  return os;
}

} // namespace dtwc::core
