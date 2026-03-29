/**
 * @file distance_matrix.hpp
 * @brief Dense symmetric distance matrix with flat array storage.
 *
 * @details Stores pairwise distances in a flat N*N array. The set() method
 * enforces symmetry by writing both (i,j) and (j,i). Uncomputed entries
 * are tracked via a separate boolean vector (not NaN sentinels), ensuring
 * correctness under -ffast-math / /fp:fast compiler flags.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include <cassert>
#include <vector>
#include <cstddef>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dtwc::core {

class DenseDistanceMatrix {
  std::vector<double> data_;
  std::vector<char> computed_;  ///< 1 if entry has been set, 0 otherwise.
  size_t n_{ 0 };

public:
  DenseDistanceMatrix() = default;
  explicit DenseDistanceMatrix(size_t n) : data_(n * n, 0.0), computed_(n * n, 0), n_(n) {}

  /// Resize and reset all entries to uncomputed.
  /// Exception-safe: allocates before updating dimension.
  void resize(size_t n)
  {
    data_.assign(n * n, 0.0);
    computed_.assign(n * n, 0);
    n_ = n;
  }

  /// Get the distance between points i and j.
  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[i * n_ + j];
  }

  /// Set the distance between points i and j (symmetric: also sets j,i).
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_);
    data_[i * n_ + j] = v;
    data_[j * n_ + i] = v;
    computed_[i * n_ + j] = 1;
    computed_[j * n_ + i] = 1;
  }

  /// Check whether the distance between i and j has been computed.
  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return computed_[i * n_ + j] != 0;
  }

  /// Number of points (matrix is size() x size()).
  size_t size() const { return n_; }

  /// Maximum computed value in the matrix.
  /// Returns 0.0 if no entries have been computed.
  double max() const
  {
    if (data_.empty()) return 0.0;
    double result = 0.0;
    bool found = false;
    for (size_t idx = 0; idx < data_.size(); ++idx) {
      if (computed_[idx]) {
        if (!found || data_[idx] > result) {
          result = data_[idx];
          found = true;
        }
      }
    }
    return found ? result : 0.0;
  }

  /// Count the number of computed entries in the matrix.
  size_t count_computed() const
  {
    size_t count = 0;
    for (char c : computed_)
      if (c) ++count;
    return count;
  }

  /// Check whether all entries have been computed.
  bool all_computed() const
  {
    for (char c : computed_)
      if (!c) return false;
    return true;
  }

  double *raw() { return data_.data(); }
  const double *raw() const { return data_.data(); }

  /// Write the matrix to a CSV file.
  void write_csv(const std::filesystem::path &path) const
  {
    std::ofstream file(path);
    if (!file.good())
      throw std::runtime_error("Cannot open file for writing: " + path.string());
    file << std::setprecision(15);
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < n_; ++j) {
        if (j > 0) file << ',';
        if (is_computed(i, j))
          file << get(i, j);
        else
          file << "nan";
      }
      file << '\n';
    }
    if (!file.good())
      throw std::runtime_error("Write error on file: " + path.string());
  }

  /// Read the matrix from a CSV file.
  void read_csv(const std::filesystem::path &path)
  {
    std::ifstream file(path);
    if (!file.good())
      throw std::runtime_error("Cannot open file for reading: " + path.string());

    std::vector<std::vector<double>> rows;
    std::vector<std::vector<bool>> rows_valid;
    std::string line;
    while (std::getline(file, line)) {
      // Strip trailing \r for Windows line endings
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      std::vector<double> row;
      std::vector<bool> valid;
      std::istringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) {
        if (cell.empty()) continue; // skip trailing commas
        if (cell == "nan" || cell == "NaN" || cell == "NAN") {
          row.push_back(0.0);
          valid.push_back(false);
        } else {
          row.push_back(std::stod(cell));
          valid.push_back(true);
        }
      }
      rows.push_back(std::move(row));
      rows_valid.push_back(std::move(valid));
    }
    if (!rows.empty()) {
      const size_t N = rows.size();
      resize(N);
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N && j < rows[i].size(); ++j)
          if (rows_valid[i][j])
            set(i, j, rows[i][j]);
    }
  }

  /// Stream output: prints CSV format with full precision.
  friend std::ostream &operator<<(std::ostream &os, const DenseDistanceMatrix &dm)
  {
    auto old_precision = os.precision(15);
    for (size_t i = 0; i < dm.n_; ++i) {
      for (size_t j = 0; j < dm.n_; ++j) {
        if (j > 0) os << ',';
        if (dm.is_computed(i, j))
          os << dm.get(i, j);
        else
          os << "nan";
      }
      os << '\n';
    }
    os.precision(old_precision);
    return os;
  }
};

} // namespace dtwc::core
