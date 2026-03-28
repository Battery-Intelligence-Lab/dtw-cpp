/**
 * @file distance_matrix.hpp
 * @brief Dense symmetric distance matrix with flat array storage.
 *
 * @details Stores pairwise distances in a flat N*N array. The set() method
 * enforces symmetry by writing both (i,j) and (j,i). Uncomputed entries
 * are marked with NaN sentinel value.
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
  size_t n_{ 0 };
  static constexpr double NOT_COMPUTED = std::numeric_limits<double>::quiet_NaN();

public:
  DenseDistanceMatrix() = default;
  explicit DenseDistanceMatrix(size_t n) : data_(n * n, NOT_COMPUTED), n_(n) {}

  /// Resize and reset all entries to NOT_COMPUTED.
  /// Exception-safe: allocates before updating dimension.
  void resize(size_t n)
  {
    data_.assign(n * n, NOT_COMPUTED); // may throw; n_ unchanged on failure
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
  }

  /// Check whether the distance between i and j has been computed.
  /// Uses NaN sentinel: an entry is computed if it is not NaN.
  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return !std::isnan(data_[i * n_ + j]);
  }

  /// Number of points (matrix is size() x size()).
  size_t size() const { return n_; }

  /// Maximum computed value in the matrix, skipping NaN entries.
  /// Returns 0.0 if no entries have been computed.
  double max() const
  {
    if (data_.empty()) return 0.0;
    double result = 0.0;
    bool found = false;
    for (const double v : data_) {
      if (!std::isnan(v)) {
        if (!found || v > result) {
          result = v;
          found = true;
        }
      }
    }
    return found ? result : 0.0;
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
        file << get(i, j);
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
    std::string line;
    while (std::getline(file, line)) {
      // Strip trailing \r for Windows line endings
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      std::vector<double> row;
      std::istringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) {
        if (cell.empty()) continue; // skip trailing commas
        row.push_back(std::stod(cell));
      }
      rows.push_back(std::move(row));
    }
    if (!rows.empty()) {
      const size_t N = rows.size();
      resize(N);
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N && j < rows[i].size(); ++j)
          if (!std::isnan(rows[i][j]))
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
        os << dm.get(i, j);
      }
      os << '\n';
    }
    os.precision(old_precision);
    return os;
  }
};

} // namespace dtwc::core
