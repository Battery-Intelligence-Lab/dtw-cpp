/**
 * @file distance_matrix.hpp
 * @brief Dense symmetric distance matrix with packed triangular storage.
 *
 * @details Stores pairwise distances in a packed lower-triangular array of
 * size N*(N+1)/2, cutting memory use by ~50% vs a full N*N matrix.
 * Uncomputed entries are tracked via a bit-packed boolean vector,
 * ensuring correctness under -ffast-math / /fp:fast compiler flags.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <Eigen/Core>

#include <cassert>
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
#include <vector>

namespace dtwc::core {

class DenseDistanceMatrix {
  Eigen::VectorXd data_;          ///< Packed lower-triangular: N*(N+1)/2 elements.
  std::vector<bool> computed_;    ///< Bit-packed: 1 if entry has been set.
  size_t n_{ 0 };

  /// Map (i,j) to packed lower-triangular index.
  /// Normalises so that the larger index is first: tri(max,min).
  static size_t tri_index(size_t i, size_t j)
  {
    if (i < j) std::swap(i, j);
    return i * (i + 1) / 2 + j;
  }

  /// Total number of elements in packed storage for n points.
  static size_t packed_size(size_t n) { return n * (n + 1) / 2; }

public:
  DenseDistanceMatrix() = default;
  explicit DenseDistanceMatrix(size_t n)
    : data_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_size(n)))),
      computed_(packed_size(n), false),
      n_(n) {}

  /// Resize and reset all entries to uncomputed.
  void resize(size_t n)
  {
    const auto ps = packed_size(n);
    data_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(ps));
    computed_.assign(ps, false);
    n_ = n;
  }

  /// Get the distance between points i and j.
  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[static_cast<Eigen::Index>(tri_index(i, j))];
  }

  /// Set the distance between points i and j (symmetric — single write).
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_);
    const auto k = tri_index(i, j);
    data_[static_cast<Eigen::Index>(k)] = v;
    computed_[k] = true;
  }

  /// Check whether the distance between i and j has been computed.
  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return computed_[tri_index(i, j)];
  }

  /// Number of points (matrix is size() x size()).
  size_t size() const { return n_; }

  /// Maximum computed value in the matrix.
  double max() const
  {
    if (data_.size() == 0) return 0.0;
    double result = 0.0;
    bool found = false;
    for (Eigen::Index idx = 0; idx < data_.size(); ++idx) {
      if (computed_[static_cast<size_t>(idx)]) {
        if (!found || data_[idx] > result) {
          result = data_[idx];
          found = true;
        }
      }
    }
    return found ? result : 0.0;
  }

  /// Count the number of computed entries in the matrix.
  /// Returns the count in packed storage (unique pairs + diagonal).
  size_t count_computed() const
  {
    size_t count = 0;
    for (bool c : computed_)
      if (c) ++count;
    return count;
  }

  /// Check whether all entries have been computed.
  bool all_computed() const
  {
    for (bool c : computed_)
      if (!c) return false;
    return true;
  }

  /// Raw pointer to packed triangular data.
  /// Layout: lower-triangular, row i elements [i*(i+1)/2 .. i*(i+1)/2+i].
  double *raw() { return data_.data(); }
  const double *raw() const { return data_.data(); }

  /// Number of elements in packed storage.
  size_t packed_count() const { return packed_size(n_); }

  /// Expand packed triangular storage to a full N*N Eigen matrix.
  /// Useful for numpy/MATLAB export.
  Eigen::MatrixXd to_full_matrix() const
  {
    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(n_), static_cast<Eigen::Index>(n_));
    for (size_t i = 0; i < n_; ++i)
      for (size_t j = 0; j <= i; ++j) {
        const double v = data_[static_cast<Eigen::Index>(tri_index(i, j))];
        full(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = v;
        full(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = v;
      }
    return full;
  }

  /// Write the matrix to a CSV file (full N*N format for compatibility).
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
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      std::vector<double> row;
      std::vector<bool> valid;
      std::istringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) {
        if (cell.empty()) continue;
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
