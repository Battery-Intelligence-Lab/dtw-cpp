/**
 * @file distance_matrix.hpp
 * @brief Dense symmetric distance matrix with packed triangular storage.
 *
 * @details Stores pairwise distances in a packed lower-triangular array of
 * size N*(N+1)/2, cutting memory use by ~50% vs a full N*N matrix.
 * Uncomputed entries are tracked via a bit-packed boolean vector,
 * ensuring correctness under -ffast-math / /fp:fast compiler flags.
 *
 * I/O (CSV, stream, Eigen export) is in core/matrix_io.hpp.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

namespace dtwc::core {

class DenseDistanceMatrix {
  std::vector<double> data_;       ///< Packed lower-triangular: N*(N+1)/2 elements.
  std::vector<bool> computed_;     ///< Bit-packed: 1 if entry has been set.
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
    : data_(packed_size(n), 0.0),
      computed_(packed_size(n), false),
      n_(n) {}

  /// Resize and reset all entries to uncomputed.
  void resize(size_t n)
  {
    const auto ps = packed_size(n);
    data_.assign(ps, 0.0);
    computed_.assign(ps, false);
    n_ = n;
  }

  /// Get the distance between points i and j.
  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[tri_index(i, j)];
  }

  /// Set the distance between points i and j (symmetric — single write).
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_);
    const auto k = tri_index(i, j);
    data_[k] = v;
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
  double *raw() { return data_.data(); }
  const double *raw() const { return data_.data(); }

  /// Number of elements in packed storage.
  size_t packed_count() const { return data_.size(); }
};

} // namespace dtwc::core
