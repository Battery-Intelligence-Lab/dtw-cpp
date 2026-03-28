/**
 * @file distance_matrix.hpp
 * @brief Dense symmetric distance matrix with flat array storage.
 *
 * @details Stores pairwise distances in a flat N*N array. The set() method
 * enforces symmetry by writing both (i,j) and (j,i). Uncomputed entries
 * are marked with a negative sentinel value.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>

namespace dtwc::core {

class DenseDistanceMatrix {
  std::vector<double> data_;
  size_t n_{ 0 };
  static constexpr double NOT_COMPUTED = -1.0;

public:
  DenseDistanceMatrix() = default;
  explicit DenseDistanceMatrix(size_t n) : data_(n * n, NOT_COMPUTED), n_(n) {}

  /// Resize and reset all entries to NOT_COMPUTED.
  void resize(size_t n)
  {
    n_ = n;
    data_.assign(n * n, NOT_COMPUTED);
  }

  /// Get the distance between points i and j.
  double get(size_t i, size_t j) const { return data_[i * n_ + j]; }

  /// Set the distance between points i and j (symmetric: also sets j,i).
  void set(size_t i, size_t j, double v)
  {
    data_[i * n_ + j] = v;
    data_[j * n_ + i] = v;
  }

  /// Check whether the distance between i and j has been computed.
  bool is_computed(size_t i, size_t j) const { return data_[i * n_ + j] >= 0.0; }

  /// Number of points (matrix is size() x size()).
  size_t size() const { return n_; }

  /// Maximum value in the matrix.
  double max() const
  {
    if (data_.empty()) return 0.0;
    return *std::max_element(data_.begin(), data_.end());
  }

  double *raw() { return data_.data(); }
  const double *raw() const { return data_.data(); }
};

} // namespace dtwc::core
