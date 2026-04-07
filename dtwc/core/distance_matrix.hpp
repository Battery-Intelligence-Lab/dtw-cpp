/**
 * @file distance_matrix.hpp
 * @brief Dense symmetric distance matrix with packed triangular storage.
 *
 * @details Stores pairwise distances in a packed lower-triangular array of
 * size N*(N+1)/2, cutting memory use by ~50% vs a full N*N matrix.
 * Uncomputed entries use a NaN sentinel (safe for all DTW variants
 * including Soft-DTW which can return negative values).
 * No synchronization needed: parallel fills use disjoint (i,j) pairs by design.
 *
 * I/O (CSV, stream, Eigen export) is in core/matrix_io.hpp.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace dtwc::core {

/// Thread-safety contract: no locking, no atomics.
/// Parallel fills (brute-force, pruned, CUDA) partition the pair space so each
/// (i,j) is written by exactly one thread. count_computed()/all_computed() are
/// cold-path queries called only after the parallel region joins.
class DenseDistanceMatrix {
  std::vector<double> data_;   ///< Packed lower-triangular: N*(N+1)/2 elements. NaN = uncomputed.
  size_t n_{ 0 };

  static size_t tri_index(size_t i, size_t j)
  {
    if (i < j) std::swap(i, j);
    return i * (i + 1) / 2 + j;
  }

  static size_t packed_size(size_t n) { return n * (n + 1) / 2; }

public:
  DenseDistanceMatrix() = default;
  explicit DenseDistanceMatrix(size_t n)
    : data_(packed_size(n), std::numeric_limits<double>::quiet_NaN()), n_(n) {}

  void resize(size_t n)
  {
    data_.assign(packed_size(n), std::numeric_limits<double>::quiet_NaN());
    n_ = n;
  }

  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[tri_index(i, j)];
  }

  /// Set distance. Parallel fills must use disjoint (i,j) pairs — no locking needed.
  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_ && !std::isnan(v));
    data_[tri_index(i, j)] = v;
  }

  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return !std::isnan(data_[tri_index(i, j)]);
  }

  size_t size() const { return n_; }

  double max() const
  {
    double result = 0.0;
    for (double d : data_)
      if (!std::isnan(d) && d > result)
        result = d;
    return result;
  }

  size_t count_computed() const
  {
    return static_cast<size_t>(
      std::count_if(data_.begin(), data_.end(), [](double d) { return !std::isnan(d); }));
  }

  bool all_computed() const
  {
    return std::none_of(data_.begin(), data_.end(), [](double d) { return std::isnan(d); });
  }

  double *raw() { return data_.data(); }
  const double *raw() const { return data_.data(); }
  size_t packed_count() const { return data_.size(); }
};

} // namespace dtwc::core
