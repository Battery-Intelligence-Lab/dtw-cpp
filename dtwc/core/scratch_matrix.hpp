/**
 * @file scratch_matrix.hpp
 * @brief Minimal column-major 2D matrix that owns its memory.
 *
 * @details Replaces arma::Mat for scratch buffers in DTW computation.
 * Column-major layout for cache-friendly column-sweep patterns in DTW.
 * This matches Armadillo's memory layout (column-major / Fortran order).
 *
 * @date 28 Mar 2026
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace dtwc::core {

template <typename T>
class ScratchMatrix {
  std::vector<T> data_;
  size_t rows_{ 0 }, cols_{ 0 };

public:
  ScratchMatrix() = default;
  ScratchMatrix(size_t r, size_t c) : data_(r * c), rows_(r), cols_(c) {}
  ScratchMatrix(size_t r, size_t c, T val) : data_(r * c, val), rows_(r), cols_(c) {}

  /// Exception-safe resize: allocates new storage before updating dimensions.
  /// If allocation throws, rows_ and cols_ remain unchanged.
  void resize(size_t r, size_t c)
  {
    data_.resize(r * c); // may throw; dimensions unchanged on failure
    rows_ = r;
    cols_ = c;
  }

  /// Column-major access: data_[j * rows_ + i]
  T &operator()(size_t i, size_t j)
  {
    assert(i < rows_ && j < cols_);
    return data_[j * rows_ + i];
  }

  /// Column-major access (const): data_[j * rows_ + i]
  const T &operator()(size_t i, size_t j) const
  {
    assert(i < rows_ && j < cols_);
    return data_[j * rows_ + i];
  }

  T *raw() { return data_.data(); }
  const T *raw() const { return data_.data(); }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  void fill(T val) { std::fill(data_.begin(), data_.end(), val); }
};

} // namespace dtwc::core
