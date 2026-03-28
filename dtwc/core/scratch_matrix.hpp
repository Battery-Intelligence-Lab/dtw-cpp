/**
 * @file scratch_matrix.hpp
 * @brief Minimal row-major 2D matrix that owns its memory.
 *
 * @details Replaces arma::Mat for scratch buffers in DTW computation.
 * Row-major layout for cache-friendly row-wise access patterns typical in DTW.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include <vector>
#include <algorithm>
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

  void resize(size_t r, size_t c)
  {
    rows_ = r;
    cols_ = c;
    data_.resize(r * c);
  }

  T &operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }
  const T &operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

  T *raw() { return data_.data(); }
  const T *raw() const { return data_.data(); }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  void fill(T val) { std::fill(data_.begin(), data_.end(), val); }
};

} // namespace dtwc::core
