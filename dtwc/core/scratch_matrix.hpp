/**
 * @file scratch_matrix.hpp
 * @brief Column-major 2D scratch matrix backed by Eigen.
 *
 * @details Thin alias over Eigen::Matrix for scratch buffers in DTW
 * computation. Column-major layout for cache-friendly column-sweep
 * patterns. Provides a `.fill()` method that Eigen lacks.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <Eigen/Core>
#include <cstddef>

namespace dtwc::core {

/**
 * @brief Column-major scratch matrix backed by Eigen for aligned SIMD-ready
 *        allocation.
 *
 * Drop-in replacement for the previous custom ScratchMatrix. Adds a `fill()`
 * convenience method that Eigen's Matrix does not natively expose.
 */
template <typename T>
class ScratchMatrix : private Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
  using Base = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

public:
  using Base::operator();
  using Base::data;
  using Base::size;
  using Base::rows;
  using Base::cols;
  ScratchMatrix() = default;
  ScratchMatrix(size_t r, size_t c) : Base(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) {}
  ScratchMatrix(size_t r, size_t c, T val) : Base(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c))
  {
    this->setConstant(val);
  }

  void resize(size_t r, size_t c)
  {
    Base::resize(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
  }

  void fill(T val) { this->setConstant(val); }

  T *raw() { return this->data(); }
  const T *raw() const { return this->data(); }

  bool empty() const { return this->size() == 0; }
};

} // namespace dtwc::core
