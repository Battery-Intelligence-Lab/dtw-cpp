/**
 * @file scratch_matrix.hpp
 * @brief Column-major 2D scratch matrix for DTW working buffers.
 *
 * @details Owns a single column-major buffer. Column-major suits the
 * column-sweep access pattern of the DTW kernels.
 *
 * Previously this privately inherited `Eigen::Matrix` (ledger X-27). Eigen was
 * the project's only copyleft dependency (MPL-2.0) and the only MPL obligation
 * in the Python wheel, and it was carried for these two uses alone, so it is
 * gone. The Eigen version was justified in this header as "aligned SIMD-ready
 * allocation"; X-04 measured that claim and it does not hold — none of the six
 * DTW kernel loops vectorise, the recurrence being reported by clang as *unsafe
 * dependent memory operations*, so no vector instruction was consuming the
 * alignment.
 *
 * Two properties of the Eigen original are load-bearing and are preserved
 * deliberately, because `core/dtw_kernel.hpp` holds one of these `thread_local`
 * and resizes it on every DTW evaluation:
 *
 *  - **resize does not initialise.** `Eigen::Matrix::resize` leaves contents
 *    undefined; `std::vector::resize` would value-initialise, adding an
 *    O(rows × cols) zero-fill to every call. `make_unique_for_overwrite` keeps
 *    the buffer uninitialised, so callers that fill before reading pay nothing.
 *  - **resize is grow-only.** The buffer is reallocated only when the request
 *    exceeds the capacity already held, so a thread_local instance reaches its
 *    working size once and then stops allocating.
 *
 * Contents are undefined after `resize`, exactly as before. Call `fill()` if you
 * need a known state.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

namespace dtwc::core {

/**
 * @brief Column-major scratch matrix with uninitialised, grow-only storage.
 */
template <typename T>
class ScratchMatrix
{
public:
  /// Signed, like the Eigen index this replaces, so call sites that cast to
  /// `int` do not become sign-conversion warnings.
  using Index = std::ptrdiff_t;

  ScratchMatrix() = default;

  ScratchMatrix(std::size_t r, std::size_t c) { resize(r, c); }

  ScratchMatrix(std::size_t r, std::size_t c, T val)
  {
    resize(r, c);
    fill(val);
  }

  /// Grow-only. Contents are undefined afterwards, as with the Eigen original.
  void resize(std::size_t r, std::size_t c)
  {
    const auto needed = static_cast<Index>(r * c);
    if (needed > capacity_) {
      buffer_ = std::make_unique_for_overwrite<T[]>(static_cast<std::size_t>(needed));
      capacity_ = needed;
    }
    rows_ = static_cast<Index>(r);
    cols_ = static_cast<Index>(c);
  }

  T &operator()(Index i, Index j) noexcept { return buffer_[i + j * rows_]; }

  const T &operator()(Index i, Index j) const noexcept { return buffer_[i + j * rows_]; }

  T *data() noexcept { return buffer_.get(); }
  const T *data() const noexcept { return buffer_.get(); }

  T *raw() noexcept { return buffer_.get(); }
  const T *raw() const noexcept { return buffer_.get(); }

  Index rows() const noexcept { return rows_; }
  Index cols() const noexcept { return cols_; }
  Index size() const noexcept { return rows_ * cols_; }

  void fill(T val) { std::fill_n(buffer_.get(), static_cast<std::size_t>(rows_ * cols_), val); }

  bool empty() const noexcept { return rows_ * cols_ == 0; }

private:
  std::unique_ptr<T[]> buffer_;
  /// Signed throughout, like Eigen's Index, so indices the kernels cast to `int`
  /// need no zero-extend. This was tried as a fix for the ~5% BM_dtwFull
  /// regression this class carries against the Eigen original and **did not
  /// help** — the cause is still unidentified. Kept because it is the closer
  /// match to what it replaces, not because it bought anything.
  /// See .claude/baselines/2026-09-22-x27-drop-eigen-band.md.
  Index capacity_{ 0 };
  Index rows_{ 0 };
  Index cols_{ 0 };
};

} // namespace dtwc::core
