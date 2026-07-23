/**
 * @file dtw_cost.hpp
 * @brief Cost functors for the unified DTW kernel.
 *
 * @details A Cost functor models the pointwise cost at cell (row, col):
 *            T operator()(size_t row, size_t col) const noexcept;
 *
 *          Both indices address the post-orientation short and long series:
 *          `row` (the first argument) indexes the short side and `col` (the
 *          second) indexes the long side.
 *          Wrappers that want the "roll over the shorter side" optimisation
 *          should swap (x, y) before constructing the Cost functor — the
 *          linear kernel iterates the long side outside while rolling a
 *          short-side buffer; the banded kernel iterates the short side
 *          outside while rolling a long-side buffer.
 *
 *          Position-agnostic costs (L1, SquaredL2) capture the two span
 *          pointers and return `metric(a[row], b[col])`. Position-aware
 *          costs (WeightedL1 for WDTW) additionally index a weight table
 *          by `|row - col|`.
 *
 *          This header also keeps the existing `MetricType`→functor dispatch
 *          (formerly in warping.hpp detail:: namespace) so any caller can
 *          pick a cost functor from a runtime metric enum.
 *
 * @date 2026-04-12
 */

#pragma once

#include "dtw_options.hpp" // for MetricType
#include "../missing_utils.hpp" // for is_missing — bitwise NaN, safe under -ffast-math

#include <cmath>     // std::abs
#include <cstddef>   // size_t
#include <limits>    // std::numeric_limits (AROW NaN sentinel)

namespace dtwc::core {

// ===========================================================================
// Position-agnostic scalar pointwise distances (for standalone / test use)
// ===========================================================================

/// L1 (absolute difference).
struct L1Dist {
  template <typename T>
  T operator()(T a, T b) const noexcept { return std::abs(a - b); }
};

/// Squared L2: (a - b)^2.
struct SquaredL2Dist {
  template <typename T>
  T operator()(T a, T b) const noexcept { const T d = a - b; return d * d; }
};

/// Multivariate L1 across `ndim` channels: sum of |a[d] - b[d]|.
struct MVL1Dist {
  template <typename T>
  T operator()(const T* a, const T* b, std::size_t ndim) const noexcept {
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) sum += std::abs(a[d] - b[d]);
    return sum;
  }
};

/// Multivariate Squared L2 across `ndim` channels: sum of (a[d] - b[d])^2.
struct MVSquaredL2Dist {
  template <typename T>
  T operator()(const T* a, const T* b, std::size_t ndim) const noexcept {
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) {
      const T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};

// -----------------------------------------------------------------------------
// Metric -> cost-functor dispatch lives in EXACTLY ONE place: dtwc::detail in
// warping.hpp (both dispatch_metric and dispatch_mv_metric).
//
// Task R1: a duplicate `core::dispatch_mv_metric` (and the `core::MVL2Dist`
// Euclidean functor it selected) used to live here. It had ZERO call sites and
// had DIVERGED from the live dispatcher — it mapped MetricType::L2 to a true
// Euclidean cost while the live warping.hpp dispatcher still aliased L2 -> L1,
// so the "L2 is Euclidean" fix (task 0.6) was inert. Both were deleted and the
// Euclidean functor was migrated to dtwc::detail::MVL2Dist (warping.hpp), which
// the single live dispatcher now selects. A second dispatcher is precisely the
// hazard that caused this bug, so this file intentionally hosts none.
// -----------------------------------------------------------------------------

// ===========================================================================
// Cost functors for the unified DTW kernel — index-based (i, j)
// ===========================================================================

/// L1 cost at cell (row, col): |x[row] - y[col]|.
template <typename T>
struct SpanL1Cost {
  const T* x;
  const T* y;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    return std::abs(x[row] - y[col]);
  }
};

/// Squared-L2 cost at cell (row, col): (x[row] - y[col])^2.
template <typename T>
struct SpanSquaredL2Cost {
  const T* x;
  const T* y;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T d = x[row] - y[col];
    return d * d;
  }
};

/// WDTW cost at cell (row, col): weights[|row - col|] * |x[row] - y[col]|.
/// `weights` must be indexable at least up to max(row, col).
template <typename T>
struct SpanWeightedL1Cost {
  const T* x;
  const T* y;
  const T* weights;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const std::size_t d = (row > col) ? (row - col) : (col - row);
    return weights[d] * std::abs(x[row] - y[col]);
  }
};

/// Multivariate L1 at (row, col): sum over d of |x[row*ndim+d] - y[col*ndim+d]|.
template <typename T>
struct SpanMVL1Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) sum += std::abs(a[d] - b[d]);
    return sum;
  }
};

/// Multivariate Squared-L2 at (row, col).
template <typename T>
struct SpanMVSquaredL2Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) {
      const T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};

/// Multivariate WDTW (weighted L1) at (row, col).
template <typename T>
struct SpanMVWeightedL1Cost {
  const T* x;
  const T* y;
  const T* weights;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const std::size_t d_ij = (row > col) ? (row - col) : (col - row);
    const T w = weights[d_ij];
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) sum += std::abs(a[d] - b[d]);
    return w * sum;
  }
};

// ===========================================================================
// NaN-aware cost functors (ZeroCost missing-data strategy)
// Pairs where either operand is NaN contribute 0 cost — the warping path can
// "pass through" missing regions without penalty.
// ===========================================================================

template <typename T>
struct SpanNanAwareL1Cost {
  const T* x;
  const T* y;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T a = x[row];
    const T b = y[col];
    if (is_missing(a) || is_missing(b)) return T(0);
    return std::abs(a - b);
  }
};

template <typename T>
struct SpanNanAwareSquaredL2Cost {
  const T* x;
  const T* y;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T a = x[row];
    const T b = y[col];
    if (is_missing(a) || is_missing(b)) return T(0);
    const T d = a - b;
    return d * d;
  }
};

/// Multivariate NaN-aware L1: skips channels where either operand is NaN.
template <typename T>
struct SpanMVNanAwareL1Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      sum += std::abs(a[d] - b[d]);
    }
    return sum;
  }
};

/// Multivariate NaN-aware L2: Euclidean norm over comparable channels.
template <typename T>
struct SpanMVNanAwareL2Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      const T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }
};

template <typename T>
struct SpanMVNanAwareSquaredL2Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    for (std::size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      const T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};

// ===========================================================================
// NaN-propagating cost functors (AROW missing-data strategy)
// When either operand is NaN, the cost is NaN — the AROW Cell policy
// interprets this as a "missing pair" signal and propagates the diagonal
// predecessor with zero additional cost.
// ===========================================================================

template <typename T>
struct SpanAROWL1Cost {
  const T* x;
  const T* y;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T a = x[row];
    const T b = y[col];
    if (is_missing(a) || is_missing(b)) return std::numeric_limits<T>::quiet_NaN();
    return std::abs(a - b);
  }
};

template <typename T>
struct SpanAROWSquaredL2Cost {
  const T* x;
  const T* y;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T a = x[row];
    const T b = y[col];
    if (is_missing(a) || is_missing(b)) return std::numeric_limits<T>::quiet_NaN();
    const T d = a - b;
    return d * d;
  }
};

/// Multivariate AROW L1 cost. Signals "missing pair" (returns NaN) only when
/// no channel pair is comparable — i.e. every channel has at least one NaN
/// operand. Otherwise sums |x[d] - y[d]| over comparable channels (per-channel
/// skip, matching SpanMVNanAwareL1Cost).
///
/// Design note: direct scalar->MV lift ("any channel missing -> trigger AROW")
/// would discard usable per-channel information. This per-channel-skip
/// semantics preserves the scalar AROW recurrence when ndim = 1 (a single
/// missing channel = no comparable channels = NaN -> diagonal carry) and
/// mirrors the ZeroCost MV behaviour elsewhere in the kernel family.
template <typename T>
struct SpanMVAROWL1Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    std::size_t comparable = 0;
    for (std::size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      sum += std::abs(a[d] - b[d]);
      ++comparable;
    }
    return (comparable == 0) ? std::numeric_limits<T>::quiet_NaN() : sum;
  }
};

template <typename T>
struct SpanMVAROWSquaredL2Cost {
  const T* x;
  const T* y;
  std::size_t ndim;
  T operator()(std::size_t row, std::size_t col) const noexcept {
    const T* a = x + row * ndim;
    const T* b = y + col * ndim;
    T sum = T(0);
    std::size_t comparable = 0;
    for (std::size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      const T diff = a[d] - b[d];
      sum += diff * diff;
      ++comparable;
    }
    return (comparable == 0) ? std::numeric_limits<T>::quiet_NaN() : sum;
  }
};

} // namespace dtwc::core
