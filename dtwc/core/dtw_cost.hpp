/**
 * @file dtw_cost.hpp
 * @brief Cost functors for the unified DTW kernel.
 *
 * @details A Cost functor models the pointwise cost at cell (row, col):
 *            T operator()(size_t row, size_t col) const noexcept;
 *
 *          `row` indexes the outer-loop series (passed as the first series
 *          to the kernel), `col` indexes the inner-loop series (second).
 *          Wrappers that want the "roll over the shorter side" optimisation
 *          should swap (x, y) before constructing the Cost functor — the
 *          kernel always rolls over the second (`col`) dimension.
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
#include <utility>   // std::forward

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

/// Runtime-metric → compile-time-functor bridge (scalar).
template <typename Fn>
auto dispatch_metric(MetricType m, Fn&& fn) -> decltype(fn(L1Dist{})) {
  switch (m) {
    case MetricType::SquaredL2: return std::forward<Fn>(fn)(SquaredL2Dist{});
    case MetricType::L2:
    case MetricType::L1:
    default:                    return std::forward<Fn>(fn)(L1Dist{});
  }
}

/// Runtime-metric → compile-time-functor bridge (multivariate).
template <typename Fn>
auto dispatch_mv_metric(MetricType m, Fn&& fn) -> decltype(fn(MVL1Dist{})) {
  switch (m) {
    case MetricType::SquaredL2: return std::forward<Fn>(fn)(MVSquaredL2Dist{});
    case MetricType::L2:
    case MetricType::L1:
    default:                    return std::forward<Fn>(fn)(MVL1Dist{});
  }
}

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

} // namespace dtwc::core
