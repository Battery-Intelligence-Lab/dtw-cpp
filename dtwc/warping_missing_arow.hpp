/**
 * @file warping_missing_arow.hpp
 * @brief DTW-AROW: diagonal-only alignment for missing values (NaN-aware).
 *
 * @details Implements the DTW-AROW recurrence from Yurtman et al. (ECML-PKDD 2023).
 * When x[i] or y[j] is NaN, the warping path is restricted to the DIAGONAL direction
 * only (one-to-one alignment), preventing many-to-one "free stretching" through
 * missing regions.
 *
 * Standard DTW recurrence:
 *   C(i,j) = min(C(i-1,j-1), C(i-1,j), C(i,j-1)) + d(x[i], y[j])
 *
 * AROW recurrence:
 *   if is_missing(x[i]) or is_missing(y[j]):
 *       C(i,j) = C(i-1,j-1)    // diagonal only, zero local cost
 *   else:
 *       C(i,j) = min(C(i-1,j-1), C(i-1,j), C(i,j-1)) + d(x[i], y[j])
 *
 * Boundary treatment: at boundaries (first row/column), only one direction of
 * movement is available (no many-to-one cheating possible), so missing values
 * at boundaries propagate from the single available predecessor with zero cost
 * (NOT set to +inf, which would cascade and make the matrix unreachable).
 *
 * Implementation: these wrappers delegate to the unified DTW kernel
 * (`core::dtw_kernel_{full,linear,banded}`) parameterised on
 * `SpanAROW*Cost` (NaN-propagating pointwise cost) + `AROWCell` (diagonal-
 * carry recurrence). The legacy hand-rolled AROW impls lived here pre-Phase 3;
 * they were folded into the unified kernel family with bit-for-bit cross-
 * validation on {no-NaN, interior-NaN, leading-NaN, trailing-NaN, all-NaN}
 * inputs over bands {1..4}.
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

#include "settings.hpp"
#include "core/dtw_kernel.hpp"   // dtw_kernel_full / _linear / _banded, AROWCell
#include "core/dtw_cost.hpp"     // SpanAROWL1Cost / SpanAROWSquaredL2Cost
#include "core/dtw_options.hpp"  // core::MetricType

#include <algorithm>   // std::min, std::max
#include <cstddef>     // size_t
#include <span>
#include <vector>

namespace dtwc {

namespace detail {

/// Orient (x, y) so the second side is at least as long. The unified kernel
/// requires `n_short <= n_long`; AROW costs (L1/SquaredL2) are symmetric so
/// swapping is a no-op on the result.
template <typename T>
struct ArowOriented {
  const T* short_ptr;
  const T* long_ptr;
  std::size_t n_short;
  std::size_t n_long;
};

template <typename T>
ArowOriented<T> arow_orient(const T* x, std::size_t nx,
                            const T* y, std::size_t ny) noexcept
{
  if (nx <= ny) return {x, y, nx, ny};
  return {y, x, ny, nx};
}

} // namespace detail

// =========================================================================
//  Public API â€” DTW-AROW
// =========================================================================

/**
 * @brief Computes DTW-AROW distance (linear space, O(min(m,n)) memory).
 *
 * @details When x[i] or y[j] is NaN, the warping path is restricted to the
 * diagonal direction only (zero local cost), preventing free stretching through
 * missing regions. Uses O(min(m,n)) space â€” no backtracking.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW-AROW distance.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_L(const data_t* x, std::size_t nx, const data_t* y, std::size_t ny,
                 core::MetricType metric = core::MetricType::L1)
{
  const auto o = detail::arow_orient(x, nx, y, ny);
  if (metric == core::MetricType::SquaredL2) {
    core::SpanAROWSquaredL2Cost<data_t> cost{o.short_ptr, o.long_ptr};
    return core::dtw_kernel_linear<data_t>(o.n_short, o.n_long, cost, core::AROWCell{});
  }
  core::SpanAROWL1Cost<data_t> cost{o.short_ptr, o.long_ptr};
  return core::dtw_kernel_linear<data_t>(o.n_short, o.n_long, cost, core::AROWCell{});
}

/**
 * @brief Computes DTW-AROW distance (full matrix, O(m*n) memory).
 *
 * @details Same recurrence as dtwAROW_L but stores the full cost matrix for
 * debugging and verification. The result is identical to dtwAROW_L.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW-AROW distance.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW(const data_t* x, std::size_t nx, const data_t* y, std::size_t ny,
               core::MetricType metric = core::MetricType::L1)
{
  const auto o = detail::arow_orient(x, nx, y, ny);
  if (metric == core::MetricType::SquaredL2) {
    core::SpanAROWSquaredL2Cost<data_t> cost{o.short_ptr, o.long_ptr};
    return core::dtw_kernel_full<data_t>(o.n_short, o.n_long, cost, core::AROWCell{});
  }
  core::SpanAROWL1Cost<data_t> cost{o.short_ptr, o.long_ptr};
  return core::dtw_kernel_full<data_t>(o.n_short, o.n_long, cost, core::AROWCell{});
}

/**
 * @brief Computes DTW-AROW distance with Sakoe-Chiba band constraint.
 *
 * @details Restricts the warping path to a band of width @p band around the
 * diagonal, in addition to the AROW missing-value constraint. When band < 0,
 * falls back to dtwAROW_L (unbanded linear-space).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param band Sakoe-Chiba bandwidth. Negative means unconstrained.
 * @param metric Pointwise distance metric (default: L1).
 * @return The banded DTW-AROW distance.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_banded(const data_t* x, std::size_t nx, const data_t* y, std::size_t ny,
                      int band = settings::DEFAULT_BAND,
                      core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwAROW_L<data_t>(x, nx, y, ny, metric);
  const auto m_short = std::min(nx, ny);
  const auto m_long  = std::max(nx, ny);
  if (m_short <= 1 || m_long <= static_cast<std::size_t>(band) + 1)
    return dtwAROW_L<data_t>(x, nx, y, ny, metric);

  const auto o = detail::arow_orient(x, nx, y, ny);
  if (metric == core::MetricType::SquaredL2) {
    core::SpanAROWSquaredL2Cost<data_t> cost{o.short_ptr, o.long_ptr};
    return core::dtw_kernel_banded<data_t>(o.n_short, o.n_long, band, cost, core::AROWCell{});
  }
  core::SpanAROWL1Cost<data_t> cost{o.short_ptr, o.long_ptr};
  return core::dtw_kernel_banded<data_t>(o.n_short, o.n_long, band, cost, core::AROWCell{});
}

// =========================================================================
//  Span + vector convenience overloads
// =========================================================================

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_L(std::span<const data_t> x, std::span<const data_t> y,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_L<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW(std::span<const data_t> x, std::span<const data_t> y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_banded(std::span<const data_t> x, std::span<const data_t> y,
                      int band = settings::DEFAULT_BAND,
                      core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_banded<data_t>(x.data(), x.size(), y.data(), y.size(), band, metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_L<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_banded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      int band = settings::DEFAULT_BAND,
                      core::MetricType metric = core::MetricType::L1)
{
  return dtwAROW_banded<data_t>(x.data(), x.size(), y.data(), y.size(), band, metric);
}

} // namespace dtwc

