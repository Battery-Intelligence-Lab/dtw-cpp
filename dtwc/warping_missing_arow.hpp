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
 * (`core::run_dtw`; the full-matrix dtwAROW to `core::dtw_kernel_full`)
 * parameterised on `SpanAROW*Cost` (NaN-propagating pointwise cost) +
 * `AROWCell` (diagonal-carry recurrence). The legacy hand-rolled AROW impls lived here pre-Phase 3;
 * they were folded into the unified kernel family with bit-for-bit cross-
 * validation on {no-NaN, interior-NaN, leading-NaN, trailing-NaN, all-NaN}
 * inputs over bands {1..4}.
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * Unchecked per-pair layer (see warping.hpp): ±inf is not rejected here;
 * dtwc::distance::arow is the checked entry point.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#pragma once

#include "base/settings.hpp"
#include "core/dtw_kernel.hpp"   // run_dtw, dtw_kernel_full, AROWCell
#include "core/dtw_cost.hpp"     // SpanAROWL1Cost / SpanAROWSquaredL2Cost
#include "core/dtw_options.hpp"  // core::MetricType

#include <cstddef>     // size_t
#include <span>
#include <vector>

namespace dtwc {

// =========================================================================
//  Public API â€” DTW-AROW
// =========================================================================

/**
 * @brief Computes DTW-AROW distance with Sakoe-Chiba band constraint.
 *
 * @details Restricts the warping path to the fixed window
 * `|i-j| <= band`, in addition to the AROW missing-value constraint. A
 * non-negative band narrower than `|nx-ny|` has no endpoint-preserving path.
 * A negative band is unconstrained (dtwAROW_L).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param band Fixed Sakoe-Chiba diagonal half-width in samples. Negative
 *             means unconstrained.
 * @param metric Pointwise distance metric (default: L1).
 * @return The banded DTW-AROW distance, or
 *         `numeric_limits<data_t>::max()` for an infeasible window.
 */
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwAROW_banded(const data_t* x, std::size_t nx, const data_t* y, std::size_t ny,
                      int band = settings::DEFAULT_BAND,
                      core::MetricType metric = core::MetricType::L1)
{
  // A univariate L2 cost is L1: sqrt((a-b)^2) == |a-b|.
  if (metric == core::MetricType::SquaredL2)
    return core::run_dtw<core::SpanAROWSquaredL2Cost>(x, nx, y, ny, band, core::AROWCell{}, data_t(-1));
  return core::run_dtw<core::SpanAROWL1Cost>(x, nx, y, ny, band, core::AROWCell{}, data_t(-1));
}

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
  return dtwAROW_banded<data_t>(x, nx, y, ny, -1, metric);
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
  core::orient(x, nx, y, ny);
  if (metric == core::MetricType::SquaredL2)
    return core::dtw_kernel_full<data_t>(nx, ny, core::SpanAROWSquaredL2Cost<data_t>{ x, y }, core::AROWCell{});
  return core::dtw_kernel_full<data_t>(nx, ny, core::SpanAROWL1Cost<data_t>{ x, y }, core::AROWCell{});
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

