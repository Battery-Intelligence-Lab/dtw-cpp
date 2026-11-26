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
 * (`core::run_dtw`) parameterised on `SpanL1Cost` / `SpanSquaredL2Cost`
 * (IEEE-754 NaN propagation) + `AROWCell` (diagonal-carry recurrence).
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
#include "core/dtw_kernel.hpp"   // run_dtw, AROWCell
#include "core/dtw_cost.hpp"     // SpanL1Cost / SpanSquaredL2Cost
#include "core/dtw_options.hpp"  // core::MetricType
#include "core/series.hpp"       // core::SeriesPair

#include <cstddef>     // size_t
#include <span>
#include <vector>

namespace dtwc {

// =========================================================================
//  Public API — DTW-AROW
// =========================================================================

/**
 * @brief Computes DTW-AROW distance with Sakoe-Chiba band constraint.
 *
 * @details Restricts the warping path to the fixed window
 * `|i-j| <= band`, in addition to the AROW missing-value constraint. A
 * non-negative band narrower than `|nx-ny|` has no endpoint-preserving path.
 * A negative band, the default, is unconstrained (linear space, O(min(m,n))
 * memory, no backtracking).
 *
 * When x[i] or y[j] is NaN, the warping path is restricted to the diagonal
 * direction only (zero local cost), preventing free stretching through
 * missing regions.
 *
 * @tparam X, Y The series: contiguous ranges of one floating-point type (core::SeriesPair),
 *              which may contain NaN for missing values.
 * @param x First series.
 * @param y Second series.
 * @param band Fixed Sakoe-Chiba diagonal half-width in samples. Negative
 *             means unconstrained.
 * @param metric Pointwise distance metric (default: L1).
 * @return The banded DTW-AROW distance, or
 *         `numeric_limits<T>::max()` for an infeasible window.
 */
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> dtwAROW_banded(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                                       core::MetricType metric = core::MetricType::L1)
{
  using value_t = core::series_value_t<X>;
  const std::span<const value_t> xs{ x }, ys{ y };
  // A univariate L2 cost is L1: sqrt((a-b)^2) == |a-b|.
  if (metric == core::MetricType::SquaredL2)
    return core::run_dtw<core::SpanSquaredL2Cost>(xs.data(), xs.size(), ys.data(), ys.size(), band,
                                                  core::AROWCell{}, value_t(-1));
  return core::run_dtw<core::SpanL1Cost>(xs.data(), xs.size(), ys.data(), ys.size(), band, core::AROWCell{},
                                         value_t(-1));
}

} // namespace dtwc

