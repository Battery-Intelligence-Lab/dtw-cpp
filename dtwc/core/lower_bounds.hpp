/**
 * @file lower_bounds.hpp
 * @brief Compile-time metric x lower-bound compatibility matrix.
 *
 * @details Records which (metric, lower-bound) combinations consumers may
 *          select. Implementations live in lower_bound_impl.hpp. F47 records
 *          that the current SquaredL2/LB_Kim predicate is broader than the
 *          L1-valued implementation, so this matrix is audited behavior rather
 *          than a blanket mathematical proof.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "distance_metric.hpp"

namespace dtwc::core {

// -------------------------------------------------------------------
//  LB_Keogh validity: only for L1 and SquaredL2
// -------------------------------------------------------------------

template <typename Metric>
inline constexpr bool lb_keogh_valid = false;

template <>
inline constexpr bool lb_keogh_valid<L1Metric> = true;

template <>
inline constexpr bool lb_keogh_valid<SquaredL2Metric> = true;

// L2Metric is identical to L1Metric for scalars (both compute |a-b|)
template <>
inline constexpr bool lb_keogh_valid<L2Metric> = true;

// -------------------------------------------------------------------
//  LB_Kim validity: requires monotone pointwise metric
//  (if |a-c| <= |a-b| then d(a,c) <= d(a,b))
//  Default false — explicitly opt in per metric.
// -------------------------------------------------------------------

template <typename Metric>
inline constexpr bool lb_kim_valid = false;

template <>
inline constexpr bool lb_kim_valid<L1Metric> = true;

template <>
inline constexpr bool lb_kim_valid<SquaredL2Metric> = true;

template <>
inline constexpr bool lb_kim_valid<L2Metric> = true;

} // namespace dtwc::core
