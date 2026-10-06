/**
 * @file dtw_dispatch.hpp
 * @brief Single-point DTW function resolver for Problem::rebind_dtw_fn.
 *
 * @details Collapses the per-(variant x missing_strategy x ndim) nested
 *          dispatch into one templated `resolve_dtw_fn<T>`. Resolution runs
 *          *once*, when a Problem's distance settings or series change; the
 *          returned std::function dispatches with zero branching across the
 *          {variant, missing_strategy, ndim} axes per call (the only per-call
 *          branches are length-dependent choices inside individual wrappers).
 *
 *          The returned functions hold copies of the settings they read, so
 *          they stay valid when the Problem that bound them moves, and parallel
 *          callers only read them.
 *
 *          Explicit instantiations for `T = data_t` (float64) and `T = float`
 *          live in dtw_dispatch.cpp, so adding a new variant only touches
 *          dtw_dispatch.cpp.
 */

#pragma once

#include "../base/settings.hpp" // data_t
#include "dtw_options.hpp"      // DistanceConfig

#include <functional>
#include <span>

namespace dtwc {
struct Data;
}

namespace dtwc::core {

/// The per-pair DTW distance function for `config`, templated on element type
/// (T = data_t or float). `data` sizes the WDTW weight table to the lengths of
/// the series the function will meet; no other variant reads it.
/// `config` must have passed validate(config, T is float): nothing here checks it.
template <typename T>
std::function<double(std::span<const T>, std::span<const T>)>
resolve_dtw_fn(const DistanceConfig &config, const Data &data);

/// The brute-force fill's block function: x and W = core::dtw_lanes<T> series
/// of x's length in `ys`, their W distances out, each what resolve_dtw_fn's
/// function returns for that pair (dtw_kernel_lanes) on series the fill admits
/// (core::LanesCell), bit for bit unless the compiler contracts a multiply-add
/// in one kernel and not the other. Empty unless Standard DTW,
/// MissingStrategy::Error and univariate series; the fill then goes pair by pair.
template <typename T>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
resolve_dtw_block_fn(const DistanceConfig &config);

// Instantiated in dtw_dispatch.cpp and dtw_lanes.cpp — no other Ts are supported.
extern template std::function<double(std::span<const data_t>, std::span<const data_t>)>
resolve_dtw_fn<data_t>(const DistanceConfig &, const Data &);
extern template std::function<double(std::span<const float>, std::span<const float>)>
resolve_dtw_fn<float>(const DistanceConfig &, const Data &);
extern template std::function<void(std::span<const data_t>, std::span<const std::span<const data_t>>,
                                   std::span<double>)>
resolve_dtw_block_fn<data_t>(const DistanceConfig &);
extern template std::function<void(std::span<const float>, std::span<const std::span<const float>>,
                                   std::span<double>)>
resolve_dtw_block_fn<float>(const DistanceConfig &);

} // namespace dtwc::core
