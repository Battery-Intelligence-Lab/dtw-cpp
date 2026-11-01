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
 * @author Volkan Kumtepeli
 */

#pragma once

#include "../base/settings.hpp" // data_t
#include "dtw_options.hpp"      // DistanceConfig
#include "storage.hpp"          // Precision

#include <cstddef>
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

/// The brute-force fill's block function: x and W = core::dtw_lanes<T> series in
/// `ys`, none empty, of any lengths when config.band < 0 and of x's length
/// otherwise; their W distances out, each what resolve_dtw_fn's function returns
/// for that pair (dtw_kernel_lanes) on series the fill admits (core::LanesCell),
/// bit for bit unless the compiler contracts a multiply-add in one kernel and not
/// the other. Empty unless Standard DTW, MissingStrategy::Error and univariate
/// series; the fill then goes pair by pair.
template <typename T>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
resolve_dtw_block_fn(const DistanceConfig &config);

/// The most the kernels of `config`'s bound distance (resolve_dtw_fn, and resolve_dtw_block_fn's lanes where it has
/// them) hold a thread for series of up to `length` time steps of `precision`: their thread_local scratch, grown exactly
/// (core/thread_scratch.hpp), from each buffer's growth rule. run_dtw's two rolling columns: dtw_linear's of the shorter
/// series (without a band; under one, only of a pair it covers whole, the longer series at most band + 1 long), and
/// dtw_banded's of the longer; the lanes' 2 x `length` rows of dtw_lanes<T> values; the missing data's and DDTW's two
/// copies (DDTW's of the series' values, every feature), the independent channels' two, WDTW's weight table, MSM's two
/// rows of the shorter series, TWE's two of it and one more; Soft-DTW's value one column, never its gradient's
/// matrices. A planner adds the algorithms' tables (fast_pam_thread_bytes, clara_assign_thread_bytes, ...).
std::size_t thread_scratch_bytes(const DistanceConfig &config, std::size_t length, Precision precision);

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
