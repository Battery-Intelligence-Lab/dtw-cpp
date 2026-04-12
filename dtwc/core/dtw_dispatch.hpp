/**
 * @file dtw_dispatch.hpp
 * @brief Single-point DTW function resolver for Problem::rebind_dtw_fn.
 *
 * @details Collapses the per-(variant x missing_strategy x ndim) nested
 *          dispatch from Problem.cpp into one templated `resolve_dtw_fn<T>`.
 *          Resolution runs *once* at rebind time; the returned std::function
 *          dispatches with zero branching across the {variant, missing_strategy,
 *          ndim} axes per call (the only per-call branches are length-dependent
 *          choices inside individual wrappers — unchanged from pre-refactor).
 *
 *          Explicit instantiations for `T = data_t` (float64) and `T = float`
 *          live in dtw_dispatch.cpp, so adding a new variant only touches
 *          dtw_dispatch.cpp.
 *
 *          Fixes the pre-existing silent-fallback bug where the float32
 *          Problem::dtw_fn_f32_ always ran Standard DTW regardless of the
 *          configured variant and missing_strategy (e.g. fast_clara's
 *          chunked-Parquet path used Standard DTW even when the Problem was
 *          set to WDTW). `resolve_dtw_fn<T>` honours both axes for both T.
 */

#pragma once

#include "../settings.hpp" // data_t

#include <functional>
#include <span>

namespace dtwc {
class Problem;
}

namespace dtwc::core {

/// Build the per-pair DTW distance function for Problem `p`, templated on
/// element type (T = data_t or float). The returned std::function reads
/// `p.band`, `p.variant_params`, `p.missing_strategy`, `p.data.ndim`, and —
/// for WDTW with T = data_t — `p.wdtw_weights_cache_` at call time.
///
/// The referenced Problem must outlive the returned function.
template <typename T>
std::function<double(std::span<const T>, std::span<const T>)>
resolve_dtw_fn(const Problem &p);

// Instantiated in dtw_dispatch.cpp — no other Ts are supported.
extern template std::function<double(std::span<const data_t>, std::span<const data_t>)>
resolve_dtw_fn<data_t>(const Problem &);
extern template std::function<double(std::span<const float>, std::span<const float>)>
resolve_dtw_fn<float>(const Problem &);

} // namespace dtwc::core
