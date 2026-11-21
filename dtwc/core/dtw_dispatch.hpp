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

#include <functional>
#include <span>

namespace dtwc {
struct Data;
}

namespace dtwc::core {

template <typename T>
struct PackedLanesView;
template <typename T>
struct PackedLanesPanel;

/// The per-pair DTW distance function for `config`, templated on element type
/// (T = data_t or float). `data` (and optional `other` for cross-collection
/// `cdist`) sizes the WDTW weight table to the series lengths the function will
/// meet; no other variant reads them.
/// `config` must have passed validate(config, T is float): nothing here checks it.
template <typename T>
std::function<double(std::span<const T>, std::span<const T>)>
resolve_dtw_fn(const DistanceConfig &config, const Data &data, const Data *other = nullptr);

/// The Soft-DTW divergence D(x, y) = sdtw(x, y) - (sdtw(x, x) + sdtw(y, y)) / 2 (Blondel, Mensch & Vert, AISTATS
/// 2021) from the raw values resolve_dtw_fn's Soft-DTW function gives: what a Problem clusters on under Soft-DTW. For
/// this library's Soft-DTW (univariate, L1 cost) D >= 0, and D = 0 only for identical series, at every gamma > 0; a
/// value below 0 is rounding and reads 0. NaN stays NaN, which std::max(0.0, diff) would read as 0.
inline double soft_dtw_divergence(double xy, double xx, double yy) noexcept
{
  const double diff = xy - (xx + yy) / 2;
  return diff < 0 ? 0.0 : diff;
}

/// The brute-force fill's block function: x and W = core::dtw_lanes<T> series in
/// `ys`, none empty, of any lengths (both unconstrained and Sakoe-Chiba banded);
/// their W distances out, each agreeing with resolve_dtw_fn's function within the
/// FMA route bound. Active for univariate Standard DTW, DDTW, and
/// MissingStrategy::Interpolate; empty otherwise (callers fall back to pair-by-pair).
template <typename T>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
resolve_dtw_block_fn(const DistanceConfig &config);

/// Pack W = core::dtw_lanes<T> series into `panel`, applying any per-series preprocessing
/// (DDTW derivative or Interpolate linear fill) once so `panel` can be reused across many queries.
template <typename T>
void pack_dtw_panel(const DistanceConfig &config, const std::span<const T> *ys, PackedLanesPanel<T> &panel);

/// Prepare a single query series `x` for `resolve_dtw_packed_block_fn<T>` (zero-copy identity for Standard DTW;
/// uses the calling thread's registered derivative/interpolate buffer for DDTW / Interpolate).
template <typename T>
std::span<const T> prepare_dtw_query(const DistanceConfig &config, std::span<const T> x);

/// Compute W distances between a prepared query `x` (`prepare_dtw_query`) and a pre-packed `panel` (`pack_dtw_panel`).
template <typename T>
std::function<void(std::span<const T>, const PackedLanesView<T> &, std::span<double>)>
resolve_dtw_packed_block_fn(const DistanceConfig &config);

// Instantiated in dtw_dispatch.cpp and dtw_lanes.cpp — no other Ts are supported.
extern template std::function<double(std::span<const data_t>, std::span<const data_t>)>
resolve_dtw_fn<data_t>(const DistanceConfig &, const Data &, const Data *);
extern template std::function<double(std::span<const float>, std::span<const float>)>
resolve_dtw_fn<float>(const DistanceConfig &, const Data &, const Data *);
extern template std::function<void(std::span<const data_t>, std::span<const std::span<const data_t>>,
                                   std::span<double>)>
resolve_dtw_block_fn<data_t>(const DistanceConfig &);
extern template std::function<void(std::span<const float>, std::span<const std::span<const float>>,
                                   std::span<double>)>
resolve_dtw_block_fn<float>(const DistanceConfig &);
extern template void pack_dtw_panel<data_t>(const DistanceConfig &, const std::span<const data_t> *,
                                            PackedLanesPanel<data_t> &);
extern template void pack_dtw_panel<float>(const DistanceConfig &, const std::span<const float> *,
                                           PackedLanesPanel<float> &);
extern template std::span<const data_t> prepare_dtw_query<data_t>(const DistanceConfig &, std::span<const data_t>);
extern template std::span<const float> prepare_dtw_query<float>(const DistanceConfig &, std::span<const float>);
extern template std::function<void(std::span<const data_t>, const PackedLanesView<data_t> &, std::span<double>)>
resolve_dtw_packed_block_fn<data_t>(const DistanceConfig &);
extern template std::function<void(std::span<const float>, const PackedLanesView<float> &, std::span<double>)>
resolve_dtw_packed_block_fn<float>(const DistanceConfig &);

} // namespace dtwc::core
