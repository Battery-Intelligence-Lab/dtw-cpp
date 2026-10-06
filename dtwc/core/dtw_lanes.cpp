/**
 * @file dtw_lanes.cpp
 * @brief Implementation of resolve_dtw_block_fn<T> (see dtw_dispatch.hpp).
 *
 * @details A translation unit of its own because lld-link's LTO backend runs the
 *          loop vectoriser but not the SLP vectoriser: under clang's ThinLTO on
 *          Windows the lane loop of dtw_kernel_lanes would compile to W scalar
 *          chains instead of W / 4 vector ones. dtwc/core/CMakeLists.txt compiles this
 *          file to native code there, where clang's own -O3 pipeline runs SLP.
 */

#include "dtw_dispatch.hpp"

#include "dtw_kernel.hpp"     // dtw_kernel_lanes, dtw_lanes, LanesCell
#include "dtw_options.hpp"    // DTWVariant, MissingStrategy, MetricType
#include "public_distance.hpp" // normalize_public_distance

#include <cmath>
#include <cstddef>
#include <functional>
#include <span>

namespace dtwc::core {

namespace {

template <typename T, typename Dist>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
block_fn(int band, Dist dist)
{
  return [band, dist](std::span<const T> x, std::span<const std::span<const T>> ys, std::span<double> out) {
    const T *y[dtw_lanes<T>];
    for (std::size_t w = 0; w < dtw_lanes<T>; ++w) y[w] = ys[w].data();
    const auto d = dtw_kernel_lanes<T>(x.data(), y, x.size(), band, dist, LanesCell{});
    for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = normalize_public_distance(d[w]);
  };
}

} // namespace

template <typename T>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
resolve_dtw_block_fn(const DistanceConfig &config)
{
  // make_standard's univariate path: its kernels run the recurrence the lanes
  // run, and these are its costs, SpanSquaredL2Cost and SpanL1Cost (univariate
  // L2 is L1), on values: the lanes interleave W series, so they cannot index one.
  if (config.variant.variant != DTWVariant::Standard
      || config.missing != MissingStrategy::Error || config.ndim != 1)
    return {};
  if (config.metric == MetricType::SquaredL2)
    return block_fn<T>(config.band, [](T a, T b) {
      const T d = a - b;
      return d * d;
    });
  return block_fn<T>(config.band, [](T a, T b) { return std::abs(a - b); });
}

template std::function<void(std::span<const data_t>, std::span<const std::span<const data_t>>,
                            std::span<double>)>
resolve_dtw_block_fn<data_t>(const DistanceConfig &);
template std::function<void(std::span<const float>, std::span<const std::span<const float>>,
                            std::span<double>)>
resolve_dtw_block_fn<float>(const DistanceConfig &);

} // namespace dtwc::core
