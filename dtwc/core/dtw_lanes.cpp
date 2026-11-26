/**
 * @file dtw_lanes.cpp
 * @brief Implementation of resolve_dtw_block_fn<T> (see dtw_dispatch.hpp).
 *
 * @details A translation unit of its own because lld-link's LTO backend runs the
 *          loop vectoriser but not the SLP vectoriser: under clang's ThinLTO on
 *          Windows the lane loop of dtw_kernel_lanes would compile to W scalar
 *          chains instead of W / 4 vector ones. dtwc/core/CMakeLists.txt compiles this
 *          file to native code there, where clang's own -O3 pipeline runs SLP.
 * @author Volkan Kumtepeli
 */

#include "dtw_dispatch.hpp"

#include "../base/missing_utils.hpp" // interpolate_linear_into, interpolate_bufs, is_missing
#include "../warping_ddtw.hpp"       // derivative_transform_inplace, derivative_bufs
#include "../warping_wdtw.hpp"       // wdtwBanded, cached_wdtw_weights
#include "dtw_kernel.hpp"            // dtw_kernel_lanes_packed, PackedLanesPanel, dtw_lanes, LanesCell, ADTWCell, AROWCell, WDTWCell
#include "dtw_options.hpp"           // DTWVariant, MissingStrategy, MetricType
#include "public_distance.hpp"       // normalize_public_distance
#include "thread_scratch.hpp"        // lanes_scratch, held_bytes

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace dtwc::core {

namespace {

struct PointwiseL1
{
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    return std::abs(a - b);
  }
};

struct PointwiseSquaredL2
{
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    const T d = a - b;
    return d * d;
  }
};

struct PointwiseZeroCostL1
{
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    return (is_missing(a) || is_missing(b)) ? T(0) : std::abs(a - b);
  }
};

struct PointwiseZeroCostSquaredL2
{
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    if (is_missing(a) || is_missing(b)) return T(0);
    const T d = a - b;
    return d * d;
  }
};

template <typename T>
std::vector<T> &wdtw_lane_unpack_buf()
{
  thread_local std::vector<T> buf;
  return buf;
}

bool supports_lanes(const DistanceConfig &c) noexcept
{
  if (c.ndim != 1) return false;
  switch (c.missing) {
  case MissingStrategy::Interpolate:
  case MissingStrategy::ZeroCost:
  case MissingStrategy::AROW:
    return c.variant.variant == DTWVariant::Standard;
  case MissingStrategy::Error:
    break;
  }
  return c.variant.variant == DTWVariant::Standard
         || c.variant.variant == DTWVariant::DDTW
         || c.variant.variant == DTWVariant::ADTW
         || c.variant.variant == DTWVariant::WDTW;
}

template <typename T>
std::span<const T> prep_series_into(const DistanceConfig &c, std::span<const T> s, std::vector<T> &buf)
{
  if (c.missing == MissingStrategy::Interpolate)
    return interpolate_linear_into<T>(s, buf);
  if (c.variant.variant == DTWVariant::DDTW) {
    derivative_transform_inplace<T>(s, buf);
    return buf;
  }
  return s;
}

template <typename T>
std::vector<T> &query_prep_buf(const DistanceConfig &c)
{
  if (c.missing == MissingStrategy::Interpolate)
    return dtwc::detail::interpolate_bufs<T>().first;
  return dtwc::detail::derivative_bufs<T>().first;
}

template <typename T>
std::vector<T> &panel_prep_buf(const DistanceConfig &c)
{
  if (c.missing == MissingStrategy::Interpolate)
    return dtwc::detail::interpolate_bufs<T>().second;
  return dtwc::detail::derivative_bufs<T>().second;
}

template <typename T>
PackedLanesView<T> pack_transformed_into(const DistanceConfig &c, const std::span<const T> *ys,
                                         std::vector<detail::LanesRow<T>> &buf)
{
  if (c.missing != MissingStrategy::Interpolate && c.variant.variant != DTWVariant::DDTW)
    return detail::pack_lanes_into<T>(ys, buf);

  constexpr std::size_t W = dtw_lanes<T>;
  std::array<std::size_t, W> ny{};
  std::size_t ny_min = std::numeric_limits<std::size_t>::max(), ny_max = 0;
  for (std::size_t w = 0; w < W; ++w) {
    ny[w] = ys[w].size();
    ny_min = std::min(ny_min, ny[w]);
    ny_max = std::max(ny_max, ny[w]);
  }
  if (ny_min == 0) return { nullptr, ny, 0, ny_max };
  if (buf.size() < ny_max) buf.resize(ny_max);
  detail::LanesRow<T> *Y = buf.data();
  std::vector<T> &tmp = panel_prep_buf<T>(c);
  for (std::size_t w = 0; w < W; ++w) {
    const std::span<const T> yw = prep_series_into<T>(c, ys[w], tmp);
    const std::size_t n = yw.size();
    std::size_t t = 0;
    for (; t < n; ++t) Y[t].v[w] = yw[t];
    const T last = yw[n - 1];
    for (; t < ny_max; ++t) Y[t].v[w] = last;
  }
  return { Y, ny, ny_min, ny_max };
}

template <typename T, typename Dist, typename Cell>
std::function<void(std::span<const T>, const PackedLanesView<T> &, std::span<double>)>
packed_block_fn(int band, Dist dist, Cell cell)
{
  return [band, dist, cell](std::span<const T> x, const PackedLanesView<T> &panel, std::span<double> out) {
    const auto d = dtw_kernel_lanes_packed<T>(x, panel, band, dist, cell);
    for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = normalize_public_distance(d[w]);
  };
}

template <typename T, typename Dist, typename Cell>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
block_fn(DistanceConfig config, Dist dist, Cell cell)
{
  return [config, dist, cell](std::span<const T> x, std::span<const std::span<const T>> ys, std::span<double> out) {
    const std::span<const T> x_prep = prepare_dtw_query<T>(config, x);
    auto &y_buf = detail::lanes_bufs<T>().first;
    const auto panel = pack_transformed_into<T>(config, ys.data(), y_buf);
    const auto d = dtw_kernel_lanes_packed<T>(x_prep, panel, config.band, dist, cell);
    for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = normalize_public_distance(d[w]);
  };
}

template <typename T, typename MakeFn>
auto dispatch_lane_kernel(const DistanceConfig &config, MakeFn &&make)
{
  if (config.missing == MissingStrategy::ZeroCost) {
    if (config.metric == MetricType::SquaredL2)
      return make(PointwiseZeroCostSquaredL2{}, LanesCell{});
    return make(PointwiseZeroCostL1{}, LanesCell{});
  }
  if (config.missing == MissingStrategy::AROW) {
    if (config.metric == MetricType::SquaredL2)
      return make(PointwiseSquaredL2{}, AROWCell{});
    return make(PointwiseL1{}, AROWCell{});
  }
  if (config.variant.variant == DTWVariant::ADTW)
    return make(PointwiseL1{}, ADTWCell<T>{ static_cast<T>(config.variant.adtw_penalty) });
  if (config.metric == MetricType::SquaredL2)
    return make(PointwiseSquaredL2{}, LanesCell{});
  return make(PointwiseL1{}, LanesCell{});
}

} // namespace

template <typename T>
void pack_dtw_panel(const DistanceConfig &config, const std::span<const T> *ys, PackedLanesPanel<T> &panel)
{
  const auto v = pack_transformed_into<T>(config, ys, panel.rows);
  panel.ny = v.ny;
  panel.ny_min = v.ny_min;
  panel.ny_max = v.ny_max;
}

template <typename T>
std::span<const T> prepare_dtw_query(const DistanceConfig &config, std::span<const T> x)
{
  if (config.missing != MissingStrategy::Interpolate && config.variant.variant != DTWVariant::DDTW)
    return x;
  if (x.empty()) return x;
  return prep_series_into<T>(config, x, query_prep_buf<T>(config));
}

template <typename T>
std::function<void(std::span<const T>, const PackedLanesView<T> &, std::span<double>)>
resolve_dtw_packed_block_fn(const DistanceConfig &config)
{
  if (!supports_lanes(config)) return {};
  if (config.variant.variant == DTWVariant::WDTW) {
    const int band = config.band;
    const T g = static_cast<T>(config.variant.wdtw_g);
    return [band, g](std::span<const T> x, const PackedLanesView<T> &panel, std::span<double> out) {
      const std::size_t nx = x.size();
      if (nx == 0 || panel.ny_min == 0) {
        for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = std::numeric_limits<double>::max();
        return;
      }
      if (nx == panel.ny_min && nx == panel.ny_max) {
        const auto &w_tab = dtwc::detail::cached_wdtw_weights<T>(static_cast<int>(nx - 1), g);
        const auto d = dtw_kernel_lanes_packed<T>(x, panel, band, PointwiseL1{}, WDTWCell<T>{ w_tab.data() });
        for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = normalize_public_distance(d[w]);
        return;
      }
      auto &buf = wdtw_lane_unpack_buf<T>();
      for (std::size_t w = 0; w < dtw_lanes<T>; ++w) {
        const std::size_t ny_w = panel.ny[w];
        if (buf.size() < ny_w) buf.resize(ny_w);
        for (std::size_t t = 0; t < ny_w; ++t) buf[t] = panel.rows[t].v[w];
        out[w] = normalize_public_distance(wdtwBanded(x, std::span<const T>(buf.data(), ny_w), band, g));
      }
    };
  }
  return dispatch_lane_kernel<T>(config, [&](auto dist, auto cell) {
    return packed_block_fn<T>(config.band, dist, cell);
  });
}

template <typename T>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
resolve_dtw_block_fn(const DistanceConfig &config)
{
  if (!supports_lanes(config)) return {};
  if (config.variant.variant == DTWVariant::WDTW) {
    const int band = config.band;
    const T g = static_cast<T>(config.variant.wdtw_g);
    return [band, g](std::span<const T> x, std::span<const std::span<const T>> ys, std::span<double> out) {
      const std::size_t nx = x.size();
      if (nx == 0) {
        for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = std::numeric_limits<double>::max();
        return;
      }
      bool equal_len = true;
      for (std::size_t w = 0; w < dtw_lanes<T>; ++w) {
        if (ys[w].empty()) {
          for (std::size_t u = 0; u < dtw_lanes<T>; ++u) out[u] = std::numeric_limits<double>::max();
          return;
        }
        equal_len = equal_len && (ys[w].size() == nx);
      }
      if (equal_len) {
        auto &y_buf = detail::lanes_bufs<T>().first;
        const auto panel = detail::pack_lanes_into<T>(ys.data(), y_buf);
        const auto &w_tab = dtwc::detail::cached_wdtw_weights<T>(static_cast<int>(nx - 1), g);
        const auto d = dtw_kernel_lanes_packed<T>(x, panel, band, PointwiseL1{}, WDTWCell<T>{ w_tab.data() });
        for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = normalize_public_distance(d[w]);
        return;
      }
      for (std::size_t w = 0; w < dtw_lanes<T>; ++w)
        out[w] = normalize_public_distance(wdtwBanded(x, ys[w], band, g));
    };
  }
  return dispatch_lane_kernel<T>(config, [&](auto dist, auto cell) {
    return block_fn<T>(config, dist, cell);
  });
}

template void pack_dtw_panel<data_t>(const DistanceConfig &, const std::span<const data_t> *,
                                     PackedLanesPanel<data_t> &);
template void pack_dtw_panel<float>(const DistanceConfig &, const std::span<const float> *,
                                    PackedLanesPanel<float> &);
template std::span<const data_t> prepare_dtw_query<data_t>(const DistanceConfig &, std::span<const data_t>);
template std::span<const float> prepare_dtw_query<float>(const DistanceConfig &, std::span<const float>);
template std::function<void(std::span<const data_t>, const PackedLanesView<data_t> &, std::span<double>)>
resolve_dtw_packed_block_fn<data_t>(const DistanceConfig &);
template std::function<void(std::span<const float>, const PackedLanesView<float> &, std::span<double>)>
resolve_dtw_packed_block_fn<float>(const DistanceConfig &);
template std::function<void(std::span<const data_t>, std::span<const std::span<const data_t>>,
                            std::span<double>)>
resolve_dtw_block_fn<data_t>(const DistanceConfig &);
template std::function<void(std::span<const float>, std::span<const std::span<const float>>,
                            std::span<double>)>
resolve_dtw_block_fn<float>(const DistanceConfig &);

std::size_t detail::lanes_scratch(bool release) noexcept
{
  auto [y32, s32] = detail::lanes_bufs<float>();
  auto [y64, s64] = detail::lanes_bufs<data_t>();
  return held_bytes(release, y32, s32, y64, s64, wdtw_lane_unpack_buf<float>(), wdtw_lane_unpack_buf<data_t>());
}

} // namespace dtwc::core
