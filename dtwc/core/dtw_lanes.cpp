/**
 * @file dtw_lanes.cpp
 * @brief Implementation of resolve_dtw_block_fn<T> (see dtw_dispatch.hpp).
 *
 * @details A translation unit of its own because lld-link's LTO backend runs the
 *          loop vectoriser but not the SLP vectoriser: under clang's ThinLTO on
 *          Windows the lane loop of dtw_kernel_lanes would compile to W scalar
 *          chains instead of W / 4 vector ones. dtwc/core/CMakeLists.txt compiles this
 *          file to native code there, where clang's own -O3 pipeline runs SLP.
 *
 * @author Volkan Kumtepeli
 * @date 29 Sep 2026
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

enum class LanePrep { None, DDTW, Interpolate };

constexpr LanePrep lane_prep_of(const DistanceConfig &c) noexcept
{
  if (c.missing == MissingStrategy::Interpolate) return LanePrep::Interpolate;
  if (c.variant.variant == DTWVariant::DDTW) return LanePrep::DDTW;
  return LanePrep::None;
}

template <LanePrep Prep, typename T>
std::span<const T> prep_series_into(std::span<const T> s, std::vector<T> &buf)
{
  if constexpr (Prep == LanePrep::Interpolate) {
    return interpolate_linear_into<T>(s, buf);
  } else if constexpr (Prep == LanePrep::DDTW) {
    derivative_transform_inplace<T>(s, buf);
    return buf;
  } else {
    (void)buf;
    return s;
  }
}

template <LanePrep Prep, typename T>
std::span<const T> prepare_query_impl(std::span<const T> x)
{
  if constexpr (Prep == LanePrep::None) {
    return x;
  } else {
    if (x.empty()) return x;
    auto &buf = (Prep == LanePrep::Interpolate)
                  ? dtwc::detail::interpolate_bufs<T>().first
                  : dtwc::detail::derivative_bufs<T>().first;
    return prep_series_into<Prep, T>(x, buf);
  }
}

template <LanePrep Prep, typename T>
PackedLanesView<T> pack_transformed_into(const std::span<const T> *ys, std::vector<detail::LanesRow<T>> &buf)
{
  if constexpr (Prep == LanePrep::None) {
    return detail::pack_lanes_into<T>(ys, buf);
  } else {
    auto &tmp = (Prep == LanePrep::Interpolate)
                  ? dtwc::detail::interpolate_bufs<T>().second
                  : dtwc::detail::derivative_bufs<T>().second;
    return detail::pack_lanes_with<T>(ys, buf, [&tmp](std::span<const T> s) {
      return prep_series_into<Prep, T>(s, tmp);
    });
  }
}

template <typename T>
void run_wdtw_equal_len_lanes(std::span<const T> x, const PackedLanesView<T> &panel,
                              int band, T g, std::span<double> out)
{
  const auto &w_tab = dtwc::detail::cached_wdtw_weights<T>(static_cast<int>(x.size() - 1), g);
  const auto d = dtw_kernel_lanes_packed<T>(x, panel, band, PointwiseL1{}, WDTWCell<T>{ w_tab.data() });
  for (std::size_t w = 0; w < dtw_lanes<T>; ++w) out[w] = normalize_public_distance(d[w]);
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

template <LanePrep Prep, typename T, typename Dist, typename Cell>
std::function<void(std::span<const T>, std::span<const std::span<const T>>, std::span<double>)>
block_fn(int band, Dist dist, Cell cell)
{
  return [band, dist, cell](std::span<const T> x, std::span<const std::span<const T>> ys, std::span<double> out) {
    const std::span<const T> x_prep = prepare_query_impl<Prep, T>(x);
    auto &y_buf = detail::lanes_bufs<T>().first;
    const auto panel = pack_transformed_into<Prep, T>(ys.data(), y_buf);
    const auto d = dtw_kernel_lanes_packed<T>(x_prep, panel, band, dist, cell);
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
  const PackedLanesView<T> v = [&] {
    switch (lane_prep_of(config)) {
    case LanePrep::Interpolate: return pack_transformed_into<LanePrep::Interpolate, T>(ys, panel.rows);
    case LanePrep::DDTW:        return pack_transformed_into<LanePrep::DDTW, T>(ys, panel.rows);
    case LanePrep::None:        break;
    }
    return pack_transformed_into<LanePrep::None, T>(ys, panel.rows);
  }();
  panel.ny = v.ny;
  panel.ny_min = v.ny_min;
  panel.ny_max = v.ny_max;
}

template <typename T>
std::span<const T> prepare_dtw_query(const DistanceConfig &config, std::span<const T> x)
{
  switch (lane_prep_of(config)) {
  case LanePrep::Interpolate: return prepare_query_impl<LanePrep::Interpolate, T>(x);
  case LanePrep::DDTW:        return prepare_query_impl<LanePrep::DDTW, T>(x);
  case LanePrep::None:        break;
  }
  return x;
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
        run_wdtw_equal_len_lanes<T>(x, panel, band, g, out);
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
        run_wdtw_equal_len_lanes<T>(x, panel, band, g, out);
        return;
      }
      for (std::size_t w = 0; w < dtw_lanes<T>; ++w)
        out[w] = normalize_public_distance(wdtwBanded(x, ys[w], band, g));
    };
  }
  const LanePrep prep = lane_prep_of(config);
  const int band = config.band;
  return dispatch_lane_kernel<T>(config, [&](auto dist, auto cell) {
    switch (prep) {
    case LanePrep::Interpolate: return block_fn<LanePrep::Interpolate, T>(band, dist, cell);
    case LanePrep::DDTW:        return block_fn<LanePrep::DDTW, T>(band, dist, cell);
    case LanePrep::None:        break;
    }
    return block_fn<LanePrep::None, T>(band, dist, cell);
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
  return held_bytes(release, y32, s32, y64, s64,
                    detail::lanes_end_buf<float>(), detail::lanes_end_buf<data_t>(),
                    wdtw_lane_unpack_buf<float>(), wdtw_lane_unpack_buf<data_t>());
}

} // namespace dtwc::core
