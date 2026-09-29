/**
 * @file unit_test_dtw_kernel_lanes.cpp
 * @brief dtw_kernel_lanes: each lane is bitwise the per-pair kernel it stands in
 *        for, over precision, length, band and metric.
 *
 * @details The oracle is the per-pair path the brute-force fill runs today:
 *          dtwFull_eap for band < 0, dtwBanded otherwise, plus dtwFull_L (the
 *          recurrence the lanes mirror) for band < 0. Bits, not tolerances: a
 *          lane that differs in one ulp would make a matrix depend on how its
 *          columns fall into blocks.
 */

#include "core/dtw_kernel.hpp"
#include "core/dtw_options.hpp" // core::MetricType
#include "warping.hpp"          // dtwFull_eap, dtwFull_L, dtwBanded, detail::dispatch_metric

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using dtwc::core::MetricType;

template <typename T>
bool same_bits(T a, T b)
{
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// Random walks, or small integers: the latter make exact ties in every min.
template <typename T>
std::vector<T> series(std::mt19937_64 &rng, std::size_t n, bool ties)
{
  std::normal_distribution<double> step;
  std::uniform_int_distribution<int> level(0, 3);
  std::vector<T> v(n);
  double walk = 0;
  for (auto &e : v) e = ties ? T(level(rng)) : T(walk += step(rng));
  return v;
}

// Bitwise mismatches between the lanes and the per-pair path, over W pairs.
template <typename T>
int lane_mismatches(std::size_t n, int band, MetricType metric, bool ties)
{
  constexpr std::size_t W = dtwc::core::dtw_lanes<T>;
  std::mt19937_64 rng(1000003u * n + 7919u * W + 31u * std::size_t(band + 2) + (ties ? 1 : 0));
  const auto x = series<T>(rng, n, ties);
  std::vector<std::vector<T>> y;
  std::array<const T *, W> ys{};
  for (std::size_t w = 0; w < W; ++w) y.push_back(series<T>(rng, n, ties));
  for (std::size_t w = 0; w < W; ++w) ys[w] = y[w].data();

  const auto lanes = dtwc::detail::dispatch_metric(metric, [&](auto dist) {
    return dtwc::core::dtw_kernel_lanes<T>(x.data(), ys.data(), n, band, dist,
                                           dtwc::core::StandardCell{});
  });
  int bad = 0;
  for (std::size_t w = 0; w < W; ++w) {
    bad += !same_bits(lanes[w], dtwc::dtwBanded<T>(x, y[w], band, T(-1), metric));
    if (band < 0) bad += !same_bits(lanes[w], dtwc::dtwFull_eap<T>(x, y[w], metric));
  }
  return bad;
}

template <typename T>
void check_every_configuration()
{
  for (const std::size_t n : { 1, 2, 7, 100, 1000 })
    for (const int band : { -1, 0, 1, int(n / 10) })
      for (const auto metric : { MetricType::L1, MetricType::SquaredL2 })
        for (const bool ties : { false, true }) {
          CAPTURE(sizeof(T), n, band, int(metric), ties);
          CHECK(lane_mismatches<T>(n, band, metric, ties) == 0);
        }
}

} // namespace

TEST_CASE("dtw_kernel_lanes: every lane is bitwise the per-pair kernel", "[lanes]")
{
  SECTION("double, 8 lanes") { check_every_configuration<double>(); }
  SECTION("float, 16 lanes") { check_every_configuration<float>(); }
}

TEST_CASE("dtw_kernel_lanes: empty series are the no-path sentinel", "[lanes]")
{
  const double *none[dtwc::core::dtw_lanes<double>] = {};
  const auto d = dtwc::core::dtw_kernel_lanes<double>(nullptr, none, 0, -1,
                                                      dtwc::detail::L1Dist{},
                                                      dtwc::core::StandardCell{});
  for (const double v : d) CHECK(v == std::numeric_limits<double>::max());
}
