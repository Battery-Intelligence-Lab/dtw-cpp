/**
 * @file unit_test_dtw_kernel_lanes.cpp
 * @brief dtw_kernel_lanes: each lane agrees with the per-pair kernel it stands in
 *        for, over precision, length, band and metric; and the brute-force fill,
 *        which runs the lanes, agrees with the per-pair fill.
 *
 * @details The oracle is the per-pair path the brute-force fill ran before the
 *          lanes: dtwBanded, plus dtwFull_L (the recurrence the lanes mirror,
 *          and what dtwBanded runs for band < 0) called directly for band < 0.
 *          The lanes and the per-pair kernel are different code, so a compiler
 *          that contracts a multiply-add into an FMA in one of them (GCC does by
 *          default) moves the last bits: they agree within the route bound of
 *          dtw_route_bound.hpp. A pair the fill was given stays bitwise. The
 *          lanes run the fill's cell, core::LanesCell (an fminnm min on
 *          AArch64), so the ties (equal values, zero costs) exercise that min.
 */

#include "Problem.hpp"
#include "core/dtw_kernel.hpp"
#include "core/dtw_options.hpp" // core::MetricType
#include "warping.hpp"          // dtwFull_L, dtwBanded

#include "../../support/dtw_route_bound.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using dtwc::core::MetricType;
using dtwc::test_support::dtw_routes_agree;

// The pointwise costs of the per-pair path, SpanL1Cost and SpanSquaredL2Cost, on values.
constexpr auto l1 = [](auto a, auto b) { return std::abs(a - b); };
constexpr auto squared = [](auto a, auto b) {
  const auto d = a - b;
  return d * d;
};

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

// Lanes outside the route bound of the per-pair path, over W pairs.
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

  const auto lanes = metric == MetricType::SquaredL2
                       ? dtwc::core::dtw_kernel_lanes<T>(x.data(), ys.data(), n, band, squared,
                                                         dtwc::core::LanesCell{})
                       : dtwc::core::dtw_kernel_lanes<T>(x.data(), ys.data(), n, band, l1,
                                                         dtwc::core::LanesCell{});
  int bad = 0;
  for (std::size_t w = 0; w < W; ++w) {
    const T banded = dtwc::dtwBanded<T>(x, y[w], band, T(-1), metric);
    bad += !dtw_routes_agree<T>(lanes[w], banded, n, n);
    if (band < 0) {
      const T full = dtwc::dtwFull_L<T>(x, y[w], T(-1), metric);
      bad += !dtw_routes_agree<T>(lanes[w], full, n, n);
    }
  }
  return bad;
}

template <typename T>
void check_every_configuration()
{
  for (const std::size_t n : { 1u, 2u, 7u, 100u, 1000u })
    for (const int band : { -1, 0, 1, int(n / 10) })
      for (const auto metric : { MetricType::L1, MetricType::SquaredL2 })
        for (const bool ties : { false, true }) {
          CAPTURE(sizeof(T), n, band, int(metric), ties);
          CHECK(lane_mismatches<T>(n, band, metric, ties) == 0);
        }
}

} // namespace

TEST_CASE("dtw_kernel_lanes: every lane agrees with the per-pair kernel", "[lanes]")
{
  SECTION("double") { check_every_configuration<double>(); }
  SECTION("float") { check_every_configuration<float>(); }
}

TEST_CASE("dtw_kernel_lanes: empty series are the no-path sentinel", "[lanes]")
{
  const double *none[dtwc::core::dtw_lanes<double>] = {};
  const auto d = dtwc::core::dtw_kernel_lanes<double>(nullptr, none, 0, -1, l1,
                                                      dtwc::core::LanesCell{});
  for (const double v : d) CHECK(v == std::numeric_limits<double>::max());
}

namespace {

struct FillCase
{
  const char *name;
  bool f32;
  std::vector<std::size_t> lengths; // one per series
  int band;
  MetricType metric;
  std::vector<std::pair<std::size_t, std::size_t>> known; // pairs set before the fill
};

template <typename T>
dtwc::Data walks(const std::vector<std::size_t> &lengths)
{
  std::mt19937_64 rng(2026);
  std::vector<std::vector<T>> rows;
  std::vector<std::string> names;
  for (const auto n : lengths) {
    rows.push_back(series<T>(rng, n, false));
    names.push_back("s" + std::to_string(names.size()));
  }
  return dtwc::Data(std::move(rows), std::move(names));
}

// Entries (i < j) outside the route bound of the per-pair function over the
// pair, which the fill ran for every pair before the lanes; a pair set before
// the fill must keep its value, bit for bit.
int fill_mismatches(const FillCase &c)
{
  const std::size_t N = c.lengths.size();
  dtwc::Problem prob("lanes");
  prob.set_data(c.f32 ? walks<float>(c.lengths) : walks<double>(c.lengths));
  prob.set_band(c.band);
  prob.set_metric(c.metric);
  std::vector<double> expected(N * N);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j)
      expected[i * N + j] =
        c.f32 ? prob.dtw_function_f32()(prob.data().series_f32(i), prob.data().series_f32(j))
              : prob.dtw_function()(prob.series(i), prob.series(j));
  if (!c.known.empty()) {
    auto &m = prob.writable_distance_matrix();
    m.resize(N);
    for (const auto [i, j] : c.known) {
      expected[i * N + j] = 1e9 + double(i * N + j); // no distance here comes near
      m.set(i, j, expected[i * N + j]);
    }
  }
  prob.fill_distance_matrix();
  const auto &m = prob.distance_matrix();
  int bad = 0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j) {
      const double got = m.get(i, j), want = expected[i * N + j];
      const bool kept =
        std::find(c.known.begin(), c.known.end(), std::pair{ i, j }) != c.known.end();
      const bool agree = c.f32 ? dtw_routes_agree<float>(got, want, c.lengths[i], c.lengths[j])
                               : dtw_routes_agree<double>(got, want, c.lengths[i], c.lengths[j]);
      bad += !(kept ? got == want : agree);
    }
  return bad;
}

} // namespace

// 2W + 5 series (W = dtw_lanes: 21 double and 37 float on x86-64, 37 and 69 on
// AArch64) give rows of two blocks and a tail, of one block, and of a tail alone.
TEST_CASE("fill_distance_matrix: the lanes fill agrees with the per-pair fill", "[lanes][fill]")
{
  constexpr std::size_t W64 = dtwc::core::dtw_lanes<double>, n64 = 2 * W64 + 5;
  constexpr std::size_t n32 = 2 * dtwc::core::dtw_lanes<float> + 5;
  std::vector<std::size_t> alternating, halves(12, 50);
  for (std::size_t i = 0; i < 21; ++i) alternating.push_back(i % 2 ? 53 : 50);
  halves.resize(24, 53); // blocks of one length, and blocks across the two
  // Row 0's first block known (skipped), two pairs of row 1's first block
  // (computed, not overwritten), and a pair of row 0's tail.
  std::vector<std::pair<std::size_t, std::size_t>> known{ { 1, 3 }, { 1, 6 }, { 0, n64 - 1 } };
  for (std::size_t j = 1; j <= W64; ++j) known.emplace_back(0, j);
  const std::vector<FillCase> cases{
    { "equal lengths, full", false, std::vector<std::size_t>(n64, 60), -1, MetricType::L1, {} },
    { "equal lengths, band 6, squared L2", false, std::vector<std::size_t>(n64, 60), 6,
      MetricType::SquaredL2, {} },
    { "float32, full", true, std::vector<std::size_t>(n32, 40), -1, MetricType::L1, {} },
    { "float32, band 4, squared L2", true, std::vector<std::size_t>(n32, 40), 4,
      MetricType::SquaredL2, {} },
    { "mixed lengths: every block falls back", false, alternating, -1, MetricType::L1, {} },
    { "two lengths: some blocks fall back", false, halves, 3, MetricType::L1, {} },
    { "partially computed", false, std::vector<std::size_t>(n64, 60), -1, MetricType::L1, known },
  };
  for (const auto &c : cases) {
    CAPTURE(c.name);
    CHECK(fill_mismatches(c) == 0);
  }
}
