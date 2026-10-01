/**
 * @file test_dtw.cpp
 * @brief Every DTW variant, metric, band, channel count and precision against the
 *        independent oracle in tests/support/dtw_oracle.hpp, through each route a
 *        caller has to the distance.
 *
 * @details Bands, registered before the first run. Where the library and the oracle
 *          perform literally the same additions, minima and absolute values in the
 *          same order (double, one channel, L1 cost; Standard, ADTW, MSM) the two
 *          agree to the bit. Everywhere else they agree within dtw_routes_agree
 *          (tests/support/dtw_route_bound.hpp), 2 (nx + ny - 1) eps(T) max(|a|, |b|): a
 *          warping path has at most nx + ny - 1 cells, each adding one rounded cost, so
 *          two evaluations of the recurrence differ by the rounding of one path's sum, and
 *          a different rounding can move a near-tie to the other path by no more. The
 *          routes differ in the last bit through contraction of x*y + z, reassociation and
 *          SIMD lanes (epsilon-level, accepted: clusterings must not change, not bits);
 *          float rows run the library in float against the oracle on the same float
 *          values widened to double.
 */

#include <dtwc.hpp>
#include <core/dtw_dispatch.hpp>

#include "../../support/deterministic_series.hpp"
#include "../../support/dtw_oracle.hpp"
#include "../../support/dtw_route_bound.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ts = dtwc::test_support;
using dtwc::Data;
using dtwc::core::DistanceConfig;
using dtwc::core::DTWVariant;
using dtwc::core::DTWVariantParams;
using dtwc::core::MetricType;
using dtwc::core::MissingStrategy;
using dtwc::core::MVMode;

namespace {

constexpr double kNoPathPublic = std::numeric_limits<double>::max();

/// One distance definition: the variant with its parameters, the metric, the channel count.
struct Config
{
  std::string name;
  DTWVariantParams params;
  MetricType metric = MetricType::L1;
  std::size_t ndim = 1;
};

std::vector<Config> configs()
{
  std::vector<Config> out;
  for (const auto metric : { MetricType::L1, MetricType::L2, MetricType::SquaredL2 }) {
    out.push_back({ "standard", {}, metric, 1 });
    out.push_back({ "standard dependent", { .mv_mode = MVMode::Dependent }, metric, 3 });
    out.push_back({ "standard independent", { .mv_mode = MVMode::Independent }, metric, 3 });
  }
  for (const std::size_t ndim : { 1, 3 }) {
    for (const auto metric : { MetricType::L1, MetricType::L2, MetricType::SquaredL2 })
      out.push_back({ "ddtw", { .variant = DTWVariant::DDTW }, metric, ndim });
    for (const double g : { 0.0, 0.3 })
      out.push_back({ "wdtw g=" + std::to_string(g), { .variant = DTWVariant::WDTW, .wdtw_g = g },
                      MetricType::L1, ndim });
    for (const double penalty : { 0.0, 0.5, 1e18 }) // 1e18: only the diagonal is affordable
      out.push_back({ "adtw penalty=" + std::to_string(penalty),
                      { .variant = DTWVariant::ADTW, .adtw_penalty = penalty }, MetricType::L1, ndim });
  }
  out.push_back({ "softdtw", { .variant = DTWVariant::SoftDTW, .sdtw_gamma = 0.7 }, MetricType::L1, 1 });
  out.push_back({ "msm", { .variant = DTWVariant::MSM, .msm_c = 0.8 }, MetricType::L1, 1 });
  out.push_back({ "twe", { .variant = DTWVariant::TWE, .twe_nu = 0.05, .twe_lambda = 0.6 },
                  MetricType::L1, 1 });
  return out;
}

std::string label(const Config &config)
{
  const char *metric = config.metric == MetricType::L1 ? "l1"
                       : config.metric == MetricType::L2 ? "l2"
                                                         : "squared l2";
  return config.name + ", " + metric + ", ndim " + std::to_string(config.ndim);
}

ts::OracleSpec oracle_spec(const Config &config, int band)
{
  ts::OracleSpec spec;
  switch (config.params.variant) {
  case DTWVariant::Standard: spec.variant = ts::OracleVariant::Standard; break;
  case DTWVariant::DDTW: spec.variant = ts::OracleVariant::DDTW; break;
  case DTWVariant::WDTW: spec.variant = ts::OracleVariant::WDTW; break;
  case DTWVariant::ADTW: spec.variant = ts::OracleVariant::ADTW; break;
  case DTWVariant::SoftDTW: spec.variant = ts::OracleVariant::SoftDTW; break;
  case DTWVariant::MSM: spec.variant = ts::OracleVariant::MSM; break;
  case DTWVariant::TWE: spec.variant = ts::OracleVariant::TWE; break;
  }
  switch (config.metric) {
  case MetricType::L1: spec.metric = ts::OracleMetric::L1; break;
  case MetricType::L2: spec.metric = ts::OracleMetric::L2; break;
  case MetricType::SquaredL2: spec.metric = ts::OracleMetric::SquaredL2; break;
  }
  spec.band = band;
  spec.ndim = config.ndim;
  spec.independent = config.params.mv_mode == MVMode::Independent && config.ndim > 1;
  spec.g = config.params.wdtw_g;
  spec.penalty = config.params.adtw_penalty;
  spec.gamma = config.params.sdtw_gamma;
  spec.c = config.params.msm_c;
  spec.nu = config.params.twe_nu;
  spec.lambda = config.params.twe_lambda;
  return spec;
}

DistanceConfig distance_config(const Config &config, int band)
{
  return { config.params, config.metric, MissingStrategy::Error, band, config.ndim };
}

/// Whether the library and the oracle perform the same arithmetic (see the file comment).
template <typename T>
bool same_arithmetic(const Config &config)
{
  const auto variant = config.params.variant;
  return std::is_same_v<T, double> && config.ndim == 1 && config.metric == MetricType::L1
         && (variant == DTWVariant::Standard || variant == DTWVariant::ADTW || variant == DTWVariant::MSM);
}

/// Soft-DTW adds up costs and softmin offsets of opposite sign (each offset is at most
/// gamma ln 3 below zero), so its rounding error follows its largest cell, not the result,
/// which can be near 0: the scale is gamma (nx + ny - 1). No other variant cancels.
double cancellation(const Config &config, std::size_t nx, std::size_t ny)
{
  const bool soft = config.params.variant == DTWVariant::SoftDTW;
  return soft ? config.params.sdtw_gamma * static_cast<double>(nx + ny - 1) : 0.0;
}

/// `got` is the library's value; `no_path` its sentinel for "no admissible path". A nonzero
/// `scale` replaces max(|a|, |b|) in the bound by max(|a|, |b|, scale), so that a result
/// near zero between large cells is judged against those cells.
template <typename T>
bool agrees(double got, double want, double no_path, std::size_t nx, std::size_t ny, bool bitwise,
            double scale = 0)
{
  if (std::isinf(want)) return got == no_path;
  if (bitwise) return got == want;
  if (scale == 0) return ts::dtw_routes_agree<T>(got, want, nx, ny);
  const double largest = std::max(std::max(std::abs(got), std::abs(want)), scale);
  return std::abs(got - want)
         <= 2.0 * static_cast<double>(nx + ny - 1) * std::numeric_limits<T>::epsilon() * largest;
}

// ---- series: lengths in steps; the empty and one-step shapes have no or one cell per row

enum class Alias { None, Copy, Same };

struct Shape
{
  std::size_t nx, ny;
  Alias alias = Alias::None;
};

constexpr Shape kShapes[] = {
  { 9, 9 }, { 9, 9, Alias::Copy }, { 9, 9, Alias::Same }, { 6, 11 }, { 11, 6 },
  { 1, 1 }, { 1, 5 },              { 5, 1 },              { 0, 4 },  { 4, 0 },
};
// -1 and -100 both mean no band; 0, 3, 5 straddle the length differences 0, 4 and 5; 50 and INT_MAX exceed every length.
constexpr int kBands[] = { -1, -100, 0, 3, 5, 50, std::numeric_limits<int>::max() };

template <typename T>
struct Pair
{
  std::vector<T> x, y;
  bool same = false; ///< y is x itself: one buffer, which the v1 kernels shortcut to 0

  const std::vector<T> &second() const { return same ? x : y; }
  std::size_t steps_x(std::size_t ndim) const { return x.size() / ndim; }
  std::size_t steps_y(std::size_t ndim) const { return second().size() / ndim; }
};

template <typename T>
Pair<T> draw(const Shape &shape, std::size_t ndim, unsigned seed)
{
  const auto series = [&](std::size_t steps, unsigned s) {
    const auto values = ts::benchmark_series(steps * ndim, s);
    return std::vector<T>(values.begin(), values.end());
  };
  Pair<T> pair{ series(shape.nx, seed), {}, shape.alias == Alias::Same };
  if (shape.alias == Alias::Copy) pair.y = pair.x;
  else if (!pair.same) pair.y = series(shape.ny, seed + 1);
  return pair;
}

template <typename T>
std::vector<double> widen(const std::vector<T> &values)
{
  return std::vector<double>(values.begin(), values.end());
}

} // namespace

TEMPLATE_TEST_CASE("every variant, metric, band and channel count agrees with the oracle",
                   "[dtw][oracle]", double, float)
{
  using T = TestType;
  std::size_t rows = 0;
  for (const auto &config : configs())
    for (std::size_t s = 0; s < std::size(kShapes); ++s) {
      const auto pair = draw<T>(kShapes[s], config.ndim, 1000 + 10 * static_cast<unsigned>(s));
      const std::span<const T> x{ pair.x }, y{ pair.second() };
      const auto xd = widen(pair.x), yd = widen(pair.second());
      const std::size_t nx = pair.steps_x(config.ndim), ny = pair.steps_y(config.ndim);

      // The series the Problem would hold; WDTW sizes its weights from them. A function
      // resolved without them (a centroid of another length) weighs on demand: same values.
      std::vector<Data> known;
      known.push_back(Data{});
      if constexpr (std::is_same_v<T, double>)
        known.push_back(Data{ std::vector<std::vector<double>>{ xd, yd }, { "x", "y" }, config.ndim });

      for (const int band : kBands) {
        const double want = ts::dtw_oracle(oracle_spec(config, band), xd, yd);
        const bool exact = same_arithmetic<T>(config);
        const auto check = [&](const char *route, double got, double no_path) {
          ++rows;
          INFO(label(config) << ", band " << band << ", shape " << s << " (" << nx << "x" << ny << "), "
                             << route << ": got " << got << ", oracle " << want);
          CHECK(agrees<T>(got, want, no_path, nx, ny, exact, cancellation(config, nx, ny)));
        };

        for (const auto &data : known)
          check("Problem's bound function",
                dtwc::core::resolve_dtw_fn<T>(distance_config(config, band), data)(x, y), kNoPathPublic);
        const auto sentinel = static_cast<double>(std::numeric_limits<T>::max());
        if (config.ndim == 1) check("distance::dtw", dtwc::distance::dtw<T>(x, y, config.params, band, config.metric), sentinel);
        if (config.ndim == 1 && config.params.variant == DTWVariant::Standard) {
          // Without a NaN the missing-data distances are Standard DTW.
          check("distance::arow", dtwc::distance::arow<T>(x, y, band, config.metric), sentinel);
          check("distance::missing", dtwc::distance::missing<T>(x, y, band, config.metric), sentinel);
        }
      }
    }
  CHECK(rows > configs().size() * std::size(kShapes) * std::size(kBands));
}

namespace {

/// A value worked out on paper from the definitions: the oracle must reproduce it (so that
/// it is not trusted blindly), and so must the library.
struct Hand
{
  Config config;
  int band;
  std::vector<double> x, y;
  double want;
};

std::vector<Hand> hand_values()
{
  const Config l1{ "standard", {}, MetricType::L1, 1 };
  const Config sq{ "standard", {}, MetricType::SquaredL2, 1 };
  constexpr double none = ts::kOracleNoPath;
  return {
    // x = {1,2,3} against y = {3,4,5,6,7}: accumulated L1 cost D(i,j)
    //         y=3  4  5  6  7
    //   x=1:    2  5  9 14 20
    //   x=2:    3  4  7 11 16
    //   x=3:    3  4  6  9 13
    // The cheapest path pairs 1 and 2 with the first 3, then 3 with every y:
    // 2+1+0+1+2+3+4 = 13, and squared 4+1+0+1+4+9+16 = 35.
    { l1, -1, { 1, 2, 3 }, { 3, 4, 5, 6, 7 }, 13 },
    { l1, -1, { 3, 4, 5, 6, 7 }, { 1, 2, 3 }, 13 },
    { sq, -1, { 1, 2, 3 }, { 3, 4, 5, 6, 7 }, 35 },

    // x = {0,1,0,2,0}, y = {0,0,0,0,0,0,2}: the end cell costs |0-2| = 2 (squared 4) on
    // every path and the 1 of x costs 1 against any y. The 2 of x meets the 2 of y for
    // free only at offset 3, so band >= 3 gives 1 + 0 + 2 = 3 (squared 1 + 0 + 4 = 5);
    // band 2 pays |2-0| = 2 (squared 4) more for it: 5 (squared 9); a band below the
    // length difference 2 admits no path.
    { l1, -1, { 0, 1, 0, 2, 0 }, { 0, 0, 0, 0, 0, 0, 2 }, 3 },
    { l1, 3, { 0, 1, 0, 2, 0 }, { 0, 0, 0, 0, 0, 0, 2 }, 3 },
    { l1, 2, { 0, 1, 0, 2, 0 }, { 0, 0, 0, 0, 0, 0, 2 }, 5 },
    { l1, 1, { 0, 1, 0, 2, 0 }, { 0, 0, 0, 0, 0, 0, 2 }, none },
    { sq, 3, { 0, 1, 0, 2, 0 }, { 0, 0, 0, 0, 0, 0, 2 }, 5 },
    { sq, 2, { 0, 1, 0, 2, 0 }, { 0, 0, 0, 0, 0, 0, 2 }, 9 },
    // One step against three has the single path along the three cells: 1+2+3 = 6; the
    // band must reach the length difference 2.
    { l1, 1, { 0 }, { 1, 2, 3 }, none },
    { l1, 2, { 0 }, { 1, 2, 3 }, 6 },

    // Accumulated DTW is no metric. L1: {0,0}-{0,1,2} costs 3 but {0,0}-{0,1} and
    // {0,1}-{0,1,2} cost 1 each. Squared: {0}-{2} costs 4 but {0}-{1} and {1}-{2} cost 1
    // each. And {0,1}-{0,0,1} costs 0 with the series different (path (0,0),(0,1),(1,2)).
    { l1, -1, { 0, 0 }, { 0, 1, 2 }, 3 },
    { l1, -1, { 0, 0 }, { 0, 1 }, 1 },
    { l1, -1, { 0, 1 }, { 0, 1, 2 }, 1 },
    { sq, -1, { 0 }, { 2 }, 4 },
    { sq, -1, { 0 }, { 1 }, 1 },
    { sq, -1, { 1 }, { 2 }, 1 },
    { l1, -1, { 0, 1 }, { 0, 0, 1 }, 0 },

    // ADTW, penalty 2, x = {0,1}, y = {0,0,1}: step (0,0)-(0,1) is horizontal, so it pays
    // 2 + |0-0|, and (0,1)-(1,2) is diagonal at |1-1| = 0: 2. Going (0,0)-(1,1)-(1,2)
    // costs |1-0| + (2 + 0) = 3.
    { { "adtw", { .variant = DTWVariant::ADTW, .adtw_penalty = 2 } }, -1, { 0, 1 }, { 0, 0, 1 }, 2 },
    // WDTW with g = 0: every weight is 1 / (1 + e^0) = 1/2, so half of the 13 above.
    { { "wdtw", { .variant = DTWVariant::WDTW, .wdtw_g = 0 } }, -1, { 1, 2, 3 }, { 3, 4, 5, 6, 7 }, 6.5 },
    // WDTW with g = 2 ln 3, x = {0,0}, y = {1,1}: m = 1, so w(0) = 1 / (1 + e^(g/2)) = 1/4 and
    // w(1) = 1 / (1 + e^(-g/2)) = 3/4. The diagonal pairs cost 1 at offset 0: 1/4 + 1/4.
    { { "wdtw", { .variant = DTWVariant::WDTW, .wdtw_g = 2 * std::log(3.0) } }, -1, { 0, 0 }, { 1, 1 }, 0.5 },
    // DDTW: the slopes of {0,1,2} are ((1-0) + (2-0)/2)/2 = 1 inside and 1 at the ends,
    // those of {0,2,4} are 2: three cells at |1-2| = 1.
    { { "ddtw", { .variant = DTWVariant::DDTW } }, -1, { 0, 1, 2 }, { 0, 2, 4 }, 3 },
    // Soft-DTW: every cost is 0 and the first row and column keep their single
    // predecessor, so only the last cell softens three zeros: -gamma ln 3.
    { { "softdtw", { .variant = DTWVariant::SoftDTW, .sdtw_gamma = 0.7 } }, -1, { 0, 0 }, { 0, 0 },
      -0.7 * std::log(3.0) },
    // MSM, c = 0.5: {0,0} to {0} is one split, and 0 lies between its neighbours: cost c.
    { { "msm", { .variant = DTWVariant::MSM, .msm_c = 0.5 } }, -1, { 0, 0 }, { 0 }, 0.5 },
    // MSM, c = 0.5, {0,2} to {1}: the only path moves 0 onto 1 (|0-1| = 1), then adds 2 beside its
    // neighbour 0 while y is at 1. 2 lies outside [0,1], so that costs c + min(|2-0|, |2-1|) = 1.5.
    { { "msm", { .variant = DTWVariant::MSM, .msm_c = 0.5 } }, -1, { 0, 2 }, { 1 }, 2.5 },
    // TWE, nu = 0.05, lambda = 0.6: {0,0} to {0} matches the first 0 at cost 0 and deletes
    // the second at |0-0| + nu + lambda.
    { { "twe", { .variant = DTWVariant::TWE, .twe_nu = 0.05, .twe_lambda = 0.6 } }, -1, { 0, 0 }, { 0 }, 0.65 },

    // Two channels per step, x = (0,0),(1,0),(0,1), y = (0,0),(0,1),(1,0). One shared path
    // (dependent) sums the channel costs in each cell:
    //          j=0 j=1 j=2        accumulated:  0 1 2
    //   i=0:     0   1   1                      1 2 1
    //   i=1:     1   2   0                      2 1 3
    //   i=2:     1   0   2
    // gives 3. Each channel on its own may warp differently (independent): channel 0
    // {0,1,0} against {0,0,1} costs 1 along (0,0),(0,1),(1,2),(2,2), channel 1
    // {0,0,1} against {0,1,0} costs 1 along (0,0),(1,0),(2,1),(2,2): 2 in total.
    { { "dependent", { .mv_mode = MVMode::Dependent }, MetricType::L1, 2 }, -1,
      { 0, 0, 1, 0, 0, 1 }, { 0, 0, 0, 1, 1, 0 }, 3 },
    { { "independent", { .mv_mode = MVMode::Independent }, MetricType::L1, 2 }, -1,
      { 0, 0, 1, 0, 0, 1 }, { 0, 0, 0, 1, 1, 0 }, 2 },
    // One step of two channels, (0,0) against (3,4): the cell cost is the metric, |3|+|4| = 7,
    // sqrt(3^2 + 4^2) = 5 and 3^2 + 4^2 = 25.
    { { "l1", {}, MetricType::L1, 2 }, -1, { 0, 0 }, { 3, 4 }, 7 },
    { { "l2", {}, MetricType::L2, 2 }, -1, { 0, 0 }, { 3, 4 }, 5 },
    { { "squared l2", {}, MetricType::SquaredL2, 2 }, -1, { 0, 0 }, { 3, 4 }, 25 },
  };
}

} // namespace

TEST_CASE("the oracle and the library reproduce the hand-computed values", "[dtw][oracle][hand]")
{
  for (const auto &hand : hand_values()) {
    const auto &config = hand.config;
    const std::size_t nx = hand.x.size() / config.ndim, ny = hand.y.size() / config.ndim;
    INFO(label(config) << ", band " << hand.band << ", " << nx << "x" << ny << ", want " << hand.want);
    const double oracle = ts::dtw_oracle(oracle_spec(config, hand.band), hand.x, hand.y);
    CHECK(agrees<double>(oracle, hand.want, ts::kOracleNoPath, nx, ny, false));

    const Data none; // WDTW weighs on demand without the series
    const double library =
      dtwc::core::resolve_dtw_fn<double>(distance_config(config, hand.band), none)(hand.x, hand.y);
    CHECK(agrees<double>(library, hand.want, kNoPathPublic, nx, ny, false));
  }
}

namespace {

/// A v1.0.0 entry point (dtwFull, dtwFull_L, dtwBanded, as vectors and as pointers),
/// or a multivariate wrapper over the same kernels.
template <typename T>
struct V1Route
{
  using Vec = std::vector<T>;
  const char *name;
  bool banded, multivariate, independent;
  std::function<T(const Vec &, const Vec &, int band, MetricType, std::size_t ndim)> call;
};

template <typename T>
std::vector<V1Route<T>> v1_routes()
{
  using Vec = std::vector<T>;
  using R = V1Route<T>;
  return {
    R{ "dtwFull", false, false, false,
       [](const Vec &x, const Vec &y, int, MetricType m, std::size_t) { return dtwc::dtwFull<T>(x, y, m); } },
    R{ "dtwFull (pointers)", false, false, false,
       [](const Vec &x, const Vec &y, int, MetricType m, std::size_t) {
         return dtwc::dtwFull<T>(x.data(), x.size(), y.data(), y.size(), m);
       } },
    R{ "dtwFull_L", false, false, false,
       [](const Vec &x, const Vec &y, int, MetricType m, std::size_t) { return dtwc::dtwFull_L<T>(x, y, T(-1), m); } },
    R{ "dtwFull_L (pointers)", false, false, false,
       [](const Vec &x, const Vec &y, int, MetricType m, std::size_t) {
         return dtwc::dtwFull_L<T>(x.data(), x.size(), y.data(), y.size(), T(-1), m);
       } },
    R{ "dtwBanded", true, false, false,
       [](const Vec &x, const Vec &y, int band, MetricType m, std::size_t) {
         return dtwc::dtwBanded<T>(x, y, band, T(-1), m);
       } },
    R{ "dtwBanded (pointers)", true, false, false,
       [](const Vec &x, const Vec &y, int band, MetricType m, std::size_t) {
         return dtwc::dtwBanded<T>(x.data(), x.size(), y.data(), y.size(), band, T(-1), m);
       } },
    R{ "dtwFull_L_mv", false, true, false,
       [](const Vec &x, const Vec &y, int, MetricType m, std::size_t nd) {
         return dtwc::dtwFull_L_mv<T>(x.data(), x.size() / nd, y.data(), y.size() / nd, nd, T(-1), m);
       } },
    R{ "dtwBanded_mv", true, true, false,
       [](const Vec &x, const Vec &y, int band, MetricType m, std::size_t nd) {
         return dtwc::dtwBanded_mv<T>(x.data(), x.size() / nd, y.data(), y.size() / nd, nd, band, T(-1), m);
       } },
    R{ "dtw_independent_mv", true, true, true,
       [](const Vec &x, const Vec &y, int band, MetricType m, std::size_t nd) {
         return dtwc::dtw_independent_mv<T>(x.data(), x.size() / nd, y.data(), y.size() / nd, nd, band, m);
       } },
  };
}

} // namespace

TEMPLATE_TEST_CASE("the v1.0.0 entry points and the multivariate wrappers agree with the oracle",
                   "[dtw][oracle][v1]", double, float)
{
  using T = TestType;
  std::size_t rows = 0;
  for (const auto &route : v1_routes<T>())
    for (const auto metric : { MetricType::L1, MetricType::L2, MetricType::SquaredL2 })
      for (const std::size_t ndim : { 1, 3 }) {
        if (ndim > 1 && !route.multivariate) continue;
        const Config config{ route.name, { .mv_mode = route.independent ? MVMode::Independent : MVMode::Dependent },
                             metric, ndim };
        for (std::size_t s = 0; s < std::size(kShapes); ++s) {
          const auto pair = draw<T>(kShapes[s], ndim, 5000 + 10 * static_cast<unsigned>(s));
          const auto xd = widen(pair.x), yd = widen(pair.second());
          const std::size_t nx = pair.steps_x(ndim), ny = pair.steps_y(ndim);
          for (const int band : route.banded ? std::span<const int>(kBands) : std::span<const int>(kBands).first(1)) {
            const double want = ts::dtw_oracle(oracle_spec(config, band), xd, yd);
            const double got = route.call(pair.x, pair.second(), band, metric, ndim);
            ++rows;
            INFO(label(config) << ", band " << band << ", shape " << s << " (" << nx << "x" << ny
                               << "), got " << got << ", oracle " << want);
            CHECK(agrees<T>(got, want, static_cast<double>(std::numeric_limits<T>::max()), nx, ny,
                            same_arithmetic<T>(config)));
          }
        }
      }
  CHECK(rows > 0);
}

TEMPLATE_TEST_CASE("Problem::fill_distance_matrix stores what the oracle computes",
                   "[dtw][oracle][problem]", double, float)
{
  // Seventeen series of one length: the fill takes the columns of their rows in blocks of 8 (float:
  // 16) through the SIMD-lane kernel. Two of other lengths go pair by pair.
  std::vector<std::size_t> kSteps(17, 12);
  kSteps.insert(kSteps.end(), { 10, 14 });
  using T = TestType;
  std::size_t pairs = 0;
  for (const auto &config : configs())
    for (const int band : { -1, 4 }) { // 4 is the longest length difference
      std::vector<std::vector<T>> series;
      std::vector<std::string> names;
      for (std::size_t i = 0; i < kSteps.size(); ++i) {
        const auto values = ts::benchmark_series(kSteps[i] * config.ndim, 300 + static_cast<unsigned>(i));
        series.emplace_back(values.begin(), values.end());
        names.push_back("s" + std::to_string(i));
      }
      const auto held = series;
      dtwc::Problem problem("dtw");
      problem.set_data(Data{ std::move(series), std::move(names), config.ndim });
      problem.set_distance(distance_config(config, band));
      problem.fill_distance_matrix();

      for (std::size_t i = 0; i < held.size(); ++i)
        for (std::size_t j = i + 1; j < held.size(); ++j) {
          const double want = ts::dtw_oracle(oracle_spec(config, band), widen(held[i]), widen(held[j]));
          const double got = problem.dist_by_ind(static_cast<dtwc::index_t>(i), static_cast<dtwc::index_t>(j));
          ++pairs;
          INFO(label(config) << ", band " << band << ", pair " << i << "," << j << ": got " << got
                             << ", oracle " << want);
          CHECK(agrees<T>(got, want, kNoPathPublic, kSteps[i], kSteps[j], false,
                          cancellation(config, kSteps[i], kSteps[j])));
        }
    }
  CHECK(pairs > 0);
}

TEST_CASE("early abandon never changes a distance it does not exceed", "[dtw][early_abandon]")
{
  // Whole numbers throughout, so every sum is exact and a threshold can sit on the distance.
  // A row whose every cell exceeds the threshold ends the call with max(); every path
  // here costs at least |1-3| = 2 in its first cell, so a threshold of 1 abandons.
  const std::vector<double> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  struct Call
  {
    const char *name;
    ts::OracleSpec spec;
    std::function<double(double threshold)> call;
  };
  const std::vector<Call> calls{
    { "dtwFull_L", { .band = -1 }, [&](double t) { return dtwc::dtwFull_L<double>(x, y, t); } },
    { "dtwBanded", { .band = 3 }, [&](double t) { return dtwc::dtwBanded<double>(x, y, 3, t); } },
    { "dtwFull_L, squared", { .metric = ts::OracleMetric::SquaredL2, .band = -1 },
      [&](double t) { return dtwc::dtwFull_L<double>(x, y, t, MetricType::SquaredL2); } },
    { "adtwFull_L", { .variant = ts::OracleVariant::ADTW, .band = -1, .penalty = 2 },
      [&](double t) { return dtwc::adtwFull_L<double>(x, y, 2.0, t); } },
    { "adtwBanded", { .variant = ts::OracleVariant::ADTW, .band = 3, .penalty = 2 },
      [&](double t) { return dtwc::adtwBanded<double>(x, y, 3, 2.0, t); } },
  };
  for (const auto &c : calls) {
    const double exact = ts::dtw_oracle(c.spec, x, y);
    INFO(c.name << ", distance " << exact);
    CHECK(c.call(-1) == exact);
    CHECK(c.call(1e12) == exact);
    CHECK(c.call(exact) == exact);
    CHECK(c.call(1) == kNoPathPublic);
  }
}

TEST_CASE("the Sakoe-Chiba window keeps size_t indices beyond INT_MAX", "[dtw][band]")
{
  constexpr auto widest = std::numeric_limits<int>::max();
  constexpr auto row = static_cast<std::size_t>(widest) + 3, columns = static_cast<std::size_t>(widest) + 10;
  using Range = std::pair<std::size_t, std::size_t>;
  CHECK(dtwc::core::dtw_band_bounds(widest, row, columns) == Range{ 3, columns }); // columns row-band .. end
  CHECK(dtwc::core::dtw_band_bounds(0, row, columns) == Range{ row, row + 1 });    // the diagonal cell
}
