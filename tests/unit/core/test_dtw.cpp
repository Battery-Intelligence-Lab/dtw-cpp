/**
 * @file test_dtw.cpp
 * @brief The oracle in tests/support/dtw_oracle.hpp reproduces values worked out on
 *        paper, one or more per variant, metric, band and multivariate mode.
 *
 * @details Two values agree to 2 (nx + ny - 1) eps max(|a|, |b|): a warping path has at most
 *          nx + ny - 1 cells, each adding one rounded cost.
 */

#include <dtwc.hpp>

#include "../../support/dtw_oracle.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace ts = dtwc::test_support;
using dtwc::core::DTWVariant;
using dtwc::core::DTWVariantParams;
using dtwc::core::MetricType;
using dtwc::core::MVMode;

namespace {

/// One distance definition: the variant with its parameters, the metric, the channel count.
struct Config
{
  std::string name;
  DTWVariantParams params;
  MetricType metric = MetricType::L1;
  std::size_t ndim = 1;
};

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

/// Whether `got` (the library, whose "no admissible path" sentinel is `no_path`) agrees with the
/// oracle: to the bit if `bitwise`, else to the bound of the file comment.
bool agrees(double got, double want, double no_path, std::size_t nx, std::size_t ny, double eps, bool bitwise)
{
  if (std::isinf(want)) return got == no_path;
  if (bitwise) return got == want;
  return std::abs(got - want) <= 2.0 * static_cast<double>(nx + ny - 1) * eps * std::max(std::abs(got), std::abs(want));
}

/// A value worked out on paper from the definitions: the oracle must reproduce it, so
/// that it is not trusted blindly.
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
    // MSM, c = 0.5, {0,2} to {1}: the only path merges 2 into its neighbour 0 against the other 1;
    // 2 is outside [0,1], so the cost is c + min(|2-0|, |2-1|) = 1.5, after |0-1| = 1.
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

TEST_CASE("the oracle reproduces the hand-computed values", "[dtw][oracle][hand]")
{
  for (const auto &hand : hand_values()) {
    const auto &config = hand.config;
    const std::size_t nx = hand.x.size() / config.ndim, ny = hand.y.size() / config.ndim;
    INFO(label(config) << ", band " << hand.band << ", " << nx << "x" << ny << ", want " << hand.want);
    const double oracle = ts::dtw_oracle(oracle_spec(config, hand.band), hand.x, hand.y);
    CHECK(agrees(oracle, hand.want, ts::kOracleNoPath, nx, ny, std::numeric_limits<double>::epsilon(), false));
  }
}
