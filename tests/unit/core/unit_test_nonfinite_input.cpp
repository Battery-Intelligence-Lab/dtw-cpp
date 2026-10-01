/**
 * @file unit_test_nonfinite_input.cpp
 * @brief FX-15: the checked distance boundary rejects NaN and ±inf once, naming
 *        the series and the position, and leaves every finite answer bit-identical.
 *
 * @details A property test over random finite pairs with one NaN, +inf or -inf
 * injected at a random position of x or y. The checked boundary is
 * dtwc::distance::* (each variant, and the dispatcher under every missing-data
 * strategy) and soft_dtw_gradient. Each must throw
 * dtwc::InvalidInput whose message says "<series>[<position>] is <value>". The
 * missing-data distances read NaN as a missing value, so there a NaN must give
 * what the unchanged wrapper gives, while ±inf is still rejected.
 *
 * Oracles: the injected position for the diagnostic, and for every answer the
 * per-pair wrapper (warping*.hpp, msm.hpp, twe.hpp, soft_dtw.hpp) called
 * directly — the boundary check must not move a single bit of a result.
 *
 * Before FX-15 these entry points handed non-finite input straight to the
 * kernels and returned NaN, the unreachable max() or an ordinary-looking number.
 */

#include <base/error.hpp>
#include <base/missing_utils.hpp>
#include <core/msm.hpp>
#include <core/twe.hpp>
#include <distance.hpp>
#include <soft_dtw.hpp>
#include <warping.hpp>
#include <warping_adtw.hpp>
#include <warping_ddtw.hpp>
#include <warping_missing.hpp>
#include <warping_missing_arow.hpp>
#include <warping_wdtw.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

using dtwc::core::DTWVariant;
using dtwc::core::MetricType;
using dtwc::core::MissingStrategy;
using Span = std::span<const double>;

constexpr int kTrials = 240;

// One pair. `band` is drawn from [-1, max(n, m)], so the unbanded route and both
// feasible and infeasible windows are exercised: the check runs before all three.
struct Pair
{
  std::vector<double> x, y;
  int band = -1;
};

// Deterministic draws independent of the standard library's (implementation-
// defined) distributions.
class Draw
{
public:
  std::size_t below(std::size_t n) { return static_cast<std::size_t>(engine_() % n); }

  double value()
  {
    return 20.0 * (static_cast<double>(engine_() >> 11) * 0x1.0p-53) - 10.0;
  }

  Pair pair()
  {
    Pair p;
    p.x.resize(2 + below(11));
    p.y.resize(2 + below(11));
    for (auto &v : p.x) v = value();
    for (auto &v : p.y) v = value();
    p.band = static_cast<int>(below(std::max(p.x.size(), p.y.size()) + 2)) - 1;
    return p;
  }

private:
  std::mt19937_64 engine_{20260924};
};

enum class Poison { NaN, PosInf, NegInf };

double poison_value(Poison p)
{
  switch (p) {
  case Poison::NaN: return std::numeric_limits<double>::quiet_NaN();
  case Poison::PosInf: return std::numeric_limits<double>::infinity();
  case Poison::NegInf: return -std::numeric_limits<double>::infinity();
  }
  return 0.0;
}

const char *poison_name(Poison p)
{
  switch (p) {
  case Poison::NaN: return "NaN";
  case Poison::PosInf: return "+inf";
  case Poison::NegInf: return "-inf";
  }
  return "?";
}

std::string show(double v)
{
  std::ostringstream out;
  out << std::setprecision(17) << v;
  return out.str();
}

bool same_bits(double a, double b)
{
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

dtwc::core::DTWVariantParams params_for(DTWVariant variant)
{
  dtwc::core::DTWVariantParams params;
  params.variant = variant;
  return params;
}

// The per-pair wrapper each variant resolves to, at DTWVariantParams' defaults.
double wrapper_for(DTWVariant variant, Span x, Span y, int band)
{
  const dtwc::core::DTWVariantParams d;
  switch (variant) {
  case DTWVariant::Standard: return dtwc::dtwBanded<double>(x, y, band, -1.0, MetricType::L1);
  case DTWVariant::DDTW: return dtwc::ddtwBanded<double>(x, y, band, MetricType::L1);
  case DTWVariant::WDTW: return dtwc::wdtwBanded<double>(x, y, band, d.wdtw_g);
  case DTWVariant::ADTW: return dtwc::adtwBanded<double>(x, y, band, d.adtw_penalty);
  case DTWVariant::SoftDTW: return dtwc::soft_dtw<double>(x, y, d.sdtw_gamma);
  case DTWVariant::MSM: return dtwc::core::msm_distance<double>(x, y, d.msm_c);
  case DTWVariant::TWE:
    return dtwc::core::twe_distance<double>(x, y, d.twe_nu, d.twe_lambda);
  }
  return 0.0;
}

double zero_cost_wrapper(Span x, Span y, int band)
{
  return dtwc::dtwMissing_banded<double>(x, y, band, -1.0, MetricType::L1);
}

double arow_wrapper(Span x, Span y, int band)
{
  return dtwc::dtwAROW_banded<double>(x, y, band, MetricType::L1);
}

double interpolate_wrapper(Span x, Span y, int band)
{
  const auto fill = [](Span s) {
    return dtwc::has_missing(s) ? dtwc::interpolate_linear(s)
                                : std::vector<double>(s.begin(), s.end());
  };
  return dtwc::dtwBanded<double>(fill(x), fill(y), band, -1.0, MetricType::L1);
}

std::vector<float> narrow(Span s) { return {s.begin(), s.end()}; }

using Fn = std::function<double(Span, Span, int)>;

// An entry point of the checked boundary and the wrapper that computes its answer.
struct Entry
{
  std::string name;
  bool nan_is_missing = false; // the missing-data distances only
  Fn call;
  Fn oracle;
};

std::vector<Entry> checked_entry_points()
{
  namespace d = dtwc::distance;
  const DTWVariant variants[] = {DTWVariant::Standard, DTWVariant::DDTW,
                                 DTWVariant::WDTW,     DTWVariant::ADTW,
                                 DTWVariant::SoftDTW,  DTWVariant::MSM,
                                 DTWVariant::TWE};
  const char *variant_names[] = {"Standard", "DDTW", "WDTW", "ADTW",
                                 "SoftDTW",  "MSM",  "TWE"};
  const auto variant_wrapper = [](DTWVariant v) -> Fn {
    return [v](Span x, Span y, int b) { return wrapper_for(v, x, y, b); };
  };
  const dtwc::core::DTWVariantParams defaults;

  std::vector<Entry> e;
  e.push_back({"distance::dtw L1", false,
               [](Span x, Span y, int b) { return d::dtw<double>(x, y, b, MetricType::L1); },
               variant_wrapper(DTWVariant::Standard)});
  e.push_back({"distance::dtw SquaredL2", false,
               [](Span x, Span y, int b) { return d::dtw<double>(x, y, b, MetricType::SquaredL2); },
               [](Span x, Span y, int b) {
                 return dtwc::dtwBanded<double>(x, y, b, -1.0, MetricType::SquaredL2);
               }});
  e.push_back({"distance::dtw<float>", false,
               [](Span x, Span y, int b) {
                 const auto xf = narrow(x), yf = narrow(y);
                 const bool self = x.data() == y.data() && x.size() == y.size();
                 return static_cast<double>(d::dtw<float>(
                   std::span<const float>{xf}, std::span<const float>{self ? xf : yf}, b,
                   MetricType::L1));
               },
               [](Span x, Span y, int b) {
                 const auto xf = narrow(x), yf = narrow(y);
                 return static_cast<double>(dtwc::dtwBanded<float>(
                   std::span<const float>{xf}, std::span<const float>{yf}, b, -1.0f,
                   MetricType::L1));
               }});
  e.push_back({"distance::ddtw", false,
               [](Span x, Span y, int b) { return d::ddtw<double>(x, y, b); },
               variant_wrapper(DTWVariant::DDTW)});
  e.push_back({"distance::wdtw", false,
               [g = defaults.wdtw_g](Span x, Span y, int b) { return d::wdtw<double>(x, y, b, g); },
               variant_wrapper(DTWVariant::WDTW)});
  e.push_back({"distance::adtw", false,
               [p = defaults.adtw_penalty](Span x, Span y, int b) {
                 return d::adtw<double>(x, y, b, p);
               },
               variant_wrapper(DTWVariant::ADTW)});
  e.push_back({"distance::soft_dtw", false,
               [g = defaults.sdtw_gamma](Span x, Span y, int) { return d::soft_dtw<double>(x, y, g); },
               variant_wrapper(DTWVariant::SoftDTW)});
  e.push_back({"distance::msm", false,
               [c = defaults.msm_c](Span x, Span y, int) { return d::msm<double>(x, y, c); },
               variant_wrapper(DTWVariant::MSM)});
  e.push_back({"distance::twe", false,
               [nu = defaults.twe_nu, lambda = defaults.twe_lambda](Span x, Span y, int) {
                 return d::twe<double>(x, y, nu, lambda);
               },
               variant_wrapper(DTWVariant::TWE)});
  e.push_back({"distance::missing", true,
               [](Span x, Span y, int b) { return d::missing<double>(x, y, b); },
               zero_cost_wrapper});
  e.push_back({"distance::arow", true,
               [](Span x, Span y, int b) { return d::arow<double>(x, y, b); },
               arow_wrapper});

  for (std::size_t i = 0; i < std::size(variants); ++i) {
    const DTWVariant v = variants[i];
    e.push_back({std::string("distance::dtw(params) ") + variant_names[i] + " / Error", false,
                 [v](Span x, Span y, int b) {
                   return d::dtw<double>(x, y, params_for(v), b, MetricType::L1,
                                         MissingStrategy::Error);
                 },
                 variant_wrapper(v)});
  }

  const struct
  {
    MissingStrategy strategy;
    const char *name;
    Fn oracle;
  } strategies[] = {{MissingStrategy::ZeroCost, "ZeroCost", zero_cost_wrapper},
                    {MissingStrategy::AROW, "AROW", arow_wrapper},
                    {MissingStrategy::Interpolate, "Interpolate", interpolate_wrapper}};
  for (const auto &s : strategies) {
    const MissingStrategy strategy = s.strategy;
    e.push_back({std::string("distance::dtw(params) Standard / ") + s.name, true,
                 [strategy](Span x, Span y, int b) {
                   return d::dtw<double>(x, y, params_for(DTWVariant::Standard), b,
                                         MetricType::L1, strategy);
                 },
                 s.oracle});
  }
  return e;
}

// Empty when `call` threw InvalidInput saying "<at> is <value>"; else what happened.
template <typename Call>
std::string rejection_problem(Call &&call, const std::string &at, const char *value)
{
  const std::string expected = at + " is " + value;
  try {
    const double d = call();
    return "returned " + show(d) + " for " + expected;
  } catch (const dtwc::InvalidInput &error) {
    const std::string what = error.what();
    if (what.find(expected) == std::string::npos)
      return "message does not say '" + expected + "': " + what;
    return {};
  } catch (const std::exception &error) {
    return std::string("not dtwc::InvalidInput: ") + error.what();
  }
}

// Tallies the property's counterexamples per entry point, keeping the first.
struct Tally
{
  explicit Tally(std::size_t n) : failures(n, 0), first(n) {}

  void record(std::size_t k, std::string problem)
  {
    ++checks;
    if (problem.empty()) return;
    if (failures[k]++ == 0) first[k] = std::move(problem);
  }

  void report(const std::vector<Entry> &table) const
  {
    REQUIRE(checks > 0);
    for (std::size_t k = 0; k < table.size(); ++k) {
      INFO(table[k].name << ": " << failures[k] << " counterexamples; first: " << first[k]);
      CHECK(failures[k] == 0);
    }
  }

  std::vector<std::size_t> failures;
  std::vector<std::string> first;
  std::size_t checks = 0;
};

} // namespace

TEST_CASE("FX-15: every checked entry point rejects NaN and +-inf naming the series and position",
          "[fx15][nonfinite][property]")
{
  const auto table = checked_entry_points();
  Tally tally(table.size());
  Draw draw;

  for (int trial = 0; trial < kTrials; ++trial) {
    Pair p = draw.pair();
    const auto poison = static_cast<Poison>(trial % 3);
    const bool in_x = (trial / 3) % 2 == 0;
    auto &target = in_x ? p.x : p.y;
    const std::size_t index = draw.below(target.size());
    target[index] = poison_value(poison);
    const std::string at = std::string(in_x ? "x[" : "y[") + std::to_string(index) + "]";

    for (std::size_t k = 0; k < table.size(); ++k) {
      if (table[k].nan_is_missing && poison == Poison::NaN) continue; // next test case
      tally.record(k, rejection_problem(
                        [&] { return table[k].call(p.x, p.y, p.band); }, at,
                        poison_name(poison)));
    }
  }
  tally.report(table);
}

TEST_CASE("FX-15: a pair of one series with itself is checked too",
          "[fx15][nonfinite][identity]")
{
  // The wrappers return 0 for (x, x) without reading x, so a NaN self-pair came
  // back as a finite zero. The check must still run, and name x first.
  const auto table = checked_entry_points();
  Tally tally(table.size());
  Draw draw;

  for (int trial = 0; trial < kTrials; ++trial) {
    Pair p = draw.pair();
    const auto poison = static_cast<Poison>(trial % 3);
    const std::size_t index = draw.below(p.x.size());
    p.x[index] = poison_value(poison);
    const std::string at = "x[" + std::to_string(index) + "]";

    for (std::size_t k = 0; k < table.size(); ++k) {
      if (table[k].nan_is_missing && poison == Poison::NaN) continue;
      tally.record(k, rejection_problem(
                        [&] { return table[k].call(p.x, p.x, p.band); }, at,
                        poison_name(poison)));
    }
  }
  tally.report(table);
}

TEST_CASE("FX-15: the missing-data distances read NaN as missing and return the unchanged value",
          "[fx15][nonfinite][missing]")
{
  const auto table = checked_entry_points();
  Tally tally(table.size());
  Draw draw;

  for (int trial = 0; trial < kTrials; ++trial) {
    Pair p = draw.pair();
    auto &target = trial % 2 == 0 ? p.x : p.y;
    target[draw.below(target.size())] = std::numeric_limits<double>::quiet_NaN();

    for (std::size_t k = 0; k < table.size(); ++k) {
      if (!table[k].nan_is_missing) continue;
      std::string problem;
      try {
        const double got = table[k].call(p.x, p.y, p.band);
        const double want = table[k].oracle(p.x, p.y, p.band);
        if (!same_bits(got, want))
          problem = "returned " + show(got) + ", wrapper " + show(want);
      } catch (const std::exception &error) {
        problem = std::string("threw: ") + error.what();
      }
      tally.record(k, std::move(problem));
    }
  }
  tally.report(table);
}

TEST_CASE("FX-15: finite input returns the unchanged wrapper's value bit for bit",
          "[fx15][nonfinite][oracle]")
{
  const auto table = checked_entry_points();
  Tally tally(table.size());
  Draw draw;

  for (int trial = 0; trial < kTrials; ++trial) {
    const Pair p = draw.pair();
    for (std::size_t k = 0; k < table.size(); ++k) {
      std::string problem;
      try {
        const double got = table[k].call(p.x, p.y, p.band);
        const double want = table[k].oracle(p.x, p.y, p.band);
        if (!same_bits(got, want))
          problem = "returned " + show(got) + ", wrapper " + show(want);
      } catch (const std::exception &error) {
        problem = std::string("threw: ") + error.what();
      }
      tally.record(k, std::move(problem));
    }
  }
  tally.report(table);
}

TEST_CASE("FX-15: soft_dtw_gradient rejects NaN and +-inf naming the series and position",
          "[fx15][nonfinite][soft_dtw][gradient]")
{
  Draw draw;
  std::size_t failures = 0;
  std::string first;

  for (int trial = 0; trial < kTrials; ++trial) {
    Pair p = draw.pair();
    const auto poison = static_cast<Poison>(trial % 3);
    const bool in_x = (trial / 3) % 2 == 0;
    auto &target = in_x ? p.x : p.y;
    const std::size_t index = draw.below(target.size());
    target[index] = poison_value(poison);
    const std::string at = std::string(in_x ? "x[" : "y[") + std::to_string(index) + "]";

    auto problem = rejection_problem(
      [&] {
        const auto gradient = dtwc::soft_dtw_gradient<double>(p.x, p.y, 1.0);
        return gradient.empty() ? 0.0 : gradient.front();
      },
      at, poison_name(poison));
    if (!problem.empty() && failures++ == 0) first = std::move(problem);
  }
  INFO("first counterexample: " << first);
  CHECK(failures == 0);
}

TEST_CASE("FX-15: an invalid parameter is reported before non-finite data",
          "[fx15][nonfinite][order]")
{
  // The boundary validates parameters first, as it did before the data check
  // existed, so a bad parameter keeps its own diagnostic whatever the data.
  namespace d = dtwc::distance;
  const std::vector<double> x{0.0, std::numeric_limits<double>::quiet_NaN()};
  const std::vector<double> y{0.0, 1.0};
  const double bad = -1.0;

  const auto message = [](auto &&call) -> std::string {
    try {
      (void)call();
    } catch (const dtwc::InvalidInput &error) {
      return error.what();
    }
    return "no dtwc::InvalidInput";
  };

  CHECK(message([&] { return d::wdtw<double>(x, y, -1, bad); })
        == "WDTW g must be finite and non-negative.");
  CHECK(message([&] { return d::adtw<double>(x, y, -1, bad); })
        == "ADTW penalty must be finite and non-negative.");
  CHECK(message([&] { return d::soft_dtw<double>(x, y, bad); })
        == "Soft-DTW gamma must be finite and positive.");
  CHECK(message([&] { return d::msm<double>(x, y, bad); })
        == "MSM c must be finite and positive.");
  CHECK(message([&] { return d::twe<double>(x, y, bad, 1.0); })
        == "TWE nu must be finite and positive.");
  CHECK(message([&] { return d::twe<double>(x, y, 0.001, bad); })
        == "TWE lambda must be finite and positive.");
  CHECK(message([&] { return dtwc::soft_dtw_gradient<double>(x, y, bad).front(); })
        == "Soft-DTW gamma must be finite and positive.");
}

TEST_CASE("FX-15: the diagnostic names the fix", "[fx15][nonfinite][message]")
{
  const std::vector<double> x{0.0, 1.0, 2.0};
  const std::vector<double> nan_y{0.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
  const std::vector<double> inf_y{0.0, 1.0, -std::numeric_limits<double>::infinity()};

  const auto message = [](auto &&call) -> std::string {
    try {
      (void)call();
    } catch (const dtwc::InvalidInput &error) {
      return error.what();
    } catch (const std::exception &error) {
      return std::string("not dtwc::InvalidInput: ") + error.what();
    }
    return "no exception";
  };

  CHECK(message([&] { return dtwc::distance::dtw<double>(x, nan_y); })
        == "distance::dtw: y[1] is NaN. NaN is a missing value only to the "
           "missing-data distances (missing, arow, or a ZeroCost, AROW or "
           "Interpolate missing strategy): use one of those, or remove it.");
  CHECK(message([&] { return dtwc::distance::missing<double>(x, inf_y); })
        == "distance::missing: y[2] is -inf. Distances are defined on finite "
           "values only: replace or remove it (NaN marks a missing value for "
           "the missing-data distances).");
}
