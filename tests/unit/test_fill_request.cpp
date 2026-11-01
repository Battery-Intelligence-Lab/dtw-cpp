/**
 * @file test_fill_request.cpp
 * @brief Problem::validate_fill_request() runs before every fill that
 *        computes a pair and in the dtw_function accessors, and names the axis
 *        it rejects.
 *
 * @details The validator's GPU checks run before any backend is called, device
 * present or not, in every build with a GPU backend (a build without one refuses
 * Device::GPU in set_device). Oracles: the length difference is computed here
 * from the input; distances come from the CPU kernels; a refused value's text is
 * require_finite's, at the series a serial scan in index order finds first.
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>
#include <core/portable_random.hpp>

#include "../support/scratch_directory.hpp"

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

std::vector<double> ramp(std::size_t n, double offset)
{
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = offset + 0.25 * static_cast<double>(i) + (i % 3 == 0 ? 0.5 : 0.0);
  return v;
}

/// Series named "a", "b", "c", ...
dtwc::Data named(std::vector<std::vector<double>> series, std::size_t ndim = 1)
{
  std::vector<std::string> names;
  for (std::size_t i = 0; i < series.size(); ++i)
    names.emplace_back(1, static_cast<char>('a' + i));
  return dtwc::Data(std::move(series), std::move(names), ndim);
}

/// The message of the Error `f` throws; fails the test when it throws none.
template <typename Error, typename F>
std::string message_of(F &&f)
{
  try {
    f();
  } catch (const Error &e) {
    return e.what();
  }
  FAIL("the expected exception was not thrown");
  return {};
}

constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr double inf_value = std::numeric_limits<double>::infinity();
constexpr double huge = 1.5e308; ///< finite, but a distance to it overflows; a float series holds 3e38 for it
constexpr std::size_t whole = std::numeric_limits<std::size_t>::max(); ///< a Poison's position: every value

/// `value` written into series `series` at `position`, or into all of its values.
struct Poison
{
  std::size_t series, position;
  double value;
};

/// Series and their names, which a view of them must outlive.
template <typename T>
struct Store
{
  std::vector<std::vector<T>> values;
  std::vector<std::string> names;
};

/// n finite series of `length` values, named s0, s1, ..., with `poisons` written in.
template <typename T>
Store<T> store(std::size_t n, std::size_t length, const std::vector<Poison> &poisons)
{
  Store<T> s;
  for (std::size_t i = 0; i < n; ++i) {
    auto &values = s.values.emplace_back(length);
    for (std::size_t t = 0; t < length; ++t)
      values[t] = static_cast<T>(0.5 * static_cast<double>((7 * i + 3 * t) % 11));
    s.names.push_back("s" + std::to_string(i));
  }
  for (const Poison &p : poisons) {
    auto &values = s.values[p.series];
    const T value = std::is_same_v<T, float> && p.value == huge ? T(3e38) : static_cast<T>(p.value);
    if (p.position == whole)
      std::fill(values.begin(), values.end(), value);
    else
      values[p.position] = value;
  }
  return s;
}

/// `s` as the Problem's series: owned (a copy), or a view of s's storage with an owner, as a mapped file's view has
/// one (FastCLARA's check then asks for the pages of the series it checks next).
template <typename T>
void hold(dtwc::Problem &prob, const Store<T> &s, bool view, std::size_t ndim = 1)
{
  if (view) {
    std::vector<std::span<const T>> spans(s.values.begin(), s.values.end());
    std::vector<std::string_view> names(s.names.begin(), s.names.end());
    prob.set_view_data(dtwc::Data(std::move(spans), std::move(names), ndim, std::make_shared<int>(0)));
  } else
    prob.set_data(dtwc::Data(std::vector<std::vector<T>>(s.values), std::vector<std::string>(s.names), ndim));
}

/// What a serial check in index order (Problem::validate_fill_request at fd8a7450) says of `s`, after `where`: the
/// first series holding ±inf, or NaN under MissingStrategy::Error, refused at its first such value by require_finite;
/// else under Interpolate the first entirely-NaN series; empty when it takes every series.
template <typename T>
std::string serial_verdict(const Store<T> &s, std::string_view where, dtwc::core::MissingStrategy missing)
{
  const bool nan_is_missing = missing != dtwc::core::MissingStrategy::Error;
  for (std::size_t i = 0; i < s.values.size(); ++i) {
    const std::span<const T> values(s.values[i]);
    if (std::any_of(values.begin(), values.end(),
                    [&](T v) { return std::isinf(v) || (std::isnan(v) && !nan_is_missing); }))
      return message_of<dtwc::InvalidInput>([&] {
        dtwc::detail::require_finite(values, "series '" + s.names[i] + "' (index " + std::to_string(i) + ")", where,
                                     nan_is_missing);
      });
  }
  if (missing == dtwc::core::MissingStrategy::Interpolate)
    for (std::size_t i = 0; i < s.values.size(); ++i)
      if (std::all_of(s.values[i].begin(), s.values[i].end(), [](T v) { return std::isnan(v); }))
        return std::string(where) + ": series '" + s.names[i] + "' (index " + std::to_string(i)
               + ") is entirely NaN, so MissingStrategy::Interpolate has nothing to interpolate from. Use ZeroCost "
                 "or AROW, or drop the series.";
  return {};
}

/// A case of the check's verdicts: poisons under a missing-data strategy, and the series the verdict names (`none`:
/// every series is taken), which the oracle must agree with.
struct Verdict
{
  const char *what;
  dtwc::core::MissingStrategy missing;
  std::vector<Poison> poisons;
  std::size_t names;
};
constexpr std::size_t none = std::numeric_limits<std::size_t>::max();

/// `route` against the oracle: the same InvalidInput text, or no exception when the oracle takes every series.
void check_route(const std::string &expected, std::size_t names, const std::function<void()> &route)
{
  if (names == none) {
    REQUIRE(expected.empty());
    CHECK_NOTHROW(route());
    return;
  }
  REQUIRE_THAT(expected, ContainsSubstring("(index " + std::to_string(names) + ")"));
  CHECK(message_of<dtwc::InvalidInput>(route) == expected);
}

} // namespace

TEST_CASE("FX-1: a band narrower than the widest length difference is rejected",
          "[fx1][band]")
{
  // Lengths 4, 10 and 6: the widest gap is a (4) against b (10).
  const std::vector<std::vector<double>> series{ ramp(4, 0.0), ramp(10, 0.5),
                                                 ramp(6, 1.0) };
  const std::size_t gap = series[1].size() - series[0].size(); // oracle: 6

  dtwc::Problem prob("band");
  prob.set_data(named(series));
  prob.set_band(2);

  const auto fill_msg =
    message_of<dtwc::InvalidInput>([&] { prob.fill_distance_matrix(); });
  CHECK_THAT(fill_msg, ContainsSubstring("Problem::fill_distance_matrix: band = 2"));
  CHECK_THAT(fill_msg, ContainsSubstring("series 'a' (index 0, length 4)"));
  CHECK_THAT(fill_msg, ContainsSubstring("series 'b' (index 1, length 10)"));
  CHECK_THAT(fill_msg, ContainsSubstring(
    "smallest feasible band is " + std::to_string(gap)));
  CHECK_THAT(fill_msg, ContainsSubstring("band = -1"));
  CHECK_FALSE(prob.is_distance_matrix_filled());

  // A matrix holding some pairs, not all, does not skip it.
  dtwc::Problem primed("band_primed");
  primed.set_data(named(series));
  primed.set_band(2);
  auto &cache = primed.writable_distance_matrix();
  cache.resize(3);
  cache.set(0, 1, 1.0);
  CHECK_THAT(message_of<dtwc::InvalidInput>([&] { primed.fill_distance_matrix(); }),
             ContainsSubstring("Problem::fill_distance_matrix: band = 2"));

  // The band the message names is feasible: every entry is the CPU kernel's.
  prob.set_band(static_cast<int>(gap));
  prob.fill_distance_matrix();
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j) {
      const double oracle = dtwc::dtwBanded<double>(
        series[static_cast<std::size_t>(i)], series[static_cast<std::size_t>(j)],
        static_cast<int>(gap));
      CHECK(oracle < 1e300);
      CHECK(prob.dist_by_ind(i, j) == oracle);
    }

  // Soft-DTW ignores the band, so a narrow band is not a request it cannot keep.
  dtwc::Problem soft("band_soft");
  soft.set_data(named(series));
  soft.set_variant(dtwc::core::DTWVariant::SoftDTW);
  soft.set_band(2);
  CHECK_NOTHROW(soft.fill_distance_matrix());
}

TEST_CASE("FX-1: a matrix holding every pair needs no feasible band; a new one re-arms",
          "[fx1][band][known]")
{
  // Lengths 4 and 10 under band 2: the pair has no warping path, but a matrix
  // that already holds it computes nothing. Oracle: the numbers put in.
  const dtwc::test_support::ScratchDirectory dir{ "fx1_known_pairs" };
  const auto band_error = [](dtwc::Problem &p) {
    return message_of<dtwc::InvalidInput>([&] { p.fill_distance_matrix(); });
  };

  dtwc::Problem prob("known_pairs");
  prob.set_data(named({ ramp(4, 0.0), ramp(10, 0.5) }));
  prob.set_band(2);

  // Through the matrix accessor: diagonal only, then the pair.
  auto &matrix = prob.writable_distance_matrix();
  matrix.resize(2);
  matrix.set(0, 0, 0.0);
  matrix.set(1, 1, 0.0);
  dtwc::save_checkpoint(prob, (dir.path / "partial").string()); // pair uncomputed
  prob.writable_distance_matrix().set(0, 1, 7.5);
  CHECK_NOTHROW(prob.fill_distance_matrix());
  CHECK(prob.is_distance_matrix_filled());
  CHECK(prob.dist_by_ind(0, 1) == 7.5);

  // A checkpoint without the pair brings the check back.
  REQUIRE(dtwc::load_checkpoint(prob, (dir.path / "partial").string()));
  CHECK_FALSE(prob.is_distance_matrix_filled());
  CHECK_THAT(band_error(prob), ContainsSubstring("Problem::fill_distance_matrix: band = 2"));

  // So does a file without it; a file with it serves the pair.
  {
    std::ofstream(dir.path / "full.csv") << "0,2.5\n2.5,0\n";
    std::ofstream(dir.path / "hole.csv") << "0,\n,0\n";
  }
  prob.read_distance_matrix(dir.path / "full.csv");
  CHECK(prob.is_distance_matrix_filled());
  CHECK_NOTHROW(prob.fill_distance_matrix());
  CHECK(prob.dist_by_ind(0, 1) == 2.5);
  prob.read_distance_matrix(dir.path / "hole.csv");
  CHECK_FALSE(prob.is_distance_matrix_filled());
  CHECK_THAT(band_error(prob), ContainsSubstring("Problem::fill_distance_matrix: band = 2"));

  // And an edit through the matrix accessor (resize() NaN-wipes every entry).
  prob.read_distance_matrix(dir.path / "full.csv");
  CHECK(prob.dist_by_ind(0, 1) == 2.5);
  prob.writable_distance_matrix().resize(2);
  CHECK_THAT(band_error(prob), ContainsSubstring("Problem::fill_distance_matrix: band = 2"));
}

TEST_CASE("FX-1: the dtw_function accessors validate the request before any pair",
          "[fx1][band][dtw_function]")
{
  // Lengths 4 and 10 under band 2: no warping path fits, and the kernels would
  // return the finite max() sentinel for the pair (oracle below).
  const std::vector<std::vector<double>> series{ ramp(4, 0.0), ramp(10, 0.5) };
  CHECK(dtwc::dtwBanded<double>(series[0], series[1], 2)
        == std::numeric_limits<double>::max());

  dtwc::Problem prob("accessor_band");
  prob.set_data(named(series));
  prob.set_band(2);
  CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function(); }),
             ContainsSubstring("Problem::dtw_function: band = 2"));

  // Float32 data through the Float32 accessors.
  std::vector<std::vector<float>> narrow;
  for (const auto &s : series) {
    auto &rounded = narrow.emplace_back();
    for (const double v : s) rounded.push_back(static_cast<float>(v));
  }
  dtwc::Problem f32("accessor_band_f32");
  f32.set_data(dtwc::Data(std::move(narrow), std::vector<std::string>{ "a", "b" }));
  f32.set_band(2);
  CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)f32.dtw_function_f32(); }),
             ContainsSubstring("Problem::dtw_function_f32: band = 2"));

  // A feasible band passes, and the function computes what the kernel does.
  prob.set_band(6);
  const auto &distance = prob.dtw_function();
  CHECK(distance(prob.series(0), prob.series(1))
        == dtwc::dtwBanded<double>(series[0], series[1], 6));
}

TEST_CASE("FX-1: a series emptied through p_vec is refused on the CPU routes",
          "[fx1][empty][dtw_function]")
{
  // set_data refuses an empty series, but p_vec edits one in place afterwards.
  // The kernels answer an empty pair with the finite max() sentinel (oracle),
  // which passes every isfinite() guard and which a clustering would sum.
  CHECK(dtwc::dtwBanded<double>(std::vector<double>{}, ramp(4, 0.0), -1)
        == std::numeric_limits<double>::max());

  dtwc::Problem prob("cpu_empty");
  prob.set_data(named({ ramp(4, 0.0), ramp(5, 0.5), ramp(6, 1.0) }));
  prob.p_vec(1).clear();
  prob.refresh_distance_matrix();

  SECTION("the fill")
  {
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { prob.fill_distance_matrix(); }),
               ContainsSubstring("Problem::fill_distance_matrix: series 1 ('b') is empty"));
    CHECK_FALSE(prob.is_distance_matrix_filled());
  }
  SECTION("the accessors")
  {
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function(); }),
               ContainsSubstring("Problem::dtw_function: series 1 ('b') is empty"));
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function_f32(); }),
               ContainsSubstring("Problem::dtw_function_f32: series 1 ('b') is empty"));
  }
  SECTION("a cost without a matrix")
  {
    prob.set_n_clusters(2);
    prob.centroids_ind = { 0, 2 };
    prob.clusters_ind = { 0, 0, 1 };
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)prob.find_total_cost(); }),
               ContainsSubstring("Problem::find_total_cost: series 1 ('b') is empty"));
  }
}

TEST_CASE("FX-15: ±inf, and NaN under MissingStrategy::Error, are rejected by position",
          "[fx15][missing][nonfinite]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  // The kernels' answers the validator replaces (independent oracle: the raw
  // banded kernel on the same values): inf and NaN, neither a distance.
  CHECK(std::isinf(dtwc::dtwBanded<double>(std::vector<double>{ 0.0, inf },
                                           std::vector<double>{ 0.0, 1.0 }, -1)));
  CHECK(std::isnan(dtwc::dtwBanded<double>(std::vector<double>{ inf },
                                           std::vector<double>{ inf }, -1)));

  SECTION("+inf on the fill")
  {
    dtwc::Problem prob("fx15_fill");
    prob.set_data(named({ { 0.0, 1.0, 2.0 }, { 0.0, 1.0, inf }, { 1.0, 1.0, 1.0 } }));
    const auto msg =
      message_of<dtwc::InvalidInput>([&] { prob.fill_distance_matrix(); });
    CHECK_THAT(msg, ContainsSubstring(
      "Problem::fill_distance_matrix: series 'b' (index 1)[2] is +inf"));
    CHECK_FALSE(prob.is_distance_matrix_filled());
  }
  SECTION("-inf through the accessor")
  {
    dtwc::Problem prob("fx15_accessor_inf");
    prob.set_data(named({ { 0.0, 1.0 }, { 1.0, 2.0 }, { -inf, 0.0 } }));
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function(); }),
               ContainsSubstring("Problem::dtw_function: series 'c' (index 2)[0] is -inf"));
  }
  SECTION("NaN through the accessors, in both precisions")
  {
    dtwc::Problem prob("fx15_accessor");
    prob.set_data(named({ { 0.0, 1.0, 2.0 }, { nan, 1.0, 2.0 } }));
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function(); }),
               ContainsSubstring("Problem::dtw_function: series 'b' (index 1)[0] is NaN"));

    dtwc::Problem f32("fx15_accessor_f32");
    f32.set_data(dtwc::Data(
      std::vector<std::vector<float>>{ { 0.f, 1.f }, { 1.f, std::numeric_limits<float>::quiet_NaN() } },
      std::vector<std::string>{ "a", "b" }));
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { (void)f32.dtw_function_f32(); }),
               ContainsSubstring("Problem::dtw_function_f32: series 'b' (index 1)[1] is NaN"));
  }
  SECTION("a multivariate position is the flat index")
  {
    dtwc::Problem prob("fx15_mv");
    prob.set_data(named({ { 0.0, 1.0, 2.0, 3.0 }, { 0.0, 1.0, 2.0, inf } }, 2));
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { prob.fill_distance_matrix(); }),
               ContainsSubstring("series 'b' (index 1)[3] is +inf"));
  }
  SECTION("a missing-data strategy takes NaN as missing, never ±inf")
  {
    dtwc::Problem prob("fx15_zero_cost");
    prob.set_data(named({ { 0.0, 1.0, 2.0 }, { 0.0, nan, 2.0 } }));
    prob.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
    CHECK_NOTHROW(prob.fill_distance_matrix());
    CHECK(prob.dist_by_ind(0, 1) == 0.0);

    dtwc::Problem inf_prob("fx15_zero_cost_inf");
    inf_prob.set_data(named({ { 0.0, 1.0, 2.0 }, { 0.0, nan, inf } }));
    inf_prob.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { inf_prob.fill_distance_matrix(); }),
               ContainsSubstring("series 'b' (index 1)[2] is +inf"));
  }
}

// MM G3: the check sweeps the series in blocks of an eighth of the memory budget, each block on every thread, and gives
// the serial check's verdict: the lowest refused series at its first refused value, ±inf (or NaN under Error) before
// any entirely-NaN series under Interpolate. 1,000 series, owned and as a view, in both precisions, the budget unknown,
// holding one block and about ten; the poisons sit on both sides of the ten blocks' edges (100, 200, ...).
TEST_CASE("MM G3: the blocked check's verdicts are the serial check's", "[mm][g3][nonfinite][missing]")
{
  using dtwc::core::MissingStrategy;
  constexpr std::size_t n = 1000, length = 16;
  const std::vector<Verdict> cases{
    { "NaN in a block's last series", MissingStrategy::Error, { { 99, 15, nan_value } }, 99 },
    { "+inf in a block's first series", MissingStrategy::Error, { { 100, 0, inf_value } }, 100 },
    { "-inf in its second", MissingStrategy::Error, { { 101, 7, -inf_value } }, 101 },
    { "two series of one block: the lower", MissingStrategy::Error, { { 205, 1, inf_value }, { 202, 9, nan_value } },
      202 },
    { "two values of one series: the first", MissingStrategy::Error,
      { { 310, 12, -inf_value }, { 310, 4, nan_value } }, 310 },
    { "two blocks: the lower", MissingStrategy::Error, { { 899, 3, -inf_value }, { 150, 2, inf_value } }, 150 },
    { "±inf after an entirely-NaN series", MissingStrategy::Interpolate,
      { { 10, whole, nan_value }, { 250, 5, inf_value } }, 250 },
    { "±inf above an entirely-NaN series in its block", MissingStrategy::Interpolate,
      { { 300, whole, nan_value }, { 350, 1, -inf_value } }, 350 },
    { "the lowest entirely-NaN series", MissingStrategy::Interpolate,
      { { 420, whole, nan_value }, { 130, whole, nan_value }, { 60, 3, nan_value } }, 130 },
    { "NaN under Interpolate", MissingStrategy::Interpolate, { { 60, 3, nan_value }, { 199, 15, nan_value } }, none },
    { "NaN under ZeroCost", MissingStrategy::ZeroCost, { { 60, 3, nan_value }, { 100, whole, nan_value } }, none },
    { "±inf under ZeroCost", MissingStrategy::ZeroCost,
      { { 60, 3, nan_value }, { 700, 0, -inf_value }, { 640, 2, nan_value } }, 700 },
    { "NaN under AROW", MissingStrategy::AROW, { { 60, 3, nan_value }, { 100, whole, nan_value } }, none },
    { "±inf under AROW", MissingStrategy::AROW, { { 99, 0, nan_value }, { 101, 15, inf_value } }, 101 },
  };
  const auto run = [&](auto zero) {
    using T = decltype(zero);
    const std::size_t one_block = 8 * n * length * sizeof(T); // an eighth of it holds every series
    for (const Verdict &c : cases) {
      const auto s = store<T>(n, length, c.poisons);
      for (const bool view : { false, true })
        for (const std::size_t ram_limit : { std::size_t{ 0 }, one_block, one_block / 10 })
          for (const std::string_view where :
               { "Problem::fill_distance_matrix", "Problem::dtw_function", "Problem::dtw_function_f32" }) {
            if (c.names == none && where == "Problem::fill_distance_matrix") continue; // no pair is computed
            INFO(c.what << " (" << (view ? "a view" : "owned") << ", float" << 8 * sizeof(T) << ", ram_limit "
                        << ram_limit << ", " << where << ")");
            dtwc::Problem prob("g3");
            prob.set_missing_strategy(c.missing);
            prob.set_ram_limit(ram_limit);
            hold(prob, s, view);
            check_route(serial_verdict(s, where, c.missing), c.names, [&] {
              if (where == "Problem::fill_distance_matrix")
                prob.fill_distance_matrix();
              else if (where == "Problem::dtw_function")
                (void)prob.dtw_function();
              else
                (void)prob.dtw_function_f32();
            });
          }
    }
  };
  run(double{});
  run(float{});
}

// MM G3: the empty series, the band (a direct write to `band` included) and float32's parameters come before any
// value, in the serial check's order.
TEST_CASE("MM G3: an empty series, an infeasible band and float32's parameters come before any value",
          "[mm][g3][band][empty]")
{
  constexpr std::size_t n = 1000, length = 16;
  auto s = store<double>(n, length, { { 5, 2, inf_value }, { 640, 1, nan_value } });
  const std::vector<std::string_view> routes{ "Problem::fill_distance_matrix", "Problem::dtw_function",
                                              "Problem::dtw_function_f32" };
  const auto call = [](dtwc::Problem &prob, std::string_view where) {
    if (where == "Problem::fill_distance_matrix")
      prob.fill_distance_matrix();
    else if (where == "Problem::dtw_function")
      (void)prob.dtw_function();
    else
      (void)prob.dtw_function_f32();
  };

  SECTION("an empty series: the first")
  {
    for (const auto where : routes) {
      dtwc::Problem prob("g3_empty");
      hold(prob, s, false);
      prob.p_vec(700).clear();
      prob.p_vec(300).clear();
      prob.refresh_distance_matrix();
      CHECK(message_of<dtwc::InvalidInput>([&] { call(prob, where); })
            == std::string(where) + ": series 300 ('s300') is empty; every series needs at least one value.");
    }
  }
  SECTION("a band, also written directly")
  {
    s.values[400].resize(length + 5, 1.0); // the longest, 21
    s.values[600].resize(length - 3);      // the shortest, 13
    const auto band_text = [](std::string_view where, int band) {
      return std::string(where) + ": band = " + std::to_string(band)
             + " is narrower than the length difference between series 's600' (index 600, length 13) and series "
               "'s400' (index 400, length 21), so no warping path fits that pair. The smallest feasible band is 8; "
               "pass band >= 8, or band = -1 for full DTW.";
    };
    for (const bool view : { false, true })
      for (const auto where : routes) {
        INFO((view ? "a view, " : "owned, ") << where);
        dtwc::Problem prob("g3_band");
        hold(prob, s, view);
        prob.set_band(2);
        CHECK(message_of<dtwc::InvalidInput>([&] { call(prob, where); }) == band_text(where, 2));
        prob.set_band(-1);
        prob.band = 7; // the v1 field, taken at the next check
        CHECK(message_of<dtwc::InvalidInput>([&] { call(prob, where); }) == band_text(where, 7));
      }
  }
  SECTION("a parameter float32 cannot hold, before a value")
  {
    // A Float32 Problem refuses the parameter when it is set (test_distance_config.cpp), so FastCLARA's float32
    // route never meets it: the float32 accessor on Float64 series is where its check comes first.
    dtwc::Problem prob("g3_f32_parameter");
    hold(prob, s, false);
    auto params = prob.variant_params();
    params.variant = dtwc::core::DTWVariant::WDTW;
    params.wdtw_g = 1e-50; // float32 rounds it to 0
    prob.set_variant(params);
    CHECK(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function_f32(); })
          == "WDTW g cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(message_of<dtwc::InvalidInput>([&] { (void)prob.dtw_function(); })
          == serial_verdict(s, "Problem::dtw_function", dtwc::core::MissingStrategy::Error));
  }
}

// MM G3b: FastCLARA on the CPU checks the samples' rows before the PAMs and each block of 65,536 points just before it
// is assigned, and gives the verdict dtw_function() gave before the PAMs at fd8a7450 (the oracle): the lowest refused
// series; ±inf before an entirely-NaN series under Interpolate, which stops the pass at its block (its distances would
// throw "interpolate_linear_into"). An error the PAMs or a block's assignment raise on values the check takes (a
// series of `huge`, whose distances overflow: fast_pam's "initialization distance must be finite", the assignment's
// "non-finite nearest-medoid distance") comes after the refusal of any series not yet checked, as the oracle's check
// came before the PAMs. 70,000 series (two blocks), sample 150, 5 samples, owned and as a view, both precisions. On a
// GPU device nothing is fused: dtw_function() checks every series first, Float32 ones too (the GPU's needs come last),
// so the same verdicts, under Problem::dtw_function.
TEST_CASE("MM G3b: FastCLARA's fused check gives dtw_function()'s verdicts", "[mm][g3b][fast_clara][nonfinite]")
{
  using dtwc::core::MissingStrategy;
  constexpr std::size_t n = 70000, length = 4, block = 65536;
  dtwc::algorithms::CLARAOptions options;
  options.n_clusters = 2;
  options.sample_size = 150;
  options.n_samples = 5;
  options.random_seed = 11;
  // The rows fast_clara samples: n_samples draws in turn from one engine (draw_clara_samples).
  std::vector<bool> sampled(n);
  std::mt19937_64 engine(options.random_seed);
  for (int k = 0; k < options.n_samples; ++k)
    for (const auto i : dtwc::core::portable_sample_indices<dtwc::index_t>(n, options.sample_size, engine))
      sampled[static_cast<std::size_t>(i)] = true;
  const auto at = [&](std::size_t from, bool in_sample) {
    while (sampled[from] != in_sample) ++from;
    return from;
  };
  const std::size_t u = at(1000, false), r = at(20000, true), x = at(r + 1, false), r2 = at(r + 5000, true),
                    o = at(30000, false), v = at(40000, false), w = at(50000, false), t = at(66000, false),
                    t2 = at(68000, false);
  REQUIRE((u < r && r < x && x < r2 && r2 < o && o < v && v < w && w < block && block <= t && t < t2 && t2 < n));

  const std::vector<Verdict> cases{
    { "a sampled row above an unsampled lower one", MissingStrategy::Error,
      { { r, 1, inf_value }, { u, 2, nan_value } }, u },
    { "two sampled rows and an unsampled one between", MissingStrategy::Error,
      { { r2, 0, -inf_value }, { x, 1, inf_value }, { r, 3, nan_value } }, r },
    { "a sampled row only", MissingStrategy::Error, { { r2, 2, nan_value } }, r2 },
    { "the late block only", MissingStrategy::Error, { { t, 1, inf_value } }, t },
    { "two blocks: the lower", MissingStrategy::Error, { { t, 3, -inf_value }, { w, 0, inf_value } }, w },
    { "an entirely-NaN sampled row, ±inf later in the file", MissingStrategy::Interpolate,
      { { r, whole, nan_value }, { t, 2, -inf_value } }, t },
    { "an entirely-NaN sampled row only", MissingStrategy::Interpolate,
      { { r2, whole, nan_value }, { r, 1, nan_value } }, r2 },
    { "an entirely-NaN unsampled row only", MissingStrategy::Interpolate, { { v, whole, nan_value } }, v },
    { "an entirely-NaN unsampled row, ±inf in a later block", MissingStrategy::Interpolate,
      { { v, whole, nan_value }, { t, 0, inf_value } }, t },
    { "an entirely-NaN unsampled row, ±inf above it in its block", MissingStrategy::Interpolate,
      { { v, whole, nan_value }, { w, 3, -inf_value } }, w },
    { "entirely-NaN rows in two blocks: the lower", MissingStrategy::Interpolate,
      { { t, whole, nan_value }, { w, whole, nan_value } }, w },
    { "an entirely-NaN row in the late block only", MissingStrategy::Interpolate, { { t2, whole, nan_value } }, t2 },
    { "an overflowing row in the early block, ±inf in the late block", MissingStrategy::Error,
      { { o, whole, huge }, { t, 1, inf_value } }, t },
    { "an overflowing row in the early block, an entirely-NaN row in the late block", MissingStrategy::Interpolate,
      { { o, whole, huge }, { t, whole, nan_value } }, t },
    { "an overflowing sampled row, ±inf in an unsampled one", MissingStrategy::Error,
      { { r, whole, huge }, { v, 1, inf_value } }, v },
  };
  const auto run = [&](auto zero, bool gpu) {
    using T = decltype(zero);
    const std::string where = gpu || sizeof(T) == 8 ? "Problem::dtw_function" : "Problem::dtw_function_f32";
    for (const Verdict &c : cases) {
      const auto s = store<T>(n, length, c.poisons);
      for (const bool view : { false, true }) {
        INFO(c.what << " (" << (view ? "a view" : "owned") << ", float" << 8 * sizeof(T) << (gpu ? ", gpu" : "")
                    << ")");
        dtwc::Problem prob("g3b");
        prob.set_missing_strategy(c.missing);
        if (gpu) prob.set_device(dtwc::Device::GPU);
        hold(prob, s, view);
        check_route(serial_verdict(s, where, c.missing), c.names,
                    [&] { (void)dtwc::algorithms::fast_clara(prob, options); });
        CHECK(prob.labels().empty()); // nothing published
      }
    }
  };
  run(double{}, false);
  run(float{}, false);
#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  run(double{}, true);
  run(float{}, true);
  // A bad value in a multivariate input is the value's InvalidInput, not the GPU's univariate DeviceError.
  const auto s = store<double>(n, length, { { w, 3, inf_value } });
  dtwc::Problem prob("g3b_gpu_ndim");
  prob.set_device(dtwc::Device::GPU);
  hold(prob, s, false, 2);
  check_route(serial_verdict(s, "Problem::dtw_function", MissingStrategy::Error), w,
              [&] { (void)dtwc::algorithms::fast_clara(prob, options); });
#endif
}

TEST_CASE("FX-1: the band feasibility check counts multivariate timesteps",
          "[fx1][band][multivariate]")
{
  // ndim = 2: flat sizes 8 and 12 are 4 and 6 timesteps, a gap of 2.
  dtwc::Problem prob("band_mv");
  prob.set_data(named({ ramp(8, 0.0), ramp(12, 1.0) }, 2));
  prob.set_band(1);
  const auto msg =
    message_of<dtwc::InvalidInput>([&] { prob.fill_distance_matrix(); });
  CHECK_THAT(msg, ContainsSubstring("(index 0, length 4)"));
  CHECK_THAT(msg, ContainsSubstring("(index 1, length 6)"));
  CHECK_THAT(msg, ContainsSubstring("smallest feasible band is 2"));

  prob.set_band(2);
  CHECK_NOTHROW(prob.fill_distance_matrix());
  CHECK(prob.dist_by_ind(0, 1) < 1e300);
}

TEST_CASE("FX-1: Tier-1 cluster() rejects an infeasible band instead of summing 1.8e308",
          "[fx1][band][tier1]")
{
  const auto dataset = dtwc::load(dtwc::Dataset::series_type{
    ramp(4, 0.0), ramp(5, 0.0), ramp(12, 3.0), ramp(13, 3.0) });
  const auto msg = message_of<dtwc::InvalidInput>(
    [&] { (void)dtwc::cluster(dataset, 2, "pam", 3, "cpu"); });
  CHECK_THAT(msg, ContainsSubstring("smallest feasible band is 9"));

  const auto result = dtwc::cluster(dataset, 2, "pam", 9, "cpu");
  CHECK(result.cost() < 1e300);
}

#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
TEST_CASE("FX-1: a GPU device rejects what its kernels do not implement",
          "[fx1][gpu]")
{
  const std::vector<std::vector<double>> equal{ ramp(8, 0.0), ramp(8, 1.0),
                                                ramp(8, 2.0) };
#  if defined(DTWC_HAS_CUDA)
  const std::string backend = "CUDA";
#  else
  const std::string backend = "Metal";
#  endif
  const auto rejects = [&](dtwc::Problem &prob, std::string_view axis) {
    const auto msg = message_of<dtwc::DeviceError>(
      [&] { prob.fill_distance_matrix(); });
    CHECK_THAT(msg, ContainsSubstring("Problem::fill_distance_matrix: " + backend));
    CHECK_THAT(msg, ContainsSubstring(std::string(axis)));
    CHECK_THAT(msg, ContainsSubstring("no backend call or CPU fallback"));
    CHECK_FALSE(prob.is_distance_matrix_filled());
  };

  {
    dtwc::Problem prob("gpu_variant");
    prob.set_data(named(equal));
    prob.set_variant(dtwc::core::DTWVariant::WDTW);
    prob.set_device(dtwc::Device::GPU);
    rejects(prob, "variant = WDTW");
    // The accessor makes the same check.
    const auto accessor = message_of<dtwc::DeviceError>(
      [&] { (void)prob.dtw_function(); });
    CHECK_THAT(accessor, ContainsSubstring("Problem::dtw_function: " + backend));
  }
  {
    dtwc::Problem prob("gpu_missing");
    prob.set_data(named(equal));
    prob.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
    prob.set_device(dtwc::Device::GPU);
    rejects(prob, "missing_strategy = ZeroCost");
  }
  {
    dtwc::Problem prob("gpu_ndim");
    prob.set_data(named(equal, 2));
    prob.set_device(dtwc::Device::GPU);
    rejects(prob, "ndim = 2");
  }
  {
    // A Float32 store leaves the Float64 upload buffer empty.
    dtwc::Problem prob("gpu_f32");
    prob.set_data(dtwc::Data(std::vector<std::vector<float>>{
                               { 0.f, 1.f, 2.f }, { 1.f, 2.f, 3.f } },
                             std::vector<std::string>{ "a", "b" }));
    prob.set_device(dtwc::Device::GPU);
    rejects(prob, "precision = Float32");
  }
  {
    // A view-mode store leaves it empty too.
    const auto owner = equal;
    std::vector<std::span<const double>> spans;
    std::vector<std::string_view> names{ "a", "b", "c" };
    for (const auto &s : owner) spans.emplace_back(s);
    dtwc::Problem prob("gpu_view");
    prob.set_view_data(dtwc::Data(std::move(spans), std::move(names), 1));
    prob.set_device(dtwc::Device::GPU);
    rejects(prob, "a non-owning view");
  }
  {
    // set_data refuses an empty series, but an edit in place can empty one; a
    // kernel would read before its start (Metal did).
    dtwc::Problem prob("gpu_empty");
    prob.set_data(named(equal));
    prob.set_device(dtwc::Device::GPU);
    prob.p_vec(1).clear();
    prob.refresh_distance_matrix();
    CHECK_THAT(message_of<dtwc::InvalidInput>([&] { prob.fill_distance_matrix(); }),
               ContainsSubstring("Problem::fill_distance_matrix: series 1 ('b') is empty"));
    CHECK_FALSE(prob.is_distance_matrix_filled());
  }
}

#endif

#if defined(DTWC_HAS_METAL) && !defined(DTWC_HAS_CUDA)
TEST_CASE("FX-1: Metal rejects a GPU index and a precision it cannot honour",
          "[fx1][gpu][metal]")
{
  dtwc::Problem prob("metal_limits");
  prob.set_data(named({ ramp(8, 0.0), ramp(8, 1.0) }));
  CHECK_THAT(message_of<dtwc::DeviceError>([&] { prob.set_device(dtwc::Device::GPU, 1); }),
             ContainsSubstring("GPU index = 1"));

  prob.set_device(dtwc::Device::GPU);
  prob.set_gpu_precision(dtwc::GpuPrecision::FP64);
  const auto fp64 =
    message_of<dtwc::DeviceError>([&] { prob.fill_distance_matrix(); });
  CHECK_THAT(fp64, ContainsSubstring("precision FP64 is not implemented"));
  // The backend entry point states the same limit.
  dtwc::metal::MetalDistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;
  dtwc::core::DistanceMatrix out;
  CHECK(message_of<dtwc::DeviceError>([&] {
          (void)dtwc::metal::compute_distance_matrix_metal({ { 0.0, 1.0 }, { 1.0 } }, opts, out);
        }) == fp64);
  CHECK_FALSE(prob.is_distance_matrix_filled());
}
#endif
