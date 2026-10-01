/**
 * @file test_fill_request.cpp
 * @brief FX-1: Problem::validate_fill_request() runs before every fill that
 *        computes a pair and in the dtw_function accessors, and names the axis
 *        it rejects.
 *
 * @details Every case runs in every build. The validator's GPU checks run before
 * any backend is called — compiled or not, device present or not — so a CPU-only
 * build and a Metal build without a GPU pin them too. Oracles: the length
 * difference is computed here from the input; distances come from the CPU kernels.
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using dtwc::DistanceMatrixStrategy;

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

std::string backend_name(DistanceMatrixStrategy s)
{
  return s == DistanceMatrixStrategy::CUDA ? "CUDA" : "Metal";
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
  for (const auto &s : series) narrow.emplace_back(s.begin(), s.end());
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

TEST_CASE("FX-1: a GPU strategy rejects what its kernels do not implement",
          "[fx1][gpu]")
{
  const std::vector<std::vector<double>> equal{ ramp(8, 0.0), ramp(8, 1.0),
                                                ramp(8, 2.0) };

  for (const auto strategy : { DistanceMatrixStrategy::CUDA,
                               DistanceMatrixStrategy::Metal }) {
    const std::string backend = backend_name(strategy);
    CAPTURE(backend);
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
      prob.set_distance_strategy(strategy);
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
      prob.set_distance_strategy(strategy);
      rejects(prob, "missing_strategy = ZeroCost");
    }
    {
      dtwc::Problem prob("gpu_ndim");
      prob.set_data(named(equal, 2));
      prob.set_distance_strategy(strategy);
      rejects(prob, "ndim = 2");
    }
    {
      // A Float32 store leaves the Float64 upload buffer empty.
      dtwc::Problem prob("gpu_f32");
      prob.set_data(dtwc::Data(std::vector<std::vector<float>>{
                                 { 0.f, 1.f, 2.f }, { 1.f, 2.f, 3.f } },
                               std::vector<std::string>{ "a", "b" }));
      prob.set_distance_strategy(strategy);
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
      prob.set_distance_strategy(strategy);
      rejects(prob, "a non-owning view");
    }
  }
}

TEST_CASE("FX-1: Metal rejects a GPU index and a precision it cannot honour",
          "[fx1][gpu][metal]")
{
  dtwc::Problem prob("metal_limits");
  prob.set_data(named({ ramp(8, 0.0), ramp(8, 1.0) }));
  prob.set_distance_strategy(DistanceMatrixStrategy::Metal);

  prob.set_cuda_settings(dtwc::CUDASettings{ 1, dtwc::GpuPrecision::Auto });
  CHECK_THAT(message_of<dtwc::DeviceError>([&] { prob.fill_distance_matrix(); }),
             ContainsSubstring("GPU index = 1"));

  prob.set_cuda_settings(dtwc::CUDASettings{ 0, dtwc::GpuPrecision::FP64 });
  const auto fp64 =
    message_of<dtwc::DeviceError>([&] { prob.fill_distance_matrix(); });
#ifdef DTWC_HAS_METAL
  CHECK_THAT(fp64, ContainsSubstring("precision FP64 is not implemented"));
  // The backend entry point states the same limit.
  dtwc::metal::MetalDistMatOptions opts;
  opts.precision = dtwc::metal::MetalPrecision::FP64;
  dtwc::core::DistanceMatrix out;
  CHECK(message_of<dtwc::DeviceError>([&] {
          (void)dtwc::metal::compute_distance_matrix_metal({ { 0.0, 1.0 }, { 1.0 } }, opts, out);
        }) == fp64);
#else
  CHECK_THAT(fp64, ContainsSubstring("Metal is not compiled in"));
#endif
  CHECK_FALSE(prob.is_distance_matrix_filled());
}
