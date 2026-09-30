/**
 * @file test_cuda_correctness.cpp
 * @brief CUDA DTW correctness tests: compare GPU distance matrix against CPU reference.
 *
 * @author Volkan Kumtepeli
 * @date 01 Apr 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <dtwc.hpp>
#include <base/parallelisation.hpp>

#include "gpu_fixed_band_oracle.hpp"
#include "../support/deterministic_series.hpp"

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#include <cuda/launch_prep.hpp>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>  // std::iota (MSVC STL does not include it transitively)
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // GlobalMemoryStatusEx: A15's host-memory check
#else
#include <fstream>
#endif

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_CUDA

TEST_CASE("CUDA not available", "[cuda]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA tests skipped");
}

#else // DTWC_HAS_CUDA

namespace {

constexpr auto generate_random_series =
  &dtwc::test_support::accelerator_series_set;

/// Physical memory the host can give a new allocation now, in bytes: Windows'
/// available physical memory, Linux's MemAvailable; 0 when unknown.
std::uint64_t available_host_memory()
{
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  return GlobalMemoryStatusEx(&status) ? status.ullAvailPhys : 0;
#else
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);)
    if (line.rfind("MemAvailable:", 0) == 0)
      return std::stoull(line.substr(13)) * 1024; // the value is in kB
  return 0;
#endif
}

/// The CUDA fill of `series` into a fresh packed matrix, and its distances as
/// the N x N row-major matrix the oracles below build.
struct GpuFill : dtwc::cuda::CUDADistMatResult {
  std::vector<double> matrix;
};

GpuFill gpu_fill(const std::vector<std::vector<double>> &series,
                 const dtwc::cuda::CUDADistMatOptions &opts = {})
{
  dtwc::core::DistanceMatrix packed;
  GpuFill fill{ dtwc::cuda::compute_distance_matrix_cuda(series, opts, packed), {} };
  fill.matrix = dtwc::io::to_full_matrix(packed);
  return fill;
}

/// Compute the full NxN CPU distance matrix using dtwFull_L (L1 metric).
std::vector<double> cpu_distance_matrix(
    const std::vector<std::vector<double>> &series)
{
  return dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [](const auto &left, const auto &right) {
      return dtwc::dtwFull_L<double>(left, right);
    });
}

/// The host kernel in FP32 on the float-rounded series: the same operations in
/// the same precision as the CUDA FP32 fill, so the two agree bit for bit.
std::vector<double> cpu_fp32_distance_matrix(
    const std::vector<std::vector<double>> &series)
{
  std::vector<std::vector<float>> rounded;
  for (const auto &s : series) {
    std::vector<float> r(s.size());
    std::transform(s.begin(), s.end(), r.begin(),
                   [](double v) { return static_cast<float>(v); });
    rounded.push_back(std::move(r));
  }
  return dtwc::test_support::symmetric_zero_diagonal_matrix(
    rounded,
    [](const auto &left, const auto &right) {
      return static_cast<double>(dtwc::dtwFull_L<float>(left, right));
    });
}

/// Compute the NxN CPU banded distance matrix using dtwBanded (L1 metric).
std::vector<double> cpu_banded_distance_matrix(
    const std::vector<std::vector<double>> &series, int band)
{
  return dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [band](const auto &left, const auto &right) {
      return dtwc::dtwBanded<double>(left, right, band);
    });
}

/// The principal pair alone takes the warp kernel; a filler series raises the
/// batch's longest length into another kernel's range.
struct F12CUDARoute {
  const char *expected_kernel;
  size_t filler_length; // 0: no filler
};

constexpr std::array<F12CUDARoute, 4> f12_cuda_routes{{
    {"warp", 0},
    {"regtile_w4", 64},
    {"regtile_w8", 129},
    {"wavefront", 257}
}};

std::vector<std::vector<double>> f12_pairwise_inventory(
    const F12CUDARoute &route)
{
  namespace oracle = dtwc::test::gpu_fixed_band;
  std::vector<std::vector<double>> series{
      oracle::principal_x(), oracle::principal_y()
  };
  if (route.filler_length > 0)
    series.push_back(oracle::filler(route.filler_length));
  return series;
}

double f12_expected_public_cost(
    const dtwc::test::gpu_fixed_band::LedgerRow &row,
    bool squared)
{
  namespace oracle = dtwc::test::gpu_fixed_band;
  if (!row.has_path) return oracle::public_no_path_sentinel;
  return squared ? row.squared_l2 : row.l1;
}

dtwc::cuda::CUDADistMatOptions f12_cuda_options(
    const dtwc::test::gpu_fixed_band::LedgerRow &row,
    bool squared,
    dtwc::cuda::CUDAPrecision precision =
        dtwc::cuda::CUDAPrecision::FP64)
{
  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = row.band;
  opts.use_squared_l2 = squared;
  opts.precision = precision;
  return opts;
}

} // anonymous namespace

TEST_CASE("F12 independent fixed-band arbiters reproduce the registered ledger",
          "[cuda][F12][oracle]")
{
  namespace oracle = dtwc::test::gpu_fixed_band;
  const auto &x = oracle::principal_x();
  const auto &y = oracle::principal_y();

  REQUIRE(x.size() == 5);
  REQUIRE(y.size() == 7);
  REQUIRE(y.size() - x.size() == 2);

  for (const auto &row : oracle::ledger) {
    CAPTURE(row.band, row.path_count, row.has_path, row.l1, row.squared_l2);
    const auto dp_l1 =
        oracle::full_matrix_oracle(x, y, row.band, false);
    const auto dp_squared =
        oracle::full_matrix_oracle(x, y, row.band, true);
    const auto enumerated = oracle::enumerate_paths(x, y, row.band);

    REQUIRE(enumerated.path_count == row.path_count);
    if (row.has_path) {
      REQUIRE(dp_l1 == row.l1);
      REQUIRE(dp_squared == row.squared_l2);
      REQUIRE(enumerated.min_l1 == row.l1);
      REQUIRE(enumerated.min_squared_l2 == row.squared_l2);
    } else {
      REQUIRE(std::isinf(dp_l1));
      REQUIRE(std::isinf(dp_squared));
      REQUIRE(std::isinf(enumerated.min_l1));
      REQUIRE(std::isinf(enumerated.min_squared_l2));
    }
  }

  const auto &singleton = oracle::singleton_x();
  const auto &singleton_longer = oracle::singleton_y();
  REQUIRE(singleton.size() == 1);
  REQUIRE(singleton_longer.size() == 3);
  for (const auto &row : oracle::singleton_ledger) {
    for (const bool reversed : {false, true}) {
      CAPTURE(
          row.band, row.path_count, row.has_path, row.l1, row.squared_l2,
          reversed);
      const auto &lhs = reversed ? singleton_longer : singleton;
      const auto &rhs = reversed ? singleton : singleton_longer;
      const auto dp_l1 =
          oracle::full_matrix_oracle(lhs, rhs, row.band, false);
      const auto dp_squared =
          oracle::full_matrix_oracle(lhs, rhs, row.band, true);
      const auto enumerated = oracle::enumerate_paths(lhs, rhs, row.band);

      REQUIRE(enumerated.path_count == row.path_count);
      if (row.has_path) {
        REQUIRE(dp_l1 == row.l1);
        REQUIRE(dp_squared == row.squared_l2);
        REQUIRE(enumerated.min_l1 == row.l1);
        REQUIRE(enumerated.min_squared_l2 == row.squared_l2);
      } else {
        REQUIRE(std::isinf(dp_l1));
        REQUIRE(std::isinf(dp_squared));
        REQUIRE(std::isinf(enumerated.min_l1));
        REQUIRE(std::isinf(enumerated.min_squared_l2));
      }
    }
  }
}

TEST_CASE("F12 CUDA pairwise kernels use canonical fixed-band geometry",
          "[cuda][F12][banded][pairwise]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  namespace oracle = dtwc::test::gpu_fixed_band;
  for (const auto &route : f12_cuda_routes) {
    const auto series = f12_pairwise_inventory(route);
    const auto n = series.size();

    for (const bool squared : {false, true}) {
      for (const auto &row : oracle::ledger) {
        CAPTURE(route.expected_kernel, squared, row.band);
        const auto opts = f12_cuda_options(row, squared);
        const auto result =
            gpu_fill(series, opts);
        const auto expected = f12_expected_public_cost(row, squared);

        REQUIRE(result.n == n);
        REQUIRE(result.matrix.size() == n * n);
        REQUIRE(result.kernel_used == route.expected_kernel);
        REQUIRE(result.matrix[1] == expected);
        REQUIRE(result.matrix[n] == expected);
      }
    }
  }
}

TEST_CASE("F12 CUDA singleton route cannot bypass endpoint feasibility",
          "[cuda][F12][banded][singleton]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  namespace oracle = dtwc::test::gpu_fixed_band;
  const std::vector<std::vector<double>> pairwise_series{
      oracle::singleton_x(), oracle::singleton_y()
  };

  for (const bool squared : {false, true}) {
    for (const auto &row : oracle::singleton_ledger) {
      CAPTURE(squared, row.band);
      const auto opts = f12_cuda_options(row, squared);
      const auto expected = f12_expected_public_cost(row, squared);

      const auto pairwise =
          gpu_fill(pairwise_series, opts);
      REQUIRE(pairwise.n == 2);
      REQUIRE(pairwise.matrix.size() == 4);
      REQUIRE(pairwise.kernel_used == "warp");
      REQUIRE(pairwise.matrix[1] == expected);
      REQUIRE(pairwise.matrix[2] == expected);
    }
  }
}

TEST_CASE("F12 CUDA FP32 results translate no-path to the public double sentinel",
          "[cuda][F12][banded][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  namespace oracle = dtwc::test::gpu_fixed_band;
  const auto &below_gap = oracle::ledger.front();
  const auto &route = f12_cuda_routes.front();
  auto opts = f12_cuda_options(
      below_gap, false, dtwc::cuda::CUDAPrecision::FP32);

  const auto pairwise = gpu_fill(
      f12_pairwise_inventory(route), opts);
  REQUIRE(pairwise.kernel_used == "warp");
  REQUIRE(pairwise.matrix[1] == oracle::public_no_path_sentinel);
  REQUIRE(pairwise.matrix[2] == oracle::public_no_path_sentinel);
}

// Every range of the automatic kernel choice, at its edges, in both precisions:
// warp (L <= 32), regtile<4> (<= 128), regtile<8> (<= 256), and the wavefront's
// preload (<= 512), three-buffer (<= 1024), double-buffer (<= 2048) and long
// three-buffer modes. The oracle is the host kernel in the fill's precision.
TEST_CASE("CUDA fill matches the host kernel in every automatic kernel range",
          "[cuda][regime]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  struct Regime {
    size_t L;
    const char *kernel;
  };
  const auto regime = GENERATE(values<Regime>({
      {16, "warp"}, {32, "warp"}, {128, "regtile_w4"}, {256, "regtile_w8"},
      {257, "wavefront"}, {512, "wavefront"}, {1024, "wavefront"},
      {1025, "wavefront"}, {2048, "wavefront"}, {2049, "wavefront"}}));
  const bool fp32 = GENERATE(true, false);
  CAPTURE(regime.L, fp32);

  const auto series = generate_random_series(6, regime.L, /*seed=*/7);
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::cuda::CUDAPrecision::FP32
                        : dtwc::cuda::CUDAPrecision::FP64;
  const auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.kernel_used == regime.kernel);
  REQUIRE(gpu_result.matrix == (fp32 ? cpu_fp32_distance_matrix(series)
                                     : cpu_distance_matrix(series)));
}

// ---------------------------------------------------------------------------
// Element-wise GPU vs CPU comparison helpers
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_matches_cpu_small", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_matches_cpu_medium", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 20;
  constexpr size_t L = 200;
  auto series = generate_random_series(N, L, /*seed=*/123);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_matches_cpu_large", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 50;
  constexpr size_t L = 500;
  auto series = generate_random_series(N, L, /*seed=*/9999);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Task 0.1 regression: long series (max_L > 2048) must not drop anti-diagonal
// cells. The retired double-buffer wavefront path cached each thread's cost-
// diagonal in a fixed MAX_SI=8 register array (256 threads * 8 = 2048), so any
// anti-diagonal longer than 2048 was silently truncated -> wrong DTW. The fix
// routes max_L > 2048 to the 3-buffer path, which grid-strides every cell.
// Non-degenerate random-walk series (NOT constant / symmetric) are used so a
// dropped cell changes the result.
// ---------------------------------------------------------------------------
namespace {

std::vector<std::vector<double>> generate_random_walks(
    size_t n, size_t length, unsigned seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<double> step(0.0, 1.0);

  std::vector<std::vector<double>> series(n);
  for (auto &s : series) {
    s.resize(length);
    double acc = 0.0;
    for (auto &v : s) { acc += step(rng); v = acc; }
  }
  return series;
}

} // anonymous namespace

TEST_CASE("test_gpu_long_series_wavefront_full", "[cuda][long]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  // L = 4096 forces max_L > 2048 (the double-buffer -> 3-buffer routing).
  // Keep N small: the CPU oracle costs O(N^2 * L^2).
  constexpr size_t N = 5;
  constexpr size_t L = 4096;
  auto series = generate_random_walks(N, L, /*seed=*/20240701);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  // REGISTERED pass band (recorded before the run): max |GPU - CPU| <= 1e-9
  // over all pairs (FP64). The unfixed kernel dropped cells and produced
  // grossly wrong distances, blowing past this band.
  double max_abs_diff = 0.0;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      max_abs_diff = std::max(
          max_abs_diff, std::abs(gpu_result.matrix[i * N + j] - cpu_mat[i * N + j]));

  INFO("max |GPU - CPU| = " << max_abs_diff);
  CHECK(max_abs_diff <= 1e-9);
}

TEST_CASE("test_gpu_long_series_wavefront_banded", "[cuda][long][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  // L = 3072 > 2048 still exercises the 3-buffer routing. Fixed-band
  // membership is evaluated directly, so no boundary arrays inflate shared
  // memory beyond the three diagonal buffers.
  constexpr size_t N = 5;
  constexpr size_t L = 3072;
  constexpr int band = 64;
  auto series = generate_random_walks(N, L, /*seed=*/20240702);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  // REGISTERED pass band (recorded before the run): max |GPU - CPU| <= 1e-9.
  double max_abs_diff = 0.0;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      max_abs_diff = std::max(
          max_abs_diff, std::abs(gpu_result.matrix[i * N + j] - cpu_mat[i * N + j]));

  INFO("max |GPU - CPU| = " << max_abs_diff);
  CHECK(max_abs_diff <= 1e-9);
}

// The three FP32 diagonal buffers of L = 4095 and 4096 fit the 48 KiB default
// shared-memory limit on their own, but not with the kernel's static shared
// memory, so the launch must opt in to the larger limit.
TEST_CASE("FP32 wavefront at L = 4095 and 4096 matches the host kernel",
          "[cuda][long][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const size_t L = GENERATE(size_t{4095}, size_t{4096});
  CAPTURE(L);
  const auto series = generate_random_walks(3, L, /*seed=*/20260930);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  const auto gpu_result = gpu_fill(series, opts);
  REQUIRE(gpu_result.kernel_used == "wavefront");
  REQUIRE(gpu_result.matrix == cpu_fp32_distance_matrix(series));
}

namespace {

/// The first length whose wavefront block does not fit the device's opt-in
/// shared memory: above L = 2048 a block holds three anti-diagonals of L values
/// plus the kernel's 16 static bytes, so FP32 8447 and FP64 4224 on the RTX 4000
/// Ada (101,376 bytes). From there the wavefront keeps its anti-diagonals in
/// global memory.
size_t first_global_length(bool fp32)
{
  int max_shared = 0;
  REQUIRE(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
          == cudaSuccess);
  return (static_cast<size_t>(max_shared) - 16) / (3 * (fp32 ? 4 : 8)) + 1;
}

} // anonymous namespace

// One length below the limit the anti-diagonals stay in shared memory, at the
// limit they go to global memory; both fills are the host kernel's bit for bit,
// and the same thread's next shared-memory fill is too.
TEST_CASE("CUDA wavefront keeps its anti-diagonals in global memory from the shared-memory limit",
          "[cuda][long]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const bool fp32 = GENERATE(true, false);
  const size_t first_global = first_global_length(fp32);
  CAPTURE(fp32, first_global);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::cuda::CUDAPrecision::FP32 : dtwc::cuda::CUDAPrecision::FP64;
  const auto host = [fp32](const auto &series) {
    return fp32 ? cpu_fp32_distance_matrix(series) : cpu_distance_matrix(series);
  };
  for (const size_t L : { first_global - 1, first_global }) {
    CAPTURE(L);
    const auto series = generate_random_walks(2, L, /*seed=*/41);
    const auto gpu_result = gpu_fill(series, opts);
    CHECK(gpu_result.kernel_used == (L < first_global ? "wavefront" : "wavefront_global"));
    CHECK(gpu_result.matrix == host(series));
  }
  const auto series = generate_random_series(4, 300, /*seed=*/9);
  REQUIRE(gpu_fill(series, opts).matrix == host(series));
}

// Series longer than the shared-memory limit fill on the GPU (FP32 L 8447 and
// 12,000, FP64 L 4224 and 9406, one past data/dummy's longest; each refused
// before the global-memory wavefront). Lengths are mixed down to 1 sample, so
// pairs run in both orientations and most are padded; every distance is the
// host kernel's, bit for bit, in the fill's precision.
TEST_CASE("CUDA fill of series beyond the shared-memory limit matches the host kernel",
          "[cuda][long]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  struct Case {
    size_t L;
    bool fp32;
  };
  const auto c = GENERATE(values<Case>({ { 8447, true }, { 12000, true },
                                         { 4224, false }, { 9406, false } }));
  CAPTURE(c.L, c.fp32);
  const size_t lengths[] = { c.L, c.L - 1, c.L - 700, 3 * c.L / 4, c.L / 2 + 1, 2049, 300, 1 };
  auto series = generate_random_walks(std::size(lengths), c.L, /*seed=*/20261001);
  for (size_t k = 0; k < series.size(); ++k) series[k].resize(lengths[k]);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = c.fp32 ? dtwc::cuda::CUDAPrecision::FP32 : dtwc::cuda::CUDAPrecision::FP64;
  const auto gpu_result = gpu_fill(series, opts);
  CHECK(gpu_result.kernel_used
        == (c.L < first_global_length(c.fp32) ? "wavefront" : "wavefront_global"));
  REQUIRE(gpu_result.matrix
          == (c.fp32 ? cpu_fp32_distance_matrix(series) : cpu_distance_matrix(series)));
}

// More pairs than the device holds blocks, so each block runs several pairs in
// its own slice of the global scratch: one series at the limit and 63 of 1 to
// 300 samples, which keeps the host oracle cheap.
TEST_CASE("CUDA global-memory wavefront runs many pairs per block",
          "[cuda][long]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const bool fp32 = GENERATE(true, false);
  CAPTURE(fp32);
  auto series = generate_random_walks(64, first_global_length(fp32), /*seed=*/20261002);
  for (size_t k = 1; k < series.size(); ++k) series[k].resize(1 + (k * 37) % 300);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::cuda::CUDAPrecision::FP32 : dtwc::cuda::CUDAPrecision::FP64;
  const auto gpu_result = gpu_fill(series, opts);
  CHECK(gpu_result.kernel_used == "wavefront_global");
  REQUIRE(gpu_result.matrix
          == (fp32 ? cpu_fp32_distance_matrix(series) : cpu_distance_matrix(series)));
}

// The Problem's fill hands its matrix to the backend, which sizes it only after
// its own refusals: a device index past the last device leaves no matrix
// allocated (Problem used to size it first, which at N = 65,537 is 17 GB of NaN).
TEST_CASE("A refused CUDA fill leaves the Problem's matrix unallocated", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  dtwc::Problem prob("refused");
  prob.set_data(dtwc::Data{ generate_random_walks(2, 100, /*seed=*/41), { "s0", "s1" } });
  prob.set_device(dtwc::Device::GPU, device_count);
  REQUIRE_THROWS_MATCHES(prob.fill_distance_matrix(), dtwc::DeviceError,
                         MessageMatches(ContainsSubstring("invalid device ordinal")));
  CHECK(std::as_const(prob).distance_matrix().size() == 0);
}

// The wavefront's dynamic shared-memory limit is one value per kernel and
// device, shared by every host thread. Set on each launch, a thread at L = 4096
// lowered it under another thread's L = 8000 launch ("invalid argument").
TEST_CASE("Two host threads filling at different long lengths do not fail each other",
          "[cuda][long][threads]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  const auto long_pair = generate_random_series(2, 8000, /*seed=*/31);
  const auto short_pair = generate_random_series(2, 4096, /*seed=*/32);
  std::array<int, 2> failures{}; // one slot per thread
  const auto fill = [&](size_t slot, const std::vector<std::vector<double>> &series) {
    for (int round = 0; round < 50; ++round) {
      try {
        (void)gpu_fill(series, opts);
      } catch (const std::exception &) {
        ++failures[slot];
      }
    }
  };
  std::thread first(fill, size_t{0}, std::cref(long_pair));
  std::thread second(fill, size_t{1}, std::cref(short_pair));
  first.join();
  second.join();
  CHECK(failures[0] == 0);
  CHECK(failures[1] == 0);
}

// ---------------------------------------------------------------------------
// Structural properties of the distance matrix
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_diagonal_zero", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 80;
  auto series = generate_random_series(N, L, /*seed=*/5555);

  auto gpu_result = gpu_fill(series);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    INFO("i=" << i);
    REQUIRE(gpu_result.matrix[i * N + i] == 0.0);
  }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_single_series", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  auto series = generate_random_series(1, 30, /*seed=*/11);

  auto gpu_result = gpu_fill(series);

  REQUIRE(gpu_result.n == 1);
  REQUIRE(gpu_result.matrix.size() == 1);
  REQUIRE(gpu_result.matrix[0] == 0.0);
  REQUIRE(gpu_result.kernel_used == "none");
}

TEST_CASE("test_gpu_two_identical", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  auto series = generate_random_series(1, 60, /*seed=*/22);
  // Duplicate the single series so we have two identical ones.
  series.push_back(series[0]);

  auto gpu_result = gpu_fill(series);

  REQUIRE(gpu_result.n == 2);
  REQUIRE(gpu_result.matrix.size() == 4);

  // Diagonal must be zero.
  REQUIRE(gpu_result.matrix[0 * 2 + 0] == 0.0);
  REQUIRE(gpu_result.matrix[1 * 2 + 1] == 0.0);

  // Off-diagonal: identical series should have zero distance.
  REQUIRE(gpu_result.matrix[0 * 2 + 1] == 0.0);
  REQUIRE(gpu_result.matrix[1 * 2 + 0] == 0.0);
}

// Series that are all empty have no distance to compute; an all-zero matrix
// would read as N identical series.
TEST_CASE("CUDA fill of all-empty series is InvalidInput and no zero matrix", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const std::vector<std::vector<double>> empty(3);
  dtwc::core::DistanceMatrix untouched;
  REQUIRE_THROWS_AS(dtwc::cuda::compute_distance_matrix_cuda(empty, {}, untouched),
                    dtwc::InvalidInput);
  CHECK(untouched.size() == 0);
}

// A CUDA call that fails is reported once, by the fill that made it. The
// runtime also keeps the error as the thread's last error, which each launch is
// checked with, so the thread's next fill used to fail with it: here
// cudaSetDevice's "invalid device ordinal" for a device index past the last.
TEST_CASE("A fill the CUDA runtime refuses leaves the thread's next fill unaffected", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  const auto series = generate_random_series(4, 60, /*seed=*/13);
  const auto before = gpu_fill(series); // the thread's CUDA context exists from here

  dtwc::cuda::CUDADistMatOptions missing_device;
  missing_device.device_id = device_count;
  dtwc::core::DistanceMatrix untouched;
  REQUIRE_THROWS_MATCHES(
      dtwc::cuda::compute_distance_matrix_cuda(series, missing_device, untouched),
      dtwc::DeviceError, MessageMatches(ContainsSubstring("invalid device ordinal")));
  CHECK(untouched.size() == 0);

  REQUIRE(gpu_fill(series).matrix == before.matrix);
}

// ---------------------------------------------------------------------------
// Banded DTW: GPU vs CPU comparison
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_banded_matches_cpu_small", "[cuda][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  constexpr int band = 5;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_matches_cpu_medium", "[cuda][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 200;
  constexpr int band = 20;
  auto series = generate_random_series(N, L, /*seed=*/123);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_narrow_band", "[cuda][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 100;
  constexpr int band = 2;
  auto series = generate_random_series(N, L, /*seed=*/555);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_negative_is_full_dtw", "[cuda][banded]")
{
  // band < 0 should produce full (unconstrained) DTW, identical to default
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts_full;
  opts_full.band = -1;

  dtwc::cuda::CUDADistMatOptions opts_default;
  // default band is -1

  auto gpu_full    = gpu_fill(series, opts_full);
  auto gpu_default = gpu_fill(series, opts_default);

  REQUIRE(gpu_full.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(gpu_full.matrix[i * N + j] == gpu_default.matrix[j * N + i]);
    }
  }
}

TEST_CASE("test_gpu_banded_wide_band_equals_full", "[cuda][banded]")
{
  // A band wider than series length should give same result as full DTW
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 30;
  constexpr int band = 500; // much wider than L
  auto series = generate_random_series(N, L, /*seed=*/777);

  dtwc::cuda::CUDADistMatOptions opts_banded;
  opts_banded.band = band;
  opts_banded.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_banded = gpu_fill(series, opts_banded);
  auto cpu_full   = cpu_distance_matrix(series);

  REQUIRE(gpu_banded.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_banded.matrix[i * N + j],
                   WithinRel(cpu_full[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_unequal_lengths", "[cuda][banded]")
{
  // Test banded DTW with series of different lengths
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(314);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(6);
  // Varying lengths: 30, 50, 40, 60, 35, 55
  const size_t lens[] = {30, 50, 40, 60, 35, 55};
  for (size_t s = 0; s < 6; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  constexpr int band = 8;

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == 6);

  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * 6 + j],
                   WithinRel(cpu_mat[i * 6 + j], 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Warp-level kernel tests (short series, L <= 32)
// ---------------------------------------------------------------------------

TEST_CASE("test_warp_kernel_short_series_L8", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 20;
  constexpr size_t L = 8;
  auto series = generate_random_series(N, L, /*seed=*/1001);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_short_series_L16", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 16;
  auto series = generate_random_series(N, L, /*seed=*/2002);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_short_series_L32", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 20;
  constexpr size_t L = 32;
  auto series = generate_random_series(N, L, /*seed=*/3003);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_short_series_L1", "[cuda][warp]")
{
  // Edge case: single-element series through the warp kernel
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::vector<std::vector<double>> series = {{3.0}, {7.0}, {1.0}, {5.0}};
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);

  const size_t N = series.size();
  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double expected = std::abs(series[i][0] - series[j][0]);
      INFO("i=" << i << " j=" << j);
      REQUIRE(gpu_result.matrix[i * N + j] == Catch::Approx(expected));
      REQUIRE(gpu_result.matrix[j * N + i] == Catch::Approx(expected));
    }
  }
}

TEST_CASE("test_warp_kernel_variable_short_lengths", "[cuda][warp]")
{
  // Variable lengths all <= 32 to exercise the warp kernel
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(4004);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  const size_t lens[] = {5, 10, 15, 20, 25, 30, 8, 12};
  for (size_t s = 0; s < 8; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  const size_t N = series.size();
  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_banded_short_series", "[cuda][warp][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 12;
  constexpr size_t L = 20;
  constexpr int band = 3;
  auto series = generate_random_series(N, L, /*seed=*/5005);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_fp32_short_series", "[cuda][warp][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 24;
  auto series = generate_random_series(N, L, /*seed=*/6006);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_warp_kernel_many_pairs_short_series", "[cuda][warp]")
{
  // Stress test: many pairs to ensure multi-block dispatch works
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 50;   // 1225 pairs, ~154 blocks of 8
  constexpr size_t L = 16;
  auto series = generate_random_series(N, L, /*seed=*/7007);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.pairs_computed == N * (N - 1) / 2);

  // Spot-check 25 pairs against CPU
  for (size_t k = 0; k < 25; ++k) {
    size_t i = k;
    size_t j = N - 1 - k;
    double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j]);
    double gpu_d = gpu_result.matrix[i * N + j];
    INFO("k=" << k << " i=" << i << " j=" << j);
    REQUIRE_THAT(gpu_d, WithinRel(cpu_d, 1e-10));
  }

  // Symmetry check on a sample
  for (size_t i = 0; i < 10; ++i)
    for (size_t j = i + 1; j < 10; ++j)
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
}

TEST_CASE("test_warp_kernel_squared_l2_short_series", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 20;
  auto series = generate_random_series(N, L, /*seed=*/8008);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.use_squared_l2 = true;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                              dtwc::core::MetricType::SquaredL2);
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_d, 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// FP32 precision tests: looser tolerance due to single-precision accumulation
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_fp32_matches_cpu_small", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      // FP32 accumulation: relative tolerance ~1e-5
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_gpu_fp32_matches_cpu_medium", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 200;
  auto series = generate_random_series(N, L, /*seed=*/123);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_gpu_fp32_identical_series_zero_distance", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  auto series = generate_random_series(1, 60, /*seed=*/22);
  series.push_back(series[0]);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.n == 2);
  REQUIRE(gpu_result.matrix[0 * 2 + 1] == 0.0);
  REQUIRE(gpu_result.matrix[1 * 2 + 0] == 0.0);
}

TEST_CASE("test_gpu_fp32_banded_matches_cpu", "[cuda][fp32][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  constexpr int band = 5;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

// ---------------------------------------------------------------------------
// Auto precision
// ---------------------------------------------------------------------------

// Auto computes in FP64 only on a device that runs FP64 at least half as fast
// as FP32 (the HPC parts). A compute-capability table once gave consumer
// Blackwell (sm_120, FP64 at 1/64) FP64; the runtime's own ratio decides now.
TEST_CASE("CUDA Auto precision follows the device's FP32:FP64 throughput",
          "[cuda][precision]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  int fp32_per_fp64 = 0;
  REQUIRE(cudaDeviceGetAttribute(&fp32_per_fp64,
                                 cudaDevAttrSingleToDoublePrecisionPerfRatio, 0)
          == cudaSuccess);
  CAPTURE(fp32_per_fp64);

  const auto series = generate_random_series(6, 40, /*seed=*/11);
  const auto fill = [&](dtwc::cuda::CUDAPrecision precision) {
    dtwc::cuda::CUDADistMatOptions opts;
    opts.precision = precision;
    return gpu_fill(series, opts).matrix;
  };
  const auto fp32 = fill(dtwc::cuda::CUDAPrecision::FP32);
  const auto fp64 = fill(dtwc::cuda::CUDAPrecision::FP64);
  REQUIRE(fp32 != fp64); // this input tells the two precisions apart
  REQUIRE(fill(dtwc::cuda::CUDAPrecision::Auto)
          == (fp32_per_fp64 > 2 ? fp32 : fp64));
}

// ---------------------------------------------------------------------------
// Squared-L2 metric: GPU vs CPU
// ---------------------------------------------------------------------------

TEST_CASE("GPU matches CPU with squared-L2 metric", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/500);

  // GPU with squared L2
  dtwc::cuda::CUDADistMatOptions opts;
  opts.use_squared_l2 = true;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  // CPU reference with squared L2
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                              dtwc::core::MetricType::SquaredL2);
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_d, 1e-10));
      REQUIRE_THAT(gpu_result.matrix[j * N + i],
                   WithinRel(cpu_d, 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Variable-length series
// ---------------------------------------------------------------------------

TEST_CASE("GPU handles variable-length series", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  // Series with different lengths
  std::vector<std::vector<double>> series;
  std::mt19937 rng(600);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int len : {30, 50, 80, 100, 60, 40}) {
    std::vector<double> s(static_cast<size_t>(len));
    for (auto &v : s) v = dist(rng);
    series.push_back(std::move(s));
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  const size_t N = series.size();

  REQUIRE(gpu_result.n == N);

  // Compare with CPU
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j]);
      double gpu_d = gpu_result.matrix[i * N + j];
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_d, WithinRel(cpu_d, 1e-10));
    }
  }

  // Also check symmetry
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
}

// ---------------------------------------------------------------------------
// Edge case: single pair (N=2)
// ---------------------------------------------------------------------------

TEST_CASE("GPU single pair (N=2)", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::vector<std::vector<double>> series = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto result = gpu_fill(series, opts);

  double cpu_d = dtwc::dtwFull_L<double>(series[0], series[1]);
  REQUIRE(result.n == 2);
  REQUIRE(result.matrix[0 * 2 + 1] == Catch::Approx(cpu_d).epsilon(1e-10));
  REQUIRE(result.matrix[1 * 2 + 0] == Catch::Approx(cpu_d).epsilon(1e-10));
  REQUIRE(result.matrix[0] == 0.0);
  REQUIRE(result.matrix[3] == 0.0);
}

// ---------------------------------------------------------------------------
// Edge case: length-1 series
// ---------------------------------------------------------------------------

TEST_CASE("GPU length-1 series", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::vector<std::vector<double>> series = {{3.0}, {7.0}, {1.0}};
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto result = gpu_fill(series, opts);

  REQUIRE(result.n == 3);
  // DTW of single-element series = |a - b| (L1 metric)
  REQUIRE(result.matrix[0 * 3 + 1] == Catch::Approx(4.0));
  REQUIRE(result.matrix[0 * 3 + 2] == Catch::Approx(2.0));
  REQUIRE(result.matrix[1 * 3 + 2] == Catch::Approx(6.0));
  // Symmetry
  REQUIRE(result.matrix[1 * 3 + 0] == Catch::Approx(4.0));
  REQUIRE(result.matrix[2 * 3 + 0] == Catch::Approx(2.0));
  REQUIRE(result.matrix[2 * 3 + 1] == Catch::Approx(6.0));
  // Diagonal
  REQUIRE(result.matrix[0] == 0.0);
  REQUIRE(result.matrix[4] == 0.0);
  REQUIRE(result.matrix[8] == 0.0);
}

// ---------------------------------------------------------------------------
// Register-tiled kernel tests (medium series, 32 < L <= 256)
// ---------------------------------------------------------------------------

TEST_CASE("test_regtile_kernel_L33", "[cuda][regtile]")
{
  // Just above warp kernel threshold — exercises TILE_W=4 with minimal columns
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 33;
  auto series = generate_random_series(N, L, /*seed=*/10001);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L64", "[cuda][regtile]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 12;
  constexpr size_t L = 64;
  auto series = generate_random_series(N, L, /*seed=*/10002);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L128", "[cuda][regtile]")
{
  // Boundary: exactly at TILE_W=4 maximum (32*4=128)
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 128;
  auto series = generate_random_series(N, L, /*seed=*/10003);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L129", "[cuda][regtile]")
{
  // Just above TILE_W=4 — exercises TILE_W=8
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 129;
  auto series = generate_random_series(N, L, /*seed=*/10004);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L256", "[cuda][regtile]")
{
  // Boundary: exactly at TILE_W=8 maximum (32*8=256)
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 256;
  auto series = generate_random_series(N, L, /*seed=*/10005);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_variable_lengths", "[cuda][regtile]")
{
  // Variable lengths spanning the regtile range
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(10006);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  const size_t lens[] = {35, 50, 70, 100, 40, 80, 60, 90};
  for (size_t s = 0; s < 8; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  const size_t N = series.size();
  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_banded_L100", "[cuda][regtile][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 100;
  constexpr int band = 10;
  auto series = generate_random_series(N, L, /*seed=*/10007);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_fp32_L100", "[cuda][regtile][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 100;
  auto series = generate_random_series(N, L, /*seed=*/10008);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = gpu_fill(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_regtile_kernel_squared_l2_L80", "[cuda][regtile]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 80;
  auto series = generate_random_series(N, L, /*seed=*/10009);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.use_squared_l2 = true;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                              dtwc::core::MetricType::SquaredL2);
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_d, 1e-10));
    }
  }
}

// A15: at N = 65,537 the pair count passes INT_MAX, so the fill runs in several
// launches of consecutive pairs, each streaming its span of the packed matrix to
// the host; the CUDA fill used to refuse any N above 65,536. Series of 4 to 8
// samples keep the host oracle cheap enough to check every pair, so every launch
// boundary is checked: bit for bit in FP64 against the host kernel, and in FP32
// against the host kernel run in FP32 on the float-rounded series (the exact
// standard of the kernel-range test above).
TEST_CASE("A15 CUDA fill above N = 65,536 matches the host kernel on every pair",
          "[cuda][A15][large]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 65537;
  // The matrix is N(N+1)/2 doubles (17.2 GB); a GiB more covers the series and
  // the rest of the process. A host that cannot hold it skips, and says so.
  const std::uint64_t needed =
      dtwc::core::packed_size(N) * sizeof(double) + (std::uint64_t{ 1 } << 30);
  const std::uint64_t available = available_host_memory();
  if (available != 0 && available < needed) {
    SKIP("A15 needs " << (needed >> 20) << " MiB of available host memory for its "
         << N << "-series matrix; this host has " << (available >> 20) << " MiB");
    return;
  }
  std::vector<std::vector<double>> series(N);
  std::vector<std::vector<float>> rounded(N);
  std::vector<std::string> names(N);
  std::mt19937 rng(20260930);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  for (size_t i = 0; i < N; ++i) {
    series[i].resize(4 + i % 5);
    for (auto &v : series[i]) v = value(rng);
    rounded[i].resize(series[i].size());
    std::transform(series[i].begin(), series[i].end(), rounded[i].begin(),
                   [](double v) { return static_cast<float>(v); });
    names[i] = "s" + std::to_string(i);
  }

  dtwc::Problem prob("a15");
  prob.set_data(dtwc::Data{ std::vector<std::vector<double>>(series), std::move(names) });
  prob.set_device(dtwc::Device::GPU);
  for (const bool fp64 : { true, false }) {
    CAPTURE(fp64);
    prob.set_cuda_settings(
        dtwc::CUDASettings{ 0, fp64 ? dtwc::GpuPrecision::FP64 : dtwc::GpuPrecision::FP32 });
    prob.fill_distance_matrix();
    const auto &matrix = std::as_const(prob).distance_matrix();
    REQUIRE(matrix.size() == N);

    // One mismatch count per row: each row has one writer.
    std::vector<size_t> mismatches(N, 0);
    auto check_row = [&](size_t i) {
      for (size_t j = 0; j < i; ++j) {
        const double host = fp64
            ? dtwc::dtwFull_L<double>(series[i], series[j])
            : static_cast<double>(dtwc::dtwFull_L<float>(rounded[i], rounded[j]));
        mismatches[i] += matrix.get(i, j) != host;
      }
      mismatches[i] += matrix.get(i, i) != 0.0;
    };
    dtwc::run_openmp(check_row, N);
    const auto first_bad =
        std::find_if(mismatches.begin(), mismatches.end(), [](size_t m) { return m != 0; });
    CAPTURE(first_bad - mismatches.begin());
    REQUIRE(std::accumulate(mismatches.begin(), mismatches.end(), size_t{ 0 }) == 0);
  }
}

// Two launches in the register-tile and the wavefront families: 16,385 is the
// first N whose pair count needs a second launch. The kernel follows the
// longest series, so every 64th series has the family's length (33:
// regtile_w4; 257: the wavefront, persistent in both launches) and the rest 1
// to 4 samples, which keeps the fill and the host oracle cheap. Every pair is
// checked against the host kernel, bit for bit, and every diagonal entry is 0;
// the matrix is written only by the launches' copies, the first launch leaves
// distances in the device buffer where the second has diagonal slots, and so
// the check covers each launch's whole copied range.
TEST_CASE("CUDA fills over two launches match the host kernel in the regtile and wavefront families",
          "[cuda][launches]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  struct Family {
    size_t long_length;
    const char *kernel;
  };
  const auto family = GENERATE(values<Family>({ { 33, "regtile_w4" }, { 257, "wavefront" } }));
  CAPTURE(family.kernel);

  constexpr size_t N = 16385;
  REQUIRE(dtwc::cuda::detail::upper_triangle_pairs(N)
          > static_cast<size_t>(dtwc::cuda::detail::kMaxPairsPerLaunch));
  std::vector<std::vector<double>> series(N);
  std::mt19937 rng(16385);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  for (size_t i = 0; i < N; ++i) {
    series[i].resize(i % 64 == 0 ? family.long_length : 1 + i % 4);
    for (auto &v : series[i]) v = value(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  dtwc::core::DistanceMatrix matrix;
  const auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix);
  REQUIRE(result.kernel_used == family.kernel);
  REQUIRE(matrix.size() == N);

  // One mismatch count per row: each row has one writer.
  std::vector<size_t> mismatches(N, 0);
  auto check_row = [&](size_t i) {
    for (size_t j = 0; j < i; ++j)
      mismatches[i] += matrix.get(i, j) != dtwc::dtwFull_L<double>(series[i], series[j]);
    mismatches[i] += matrix.get(i, i) != 0.0;
  };
  dtwc::run_openmp(check_row, N);
  const auto first_bad =
      std::find_if(mismatches.begin(), mismatches.end(), [](size_t m) { return m != 0; });
  CAPTURE(first_bad - mismatches.begin());
  REQUIRE(std::accumulate(mismatches.begin(), mismatches.end(), size_t{ 0 }) == 0);
}

// ---------------------------------------------------------------------------
// Larger stress test
// ---------------------------------------------------------------------------

TEST_CASE("GPU stress test (100 series x 200 length)", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 100;
  constexpr size_t L = 200;
  auto series = generate_random_series(N, L, /*seed=*/700);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = gpu_fill(series, opts);
  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.pairs_computed == N * (N - 1) / 2);

  // Spot-check 20 pairs against CPU
  for (size_t k = 0; k < 20; ++k) {
    size_t i = k;
    size_t j = N - 1 - k;
    double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j]);
    double gpu_d = gpu_result.matrix[i * N + j];
    INFO("k=" << k << " i=" << i << " j=" << j);
    REQUIRE_THAT(gpu_d, WithinRel(cpu_d, 1e-10));
  }

  // Verify diagonal is zero
  for (size_t i = 0; i < N; ++i)
    REQUIRE(gpu_result.matrix[i * N + i] == 0.0);
}

// FX-1: a squared-L2 cache is filled by the Problem's CUDA route with the
// squared-L2 kernel (it used to be refused as external-fill-only) and matches
// the CPU squared-L2 kernels. FP64 is explicit: a persistent cache refuses Auto.
TEST_CASE("FX-1 CUDA squared-L2 cache via Problem::fill_distance_matrix",
          "[cuda][mmap][fx1]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in");
#else
  const auto series = generate_random_series(6, 40, /*seed=*/7);
  const auto cache =
    std::filesystem::temp_directory_path() / "dtwc_fx1_cuda_sql2.bin";
  for (const int band : { -1, 6 }) {
    CAPTURE(band);
    std::filesystem::remove(cache);
    dtwc::Problem prob("cuda_sql2");
    prob.set_data(dtwc::Data{ std::vector<std::vector<double>>(series),
                              { "s0", "s1", "s2", "s3", "s4", "s5" } });
    prob.set_band(band);
    prob.set_cuda_settings(dtwc::CUDASettings{ 0, dtwc::GpuPrecision::FP64 });
    prob.set_distance_strategy(dtwc::DistanceMatrixStrategy::CUDA);
    prob.use_mmap_distance_matrix(cache, dtwc::core::MetricType::SquaredL2);
    prob.fill_distance_matrix();
    for (size_t i = 0; i < series.size(); ++i)
      for (size_t j = i + 1; j < series.size(); ++j) {
        const double oracle = band < 0
          ? dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                    dtwc::core::MetricType::SquaredL2)
          : dtwc::dtwBanded<double>(series[i], series[j], band, -1.0,
                                    dtwc::core::MetricType::SquaredL2);
        REQUIRE_THAT(prob.dist_by_ind(int(i), int(j)), WithinRel(oracle, 1e-9));
      }
  }
  std::filesystem::remove(cache);
#endif
}

// IF-2 S2: the metric is the Problem's (set_metric), so a dense CUDA fill
// computes squared L2 without a cache; FP64 matches the CPU kernels to 1e-9,
// FP32 to the Metal tests' FP32 band.
TEST_CASE("IF-2 CUDA dense squared-L2 via Problem::set_metric",
          "[cuda][metric][if2]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
  const auto series = generate_random_series(6, 40, /*seed=*/9);
  for (const auto precision : { dtwc::GpuPrecision::FP32, dtwc::GpuPrecision::FP64 })
    for (const int band : { -1, 6 }) {
      CAPTURE(precision, band);
      dtwc::Problem prob("cuda_dense_sql2");
      prob.set_data(dtwc::Data{ std::vector<std::vector<double>>(series),
                                { "s0", "s1", "s2", "s3", "s4", "s5" } });
      prob.set_band(band);
      prob.set_cuda_settings(dtwc::CUDASettings{ 0, precision });
      prob.set_device(dtwc::Device::GPU);
      prob.set_metric(dtwc::core::MetricType::SquaredL2);
      prob.fill_distance_matrix();
      for (size_t i = 0; i < series.size(); ++i)
        for (size_t j = i + 1; j < series.size(); ++j) {
          const double oracle = dtwc::distance::dtw<double>(
            series[i], series[j], band, dtwc::core::MetricType::SquaredL2);
          if (precision == dtwc::GpuPrecision::FP64)
            REQUIRE_THAT(prob.dist_by_ind(int(i), int(j)), WithinRel(oracle, 1e-9));
          else
            REQUIRE_THAT(prob.dist_by_ind(int(i), int(j)),
                         WithinRel(oracle, 1e-4)
                           || Catch::Matchers::WithinAbs(oracle, 1e-3));
        }
    }
}

#endif // DTWC_HAS_CUDA
