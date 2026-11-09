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
#include <algorithms/detail/fast_clara_assign.hpp>
#include <base/parallelisation.hpp>
#include <cli/run.hpp>
#include <config.hpp>
#include <core/thread_scratch.hpp>

#include "gpu_fixed_band_oracle.hpp"
#include "../support/deterministic_series.hpp"
#include "../support/dtw_route_bound.hpp"

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#include <cuda/launch_prep.hpp>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>  // std::iota (MSVC STL does not include it transitively)
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef DTWC_HAS_CUDA

TEST_CASE("CUDA not available", "[cuda]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA tests skipped");
}

#else // DTWC_HAS_CUDA

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::WithinRel;

namespace {

constexpr auto generate_random_series =
  &dtwc::test_support::accelerator_series_set;

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

/// compute_medoid_distances_cuda on series and medoids held as vectors, their blocks padded by this thread's team.
dtwc::cuda::CUDADistMatResult medoid_distances(
    const std::vector<std::vector<double>> &series, const std::vector<std::vector<double>> &medoids,
    const dtwc::cuda::CUDADistMatOptions &opts,
    const std::function<void(size_t, size_t, std::span<const double>)> &consume)
{
  const std::vector<std::span<const double>> spans(medoids.begin(), medoids.end());
  return dtwc::cuda::compute_medoid_distances_cuda(
      series.size(), [&series](size_t i) { return std::span<const double>(series[i]); }, spans, opts,
      [](size_t count, const std::function<void(size_t)> &task) { dtwc::run_openmp(task, count); }, consume);
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
    dtwc::GpuPrecision precision =
        dtwc::GpuPrecision::FP64)
{
  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = row.band;
  opts.use_squared_l2 = squared;
  opts.precision = precision;
  return opts;
}

/// The first length that takes the global-memory wavefront on device 0. Up to
/// L = 2048 the wavefront keeps its anti-diagonals in shared memory; above it,
/// only while three blocks fit an SM's shared memory, each three anti-diagonals
/// of L values, the kernel's 16 static bytes and the runtime's reserved bytes:
/// on the RTX 4000 Ada (100 KB per SM, 1 KB reserved) FP32 2758 and FP64 2049.
size_t first_global_length(bool fp32)
{
  int sm_shared = 0;
  int reserved = 0;
  REQUIRE(cudaDeviceGetAttribute(&sm_shared, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0)
          == cudaSuccess);
  REQUIRE(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, 0)
          == cudaSuccess);
  const size_t block_room = static_cast<size_t>(sm_shared) / 3 - 16 - static_cast<size_t>(reserved);
  return std::max<size_t>(2049, block_room / (3 * (fp32 ? 4 : 8)) + 1);
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
      below_gap, false, dtwc::GpuPrecision::FP32);

  const auto pairwise = gpu_fill(
      f12_pairwise_inventory(route), opts);
  REQUIRE(pairwise.kernel_used == "warp");
  REQUIRE(pairwise.matrix[1] == oracle::public_no_path_sentinel);
  REQUIRE(pairwise.matrix[2] == oracle::public_no_path_sentinel);
}

// Every range of the automatic kernel choice, at its edges, in both precisions:
// warp (L <= 32), regtile<4> (<= 128), regtile<8> (<= 256), and the wavefront's
// preload (<= 512), three-buffer (<= 1024), double-buffer (<= 2048) and long
// three-buffer modes, the last in global memory where three shared blocks do not
// fit an SM. The oracle is the host kernel in the fill's precision.
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
  opts.precision = fp32 ? dtwc::GpuPrecision::FP32
                        : dtwc::GpuPrecision::FP64;
  const auto gpu_result = gpu_fill(series, opts);

  REQUIRE(gpu_result.kernel_used
          == (regime.L < first_global_length(fp32) ? regime.kernel : "wavefront_global"));
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
// Regression: long series (max_L > 2048) must not drop anti-diagonal
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
// memory, so a shared-memory launch must opt in to the larger limit. That route
// takes these lengths where three such blocks fit an SM (an A100 or H100); an
// RTX 4000 Ada runs them in global memory.
TEST_CASE("FP32 wavefront at L = 4095 and 4096 matches the host kernel",
          "[cuda][long][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const size_t L = GENERATE(size_t{4095}, size_t{4096});
  CAPTURE(L);
  const auto series = generate_random_walks(3, L, /*seed=*/20260930);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP32;
  const auto gpu_result = gpu_fill(series, opts);
  REQUIRE(gpu_result.kernel_used
          == (L < first_global_length(true) ? "wavefront" : "wavefront_global"));
  REQUIRE(gpu_result.matrix == cpu_fp32_distance_matrix(series));
}

// One length below the first global length the anti-diagonals stay in shared
// memory, from it they go to global memory; both fills are the host kernel's
// bit for bit, and the same thread's next shared-memory fill is too.
TEST_CASE("CUDA wavefront keeps its anti-diagonals in global memory where three shared blocks do not fit an SM",
          "[cuda][long]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const bool fp32 = GENERATE(true, false);
  const size_t first_global = first_global_length(fp32);
  CAPTURE(fp32, first_global);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::GpuPrecision::FP32 : dtwc::GpuPrecision::FP64;
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
  opts.precision = c.fp32 ? dtwc::GpuPrecision::FP32 : dtwc::GpuPrecision::FP64;
  const auto gpu_result = gpu_fill(series, opts);
  CHECK(gpu_result.kernel_used
        == (c.L < first_global_length(c.fp32) ? "wavefront" : "wavefront_global"));
  REQUIRE(gpu_result.matrix
          == (c.fp32 ? cpu_fp32_distance_matrix(series) : cpu_distance_matrix(series)));
}

// More pairs than the grid has blocks, so each block runs several pairs in its
// own slice of the global scratch: one long series and 63 of 1 to 300 samples,
// which keeps the host oracle cheap. At FP64 L = 8448 the slices of the resident
// blocks pass the L2 (40 MB on the RTX 4000 Ada), so the grid is cut to the
// slices the L2 holds.
TEST_CASE("CUDA global-memory wavefront runs many pairs per block",
          "[cuda][long]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const bool fp32 = GENERATE(true, false);
  const size_t L = fp32 ? first_global_length(true)
                         : std::max<size_t>(8448, first_global_length(false));
  CAPTURE(fp32, L);
  auto series = generate_random_walks(64, L, /*seed=*/20261002);
  for (size_t k = 1; k < series.size(); ++k) series[k].resize(1 + (k * 37) % 300);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::GpuPrecision::FP32 : dtwc::GpuPrecision::FP64;
  const auto gpu_result = gpu_fill(series, opts);
  CHECK(gpu_result.kernel_used == "wavefront_global");
  REQUIRE(gpu_result.matrix
          == (fp32 ? cpu_fp32_distance_matrix(series) : cpu_distance_matrix(series)));
}

// The global-memory wavefront runs the shared kernel's band and squared-L2 cost
// too: FP64 at the limit, a band of 64 against the host's banded kernel bit for
// bit, and squared L2 against the host kernel within the relative 1e-10 of the
// other squared-L2 tests (a host compiler may contract the square into an FMA).
TEST_CASE("CUDA global-memory wavefront matches the host kernel with a band and with squared L2",
          "[cuda][long][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  const auto series = generate_random_walks(4, first_global_length(false), /*seed=*/20261003);
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;

  opts.band = 64;
  const auto banded = gpu_fill(series, opts);
  CHECK(banded.kernel_used == "wavefront_global");
  CHECK(banded.matrix == cpu_banded_distance_matrix(series, 64));

  opts.band = -1;
  opts.use_squared_l2 = true;
  const auto squared = gpu_fill(series, opts);
  CHECK(squared.kernel_used == "wavefront_global");
  for (size_t i = 0; i < series.size(); ++i)
    for (size_t j = 0; j < i; ++j) {
      CAPTURE(i, j);
      CHECK_THAT(squared.matrix[i * series.size() + j],
                 WithinRel(dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                                   dtwc::core::MetricType::SquaredL2),
                           1e-10));
    }
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
  CHECK(prob.distance_matrix().size() == 0);
}

// The wavefront's dynamic shared-memory limit is one value per kernel and
// device, shared by every host thread. Set on each launch, a thread at L = 4096
// lowered it under another thread's L = 8000 launch ("invalid argument"). Where
// three shared blocks do not fit an SM (an RTX 4000 Ada) both lengths take the
// global-memory wavefront; the threads must still not fail each other.
TEST_CASE("Two host threads filling at different long lengths do not fail each other",
          "[cuda][long][threads]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP32;
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
  opts.precision = dtwc::GpuPrecision::FP64;

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
  opts.precision = dtwc::GpuPrecision::FP64;

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
  opts.precision = dtwc::GpuPrecision::FP64;

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
  opts_banded.precision = dtwc::GpuPrecision::FP64;

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

  std::mt19937 rng(314); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
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
  opts.precision = dtwc::GpuPrecision::FP64;

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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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

  std::mt19937 rng(4004); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  const size_t lens[] = {5, 10, 15, 20, 25, 30, 8, 12};
  for (size_t s = 0; s < 8; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;

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
  opts.precision = dtwc::GpuPrecision::FP32;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP32;
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
  opts.precision = dtwc::GpuPrecision::FP32;
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
  opts.precision = dtwc::GpuPrecision::FP32;
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
  opts.precision = dtwc::GpuPrecision::FP32;

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
  const auto fill = [&](dtwc::GpuPrecision precision) {
    dtwc::cuda::CUDADistMatOptions opts;
    opts.precision = precision;
    return gpu_fill(series, opts).matrix;
  };
  const auto fp32 = fill(dtwc::GpuPrecision::FP32);
  const auto fp64 = fill(dtwc::GpuPrecision::FP64);
  REQUIRE(fp32 != fp64); // this input tells the two precisions apart
  REQUIRE(fill(dtwc::GpuPrecision::Auto)
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  std::mt19937 rng(600); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int len : {30, 50, 80, 100, 60, 40}) {
    std::vector<double> s(static_cast<size_t>(len));
    for (auto &v : s) v = dist(rng);
    series.push_back(std::move(s));
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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

  std::mt19937 rng(10006); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  const size_t lens[] = {35, 50, 70, 100, 40, 80, 60, 90};
  for (size_t s = 0; s < 8; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;

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
  opts.precision = dtwc::GpuPrecision::FP32;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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

// At N = 65,537 the pair count passes INT_MAX, so the fill runs in several
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
  const std::uint64_t available = dtwc::core::available_memory(); // 0: unknown
  if (available != 0 && available < needed) {
    SKIP("A15 needs " << (needed >> 20) << " MiB of available host memory for its "
         << N << "-series matrix; this host has " << (available >> 20) << " MiB");
    return;
  }
  std::vector<std::vector<double>> series(N);
  std::vector<std::vector<float>> rounded(N);
  std::vector<std::string> names(N);
  std::mt19937 rng(20260930); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
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
    prob.set_gpu_precision(fp64 ? dtwc::GpuPrecision::FP64 : dtwc::GpuPrecision::FP32);
    prob.fill_distance_matrix();
    const auto &matrix = prob.distance_matrix();
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
  std::mt19937 rng(16385); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  for (size_t i = 0; i < N; ++i) {
    series[i].resize(i % 64 == 0 ? family.long_length : 1 + i % 4);
    for (auto &v : series[i]) v = value(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;
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
  opts.precision = dtwc::GpuPrecision::FP64;
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

// A squared-L2 cache is filled by the Problem's CUDA route with the
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
    prob.set_gpu_precision(dtwc::GpuPrecision::FP64);
    prob.set_device(dtwc::Device::GPU);
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

// The metric is the Problem's (set_metric), so a dense CUDA fill
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
      prob.set_gpu_precision(precision);
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

// compute_medoid_distances_cuda: each series against each of k medoid series
// on the fill's kernels and route rule, in every automatic kernel range, both
// precisions, without and with a band, against the host kernel in the same
// precision within the route bound (dtw_route_bound.hpp). The series are
// ragged and the longest picks the kernel. Series 4 is medoid 0, at exactly 0
// from it; medoids 1 and 2 are both series 9, so every series is at exactly
// equal distances from them: the ties the assignment breaks.
TEST_CASE("CUDA medoid distances match the host kernel in every kernel range",
          "[cuda][medoids]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  struct Regime {
    size_t L;
    const char *kernel;
  };
  const auto regime = GENERATE(values<Regime>({
      {16, "warp"}, {128, "regtile_w4"}, {256, "regtile_w8"}, {512, "wavefront"},
      {1024, "wavefront"}, {2048, "wavefront"}, {2049, "wavefront"}, {2758, "wavefront"}}));
  const bool fp32 = GENERATE(true, false);
  const int band = GENERATE_COPY(-1, static_cast<int>(regime.L / 4));
  CAPTURE(regime.L, fp32, band);

  // Lengths L - L/4 .. L, series 0 the longest: every band here has a path.
  std::mt19937 rng(static_cast<unsigned>(regime.L));
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<std::vector<double>> series(13);
  for (size_t i = 0; i < series.size(); ++i) {
    series[i].resize(regime.L - (i * 7919) % (regime.L / 4 + 1));
    for (auto &v : series[i]) v = value(rng);
  }
  const std::vector<std::vector<double>> medoids{ series[4], series[9], series[9] };
  const size_t k = medoids.size();

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::GpuPrecision::FP32 : dtwc::GpuPrecision::FP64;
  opts.band = band;
  std::vector<double> d(series.size() * k);
  size_t next = 0;
  const auto result = medoid_distances(
      series, medoids, opts, [&](size_t first, size_t count, std::span<const double> block) {
        REQUIRE(first == next);
        REQUIRE(block.size() == count * k);
        std::copy(block.begin(), block.end(), d.begin() + static_cast<std::ptrdiff_t>(first * k));
        next = first + count;
      });
  REQUIRE(next == series.size());
  REQUIRE(result.kernel_used
          == (regime.L < first_global_length(fp32) ? regime.kernel : "wavefront_global"));
  REQUIRE(result.pairs_computed == series.size() * k);

  for (size_t i = 0; i < series.size(); ++i)
    for (size_t m = 0; m < k; ++m) {
      CAPTURE(i, m);
      const auto &x = series[i];
      const auto &y = medoids[m];
      if (fp32) {
        std::vector<float> xf(x.size()), yf(y.size());
        std::transform(x.begin(), x.end(), xf.begin(), [](double v) { return static_cast<float>(v); });
        std::transform(y.begin(), y.end(), yf.begin(), [](double v) { return static_cast<float>(v); });
        CHECK(dtwc::test_support::dtw_routes_agree<float>(
            d[i * k + m], dtwc::dtwBanded<float>(xf, yf, band), x.size(), y.size()));
      } else {
        CHECK(dtwc::test_support::dtw_routes_agree<double>(
            d[i * k + m], dtwc::dtwBanded<double>(x, y, band), x.size(), y.size()));
      }
    }
  CHECK(d[4 * k + 0] == 0.0);
  for (size_t i = 0; i < series.size(); ++i) {
    CAPTURE(i);
    CHECK(d[i * k + 1] == d[i * k + 2]);
  }
}

// A block holds at most kMaxPairsPerLaunch distances (and samples): with 4096
// medoids that is 32,768 series, so 32,769 take two blocks, handed over in
// order, every distance the host kernel's. The refusals come first: all-empty
// series hand over no block.
TEST_CASE("CUDA medoid distances arrive in blocks of at most kMaxPairsPerLaunch",
          "[cuda][medoids][blocks]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  bool handed = false;
  const auto none_expected = [&](size_t, size_t, std::span<const double>) { handed = true; };
  CHECK_THROWS_AS(medoid_distances({ {}, {} }, { {} }, {}, none_expected),
                  dtwc::InvalidInput);
  CHECK_FALSE(handed);

  constexpr size_t k = 4096;
  const size_t block = static_cast<size_t>(dtwc::cuda::detail::kMaxPairsPerLaunch) / k;
  std::mt19937 rng(4096); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  const auto make = [&](size_t count) {
    std::vector<std::vector<double>> made(count);
    for (size_t i = 0; i < count; ++i) {
      made[i].resize(1 + i % 2);
      for (auto &v : made[i]) v = value(rng);
    }
    return made;
  };
  const auto series = make(block + 1);
  const auto medoids = make(k);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::GpuPrecision::FP64;
  std::vector<std::pair<size_t, size_t>> blocks;
  std::vector<size_t> mismatches(series.size(), 0); // one slot per series: one writer each
  (void)medoid_distances(
      series, medoids, opts, [&](size_t first, size_t count, std::span<const double> d) {
        blocks.emplace_back(first, count);
        auto check = [&](size_t i) {
          const auto &x = series[first + i];
          for (size_t m = 0; m < k; ++m)
            mismatches[first + i] += !dtwc::test_support::dtw_routes_agree<double>(
                d[i * k + m], dtwc::dtwFull_L<double>(x, medoids[m]), x.size(), medoids[m].size());
        };
        dtwc::run_openmp(check, count);
      });
  CHECK(blocks == std::vector<std::pair<size_t, size_t>>{ { 0, block }, { block, 1 } });
  REQUIRE(std::accumulate(mismatches.begin(), mismatches.end(), size_t{ 0 }) == 0);
}

// FastCLARA on a CUDA device: its sample matrices fill on the GPU and its
// assignment runs there (compute_medoid_distances_cuda), as on the CPU. FP64
// against the CPU: the same labels and medoids, the cost within 1e-12
// relative. FP32 against the CPU in Float32: the same labels and medoids on
// these well-separated groups, the cost within the route bound. The data:
// three groups of ragged noisy series and twelve exact duplicates.
TEST_CASE("FastCLARA on a CUDA device assigns its series as the CPU does", "[cuda][clara]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(20261001); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> noise(-0.5, 0.5);
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < 240; ++i) {
    std::vector<double> s(40 + i % 21);
    for (auto &v : s) v = 6.0 * static_cast<double>(i % 3) + noise(rng);
    series.push_back(std::move(s));
  }
  for (size_t i = 0; i < 12; ++i) series.push_back(series[i * 7]);

  dtwc::algorithms::CLARAOptions clara;
  clara.n_clusters = 3;
  clara.sample_size = 40;
  clara.n_samples = 3;
  const auto run = [&](dtwc::Device device, dtwc::GpuPrecision precision, bool float32_series, bool view = false) {
    std::vector<std::string> names(series.size());
    dtwc::Problem prob("clara");
    if (view) {
      std::vector<std::span<const double>> spans(series.begin(), series.end());
      std::vector<std::string_view> name_views(names.begin(), names.end());
      prob.set_view_data(dtwc::Data(std::move(spans), std::move(name_views), 1, std::make_shared<int>(0)));
    } else if (float32_series) {
      std::vector<std::vector<float>> rounded;
      for (const auto &s : series) {
        auto &r = rounded.emplace_back(s.size());
        std::transform(s.begin(), s.end(), r.begin(), [](double v) { return static_cast<float>(v); });
      }
      prob.set_data(dtwc::Data(std::move(rounded), std::move(names)));
    } else {
      prob.set_data(dtwc::Data(std::vector<std::vector<double>>(series), std::move(names)));
    }
    prob.set_gpu_precision(precision);
    prob.set_device(device);
    return dtwc::algorithms::fast_clara(prob, clara);
  };

  const auto cpu = run(dtwc::Device::CPU, dtwc::GpuPrecision::Auto, false);
  const auto gpu = run(dtwc::Device::GPU, dtwc::GpuPrecision::FP64, false);
  CHECK(gpu.labels == cpu.labels);
  CHECK(gpu.medoid_indices == cpu.medoid_indices);
  CHECK_THAT(gpu.total_cost, WithinRel(cpu.total_cost, 1e-12));
  // A view of the same series (a mapped Arrow IPC file's) leaves p_vec empty: the assignment reads its series
  // and medoids where they lie, so the owned run's labels, medoids and cost, bit for bit.
  const auto view = run(dtwc::Device::GPU, dtwc::GpuPrecision::FP64, false, true);
  CHECK(view.labels == gpu.labels);
  CHECK(view.medoid_indices == gpu.medoid_indices);
  CHECK(view.total_cost == gpu.total_cost);

  const auto cpu32 = run(dtwc::Device::CPU, dtwc::GpuPrecision::Auto, true);
  const auto gpu32 = run(dtwc::Device::GPU, dtwc::GpuPrecision::FP32, false);
  CHECK(gpu32.labels == cpu32.labels);
  CHECK(gpu32.medoid_indices == cpu32.medoid_indices);
  // Every point's distance agrees within the route bound of its pair, so the sum
  // agrees within the bound of the longest pair (2 * 60 - 1 samples).
  CHECK(dtwc::test_support::dtw_routes_agree<float>(gpu32.total_cost, cpu32.total_cost, 60, 60));
}

namespace {

/// HG0's fixtures, seeds fixed: 307 series, row i a benchmark_series draw (seed 20261109 + i) in [-1, 1) moved to 0, 4
/// or 8 by its group i % 3, of 17 + (i * 119) % 195 samples (`narrow` false: 17 to 211, every length once in a cycle
/// of 195 rows) or of 126 + i % 3 (`narrow`: 126 to 128, where a band of 2 leaves every pair a path).
std::vector<std::vector<double>> hg0_series(bool narrow)
{
  std::vector<std::vector<double>> series(307);
  for (std::size_t i = 0; i < series.size(); ++i) {
    series[i] = dtwc::test_support::benchmark_series(narrow ? 126 + i % 3 : 17 + (i * 119) % 195,
                                                     static_cast<unsigned>(20261109 + i));
    for (auto &v : series[i]) v += 4.0 * static_cast<double>(i % 3);
  }
  return series;
}

/// Three samples of three medoids, one of each group a sample: sample s's are slots 3s .. 3s + 2.
const std::vector<dtwc::index_t> hg0_medoids{ 3, 154, 290, 12, 100, 305, 0, 7, 8 };
constexpr std::size_t hg0_k = 3;

/// The fixtures' names, s0 .. s306.
const std::vector<std::string> hg0_names = [] {
  std::vector<std::string> names(307);
  for (std::size_t i = 0; i < names.size(); ++i) names[i] = "s" + std::to_string(i);
  return names;
}();

/// `prob` holding `series`, an owned copy of them or (`view`) a view of them, as a mapped file's (spans into `series`,
/// which outlives `prob`).
void hg0_configure(dtwc::Problem &prob, const std::vector<std::vector<double>> &series, int band,
                   dtwc::core::MetricType metric, dtwc::Device device, dtwc::GpuPrecision precision, bool view = false)
{
  if (view)
    prob.set_view_data(dtwc::Data(std::vector<std::span<const double>>(series.begin(), series.end()),
                                  std::vector<std::string_view>(hg0_names.begin(), hg0_names.end()), 1,
                                  std::make_shared<int>(0)));
  else
    prob.set_data(dtwc::Data(std::vector<std::vector<double>>(series), std::vector<std::string>(hg0_names)));
  prob.set_band(band);
  prob.set_metric(metric);
  prob.set_gpu_precision(precision);
  prob.set_device(device);
}

/// A pass's labels, each sample's nearest distances ([s * N + p]) and each sample's objective.
struct Hg0Pass {
  std::vector<std::vector<dtwc::index_t>> labels;
  std::vector<double> nearest;
  std::vector<double> costs;
};

Hg0Pass hg0_empty_pass(std::size_t n)
{
  const std::size_t samples = hg0_medoids.size() / hg0_k;
  return { std::vector<std::vector<dtwc::index_t>>(samples, std::vector<dtwc::index_t>(n)),
           std::vector<double>(samples * n), {} };
}

/// FastCLARA's CPU pass (fast_clara.cpp): assign_in_blocks over assign_points_cpu, the nearest distances taken from
/// each block's `best`.
Hg0Pass hg0_cpu_pass(dtwc::Problem &prob)
{
  const std::size_t n = prob.size(), samples = hg0_medoids.size() / hg0_k;
  Hg0Pass pass = hg0_empty_pass(n);
  std::vector<dtwc::core::detail::OrderedMedoidObjective> costs(samples);
  const auto &data = prob.data();
  const auto &pair = prob.dtw_function();
  dtwc::algorithms::detail::assign_in_blocks(
    0, n, costs, [&](dtwc::index_t first, std::size_t count, std::span<double> best) {
      dtwc::algorithms::detail::assign_points_cpu<double>(
        first, count, best, hg0_medoids, pass.labels, prob.distance(), pair,
        [&](dtwc::index_t p) { return data.series(static_cast<std::size_t>(p)); },
        [&](dtwc::index_t j) {
          return data.series(static_cast<std::size_t>(hg0_medoids[static_cast<std::size_t>(j)]));
        });
      for (std::size_t s = 0; s < samples; ++s)
        std::copy_n(best.data() + s * count, count, pass.nearest.data() + s * n + static_cast<std::size_t>(first));
    });
  for (const auto &cost : costs) pass.costs.push_back(cost.value());
  return pass;
}

/// Every distance of the medoid entry, [p * slots + j], on `prob`'s series and device settings, as assign_on_gpu asks
/// for them, within `staging` host bytes; `kernel` the kernel it ran and `blocks` the blocks it handed over, each
/// checked to follow the one before.
std::vector<double> hg0_gpu_distances(const dtwc::Problem &prob, std::size_t staging, std::string &kernel,
                                      std::size_t &blocks)
{
  const dtwc::Data &data = prob.data();
  std::vector<std::span<const double>> medoid_series;
  medoid_series.reserve(hg0_medoids.size());
  for (const dtwc::index_t m : hg0_medoids) medoid_series.push_back(data.series(static_cast<std::size_t>(m)));
  dtwc::cuda::CUDADistMatOptions opts;
  opts.device_id = prob.device().second;
  opts.precision = prob.gpu_precision();
  opts.band = prob.distance().band;
  opts.use_squared_l2 = prob.distance().metric == dtwc::core::MetricType::SquaredL2;
  opts.host_staging_bytes = staging;
  const std::size_t n = static_cast<std::size_t>(data.size()), slots = hg0_medoids.size();
  std::vector<double> d(n * slots);
  std::size_t next = 0;
  blocks = 0;
  kernel = dtwc::cuda::compute_medoid_distances_cuda(
             n, [&data](std::size_t i) { return data.series(i); }, medoid_series, opts,
             [](std::size_t count, const std::function<void(std::size_t)> &task) { dtwc::run_openmp(task, count); },
             [&](std::size_t first, std::size_t count, std::span<const double> block) {
               REQUIRE(first == next);
               REQUIRE(block.size() == count * slots);
               std::copy(block.begin(), block.end(), d.begin() + static_cast<std::ptrdiff_t>(first * slots));
               next = first + count;
               ++blocks;
             })
             .kernel_used;
  REQUIRE(next == n);
  return d;
}

/// assign_on_gpu's pass on `prob` (a GPU device) within `staging` host bytes: its labels and objectives, and each
/// label's distance from the medoid entry (0 at a medoid itself, as assign_point takes it).
Hg0Pass hg0_gpu_pass(const dtwc::Problem &prob, std::size_t staging, std::string &kernel, std::size_t &blocks)
{
  const dtwc::Data &data = prob.data();
  const std::size_t n = static_cast<std::size_t>(data.size()), samples = hg0_medoids.size() / hg0_k;
  const std::size_t slots = hg0_medoids.size();
  Hg0Pass pass = hg0_empty_pass(n);
  std::vector<std::span<const double>> medoid_series;
  medoid_series.reserve(hg0_medoids.size());
  for (const dtwc::index_t m : hg0_medoids) medoid_series.push_back(data.series(static_cast<std::size_t>(m)));
  std::vector<dtwc::core::detail::OrderedMedoidObjective> costs(samples);
  dtwc::algorithms::detail::assign_on_gpu(prob, data, 0, medoid_series, hg0_medoids, pass.labels, costs, staging);
  for (const auto &cost : costs) pass.costs.push_back(cost.value());
  const auto d = hg0_gpu_distances(prob, staging, kernel, blocks);
  for (std::size_t s = 0; s < samples; ++s)
    for (std::size_t p = 0; p < n; ++p) {
      const std::size_t slot = s * hg0_k + static_cast<std::size_t>(pass.labels[s][p]);
      pass.nearest[s * n + p] = static_cast<dtwc::index_t>(p) == hg0_medoids[slot] ? 0.0 : d[p * slots + slot];
    }
  return pass;
}

/// The Problem's distance matrix, filled on its device: its packed values.
std::vector<double> hg0_filled(dtwc::Problem &prob)
{
  prob.fill_distance_matrix();
  std::vector<double> values;
  prob.distance_matrix().visit([&](const auto *v) { values.assign(v, v + prob.distance_matrix().packed_count()); });
  return values;
}

bool same_bits(const std::vector<double> &a, const std::vector<double> &b)
{
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

} // anonymous namespace

// HG0, the bit-equality the hybrid assignment (HG3) rests on, as a regression test: on ragged, non-dyadic fixtures
// (307 series; 17 to 211 samples unbanded, 126 to 128 at band 2 and unbanded), FP64 L1, on one thread and on the
// default team, the GPU's assignment (assign_on_gpu) gives the CPU pass's labels, nearest distances and objectives
// (assign_in_blocks over assign_points_cpu, on the same medoids), and the GPU's fill the CPU fill's matrix, of owned
// series and of a view of them, byte for byte (memcmp, no tolerance). The CPU pass takes both its routes here: its
// lanes unbanded (dtw_lanes_packed), the per-pair function under the band, where the lanes take only points as long as
// the medoid. The kernel is named, so
// a mutation of its accumulation is known to reach the route that ran. The GPU's pipeline (HG1) runs one block of
// every series, blocks of 50 (a host budget of exactly their bytes, cuda::detail::medoid_blocks: two series slots,
// three distance slots, seven blocks) and blocks of one series (a budget of one byte: 307 blocks), the same bits.
TEST_CASE("HG0: the GPU's FP64 L1 assignment and fill equal the CPU's bit for bit", "[cuda][clara][hg0]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
  struct Shape {
    bool narrow;
    int band;
    const char *kernel;
    std::size_t max_L;
  };
  const auto shape = GENERATE(values<Shape>({ { false, -1, "regtile_w8", 211 }, { true, 2, "regtile_w4", 128 },
                                              { true, -1, "regtile_w4", 128 } }));
  const int threads = GENERATE(1, 0); // 0: the default team
  const std::size_t block = GENERATE(std::size_t{ 307 }, std::size_t{ 50 }, std::size_t{ 1 });
  CAPTURE(shape.narrow, shape.band, threads, block);
  std::optional<dtwc::TeamCap> cap;
  if (threads == 1) cap.emplace(1);
  const std::size_t blocks = (307 + block - 1) / block;
  const std::size_t staging =
    block == 307 ? 0
    : block == 1 ? 1
                 : dtwc::cuda::detail::MedoidBlocks{ block, 2, 3 }.host_bytes(hg0_medoids.size(), shape.max_L, 8);
  REQUIRE(dtwc::cuda::detail::medoid_blocks(307, hg0_medoids.size(), shape.max_L, 8, staging).block == block);

  const auto series = hg0_series(shape.narrow);
  dtwc::Problem cpu("hg0_cpu"), gpu("hg0_gpu");
  hg0_configure(cpu, series, shape.band, dtwc::core::MetricType::L1, dtwc::Device::CPU, dtwc::GpuPrecision::FP64);
  hg0_configure(gpu, series, shape.band, dtwc::core::MetricType::L1, dtwc::Device::GPU, dtwc::GpuPrecision::FP64);

  std::string kernel;
  std::size_t handed = 0;
  const Hg0Pass on_cpu = hg0_cpu_pass(cpu);
  const Hg0Pass on_gpu = hg0_gpu_pass(gpu, staging, kernel, handed);
  REQUIRE(kernel == shape.kernel);
  REQUIRE(handed == blocks);
  CHECK(on_gpu.labels == on_cpu.labels);
  CHECK(same_bits(on_gpu.nearest, on_cpu.nearest));
  CHECK(same_bits(on_gpu.costs, on_cpu.costs));
  if (block == 307) {
    const auto on_cpu_fill = hg0_filled(cpu);
    CHECK(same_bits(hg0_filled(gpu), on_cpu_fill));
    dtwc::Problem view("hg0_view"); // the fill reads a view's series where they lie (HG2), as it reads owned ones
    hg0_configure(view, series, shape.band, dtwc::core::MetricType::L1, dtwc::Device::GPU, dtwc::GpuPrecision::FP64,
                  true);
    REQUIRE(view.data().is_view());
    CHECK(same_bits(hg0_filled(view), on_cpu_fill));
  }
}

// What HG0's bit-equality does not cover, recorded as facts: on HG0's narrow fixture, the GPU's FP32 L1 distances
// (against the CPU's FP64) and its FP64 squared-L2 distances agree with the CPU's per-pair function within the route
// bound (dtw_route_bound.hpp) of each pair, and not bit for bit: on 2026-11-09 (cl 19.50 + nvcc 13.0, an RTX 4000
// Ada) 2754 of the 2763 FP32 distances differed at band 2 and unbanded, and 646 squared-L2 ones unbanded; at band 2
// none of the squared-L2 ones did, which is recorded here and not pinned.
TEST_CASE("HG0: the GPU's FP32 and squared-L2 distances differ from the CPU's within the route bound",
          "[cuda][clara][hg0]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
  struct Route {
    dtwc::GpuPrecision precision;
    dtwc::core::MetricType metric;
  };
  const auto route = GENERATE(values<Route>({ { dtwc::GpuPrecision::FP32, dtwc::core::MetricType::L1 },
                                              { dtwc::GpuPrecision::FP64, dtwc::core::MetricType::SquaredL2 } }));
  const int band = GENERATE(2, -1);
  CAPTURE(route.precision, route.metric, band);
  const auto series = hg0_series(true);
  dtwc::Problem cpu("hg0_cpu"), gpu("hg0_gpu");
  hg0_configure(cpu, series, band, route.metric, dtwc::Device::CPU, dtwc::GpuPrecision::FP64);
  hg0_configure(gpu, series, band, route.metric, dtwc::Device::GPU, route.precision);

  std::string kernel;
  std::size_t blocks = 0;
  const auto d = hg0_gpu_distances(gpu, 0, kernel, blocks);
  const auto &pair = cpu.dtw_function();
  const std::size_t slots = hg0_medoids.size();
  std::size_t outside = 0, differ = 0;
  for (std::size_t p = 0; p < series.size(); ++p)
    for (std::size_t j = 0; j < slots; ++j) {
      const auto &medoid = series[static_cast<std::size_t>(hg0_medoids[j])];
      const double on_cpu = pair(series[p], medoid), on_gpu = d[p * slots + j];
      const std::size_t nx = series[p].size(), ny = medoid.size();
      const bool agree = route.precision == dtwc::GpuPrecision::FP32
                           ? dtwc::test_support::dtw_routes_agree<float>(on_gpu, on_cpu, nx, ny)
                           : dtwc::test_support::dtw_routes_agree<double>(on_gpu, on_cpu, nx, ny);
      outside += !agree;
      differ += std::memcmp(&on_cpu, &on_gpu, sizeof(double)) != 0;
    }
  CAPTURE(differ);
  CHECK(outside == 0);
  if (route.metric == dtwc::core::MetricType::L1 || band < 0) CHECK(differ > 0);
}

namespace {

/// HG3's fixture: `n` series, more than one calibration block (clara_block_points), in three groups (moved to 0, 4
/// or 8 by i % 3) of uniform noise in [-0.5, 0.5), seed fixed; lengths 17 to 48 (`narrow` false, each length once in a
/// cycle of 32 rows) or 30 to 32 (`narrow`: a band of 2 leaves every pair a path).
std::vector<std::vector<double>> hg3_series(bool narrow, std::size_t n = 70000)
{
  std::mt19937 rng(20261109); // NOLINT(cert-msc51-cpp): fixed seed, reproducible.
  std::uniform_real_distribution<double> noise(-0.5, 0.5);
  std::vector<std::vector<double>> series(n);
  for (std::size_t i = 0; i < series.size(); ++i) {
    series[i].resize(narrow ? 30 + i % 3 : 17 + (i * 7) % 32);
    for (auto &v : series[i]) v = 4.0 * static_cast<double>(i % 3) + noise(rng);
  }
  return series;
}

/// FastCLARA (k 3, three samples of 40) on `series` on `device`, its GPU's host buffers within blocks of 8,192 series
/// (nine for the whole input), the GPU's share `share` on cpu+gpu.
dtwc::core::ClusteringResult hg3_run(const std::vector<std::vector<double>> &series, int band,
                                     dtwc::core::MetricType metric, dtwc::Device device, dtwc::GpuPrecision precision,
                                     double share)
{
  dtwc::Problem prob("hg3");
  prob.set_data(dtwc::Data(std::vector<std::vector<double>>(series), std::vector<std::string>(series.size())));
  prob.set_band(band);
  prob.set_metric(metric);
  prob.set_gpu_precision(precision);
  prob.set_device(device);
  dtwc::algorithms::CLARAOptions clara;
  clara.n_clusters = 3;
  clara.sample_size = 40;
  clara.n_samples = 3;
  clara.gpu_staging_bytes = dtwc::cuda::detail::MedoidBlocks{ 8192, 2, 3 }.host_bytes(9, 48, 8);
  clara.gpu_share = share;
  return dtwc::algorithms::fast_clara(prob, clara);
}

bool same_result(const dtwc::core::ClusteringResult &a, const dtwc::core::ClusteringResult &b)
{
  return a.labels == b.labels && a.medoid_indices == b.medoid_indices
         && std::memcmp(&a.total_cost, &b.total_cost, sizeof(double)) == 0;
}

} // anonymous namespace

// HG3, the hybrid's contract (DECISIONS 11-09): under FP64 L1 the GPU's distances are the CPU's bit for bit (HG0), so
// cpu+gpu's labels, medoids and cost equal the CPU-only and the GPU-only runs whatever the split: the GPU's share forced
// to 0, 0.5 and 1 and measured, compared by memcmp. Every shape (70,000 and 140,000 series, at least the 69,632 that
// hold both calibration blocks) runs the measured path: the GPU's block, the first 65,536 points, and the CPU's, the
// last 4,096, both stand, so the GPU's share lies between them. Forced at 0.5 the GPU's share is one call (fewer than
// 65,536 points at 70,000), at 1 its first block and then the rest. The CPU's lanes run unbanded, its per-pair
// function at band 2.
TEST_CASE("HG3: cpu+gpu's FP64 L1 FastCLARA equals the CPU's and the GPU's whatever the split", "[cuda][clara][hg3]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
  struct Shape {
    bool narrow;
    std::size_t n;
  };
  const auto shape = GENERATE(values<Shape>({ { false, 70000 }, { false, 140000 }, { true, 70000 } }));
  const bool narrow = shape.narrow;
  const int band = narrow ? 2 : -1;
  CAPTURE(narrow, band, shape.n);
  const auto series = hg3_series(narrow, shape.n);
  const auto run = [&](dtwc::Device device, double share) {
    return hg3_run(series, band, dtwc::core::MetricType::L1, device, dtwc::GpuPrecision::FP64, share);
  };
  const auto cpu = run(dtwc::Device::CPU, -1.0);
  const auto gpu = run(dtwc::Device::GPU, -1.0);
  REQUIRE(same_result(gpu, cpu));
  CHECK_FALSE(cpu.hybrid);
  for (const double share : { 0.0, -1.0, 0.5, 1.0 }) {
    CAPTURE(share);
    const auto hybrid = run(dtwc::Device::CPU_GPU, share);
    CHECK(same_result(hybrid, cpu));
    REQUIRE(hybrid.hybrid);
    if (share < 0) {
      CHECK(std::string_view(hybrid.hybrid->source) == "calibrated");
      CHECK((hybrid.hybrid->cpu_rate > 0 || dtwc::get_max_threads() == 1)); // a team of one leaves the CPU none
      CHECK(hybrid.hybrid->gpu_rate > 0);
      const double gpu_points = hybrid.hybrid->gpu_share * static_cast<double>(series.size());
      CHECK(gpu_points >= 65536.0);
      CHECK(gpu_points <= static_cast<double>(series.size() - 4096));
      std::cout << "HG3 FP64 L1 " << (narrow ? "band 2" : "unbanded") << " on " << series.size()
                << " series, measured: FastCLARA " << dtwc::algorithms::describe(*hybrid.hybrid, series.size()) << "\n";
    } else {
      CHECK(std::string_view(hybrid.hybrid->source) == "forced");
      CHECK(hybrid.hybrid->gpu_share == share);
    }
  }
  dtwc::TeamCap one(1); // the region of one thread: thread 0 drives the GPU's share, then runs every CPU item
  CHECK(same_result(run(dtwc::Device::CPU_GPU, 0.5), cpu));
}

// Where the GPU's distances are not the CPU's (FP32; squared L2, where the routes contract the product differently):
// cpu+gpu at share 1 is the GPU-only run byte for byte (the same sample fills, the same assignment), and a forced share
// repeats byte for byte; at a share of 0.5 and measured, the medoids are the GPU-only run's (the samples fill on the
// GPU), the cost lies within the route bound of the GPU-only run's, and a label differs only at a near-tie: where the
// point's CPU (FP64) distances to the two medoids lie within the route bound of each other. The near-ties are counted.
TEST_CASE("HG3: cpu+gpu under FP32 and squared L2 differs from the GPU only at near-ties", "[cuda][clara][hg3]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
  struct Route {
    dtwc::GpuPrecision precision;
    dtwc::core::MetricType metric;
  };
  const auto route = GENERATE(values<Route>({ { dtwc::GpuPrecision::FP32, dtwc::core::MetricType::L1 },
                                              { dtwc::GpuPrecision::FP64, dtwc::core::MetricType::SquaredL2 } }));
  const bool fp32 = route.precision == dtwc::GpuPrecision::FP32;
  CAPTURE(fp32);
  const auto series = hg3_series(false);
  const auto run = [&](dtwc::Device device, double share) {
    return hg3_run(series, -1, route.metric, device, route.precision, share);
  };
  const auto gpu = run(dtwc::Device::GPU, -1.0);
  CHECK(same_result(run(dtwc::Device::CPU_GPU, 1.0), gpu));
  const auto half = run(dtwc::Device::CPU_GPU, 0.5);
  CHECK(same_result(run(dtwc::Device::CPU_GPU, 0.5), half));

  dtwc::Problem cpu("hg3_cpu"); // the CPU's FP64 distance function on the same settings
  cpu.set_data(dtwc::Data(std::vector<std::vector<double>>(series), std::vector<std::string>(series.size())));
  cpu.set_metric(route.metric);
  const auto &pair = cpu.dtw_function();
  const auto agree = [fp32](double a, double b, std::size_t nx, std::size_t ny) {
    return fp32 ? dtwc::test_support::dtw_routes_agree<float>(a, b, nx, ny)
                : dtwc::test_support::dtw_routes_agree<double>(a, b, nx, ny);
  };
  for (const double share : { 0.5, -1.0 }) {
    CAPTURE(share);
    const auto hybrid = share == 0.5 ? half : run(dtwc::Device::CPU_GPU, share);
    REQUIRE(hybrid.medoid_indices == gpu.medoid_indices);
    CHECK(agree(hybrid.total_cost, gpu.total_cost, 48, 48));
    std::size_t near_ties = 0, other = 0;
    for (std::size_t p = 0; p < series.size(); ++p) {
      if (hybrid.labels[p] == gpu.labels[p]) continue;
      const auto &a = series[static_cast<std::size_t>(gpu.medoid_indices[static_cast<std::size_t>(hybrid.labels[p])])];
      const auto &b = series[static_cast<std::size_t>(gpu.medoid_indices[static_cast<std::size_t>(gpu.labels[p])])];
      const bool tie = agree(pair(series[p], a), pair(series[p], b), series[p].size(), std::max(a.size(), b.size()));
      near_ties += tie;
      other += !tie;
    }
    std::cout << "HG3 " << (fp32 ? "FP32 L1" : "FP64 squared L2") << " share " << share << ": " << near_ties
              << " labels differ from the GPU-only run, each at a near-tie\n";
    CAPTURE(near_ties);
    CHECK(other == 0);
  }
}


TEST_CASE("CUDA's launch workspace is its thread's scratch: made by a GPU fill, its pinned and device buffers "
          "given back by the release, as dtwc::run gives them back",
          "[cuda][thread_scratch]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }
  std::vector<std::vector<double>> series(8);
  for (std::size_t i = 0; i < series.size(); ++i)
    for (std::size_t t = 0; t < 300; ++t) series[i].push_back(std::sin(0.37 * double(i + 1) + 0.11 * double(t)));
  dtwc::core::release_thread_scratch();
  REQUIRE(dtwc::cuda::thread_workspace_bytes() == 0);
  const auto first = gpu_fill(series); // this thread's workspace stages the series, its device buffers beside
  // the series flattened, as floats where the GPU computes in FP32, and more on the device
  CHECK(dtwc::cuda::thread_workspace_bytes() > 8 * 300 * sizeof(float));
  dtwc::core::release_thread_scratch(); // every buffer freed
  CHECK(dtwc::cuda::thread_workspace_bytes() == 0);
  const auto again = gpu_fill(series); // made again
  CHECK(dtwc::cuda::thread_workspace_bytes() > 8 * 300 * sizeof(float));
  CHECK(again.matrix == first.matrix);
  // dtwc::run on the GPU leaves none.
  dtwc::Config config;
  config.output.clear();
  config.k = 2;
  config.method = dtwc::Method::PAM;
  config.device = dtwc::Device::GPU;
  config.band = -1;
  std::vector<std::string> names(series.size());
  for (std::size_t i = 0; i < names.size(); ++i) names[i] = "s" + std::to_string(i);
  auto copy = series;
  (void)dtwc::run(config, dtwc::Data(std::move(copy), std::move(names), 1));
  CHECK(dtwc::cuda::thread_workspace_bytes() == 0);
}

#endif // DTWC_HAS_CUDA
