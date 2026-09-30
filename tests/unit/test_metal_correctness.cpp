/**
 * @file test_metal_correctness.cpp
 * @brief Metal DTW correctness tests: compare GPU distance matrix against CPU reference.
 *
 * @details Mirrors test_cuda_correctness.cpp. FP32 rounding tolerance is
 *          looser than CUDA FP64 but still tight enough to catch kernel bugs.
 *
 * @date 2026-04-12
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>

#include "gpu_fixed_band_oracle.hpp"
#include "../support/deterministic_series.hpp"

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_METAL

TEST_CASE("Metal not available", "[metal]")
{
  SKIP("DTWC_HAS_METAL not defined; Metal tests skipped");
}

#else // DTWC_HAS_METAL

namespace {

namespace fixed_band = dtwc::test::gpu_fixed_band;

constexpr auto generate_random_series =
  &dtwc::test_support::accelerator_series_set;

std::vector<double> cpu_distance_matrix(
    const std::vector<std::vector<double>> &series)
{
  return dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [](const auto &left, const auto &right) {
      return dtwc::dtwFull_L<double>(left, right);
    });
}

std::vector<double> cpu_banded_matrix(
    const std::vector<std::vector<double>> &series, int band)
{
  return dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [band](const auto &left, const auto &right) {
      return dtwc::dtwBanded<double>(left, right, band);
    });
}

} // namespace

TEST_CASE("Metal backend is available", "[metal]")
{
  if (!dtwc::metal::metal_available()) {
    SKIP("Metal initialization failed on this machine: "
         << dtwc::metal::metal_device_info());
  }
  INFO("Metal device: " << dtwc::metal::metal_device_info());
  REQUIRE(dtwc::metal::metal_available());
}

TEST_CASE("Metal pairwise fixed-band routes match the independent F12 oracle",
          "[metal][banded][F12]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  // Registered before any real-Metal execution in
  // .claude/baselines/2026-07-24-f12-gpu-fixed-band-parity.md. The literal
  // costs and sentinel come from the test-only full-matrix oracle, never the
  // CPU rolling-buffer implementation.
  //
  // The principal pair alone takes the threadgroup wavefront. A 64-sample
  // filler makes bands 1-3 tight enough (band * 20 < max_L) for banded_row. A
  // 4096-sample filler overflows the 32 KB threadgroup memory of Apple GPUs, so
  // the unbounded band runs on the device-memory wavefront.
  struct Route {
    const char *kernel_name;
    std::size_t filler_length; // 0: no filler
    bool takes_narrow_bands;   // the rows with band <= 3
    bool takes_int_max;        // the row with band = INT_MAX
  };
  constexpr std::array routes{
      Route{"wavefront", 0, true, true},
      Route{"banded_row", 64, true, false},
      Route{"wavefront_global", 4096, false, true},
  };

  for (const auto &route : routes) {
    std::vector<std::vector<double>> series{
        fixed_band::principal_x(),
        fixed_band::principal_y(),
    };
    if (route.filler_length > 0)
      series.push_back(fixed_band::filler(route.filler_length));
    const std::size_t n = series.size();
    for (const bool squared : {false, true}) {
      for (const auto &row : fixed_band::ledger) {
        const bool int_max = row.band == std::numeric_limits<int>::max();
        if (int_max ? !route.takes_int_max : !route.takes_narrow_bands)
          continue;

        const double oracle = fixed_band::full_matrix_oracle(
            series[0], series[1], row.band, squared);
        const double registered = squared ? row.squared_l2 : row.l1;
        INFO("kernel=" << route.kernel_name << " band=" << row.band
             << " squared=" << squared);
        if (row.has_path) {
          REQUIRE_FALSE(std::isinf(oracle));
          REQUIRE(oracle == registered);
        } else {
          REQUIRE(std::isinf(oracle));
        }
        const double expected =
            row.has_path ? registered : fixed_band::public_no_path_sentinel;
        // The CPU kernel on the same inputs is the reference for Metal's
        // geometry and for its unreachable-value sentinel (F12, Metal half).
        const double cpu = dtwc::dtwBanded<double>(
            series[0], series[1], row.band, -1.0,
            squared ? dtwc::core::MetricType::SquaredL2
                    : dtwc::core::MetricType::L1);
        REQUIRE(cpu == expected);

        dtwc::metal::MetalDistMatOptions opts;
        opts.band = row.band;
        opts.use_squared_l2 = squared;
        const auto gpu =
            dtwc::metal::compute_distance_matrix_metal(series, opts);

        REQUIRE(gpu.kernel_used == route.kernel_name);
        REQUIRE(gpu.n == n);
        REQUIRE(gpu.pairs_computed == n * (n - 1) / 2);
        REQUIRE(gpu.matrix.size() == n * n);
        REQUIRE(gpu.matrix[0] == 0.0);
        REQUIRE(gpu.matrix[n + 1] == 0.0);
        REQUIRE(gpu.matrix[1] == cpu);
        REQUIRE(gpu.matrix[n] == cpu);
      }
    }
  }
}

TEST_CASE("Metal unbanded DTW matches CPU on small random series", "[metal]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  const size_t N = 8;
  const size_t L = 32;
  auto series = generate_random_series(N, L, 42);

  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = -1;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  REQUIRE(gpu.n == N);
  REQUIRE(gpu.matrix.size() == N * N);

  // FP32 accumulation over L iterations -> relative tolerance ~1e-5.
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-4) || WithinAbs(cpu[i * N + j], 1e-3));
    }
    CAPTURE(i);
    REQUIRE(gpu.matrix[i * N + i] == 0.0); // diagonal
  }
}

TEST_CASE("Metal handles N=2 smallest possible matrix", "[metal]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  std::vector<std::vector<double>> series = {
      {1.0, 2.0, 3.0, 4.0, 5.0},
      {2.0, 3.0, 4.0, 5.0, 6.0},
  };

  dtwc::metal::MetalDistMatOptions opts;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  REQUIRE(gpu.n == 2);
  // Two shifted ramps: DTW L1 distance should be small (aligned by warping).
  double d_cpu = dtwc::dtwFull_L<double>(series[0], series[1]);
  REQUIRE_THAT(gpu.matrix[0 * 2 + 1], WithinRel(d_cpu, 1e-4) || WithinAbs(d_cpu, 1e-3));
  REQUIRE(gpu.matrix[1 * 2 + 0] == gpu.matrix[0 * 2 + 1]); // symmetric
}

TEST_CASE("Metal unbanded DTW on longer series", "[metal]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  const size_t N = 5;
  const size_t L = 200;
  auto series = generate_random_series(N, L, 7);

  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  REQUIRE(gpu.n == N);
  // With L=200 and ~O(L^2) FP32 ops per pair, accumulated error can be
  // larger — widen tolerance slightly.
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
    }
  }
}

TEST_CASE("Metal regtile_w4 kernel matches CPU for max_L <= 128", "[metal][regtile]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // Three lengths covering the TILE_W=4 lane-coverage range (32 * 4 = 128).
  // N=32 stresses the 8-pair-per-threadgroup tile layout (multiple full tgs).
  for (size_t L : {size_t{64}, size_t{100}, size_t{128}}) {
    const size_t N = 32;
    auto series = generate_random_series(N, L, 0xA11C0DE);
    auto cpu = cpu_distance_matrix(series);

    dtwc::metal::MetalDistMatOptions opts;
    opts.band = -1;
    auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

    INFO("L=" << L << " kernel_used=" << gpu.kernel_used);
    REQUIRE(gpu.kernel_used == "regtile_w4");
    REQUIRE(gpu.n == N);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        CAPTURE(L, i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
        REQUIRE_THAT(gpu.matrix[i * N + j],
                     WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
      }
      REQUIRE(gpu.matrix[i * N + i] == 0.0);
    }
  }
}

TEST_CASE("Metal regtile_w8 kernel matches CPU for max_L in (128, 256]", "[metal][regtile]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  for (size_t L : {size_t{192}, size_t{256}}) {
    const size_t N = 16;
    auto series = generate_random_series(N, L, 0xBE57CAFE);
    auto cpu = cpu_distance_matrix(series);

    dtwc::metal::MetalDistMatOptions opts;
    opts.band = -1;
    auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

    INFO("L=" << L << " kernel_used=" << gpu.kernel_used);
    REQUIRE(gpu.kernel_used == "regtile_w8");
    REQUIRE(gpu.n == N);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        CAPTURE(L, i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
        REQUIRE_THAT(gpu.matrix[i * N + j],
                     WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
      }
    }
  }
}

TEST_CASE("Metal regtile handles uneven tile cover (N_len not multiple of TILE_W)", "[metal][regtile]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // L=127 leaves a partial w4 tile at the end (32 threads * 4 - 127 = 1 unused slot).
  // L=255 leaves a partial w8 tile (32*8 - 255 = 1 unused slot).
  for (size_t L : {size_t{127}, size_t{255}}) {
    const size_t N = 6;
    auto series = generate_random_series(N, L, 0xDEADBEEF);
    auto cpu = cpu_distance_matrix(series);

    dtwc::metal::MetalDistMatOptions opts;
    auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
    INFO("L=" << L << " kernel_used=" << gpu.kernel_used);
    REQUIRE((gpu.kernel_used == "regtile_w4" || gpu.kernel_used == "regtile_w8"));
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        CAPTURE(L, i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
        REQUIRE_THAT(gpu.matrix[i * N + j],
                     WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
      }
    }
  }
}

TEST_CASE("Metal regtile with asymmetric (variable-length) series", "[metal][regtile]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // Variable lengths force the kernel's rows=short / cols=long orientation logic
  // to matter. All lengths stay within max_L=128 so regtile_w4 fires.
  std::mt19937 rng(0xCAFEBABE);
  std::uniform_int_distribution<int> len_dist(32, 120);
  std::uniform_real_distribution<double> val_dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  for (auto &s : series) {
    const int L = len_dist(rng);
    s.resize(L);
    for (auto &v : s) v = val_dist(rng);
  }

  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
  INFO("kernel_used=" << gpu.kernel_used);
  REQUIRE(gpu.kernel_used == "regtile_w4");
  const size_t N = series.size();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, series[i].size(), series[j].size(),
              cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
    }
  }
}

TEST_CASE("Metal pushes max_L towards the threadgroup memory cap", "[metal]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // Longest length that fits in the 32 KB cap: 3 * L * 4 <= 32768 -> L <= 2730.
  // Pick 2000 to stay comfortably inside the cap.
  const size_t N = 3;
  const size_t L = 2000;
  auto series = generate_random_series(N, L, 123);

  dtwc::metal::MetalDistMatOptions opts;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  REQUIRE(gpu.n == N);
  REQUIRE(gpu.pairs_computed == N * (N - 1) / 2);
  // Sanity: all off-diagonals are positive and finite; diagonals are zero.
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i == j) REQUIRE(gpu.matrix[i * N + j] == 0.0);
      else REQUIRE(gpu.matrix[i * N + j] > 0.0);
    }
  }
}

TEST_CASE("Metal row-major banded kernel matches CPU banded reference", "[metal][banded]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  // Mid-length so we can run in a single correctness pass.
  //   band*4 = 20 < L = 400  -> row-major banded dispatcher fires.
  const size_t N = 6;
  const size_t L = 400;
  const int band = 5;
  auto series = generate_random_series(N, L, 2024);

  auto cpu = cpu_banded_matrix(series, band);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  INFO("kernel_used=" << gpu.kernel_used);
  REQUIRE(gpu.kernel_used == "banded_row");
  REQUIRE(gpu.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
    }
  }
}

TEST_CASE("Metal row-major banded kernel handles long series tight band", "[metal][banded]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  // 2730-cap wavefront would still fire here (L > 2730); tight band triggers
  // row-major path regardless. Small N keeps the correctness test fast.
  const size_t N = 3;
  const size_t L = 5000;
  const int band = 50;
  auto series = generate_random_series(N, L, 99);

  auto cpu = cpu_banded_matrix(series, band);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  INFO("kernel_used=" << gpu.kernel_used);
  REQUIRE(gpu.kernel_used == "banded_row");

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1.0));
    }
  }
}

TEST_CASE("Metal wide band still routes to wavefront kernel", "[metal][banded]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // band*4 >= L  -> stays on wavefront, not banded-row.
  const size_t N = 4;
  const size_t L = 200;
  const int band = 100; // band*4 = 400 >= L=200 -> wavefront
  auto series = generate_random_series(N, L, 33);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
  INFO("kernel_used=" << gpu.kernel_used);
  REQUIRE(gpu.kernel_used == "wavefront");
}

TEST_CASE("Metal wavefront NxN banded matches CPU dtwBanded", "[metal][banded]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // L=200, band=20: band*20=400 > L=200 so NOT banded-row -> wavefront path.
  const size_t N = 4;
  const size_t L = 200;
  const int band = 20;
  auto series = generate_random_series(N, L, 919);

  auto cpu = cpu_banded_matrix(series, band);
  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
  INFO("kernel_used=" << gpu.kernel_used);
  REQUIRE(gpu.kernel_used == "wavefront");
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
    }
  }
}

TEST_CASE("Metal handles long series via global-memory kernel", "[metal]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // 10 000 > 2730 (threadgroup cap), so the dispatcher must pick the global
  // kernel. Verify correctness against CPU on a small N to keep the test fast.
  const size_t N = 3;
  const size_t L = 10000;
  auto series = generate_random_series(N, L, 456);

  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  opts.verbose = false;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
  REQUIRE(gpu.n == N);
  REQUIRE(gpu.pairs_computed == N * (N - 1) / 2);

  // FP32 accumulation over L=10000 -> loose tolerance (relative 1e-3, absolute 1e-1).
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1.0));
    }
  }
}

#endif // DTWC_HAS_METAL
