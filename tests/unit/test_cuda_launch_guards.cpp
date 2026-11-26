/**
 * @file test_cuda_launch_guards.cpp
 * @brief CUDA launch preconditions are typed, loud, and
 *        assertable without a GPU.
 *
 * The host seam (`cuda/launch_prep.hpp`) is deliberately free of CUDA headers,
 * so these cases compile and RUN in the CUDA-OFF canonical gate as well as in
 * build/cuda-verify. Two properties are under test:
 *
 *   PAIRS   — the pair count `N*(N-1)/2` is evaluated in 64 bits and never
 *         narrowed: a fill splits it into launches of at most
 *         kMaxPairsPerLaunch pairs, a count an int holds. N = 65,537, the first
 *         N whose pair count passes INT_MAX, fills (test_cuda_correctness).
 *   DEVICE  — a missing CUDA device is a typed DeviceError, never an N*N matrix of
 *         zeros (a valid-looking wrong answer). The entry-point case runs only
 *         when the process sees no device: on a GPU host, execute this binary
 *         with CUDA_VISIBLE_DEVICES=-1 to exercise it.
 *
 * The automatic kernel choice (`cuda/kernel_selection.hpp`, also free of CUDA
 * headers) is pinned here too, so a build without a GPU checks its ranges.
 * @author Volkan Kumtepeli
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <dtwc.hpp>
#include <base/error.hpp>

#if __has_include(<cuda/launch_prep.hpp>)
#include <cuda/kernel_selection.hpp>
#include <cuda/launch_prep.hpp>
#define DTWC_HAS_CUDA_LAUNCH_PREP_SEAM 1
#else
#define DTWC_HAS_CUDA_LAUNCH_PREP_SEAM 0
#endif
#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

TEST_CASE("A15 CUDA pair count is 64-bit and launches split it into int counts",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using dtwc::cuda::detail::kMaxPairsPerLaunch;
  using dtwc::cuda::detail::upper_triangle_pairs;

  constexpr std::size_t int_max =
    static_cast<std::size_t>(std::numeric_limits<int>::max());
  // A launch counts its pairs in int, and the persistent wavefront's counter
  // overshoots the count by up to one per block: room to spare.
  STATIC_REQUIRE(kMaxPairsPerLaunch > 0);
  STATIC_REQUIRE(static_cast<std::size_t>(kMaxPairsPerLaunch) <= int_max / 2);

  SECTION("the count itself never overflows") {
    // 65536 is the last N whose pair count fits an int; 65537 the first that does not.
    REQUIRE(upper_triangle_pairs(0) == 0u);
    REQUIRE(upper_triangle_pairs(1) == 0u);
    REQUIRE(upper_triangle_pairs(2) == 1u);
    REQUIRE(upper_triangle_pairs(65536) == 2147450880u);
    REQUIRE(upper_triangle_pairs(65536) <= int_max);
    REQUIRE(upper_triangle_pairs(65537) == 2147516416u);
    REQUIRE(upper_triangle_pairs(65537) > int_max);
    // The 100M-series target: only a 64-bit count can represent this.
    REQUIRE(upper_triangle_pairs(100000000u) == 4999999950000000u);
  }
#endif
}

TEST_CASE("A16 missing CUDA device is a typed error, never a zero matrix",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using dtwc::cuda::detail::require_cuda_device;

  REQUIRE_NOTHROW(require_cuda_device(true, "probe"));
  REQUIRE_THROWS_AS(require_cuda_device(false, "probe"), dtwc::DeviceError);
  // DeviceError, not InvalidInput: the input was fine, the device was not.
  REQUIRE_THROWS_AS(require_cuda_device(false, "probe"), dtwc::Error);
#endif
}

// The kernels are built for compute capability 8.0 and newer (Ampere, 2021); an
// older GPU is refused with the typed error when the fill first reads the device,
// before anything is allocated, instead of failing at its first launch.
TEST_CASE("CUDA refuses a GPU older than compute capability 8.0",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using Catch::Matchers::ContainsSubstring;
  using Catch::Matchers::MessageMatches;
  using dtwc::cuda::detail::require_compute_capability;

  REQUIRE_NOTHROW(require_compute_capability(8, 0, 0));  // A30, A100
  REQUIRE_NOTHROW(require_compute_capability(8, 6, 0));  // RTX 30
  REQUIRE_NOTHROW(require_compute_capability(8, 9, 0));  // RTX 4000 Ada, L40S
  REQUIRE_NOTHROW(require_compute_capability(9, 0, 0));  // H100
  REQUIRE_NOTHROW(require_compute_capability(12, 0, 0)); // RTX 50
  REQUIRE_THROWS_MATCHES(
      require_compute_capability(7, 5, 1), dtwc::DeviceError,
      MessageMatches(ContainsSubstring("CUDA device 1 has compute capability 7.5")
                     && ContainsSubstring("needs 8.0 or newer")));
  REQUIRE_THROWS_AS(require_compute_capability(6, 0, 0), dtwc::DeviceError);
#endif
}

TEST_CASE("CUDA kernel choice follows the longest series length, pair count, and shared memory",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using dtwc::cuda::detail::KernelPath;
  using dtwc::cuda::detail::kernel_path_name;
  using dtwc::cuda::detail::select_kernel;

  constexpr int ada_sm = 48;
  constexpr std::size_t ada_block_shared = 101376;
  constexpr std::size_t hopper_block_shared = 232448;
  constexpr std::size_t overhead = 32 + 1024;
  const std::size_t many_pairs = std::size_t{ 8 } * static_cast<std::size_t>(ada_sm);
  const std::size_t few_pairs = 10;

  // High pair-count batches (and all batches with max_L <= 1024) take the unified warptile kernel:
  // TILE_W=1 for L <= 32, TILE_W=4 for 32 < L <= 128 or FP64, TILE_W=8 for FP32 L > 128.
  for (const std::size_t L : { std::size_t{ 1 }, std::size_t{ 32 } }) {
    CHECK(select_kernel(L, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW1);
    CHECK(select_kernel(L, 8, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW1);
  }
  for (const std::size_t L : { std::size_t{ 33 }, std::size_t{ 128 } }) {
    CHECK(select_kernel(L, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW4);
    CHECK(select_kernel(L, 8, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW4);
  }
  for (const std::size_t L : { std::size_t{ 129 }, std::size_t{ 256 }, std::size_t{ 257 }, std::size_t{ 1024 } }) {
    CHECK(select_kernel(L, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW8);
    CHECK(select_kernel(L, 8, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW4);
  }
  CHECK(select_kernel(2048, 4, many_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW8);
  CHECK(select_kernel(2048, 8, many_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WarpTileW4);

  // Low pair-count batches with L > 1024 (or when border buffer exceeds VRAM) fall back to wavefront.
  CHECK(select_kernel(1025, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::Wavefront);
  CHECK(select_kernel(4096, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::Wavefront);
  CHECK(select_kernel(4096, 8, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::Wavefront);
  CHECK(select_kernel(8360, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::Wavefront);
  CHECK(select_kernel(8361, 4, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WavefrontGlobal);
  CHECK(select_kernel(4180, 8, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::Wavefront);
  CHECK(select_kernel(4181, 8, few_pairs, ada_sm, true, ada_block_shared, overhead) == KernelPath::WavefrontGlobal);
  CHECK(select_kernel(100000000, 4, many_pairs, ada_sm, false, ada_block_shared, overhead)
        == KernelPath::WavefrontGlobal);
  CHECK(select_kernel(19280, 4, few_pairs, ada_sm, true, hopper_block_shared, overhead) == KernelPath::Wavefront);
  CHECK(select_kernel(19285, 4, few_pairs, ada_sm, true, hopper_block_shared, overhead) == KernelPath::WavefrontGlobal);

  CHECK(kernel_path_name(KernelPath::WarpTileW1) == "warptile_w1");
  CHECK(kernel_path_name(KernelPath::WarpTileW4) == "warptile_w4");
  CHECK(kernel_path_name(KernelPath::WarpTileW8) == "warptile_w8");
  CHECK(kernel_path_name(KernelPath::Wavefront) == "wavefront");
  CHECK(kernel_path_name(KernelPath::WavefrontGlobal) == "wavefront_global");
#endif
}

TEST_CASE("CUDA medoid_blocks caps double-buffered device output slots at kMaxPairsPerLaunch",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using dtwc::cuda::detail::kMaxPairsPerLaunch;
  using dtwc::cuda::detail::medoid_blocks;

  constexpr auto max_slots = static_cast<std::size_t>(kMaxPairsPerLaunch);

  // Single-block launch uses 1 series slot and fits within kMaxPairsPerLaunch.
  const auto single = medoid_blocks(64, 8, 32, sizeof(double));
  CHECK(single.block == 64u);
  CHECK(single.series_slots == 1u);
  CHECK(single.series_slots * single.block * 8u <= max_slots);

  // Multi-block launches use 2 series slots (double-buffered d_out = 2 * block * k);
  // total device slots must never exceed kMaxPairsPerLaunch.
  for (const auto &[n, k, max_L] : {
         std::tuple<std::size_t, std::size_t, std::size_t>{ 200000u, 1024u, 64u },
         std::tuple<std::size_t, std::size_t, std::size_t>{ 600000u, 256u, 128u },
         std::tuple<std::size_t, std::size_t, std::size_t>{ 2000000u, 128u, 512u } }) {
    const auto mb = medoid_blocks(n, k, max_L, sizeof(double));
    CAPTURE(n, k, max_L, mb.block, mb.series_slots, mb.distance_slots);
    CHECK(mb.series_slots == 2u);
    CHECK(mb.series_slots * mb.block * k <= max_slots);
  }
#endif
}

#ifdef DTWC_HAS_CUDA

TEST_CASE("A16 CUDA entry points refuse to answer without a device",
          "[cuda][launch_guard][no_device]")
{
  // Runs only when the process sees no device. On a GPU host, execute it with
  // CUDA_VISIBLE_DEVICES=-1 — that is the only way to exercise the
  // "device missing" contract on hardware that has one.
  if (dtwc::cuda::cuda_available()) {
    SKIP("A CUDA device is present; rerun with CUDA_VISIBLE_DEVICES=-1 to exercise this");
    return;
  }

  const std::vector<std::vector<double>> series{ { 1.0, 2.0, 3.0 },
                                                 { 2.0, 3.0, 4.0 } };

  dtwc::core::DistanceMatrix out;
  REQUIRE_THROWS_AS(dtwc::cuda::compute_distance_matrix_cuda(series, {}, out),
                    dtwc::DeviceError);
  CHECK(out.size() == 0);
}

TEST_CASE("A16 CUDA entry points still answer normally on a real device",
          "[cuda][launch_guard][device]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device");
    return;
  }

  const std::vector<std::vector<double>> series{
    { 1.0, 2.0, 3.0, 4.0 }, { 1.0, 2.0, 3.0, 5.0 }, { 4.0, 3.0, 2.0, 1.0 }
  };
  dtwc::core::DistanceMatrix out;
  const auto result = dtwc::cuda::compute_distance_matrix_cuda(series, {}, out);
  REQUIRE(result.n == 3u);
  REQUIRE(out.size() == 3u);
  CHECK(result.kernel_used != "none");
  CHECK(out.get(0, 1) > 0.0);
}

#else

TEST_CASE("CUDA launch-guard device cases require a CUDA build",
          "[cuda][launch_guard][device]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA device launch-guard cases skipped");
}

#endif // DTWC_HAS_CUDA
