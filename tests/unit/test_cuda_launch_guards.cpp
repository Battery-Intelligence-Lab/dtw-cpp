/**
 * @file test_cuda_launch_guards.cpp
 * @brief A15/A16 contract: CUDA launch preconditions are typed, loud, and
 *        assertable without a GPU.
 *
 * The host seam (`cuda/launch_prep.hpp`) is deliberately free of CUDA headers,
 * so these cases compile and RUN in the CUDA-OFF canonical gate as well as in
 * build/cuda-verify. Two properties are under test:
 *
 *   A15 — the pair count `N*(N-1)/2` is evaluated in 64 bits and never
 *         narrowed: a fill splits it into launches of at most
 *         kMaxPairsPerLaunch pairs, a count an int holds. N = 65,537, the first
 *         N whose pair count passes INT_MAX, fills (test_cuda_correctness).
 *   A16 — a missing CUDA device is a typed DeviceError, never an N*N matrix of
 *         zeros (a valid-looking wrong answer). The entry-point case runs only
 *         when the process sees no device: on a GPU host, execute this binary
 *         with CUDA_VISIBLE_DEVICES=-1 to exercise it.
 *
 * The automatic kernel choice (`cuda/kernel_selection.hpp`, also free of CUDA
 * headers) is pinned here too, so a build without a GPU checks its ranges.
 */

#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("CUDA kernel choice follows the longest series length and the shared memory",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using dtwc::cuda::detail::KernelPath;
  using dtwc::cuda::detail::kernel_path_name;
  using dtwc::cuda::detail::select_kernel;

  // Each range's kernel beats every other kernel that accepts the range by at
  // least 15 % on the RTX 4000 Ada (.claude/baselines/2026-09-29-w4a-cuda-kernel-ab.md).
  // That device gives a wavefront block 101,376 bytes of shared memory, 16 of
  // them the kernel's own: a block's anti-diagonals (three of L samples above
  // L = 2048) leave shared memory for global memory from FP32 L = 8447 and
  // FP64 L = 4224.
  constexpr std::size_t shared = 101376 - 16;
  for (const std::size_t sample : { std::size_t{ 4 }, std::size_t{ 8 } }) {
    CHECK(select_kernel(1, sample, shared) == KernelPath::Warp);
    CHECK(select_kernel(32, sample, shared) == KernelPath::Warp);
    CHECK(select_kernel(33, sample, shared) == KernelPath::RegTileW4);
    CHECK(select_kernel(128, sample, shared) == KernelPath::RegTileW4);
    CHECK(select_kernel(129, sample, shared) == KernelPath::RegTileW8);
    CHECK(select_kernel(256, sample, shared) == KernelPath::RegTileW8);
    CHECK(select_kernel(257, sample, shared) == KernelPath::Wavefront);
    CHECK(select_kernel(2048, sample, shared) == KernelPath::Wavefront);
    CHECK(select_kernel(2049, sample, shared) == KernelPath::Wavefront);
  }
  CHECK(select_kernel(8446, 4, shared) == KernelPath::Wavefront);
  CHECK(select_kernel(8447, 4, shared) == KernelPath::WavefrontGlobal);
  CHECK(select_kernel(4223, 8, shared) == KernelPath::Wavefront);
  CHECK(select_kernel(4224, 8, shared) == KernelPath::WavefrontGlobal);
  CHECK(select_kernel(100000000, 4, shared) == KernelPath::WavefrontGlobal);
  CHECK(kernel_path_name(KernelPath::Warp) == "warp");
  CHECK(kernel_path_name(KernelPath::RegTileW4) == "regtile_w4");
  CHECK(kernel_path_name(KernelPath::RegTileW8) == "regtile_w8");
  CHECK(kernel_path_name(KernelPath::Wavefront) == "wavefront");
  CHECK(kernel_path_name(KernelPath::WavefrontGlobal) == "wavefront_global");
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
