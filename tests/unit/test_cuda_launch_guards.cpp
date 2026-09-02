/**
 * @file test_cuda_launch_guards.cpp
 * @brief A15/A16 contract: CUDA launch preconditions are typed, loud, and
 *        assertable without a GPU.
 *
 * The host seam (`cuda/launch_prep.hpp`) is deliberately free of CUDA headers,
 * so these cases compile and RUN in the CUDA-OFF canonical gate as well as in
 * build/cuda-verify. Two properties are under test:
 *
 *   A15 — a pair count above INT_MAX is rejected before anything is allocated
 *         or narrowed. `N*(N-1)/2` must be evaluated in 64 bits: the smallest
 *         failing N is 65537, whose N*N result matrix (34 GB) must never be
 *         allocated in order to discover the problem.
 *   A16 — a missing CUDA device is a typed DeviceError, never an N*N matrix of
 *         zeros (a valid-looking wrong answer). The entry-point case runs only
 *         when the process sees no device: on a GPU host, execute this binary
 *         with CUDA_VISIBLE_DEVICES=-1 to exercise it.
 */

#include <catch2/catch_test_macros.hpp>

#include <dtwc.hpp>
#include <error.hpp>

#if __has_include(<cuda/launch_prep.hpp>)
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

TEST_CASE("A15 CUDA pair-count guard is host-testable and 64-bit",
          "[cuda][launch_guard][host]")
{
#if !DTWC_HAS_CUDA_LAUNCH_PREP_SEAM
  FAIL("CUDA launch preconditions must be exposed through a host-testable seam");
#else
  using dtwc::cuda::detail::kMaxPairsPerLaunch;
  using dtwc::cuda::detail::require_pair_count_fits;
  using dtwc::cuda::detail::upper_triangle_pairs;

  constexpr std::size_t int_max =
    static_cast<std::size_t>(std::numeric_limits<int>::max());
  REQUIRE(kMaxPairsPerLaunch == int_max);

  SECTION("the count itself never overflows") {
    // 65536 is the last N that fits; 65537 is the first that does not.
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

  SECTION("the guard admits what fits and rejects what does not") {
    REQUIRE_NOTHROW(require_pair_count_fits(upper_triangle_pairs(65536), "probe"));
    REQUIRE_NOTHROW(require_pair_count_fits(kMaxPairsPerLaunch, "probe"));
    REQUIRE_THROWS_AS(require_pair_count_fits(kMaxPairsPerLaunch + 1, "probe"),
                      dtwc::InvalidInput);
    REQUIRE_THROWS_AS(require_pair_count_fits(upper_triangle_pairs(65537), "probe"),
                      dtwc::InvalidInput);
    // Every dtwc error stays catchable through the standard hierarchy.
    REQUIRE_THROWS_AS(require_pair_count_fits(upper_triangle_pairs(100000000u), "probe"),
                      std::runtime_error);
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

#ifdef DTWC_HAS_CUDA

TEST_CASE("A15 CUDA entry points reject an over-large N before allocating",
          "[cuda][launch_guard][device]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device");
    return;
  }

  // 65537 length-1 series: a few MB of host memory. Before the guard the first
  // act of these entry points was to size an N*N matrix (34 GB) or to narrow
  // the pair count to a negative int inside the LB_Keogh pre-pass.
  const std::vector<std::vector<double>> series(65537, std::vector<double>{ 1.0 });
  REQUIRE(dtwc::cuda::detail::upper_triangle_pairs(series.size())
          > dtwc::cuda::detail::kMaxPairsPerLaunch);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = 4;
  opts.use_lb_keogh = true; // the pre-pass is what used to truncate first

  REQUIRE_THROWS_AS(dtwc::cuda::compute_distance_matrix_cuda(series, opts),
                    dtwc::InvalidInput);
  REQUIRE_THROWS_AS(dtwc::cuda::compute_lb_keogh_cuda(series, 4),
                    dtwc::InvalidInput);
}

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
  const std::vector<double> query{ 1.0, 2.0, 3.0 };

  REQUIRE_THROWS_AS(dtwc::cuda::compute_distance_matrix_cuda(series, {}),
                    dtwc::DeviceError);
  REQUIRE_THROWS_AS(dtwc::cuda::compute_lb_keogh_cuda(series, 1),
                    dtwc::DeviceError);
  REQUIRE_THROWS_AS(dtwc::cuda::compute_dtw_one_vs_all(series, 0, {}),
                    dtwc::DeviceError);
  REQUIRE_THROWS_AS(dtwc::cuda::compute_dtw_one_vs_all(query, series, {}),
                    dtwc::DeviceError);
  REQUIRE_THROWS_AS(dtwc::cuda::compute_dtw_k_vs_all(series, { 0 }, {}),
                    dtwc::DeviceError);
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
  const auto result = dtwc::cuda::compute_distance_matrix_cuda(series, {});
  REQUIRE(result.n == 3u);
  REQUIRE(result.matrix.size() == 9u);
  CHECK(result.kernel_used != "none");
  CHECK(result.matrix[1] > 0.0);
}

#else

TEST_CASE("CUDA launch-guard device cases require a CUDA build",
          "[cuda][launch_guard][device]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA device launch-guard cases skipped");
}

#endif // DTWC_HAS_CUDA
