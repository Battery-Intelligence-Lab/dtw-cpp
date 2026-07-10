/**
 * @file test_cuda_kernel_override.cpp
 * @brief M50 contract tests for CUDA kernel selection and truthful reporting.
 *
 * The host contract is intentionally independent of the CUDA toolchain.  The
 * real-device gate verifies dispatch through the result discriminator rather
 * than timing, and compares every forced/fallback result with a CPU oracle.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>
#include <enums/KernelOverride.hpp>

#if __has_include(<cuda/kernel_selection.hpp>)
#include <cuda/kernel_selection.hpp>
#define DTWC_HAS_CUDA_KERNEL_SELECTION_SEAM 1
#else
#define DTWC_HAS_CUDA_KERNEL_SELECTION_SEAM 0
#endif
#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

TEST_CASE("M50 CUDA kernel selection is host-testable and truthful",
          "[cuda][kernel_override][host][m50]")
{
#if !DTWC_HAS_CUDA_KERNEL_SELECTION_SEAM
  FAIL("CUDA kernel selection must be exposed through a host-testable seam");
#else
  using dtwc::KernelOverride;
  using dtwc::cuda::detail::KernelPath;
  using dtwc::cuda::detail::kernel_path_name;
  using dtwc::cuda::detail::select_kernel;

  struct SelectionCase {
    std::size_t max_length;
    KernelOverride requested;
    KernelPath expected;
    bool fell_back_to_auto;
  };

  const std::array auto_cases{
      SelectionCase{1, KernelOverride::Auto, KernelPath::Warp, false},
      SelectionCase{32, KernelOverride::Auto, KernelPath::Warp, false},
      SelectionCase{33, KernelOverride::Auto, KernelPath::RegTileW4, false},
      SelectionCase{128, KernelOverride::Auto, KernelPath::RegTileW4, false},
      SelectionCase{129, KernelOverride::Auto, KernelPath::RegTileW8, false},
      SelectionCase{256, KernelOverride::Auto, KernelPath::RegTileW8, false},
      SelectionCase{257, KernelOverride::Auto, KernelPath::Wavefront, false},
  };

  const std::array override_cases{
      // CUDA has a shared-memory wavefront implementation at every length.
      SelectionCase{16, KernelOverride::Wavefront, KernelPath::Wavefront, false},
      SelectionCase{256, KernelOverride::Wavefront, KernelPath::Wavefront, false},
      SelectionCase{1024, KernelOverride::Wavefront, KernelPath::Wavefront, false},

      // CUDA register tiling is implemented only through length 256.
      SelectionCase{16, KernelOverride::RegTile, KernelPath::RegTileW4, false},
      SelectionCase{128, KernelOverride::RegTile, KernelPath::RegTileW4, false},
      SelectionCase{129, KernelOverride::RegTile, KernelPath::RegTileW8, false},
      SelectionCase{256, KernelOverride::RegTile, KernelPath::RegTileW8, false},
      SelectionCase{257, KernelOverride::RegTile, KernelPath::Wavefront, true},

      // These public selectors have no distinct CUDA implementation.
      SelectionCase{16, KernelOverride::WavefrontGlobal, KernelPath::Warp, true},
      SelectionCase{64, KernelOverride::WavefrontGlobal, KernelPath::RegTileW4, true},
      SelectionCase{200, KernelOverride::BandedRow, KernelPath::RegTileW8, true},
      SelectionCase{300, KernelOverride::BandedRow, KernelPath::Wavefront, true},
  };

  for (const auto &tc : auto_cases) {
    const auto selected = select_kernel(tc.max_length, tc.requested);
    INFO("max_length=" << tc.max_length);
    CHECK(selected.path == tc.expected);
    CHECK(selected.fell_back_to_auto == tc.fell_back_to_auto);
  }
  for (const auto &tc : override_cases) {
    const auto selected = select_kernel(tc.max_length, tc.requested);
    INFO("max_length=" << tc.max_length
         << " requested=" << static_cast<int>(tc.requested));
    CHECK(selected.path == tc.expected);
    CHECK(selected.fell_back_to_auto == tc.fell_back_to_auto);
  }

  CHECK(kernel_path_name(KernelPath::Warp) == std::string_view{"warp"});
  CHECK(kernel_path_name(KernelPath::RegTileW4) == std::string_view{"regtile_w4"});
  CHECK(kernel_path_name(KernelPath::RegTileW8) == std::string_view{"regtile_w8"});
  CHECK(kernel_path_name(KernelPath::Wavefront) == std::string_view{"wavefront"});
#endif
}

#ifdef DTWC_HAS_CUDA

namespace {

std::vector<std::vector<double>> m50_series(std::size_t length)
{
  constexpr std::size_t n = 4;
  std::vector<std::vector<double>> series(n, std::vector<double>(length));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < length; ++t) {
      const int centered = static_cast<int>((17 * i + 7 * t) % 29) - 14;
      series[i][t] = static_cast<double>(centered) + 0.125 * static_cast<double>(i);
    }
  }
  return series;
}

std::vector<double> m50_cpu_matrix(const std::vector<std::vector<double>> &series)
{
  const std::size_t n = series.size();
  std::vector<double> result(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double d = dtwc::dtwFull_L<double>(series[i], series[j]);
      result[i * n + j] = d;
      result[j * n + i] = d;
    }
  }
  return result;
}

std::vector<double> m50_cpu_row(const std::vector<double> &query,
                                const std::vector<std::vector<double>> &series)
{
  std::vector<double> result(series.size());
  for (std::size_t i = 0; i < series.size(); ++i)
    result[i] = dtwc::dtwFull_L<double>(query, series[i]);
  return result;
}

void require_m50_matrix(const std::vector<double> &actual,
                        const std::vector<double> &expected)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    INFO("matrix offset=" << i);
    const double tolerance = 1e-10 * (1.0 + std::abs(expected[i]));
    CHECK_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], tolerance));
  }
}

} // namespace

TEST_CASE("M50 CUDA no-launch paths report no kernel or fallback",
          "[cuda][kernel_override][no_launch][m50]")
{
  dtwc::cuda::CUDADistMatOptions options;
  // An unsupported CUDA family would report a fallback if dispatch happened.
  // No-work returns must instead report that no kernel was launched at all.
  options.kernel_override = dtwc::KernelOverride::BandedRow;

  SECTION("pairwise") {
    const auto result = dtwc::cuda::compute_distance_matrix_cuda({{}}, options);
    CHECK(result.kernel_used == "none");
    CHECK_FALSE(result.kernel_override_fell_back);
  }

  SECTION("one-vs-N by index") {
    const std::vector<std::vector<double>> series{{}, {}};
    const auto result = dtwc::cuda::compute_dtw_one_vs_all(series, 0, options);
    CHECK(result.kernel_used == "none");
    CHECK_FALSE(result.kernel_override_fell_back);
  }

  SECTION("one-vs-N by external query") {
    const std::vector<std::vector<double>> series{{}, {}};
    const auto result = dtwc::cuda::compute_dtw_one_vs_all(
        std::vector<double>{}, series, options);
    CHECK(result.kernel_used == "none");
    CHECK_FALSE(result.kernel_override_fell_back);
  }

  SECTION("K-vs-N") {
    const std::vector<std::vector<double>> series{{}, {}};
    const auto result = dtwc::cuda::compute_dtw_k_vs_all(series, {}, options);
    CHECK(result.kernel_used == "none");
    CHECK_FALSE(result.kernel_override_fell_back);
  }
}

TEST_CASE("M50 fully pruned CUDA work reports no DTW kernel launch",
          "[cuda][kernel_override][device][no_launch][m50]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device");
    return;
  }

  constexpr std::size_t n = 4;
  constexpr std::size_t length = 16;
  std::vector<std::vector<double>> series(n, std::vector<double>(length));
  for (std::size_t i = 0; i < n; ++i)
    for (double &value : series[i]) value = 10.0 * static_cast<double>(i);

  dtwc::cuda::CUDADistMatOptions options;
  options.precision = dtwc::cuda::CUDAPrecision::FP64;
  options.kernel_override = dtwc::KernelOverride::BandedRow;
  options.use_lb_keogh = true;
  options.band = 0;
  options.lb_threshold = 0.5;

  const auto result = dtwc::cuda::compute_distance_matrix_cuda(series, options);
  CHECK(result.pairs_pruned == n * (n - 1) / 2);
  CHECK(result.pairs_computed == 0);
  CHECK(result.gpu_time_sec == 0.0);
  CHECK(result.kernel_used == "none");
  CHECK_FALSE(result.kernel_override_fell_back);
}

TEST_CASE("M50 CUDA override executes or reports the real Auto fallback",
          "[cuda][kernel_override][device][m50]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device");
    return;
  }

  using dtwc::KernelOverride;
  struct DeviceCase {
    std::size_t length;
    KernelOverride requested;
    std::string_view expected_auto;
    std::string_view expected_actual;
    bool expected_fallback;
  };

  const std::array cases{
      // Force a different implemented family than Auto would choose.
      DeviceCase{16, KernelOverride::Wavefront, "warp", "wavefront", false},
      DeviceCase{16, KernelOverride::RegTile, "warp", "regtile_w4", false},
      DeviceCase{200, KernelOverride::Wavefront, "regtile_w8", "wavefront", false},

      // Supported and unsupported boundary behavior.
      DeviceCase{200, KernelOverride::RegTile, "regtile_w8", "regtile_w8", false},
      DeviceCase{300, KernelOverride::RegTile, "wavefront", "wavefront", true},
      DeviceCase{64, KernelOverride::BandedRow, "regtile_w4", "regtile_w4", true},
      DeviceCase{64, KernelOverride::WavefrontGlobal, "regtile_w4", "regtile_w4", true},
  };

  for (const auto &tc : cases) {
    CAPTURE(tc.length, static_cast<int>(tc.requested));
    const auto series = m50_series(tc.length);
    const auto cpu = m50_cpu_matrix(series);

    dtwc::cuda::CUDADistMatOptions auto_options;
    auto_options.precision = dtwc::cuda::CUDAPrecision::FP64;
    const auto automatic = dtwc::cuda::compute_distance_matrix_cuda(series, auto_options);

    auto forced_options = auto_options;
    forced_options.kernel_override = tc.requested;
    const auto selected = dtwc::cuda::compute_distance_matrix_cuda(series, forced_options);

    CHECK(automatic.kernel_used == tc.expected_auto);
    CHECK(selected.kernel_used == tc.expected_actual);
    CHECK_FALSE(automatic.kernel_override_fell_back);
    CHECK(selected.kernel_override_fell_back == tc.expected_fallback);
    require_m50_matrix(automatic.matrix, cpu);
    require_m50_matrix(selected.matrix, cpu);
    require_m50_matrix(selected.matrix, automatic.matrix);
  }
}

TEST_CASE("M50 CUDA row and batched entry points share override dispatch",
          "[cuda][kernel_override][device][one_vs_n][k_vs_n][m50]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device");
    return;
  }

  using dtwc::KernelOverride;

  SECTION("one-vs-N by index forces Wavefront instead of Auto Warp") {
    const auto series = m50_series(16);
    const auto cpu = m50_cpu_row(series[1], series);

    dtwc::cuda::CUDADistMatOptions automatic_options;
    automatic_options.precision = dtwc::cuda::CUDAPrecision::FP64;
    const auto automatic = dtwc::cuda::compute_dtw_one_vs_all(series, 1, automatic_options);

    auto forced_options = automatic_options;
    forced_options.kernel_override = KernelOverride::Wavefront;
    const auto forced = dtwc::cuda::compute_dtw_one_vs_all(series, 1, forced_options);

    CHECK(automatic.kernel_used == "warp");
    CHECK(forced.kernel_used == "wavefront");
    CHECK_FALSE(automatic.kernel_override_fell_back);
    CHECK_FALSE(forced.kernel_override_fell_back);
    require_m50_matrix(automatic.distances, cpu);
    require_m50_matrix(forced.distances, cpu);
    require_m50_matrix(forced.distances, automatic.distances);
  }

  SECTION("one-vs-N by external query reports unsupported BandedRow fallback") {
    const auto series = m50_series(64);
    std::vector<double> query = series[2];
    for (double &value : query) value += 0.25;
    const auto cpu = m50_cpu_row(query, series);

    dtwc::cuda::CUDADistMatOptions automatic_options;
    automatic_options.precision = dtwc::cuda::CUDAPrecision::FP64;
    const auto automatic = dtwc::cuda::compute_dtw_one_vs_all(
        query, series, automatic_options);

    auto fallback_options = automatic_options;
    fallback_options.kernel_override = KernelOverride::BandedRow;
    const auto fallback = dtwc::cuda::compute_dtw_one_vs_all(
        query, series, fallback_options);

    CHECK(automatic.kernel_used == "regtile_w4");
    CHECK(fallback.kernel_used == "regtile_w4");
    CHECK_FALSE(automatic.kernel_override_fell_back);
    CHECK(fallback.kernel_override_fell_back);
    require_m50_matrix(automatic.distances, cpu);
    require_m50_matrix(fallback.distances, cpu);
    require_m50_matrix(fallback.distances, automatic.distances);
  }

  SECTION("K-vs-N forces Wavefront and preserves every requested row") {
    const auto series = m50_series(200);
    const std::vector<std::size_t> queries{0, 2, 3};
    std::vector<double> cpu;
    for (const std::size_t query : queries) {
      const auto row = m50_cpu_row(series[query], series);
      cpu.insert(cpu.end(), row.begin(), row.end());
    }

    dtwc::cuda::CUDADistMatOptions automatic_options;
    automatic_options.precision = dtwc::cuda::CUDAPrecision::FP64;
    const auto automatic = dtwc::cuda::compute_dtw_k_vs_all(
        series, queries, automatic_options);

    auto forced_options = automatic_options;
    forced_options.kernel_override = KernelOverride::Wavefront;
    const auto forced = dtwc::cuda::compute_dtw_k_vs_all(
        series, queries, forced_options);

    CHECK(automatic.kernel_used == "regtile_w8");
    CHECK(forced.kernel_used == "wavefront");
    CHECK_FALSE(automatic.kernel_override_fell_back);
    CHECK_FALSE(forced.kernel_override_fell_back);
    require_m50_matrix(automatic.distances, cpu);
    require_m50_matrix(forced.distances, cpu);
    require_m50_matrix(forced.distances, automatic.distances);
  }

  SECTION("K-vs-N reports RegTile fallback above its supported length") {
    const auto series = m50_series(300);
    const std::vector<std::size_t> queries{1, 3};
    std::vector<double> cpu;
    for (const std::size_t query : queries) {
      const auto row = m50_cpu_row(series[query], series);
      cpu.insert(cpu.end(), row.begin(), row.end());
    }

    dtwc::cuda::CUDADistMatOptions fallback_options;
    fallback_options.precision = dtwc::cuda::CUDAPrecision::FP64;
    fallback_options.kernel_override = KernelOverride::RegTile;
    const auto fallback = dtwc::cuda::compute_dtw_k_vs_all(
        series, queries, fallback_options);

    CHECK(fallback.kernel_used == "wavefront");
    CHECK(fallback.kernel_override_fell_back);
    require_m50_matrix(fallback.distances, cpu);
  }
}

#else

TEST_CASE("M50 CUDA device discriminator requires a CUDA build",
          "[cuda][kernel_override][device][m50]")
{
  SKIP("DTWC_HAS_CUDA not defined; device discriminator skipped");
}

#endif
