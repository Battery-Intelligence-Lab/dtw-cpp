/**
 * @file unit_test_variant_distmat.cpp
 * @brief Integration tests for std::variant-based distance matrix in Problem.
 *
 * Tests that Problem works correctly with both DenseDistanceMatrix (default)
 * and MmapDistanceMatrix (when forced via use_mmap_distance_matrix()).
 *
 * @date 08 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace dtwc;
namespace fs = std::filesystem;

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

namespace {
fs::path dummy_data_path() { return fs::path{DTWC_TEST_DATA_DIR} / "dummy"; }

struct ScratchCache
{
  fs::path directory;
  fs::path path;

  explicit ScratchCache(std::string_view stem)
    : directory(fs::temp_directory_path()
                / (std::string(stem) + "_"
                   + std::to_string(reinterpret_cast<std::uintptr_t>(this)))),
      path(directory / "distances.dtwcache")
  {
    fs::create_directories(directory);
  }

  ~ScratchCache()
  {
    std::error_code ec;
    fs::remove_all(directory, ec);
  }
};

Data make_data(std::vector<std::vector<data_t>> series, size_t ndim = 1)
{
  std::vector<std::string> names;
  names.reserve(series.size());
  for (size_t i = 0; i < series.size(); ++i)
    names.push_back("s" + std::to_string(i));
  return Data(std::move(series), std::move(names), ndim);
}

Data make_data_f32(std::vector<std::vector<float>> series, size_t ndim = 1)
{
  std::vector<std::string> names;
  names.reserve(series.size());
  for (size_t i = 0; i < series.size(); ++i)
    names.push_back("s" + std::to_string(i));
  return Data(std::move(series), std::move(names), ndim);
}
}

TEST_CASE("Problem uses DenseDistanceMatrix by default for small N", "[variant][distmat]")
{
  DataLoader dl(dummy_data_path());
  Problem prob("test_variant", dl);
  REQUIRE(prob.size() == 25);

  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());

  double d = prob.dist_by_ind(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(d == prob.dist_by_ind(1, 0)); // symmetry
}

TEST_CASE("Problem dense cache never survives a raw semantic configuration mutation",
          "[variant][distmat][dense][semantic_mutation]")
{
  SECTION("band")
  {
    Problem prob{"dense_band_mutation"};
    prob.set_data(make_data({{0.0, 0.0, 10.0}, {0.0, 10.0, 10.0}}));

    REQUIRE(prob.dist_by_ind(0, 1) == 0.0);
    prob.band = 0; // Legacy public-field mutation must not preserve the cached 0.

    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    REQUIRE(prob.dist_by_ind(0, 1) == 10.0);
  }

  SECTION("variant and parameters")
  {
    Problem prob{"dense_variant_mutation"};
    prob.set_data(make_data({{0.0}, {2.0}}));

    REQUIRE(prob.dist_by_ind(0, 1) == 2.0);
    core::DTWVariantParams params;
    params.variant = core::DTWVariant::WDTW;
    params.wdtw_g = 0.5;
    prob.variant_params = params; // Legacy whole-field mutation.

    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    REQUIRE(prob.dist_by_ind(0, 1) == 1.0);
  }

  SECTION("missing-data strategy")
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Problem prob{"dense_missing_mutation"};
    prob.missing_strategy = core::MissingStrategy::ZeroCost;
    prob.set_data(make_data({{0.0, nan, 2.0}, {0.0, 2.0, 2.0}}));

    REQUIRE(prob.dist_by_ind(0, 1) == 0.0);
    prob.missing_strategy = core::MissingStrategy::Interpolate;

    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    REQUIRE(prob.dist_by_ind(0, 1) == 1.0);
  }

  SECTION("distance backend and nested CUDA settings")
  {
    Problem prob{"dense_backend_mutation"};
    prob.set_data(make_data({{0.0}, {2.0}}));
    auto load_precomputed = [&] {
      auto &matrix = prob.dense_distance_matrix();
      matrix.resize(2);
      matrix.set(0, 0, 0.0);
      matrix.set(0, 1, 123.0);
      matrix.set(1, 1, 0.0);
      REQUIRE(prob.is_distance_matrix_filled());
    };

    load_precomputed();
    prob.distance_strategy = DistanceMatrixStrategy::BruteForce;
    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    REQUIRE(prob.dist_by_ind(0, 1) == 2.0);

    load_precomputed();
    prob.cuda_settings.device_id = 7; // Nested public-struct mutation.
    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    REQUIRE(prob.dist_by_ind(0, 1) == 2.0);
  }

  SECTION("every variant parameter and multivariate mode")
  {
    using Mutator = std::function<void(core::DTWVariantParams &)>;
    const std::vector<std::pair<std::string, Mutator>> mutations{
      {"variant", [](auto &p) { p.variant = core::DTWVariant::DDTW; }},
      {"wdtw_g", [](auto &p) { p.wdtw_g = 0.5; }},
      {"adtw_penalty", [](auto &p) { p.adtw_penalty = 2.0; }},
      {"sdtw_gamma", [](auto &p) { p.sdtw_gamma = 0.5; }},
      {"msm_c", [](auto &p) { p.msm_c = 2.0; }},
      {"twe_nu", [](auto &p) { p.twe_nu = 0.01; }},
      {"twe_lambda", [](auto &p) { p.twe_lambda = 2.0; }},
      {"mv_mode", [](auto &p) { p.mv_mode = core::MVMode::Independent; }},
    };

    for (const auto &[name, mutate] : mutations) {
      CAPTURE(name);
      Problem prob{"dense_variant_parameter_mutation"};
      prob.set_data(make_data({{0.0}, {2.0}}));
      auto &matrix = prob.dense_distance_matrix();
      matrix.resize(2);
      matrix.set(0, 0, 0.0);
      matrix.set(0, 1, 123.0);
      matrix.set(1, 1, 0.0);
      REQUIRE(prob.is_distance_matrix_filled());

      mutate(prob.variant_params);

      REQUIRE_FALSE(prob.is_distance_matrix_filled());
      REQUIRE(prob.dist_by_ind(0, 1) != 123.0);
    }
  }

  SECTION("const readers reject a stale raw configuration")
  {
    Problem prob{"dense_const_reader_mutation"};
    prob.set_data(make_data({{0.0, 0.0, 10.0}, {0.0, 10.0, 10.0}}));
    REQUIRE(prob.dist_by_ind(0, 1) == 0.0);
    prob.band = 0;

    const Problem &view = prob;
    REQUIRE_FALSE(view.is_distance_matrix_filled());
    REQUIRE_THROWS_WITH(
      view.dense_distance_matrix(),
      Catch::Matchers::ContainsSubstring("cached distance configuration changed"));
  }
}

TEST_CASE("Problem semantic setters preserve or invalidate precomputed distances exactly",
          "[variant][distmat][dense][semantic_mutation][setters]")
{
  Problem prob{"dense_semantic_setters"};
  prob.set_data(make_data({{0.0}, {2.0}}));
  const auto load_precomputed = [&] {
    auto &matrix = prob.dense_distance_matrix();
    matrix.resize(2);
    matrix.set(0, 0, 0.0);
    matrix.set(0, 1, 123.0);
    matrix.set(1, 1, 0.0);
    REQUIRE(prob.is_distance_matrix_filled());
  };

  load_precomputed();
  prob.set_missing_strategy(core::MissingStrategy::Error);
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 123.0);
  prob.set_variant(prob.variant_params);
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 123.0);
  prob.set_variant(core::DTWVariant::Standard);
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 123.0);
  prob.set_missing_strategy(core::MissingStrategy::Interpolate);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 2.0);

  load_precomputed();
  prob.set_distance_strategy(DistanceMatrixStrategy::BruteForce);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 2.0);

  load_precomputed();
  CUDASettings settings;
  settings.device_id = 7;
  settings.precision = 2;
  prob.set_cuda_settings(settings);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 2.0);
}

TEST_CASE("Problem uses MmapDistanceMatrix when forced", "[variant][distmat][mmap]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  auto cache_path = fs::temp_directory_path() / "dtwc_test" / "variant_mmap.dtwcache";
  fs::create_directories(cache_path.parent_path());
  if (fs::exists(cache_path)) fs::remove(cache_path);

  DataLoader dl(dummy_data_path());
  Problem prob("test_mmap", dl);

  // Force mmap mode
  prob.use_mmap_distance_matrix(cache_path);

  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());

  double d = prob.dist_by_ind(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(d == prob.dist_by_ind(1, 0));

  REQUIRE(fs::exists(cache_path));
  REQUIRE(fs::file_size(cache_path) > 0);

  // Cleanup
  fs::remove_all(cache_path.parent_path());
#endif
}

TEST_CASE("MmapDistanceMatrix warmstart via Problem", "[variant][distmat][mmap]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  auto cache_path = fs::temp_directory_path() / "dtwc_test" / "warmstart_prob.dtwcache";
  fs::create_directories(cache_path.parent_path());
  if (fs::exists(cache_path)) fs::remove(cache_path);

  double d01_original;

  // First run: fill distance matrix
  {
    DataLoader dl(dummy_data_path());
    Problem prob("test_warmstart", dl);
    prob.use_mmap_distance_matrix(cache_path);
    prob.fill_distance_matrix();
    d01_original = prob.dist_by_ind(0, 1);
  }

  // Second run: reopen - distances should persist
  {
    DataLoader dl(dummy_data_path());
    Problem prob("test_warmstart", dl);
    prob.use_mmap_distance_matrix(cache_path);
    REQUIRE(prob.is_distance_matrix_filled());
    REQUIRE(prob.dist_by_ind(0, 1) == d01_original);
  }

  // Cleanup
  fs::remove_all(cache_path.parent_path());
#endif
}

TEST_CASE("Problem mmap warmstart rejects a same-N data fingerprint mismatch",
          "[variant][distmat][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  ScratchCache cache{"dtwc_mmap_data_fingerprint"};

  {
    Problem original{"cache_data"};
    original.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 3.0}}));
    original.use_mmap_distance_matrix(cache.path);
    original.fill_distance_matrix();
  }

  Problem changed{"cache_data"};
  changed.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 30.0}}));
  REQUIRE_THROWS_WITH(
    changed.use_mmap_distance_matrix(cache.path),
    Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
#endif
}

TEST_CASE("Problem mmap warmstart fingerprints every distance configuration dimension",
          "[variant][distmat][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  const auto reopen_throws = [](std::string_view stem,
                                const std::function<void(Problem &)> &configure_original,
                                const std::function<void(Problem &)> &configure_changed,
                                Data original_data,
                                Data changed_data) {
    ScratchCache cache{stem};
    {
      Problem original{"cache_config"};
      original.set_data(std::move(original_data));
      configure_original(original);
      original.use_mmap_distance_matrix(cache.path);
      // One computed bit is enough to make stale reuse observable.
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_config"};
    changed.set_data(std::move(changed_data));
    configure_changed(changed);
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache.path),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  };

  SECTION("band")
  {
    reopen_throws(
      "dtwc_mmap_band_fingerprint",
      [](Problem &p) { p.set_band(-1); },
      [](Problem &p) { p.set_band(0); },
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("variant")
  {
    reopen_throws(
      "dtwc_mmap_variant_fingerprint",
      [](Problem &p) { p.set_variant(core::DTWVariant::Standard); },
      [](Problem &p) { p.set_variant(core::DTWVariant::ADTW); },
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("variant parameter")
  {
    reopen_throws(
      "dtwc_mmap_variant_parameter_fingerprint",
      [](Problem &p) {
        core::DTWVariantParams params;
        params.variant = core::DTWVariant::WDTW;
        params.wdtw_g = 0.05;
        p.set_variant(params);
      },
      [](Problem &p) {
        core::DTWVariantParams params;
        params.variant = core::DTWVariant::WDTW;
        params.wdtw_g = 0.5;
        p.set_variant(params);
      },
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("missing-data strategy")
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    reopen_throws(
      "dtwc_mmap_missing_fingerprint",
      [](Problem &p) { p.missing_strategy = core::MissingStrategy::ZeroCost; },
      [](Problem &p) { p.missing_strategy = core::MissingStrategy::AROW; },
      make_data({{0.0, nan, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, nan, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("pointwise metric")
  {
    ScratchCache cache{"dtwc_mmap_metric_fingerprint"};
    {
      Problem original{"cache_metric"};
      original.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
      original.use_mmap_distance_matrix(cache.path, core::MetricType::L1);
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_metric"};
    changed.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache.path, core::MetricType::SquaredL2),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  }
#endif
}

TEST_CASE("Problem mmap data fingerprint includes precision and multivariate shape",
          "[variant][distmat][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  SECTION("numeric precision")
  {
    ScratchCache cache{"dtwc_mmap_precision_fingerprint"};
    {
      Problem original{"cache_precision"};
      original.set_data(make_data({{0.0, 1.0}, {1.0, 2.0}}));
      original.use_mmap_distance_matrix(cache.path);
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_precision"};
    changed.set_data(make_data_f32({{0.0f, 1.0f}, {1.0f, 2.0f}}));
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache.path),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  }

  SECTION("ndim")
  {
    ScratchCache cache{"dtwc_mmap_ndim_fingerprint"};
    {
      Problem original{"cache_ndim"};
      original.set_data(make_data(
        {{0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 3.0, 4.0}}, 1));
      original.use_mmap_distance_matrix(cache.path);
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_ndim"};
    changed.set_data(make_data(
      {{0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 3.0, 4.0}}, 2));
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache.path),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  }
#endif
}

TEST_CASE("Problem mmap warmstart accepts an unchanged fingerprint",
          "[variant][distmat][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  ScratchCache cache{"dtwc_mmap_unchanged_fingerprint"};
  double expected{};
  {
    Problem original{"cache_unchanged"};
    original.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 4.0}}));
    original.set_band(1);
    original.use_mmap_distance_matrix(cache.path);
    expected = original.dist_by_ind(0, 1);
  }

  Problem reopened{"cache_unchanged"};
  reopened.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 4.0}}));
  reopened.set_band(1);
  reopened.use_mmap_distance_matrix(cache.path);
  REQUIRE(reopened.dist_by_ind(0, 1) == expected);
#endif
}

TEST_CASE("Problem invalidates or rejects post-bind distance-semantic mutations",
          "[variant][distmat][mmap][fingerprint][mutation]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  SECTION("set_band detaches without rewriting the old cache")
  {
    ScratchCache cache{"dtwc_mmap_post_bind_set_band"};
    Problem prob{"cache_mutation"};
    prob.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    prob.use_mmap_distance_matrix(cache.path);
    REQUIRE(prob.dist_by_ind(0, 1) >= 0.0);

    prob.set_band(0);
    REQUIRE(std::holds_alternative<core::DenseDistanceMatrix>(
      prob.distance_matrix()));
    REQUIRE_THROWS_WITH(
      prob.use_mmap_distance_matrix(cache.path),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  }

  SECTION("naked config edit is caught before a computed bit is read")
  {
    ScratchCache cache{"dtwc_mmap_post_bind_direct_band"};
    Problem prob{"cache_mutation"};
    prob.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    prob.use_mmap_distance_matrix(cache.path);
    REQUIRE(prob.dist_by_ind(0, 1) >= 0.0);

    prob.band = 0; // legacy public field; bypasses set_band deliberately
    REQUIRE_THROWS_WITH(
      prob.dist_by_ind(0, 1),
      Catch::Matchers::ContainsSubstring("bound cache fingerprint mismatch"));
  }

  SECTION("in-place series edit before first use is caught")
  {
    ScratchCache cache{"dtwc_mmap_post_bind_data_edit"};
    Problem prob{"cache_mutation"};
    prob.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    prob.use_mmap_distance_matrix(cache.path);

    // Mutate after binding but before the first use-session validation. Raw
    // edits after that first validation are unsupported; semantic setters are
    // the cache-invalidating API.
    prob.p_vec(1)[2] = 30.0;
    REQUIRE_THROWS_WITH(
      prob.dist_by_ind(0, 1),
      Catch::Matchers::ContainsSubstring("changed before first use"));
  }
#endif
}

TEST_CASE("Warm mmap cached lookups remain O(1) in series length",
          "[variant][distmat][mmap][fingerprint][performance]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  using clock = std::chrono::steady_clock;
  constexpr int repeats = 100000;

  const auto elapsed_for_length = [](size_t length, std::string_view stem) {
    ScratchCache cache{stem};
    std::vector<data_t> a(length), b(length);
    for (size_t i = 0; i < length; ++i) {
      a[i] = static_cast<double>(i % 17);
      b[i] = static_cast<double>((i + 3) % 19);
    }

    Problem prob{"cache_lookup_complexity"};
    prob.set_data(make_data({std::move(a), std::move(b)}));
    prob.use_mmap_distance_matrix(cache.path);
    // Exclude the deliberately one-time full identity validation from timing.
    auto &matrix = std::get<core::MmapDistanceMatrix>(prob.distance_matrix());
    matrix.set(0, 1, 7.0);

    const auto start = clock::now();
    double checksum = 0.0;
    for (int iteration = 0; iteration < repeats; ++iteration)
      checksum += prob.dist_by_ind(0, 1);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      clock::now() - start).count();
    REQUIRE(checksum == static_cast<double>(repeats) * 7.0);
    return elapsed;
  };

  // Registered before running: after the one-time validation, length 4096 must
  // stay within 8x the length-1 lookup plus a 20 ms scheduler allowance. A
  // per-lookup data hash performs ~819M scalar visits here and violates the band
  // by orders of magnitude; the intended path is two O(1) config/variant checks.
  const auto short_ns = elapsed_for_length(1, "dtwc_mmap_lookup_short");
  const auto long_ns = elapsed_for_length(4096, "dtwc_mmap_lookup_long");
  INFO("short_ns=" << short_ns << " long_ns=" << long_ns);
  REQUIRE(long_ns <= short_ns * 8 + 20'000'000);
#endif
}

TEST_CASE("Problem non-L1 mmap identity cannot be lazily filled by the L1 CPU path",
          "[variant][distmat][mmap][fingerprint][metric]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  ScratchCache cache{"dtwc_mmap_external_metric"};
  Problem prob{"cache_metric"};
  prob.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
  prob.use_mmap_distance_matrix(cache.path, core::MetricType::SquaredL2);

  REQUIRE_THROWS_WITH(
    prob.dist_by_ind(0, 1),
    Catch::Matchers::ContainsSubstring("external-fill-only")
      && Catch::Matchers::ContainsSubstring("lazy CPU"));
  REQUIRE(std::get<core::MmapDistanceMatrix>(prob.distance_matrix())
            .count_computed() == 0);
#endif
}

TEST_CASE("Problem rejects CUDA Auto precision for persistent mmap identity",
          "[variant][distmat][mmap][fingerprint][cuda]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  ScratchCache cache{"dtwc_mmap_cuda_auto_precision"};
  Problem prob{"cache_cuda_auto"};
  prob.set_data(make_data({{0.0, 1.0}, {1.0, 2.0}}));
  prob.distance_strategy = DistanceMatrixStrategy::CUDA;
  prob.cuda_settings.precision = 0; // Auto: runtime-hardware dependent

  REQUIRE_THROWS_WITH(
    prob.use_mmap_distance_matrix(cache.path),
    Catch::Matchers::ContainsSubstring("CUDA precision=Auto")
      && Catch::Matchers::ContainsSubstring("explicit FP32 or FP64"));
  REQUIRE_FALSE(fs::exists(cache.path));
#endif
}
