/**
 * @file unit_test_variant_distmat.cpp
 * @brief The Problem's distance matrix, on the heap and mapped: what drops a dense
 *        cache, and what the identity of a mapped cache (use_mmap_distance_matrix())
 *        covers. The distances themselves are checked against the oracle in
 *        core/test_dtw.cpp.
 *
 * @date 08 Apr 2026
 */

#include <dtwc.hpp>

#include "../support/dtw_route_bound.hpp"
#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace dtwc;
namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

namespace {

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

TEST_CASE("Problem dense cache never survives a semantic configuration change",
          "[variant][distmat][dense][semantic_mutation]")
{
  SECTION("a direct write to the v1 band field")
  {
    Problem prob{"dense_band_mutation"};
    prob.set_data(make_data({{0.0, 0.0, 10.0}, {0.0, 10.0, 10.0}}));

    prob.fill_distance_matrix();
    REQUIRE(prob.dist_by_ind(0, 1) == 0.0);
    prob.band = 0; // the v1 field; the next fill takes it

    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    prob.fill_distance_matrix();
    REQUIRE(prob.distance().band == 0);
    REQUIRE(prob.dist_by_ind(0, 1) == 10.0);
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
      auto &matrix = prob.distance_matrix();
      matrix.resize(2);
      matrix.set(0, 0, 0.0);
      matrix.set(0, 1, 123.0);
      matrix.set(1, 1, 0.0);
      prob.fill_distance_matrix(); // every pair is set: marks it complete, computes nothing
      REQUIRE(prob.is_distance_matrix_filled());

      auto params = prob.variant_params();
      mutate(params);
      prob.set_variant(params);

      REQUIRE_FALSE(prob.is_distance_matrix_filled());
      prob.fill_distance_matrix();
      REQUIRE(prob.dist_by_ind(0, 1) != 123.0);
    }
  }
}

TEST_CASE("Problem semantic setters preserve or invalidate precomputed distances exactly",
          "[variant][distmat][dense][semantic_mutation][setters]")
{
  Problem prob{"dense_semantic_setters"};
  prob.set_data(make_data({{0.0}, {2.0}}));
  const auto load_precomputed = [&] {
    auto &matrix = prob.distance_matrix();
    matrix.resize(2);
    matrix.set(0, 0, 0.0);
    matrix.set(0, 1, 123.0);
    matrix.set(1, 1, 0.0);
    prob.fill_distance_matrix(); // every pair is set: marks it complete, computes nothing
    REQUIRE(prob.is_distance_matrix_filled());
  };

  load_precomputed();
  prob.set_missing_strategy(core::MissingStrategy::Error);
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 123.0);
  prob.set_variant(prob.variant_params());
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 123.0);
  prob.set_variant(core::DTWVariant::Standard);
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dist_by_ind(0, 1) == 123.0);
  prob.set_missing_strategy(core::MissingStrategy::Interpolate);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  prob.fill_distance_matrix();
  REQUIRE(prob.dist_by_ind(0, 1) == 2.0);

  load_precomputed();
  prob.set_distance_strategy(DistanceMatrixStrategy::BruteForce);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  prob.fill_distance_matrix();
  REQUIRE(prob.dist_by_ind(0, 1) == 2.0);

  load_precomputed();
  CUDASettings settings;
  settings.device_id = 7;
  settings.precision = GpuPrecision::FP64;
  prob.set_cuda_settings(settings);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  prob.fill_distance_matrix();
  REQUIRE(prob.dist_by_ind(0, 1) == 2.0);
}

TEST_CASE("Problem mmap warmstart rejects a same-N data fingerprint mismatch",
          "[variant][distmat][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  const ScratchDirectory cache_dir{ "mmap_data_fingerprint" };
  const fs::path cache = cache_dir.path / "distances.dtwcache";

  {
    Problem original{"cache_data"};
    original.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 3.0}}));
    original.use_mmap_distance_matrix(cache);
    original.fill_distance_matrix();
  }

  Problem changed{"cache_data"};
  changed.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 30.0}}));
  REQUIRE_THROWS_WITH(
    changed.use_mmap_distance_matrix(cache),
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
    const ScratchDirectory cache_dir{ stem };
    const fs::path cache = cache_dir.path / "distances.dtwcache";
    {
      Problem original{"cache_config"};
      original.set_data(std::move(original_data));
      configure_original(original);
      original.use_mmap_distance_matrix(cache);
      // Computed bits make stale reuse observable.
      original.fill_distance_matrix();
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_config"};
    changed.set_data(std::move(changed_data));
    configure_changed(changed);
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  };

  SECTION("band")
  {
    reopen_throws(
      "mmap_band_fingerprint",
      [](Problem &p) { p.set_band(-1); },
      [](Problem &p) { p.set_band(0); },
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("variant")
  {
    reopen_throws(
      "mmap_variant_fingerprint",
      [](Problem &p) { p.set_variant(core::DTWVariant::Standard); },
      [](Problem &p) { p.set_variant(core::DTWVariant::ADTW); },
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("variant parameter")
  {
    reopen_throws(
      "mmap_variant_parameter_fingerprint",
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
      "mmap_missing_fingerprint",
      [](Problem &p) { p.set_missing_strategy(core::MissingStrategy::ZeroCost); },
      [](Problem &p) { p.set_missing_strategy(core::MissingStrategy::AROW); },
      make_data({{0.0, nan, 2.0}, {0.0, 2.0, 3.0}}),
      make_data({{0.0, nan, 2.0}, {0.0, 2.0, 3.0}}));
  }

  SECTION("pointwise metric")
  {
    const ScratchDirectory cache_dir{ "mmap_metric_fingerprint" };
    const fs::path cache = cache_dir.path / "distances.dtwcache";
    {
      Problem original{"cache_metric"};
      original.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
      original.use_mmap_distance_matrix(cache, core::MetricType::L1);
      original.fill_distance_matrix();
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_metric"};
    changed.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache, core::MetricType::SquaredL2),
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
    const ScratchDirectory cache_dir{ "mmap_precision_fingerprint" };
    const fs::path cache = cache_dir.path / "distances.dtwcache";
    {
      Problem original{"cache_precision"};
      original.set_data(make_data({{0.0, 1.0}, {1.0, 2.0}}));
      original.use_mmap_distance_matrix(cache);
      original.fill_distance_matrix();
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_precision"};
    changed.set_data(make_data_f32({{0.0f, 1.0f}, {1.0f, 2.0f}}));
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  }

  SECTION("ndim")
  {
    const ScratchDirectory cache_dir{ "mmap_ndim_fingerprint" };
    const fs::path cache = cache_dir.path / "distances.dtwcache";
    {
      Problem original{"cache_ndim"};
      original.set_data(make_data(
        {{0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 3.0, 4.0}}, 1));
      original.use_mmap_distance_matrix(cache);
      original.fill_distance_matrix();
      REQUIRE(original.dist_by_ind(0, 1) >= 0.0);
    }

    Problem changed{"cache_ndim"};
    changed.set_data(make_data(
      {{0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 3.0, 4.0}}, 2));
    REQUIRE_THROWS_WITH(
      changed.use_mmap_distance_matrix(cache),
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
  const ScratchDirectory cache_dir{ "mmap_unchanged_fingerprint" };
  const fs::path cache = cache_dir.path / "distances.dtwcache";
  double expected{};
  {
    Problem original{"cache_unchanged"};
    original.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 4.0}}));
    original.set_band(1);
    original.use_mmap_distance_matrix(cache);
    original.fill_distance_matrix();
    expected = original.dist_by_ind(0, 1);
  }

  Problem reopened{"cache_unchanged"};
  reopened.set_data(make_data({{0.0, 1.0, 2.0}, {1.0, 2.0, 4.0}}));
  reopened.set_band(1);
  reopened.use_mmap_distance_matrix(cache);
  REQUIRE(reopened.is_distance_matrix_filled());
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
    const ScratchDirectory cache_dir{ "mmap_post_bind_set_band" };
    const fs::path cache = cache_dir.path / "distances.dtwcache";
    Problem prob{"cache_mutation"};
    prob.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    prob.use_mmap_distance_matrix(cache);
    prob.fill_distance_matrix();
    REQUIRE(prob.dist_by_ind(0, 1) >= 0.0);

    prob.set_band(0);
    REQUIRE(!prob.distance_matrix().is_mapped());
    REQUIRE_THROWS_WITH(
      prob.use_mmap_distance_matrix(cache),
      Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
  }

  SECTION("a direct band write detaches at the next fill without rewriting the old cache")
  {
    const ScratchDirectory cache_dir{ "mmap_post_bind_direct_band" };
    const fs::path cache = cache_dir.path / "distances.dtwcache";
    Problem prob{"cache_mutation"};
    prob.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    prob.use_mmap_distance_matrix(cache);
    prob.fill_distance_matrix();
    const double full = prob.dist_by_ind(0, 1);

    prob.band = 0; // the v1 field; the next fill takes it
    REQUIRE_FALSE(prob.is_distance_matrix_filled());
    prob.fill_distance_matrix();
    REQUIRE(!std::as_const(prob).distance_matrix().is_mapped());

    Problem reopened{"cache_mutation"};
    reopened.set_data(make_data({{0.0, 1.0, 2.0}, {0.0, 2.0, 3.0}}));
    reopened.use_mmap_distance_matrix(cache); // still the full-DTW cache
    REQUIRE(reopened.is_distance_matrix_filled());
    REQUIRE(reopened.dist_by_ind(0, 1) == full);
  }
#endif
}

TEST_CASE("Problem non-L1 mmap identity is filled by the CPU in that metric",
          "[variant][distmat][mmap][fingerprint][metric]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  // IF-2 S2: binding a squared-L2 cache sets the Problem's metric, which the
  // CPU kernels take, so the fill computes in it (it was refused as
  // external-fill-only while the CPU computed L1 only).
  const ScratchDirectory cache_dir{ "mmap_external_metric" };
  const fs::path cache = cache_dir.path / "distances.dtwcache";
  Problem prob{"cache_metric"};
  const std::vector<data_t> x{0.0, 1.0, 2.0}, y{0.0, 3.0, 5.0}; // L1 5, squared L2 11
  prob.set_data(make_data({x, y}));
  prob.use_mmap_distance_matrix(cache, core::MetricType::SquaredL2);
  REQUIRE(prob.metric() == core::MetricType::SquaredL2);

  prob.fill_distance_matrix();
  REQUIRE(std::as_const(prob).distance_matrix().is_mapped());
  // The fill may run the SIMD lanes, the free function the per-pair kernel.
  REQUIRE(test_support::dtw_routes_agree<data_t>(
    prob.dist_by_ind(0, 1), distance::dtw<data_t>(x, y, -1, core::MetricType::SquaredL2),
    x.size(), y.size()));
  REQUIRE(prob.dist_by_ind(0, 1) != distance::dtw<data_t>(x, y));
#endif
}

TEST_CASE("Problem rejects CUDA Auto precision for persistent mmap identity",
          "[variant][distmat][mmap][fingerprint][cuda]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  const ScratchDirectory cache_dir{ "mmap_cuda_auto_precision" };
  const fs::path cache = cache_dir.path / "distances.dtwcache";
  Problem prob{"cache_cuda_auto"};
  prob.set_data(make_data({{0.0, 1.0}, {1.0, 2.0}}));
  prob.set_distance_strategy(DistanceMatrixStrategy::CUDA); // precision stays Auto, the default

  REQUIRE_THROWS_WITH(
    prob.use_mmap_distance_matrix(cache),
    Catch::Matchers::ContainsSubstring("CUDA precision=Auto")
      && Catch::Matchers::ContainsSubstring("explicit FP32 or FP64"));
  REQUIRE_FALSE(fs::exists(cache));
#endif
}
