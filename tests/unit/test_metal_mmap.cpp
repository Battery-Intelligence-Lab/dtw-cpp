/**
 * @file test_metal_mmap.cpp
 * @brief Verify the Metal backend writes correctly into the distance matrix
 *        on the heap and mapped to a file.
 *
 *        Problem::fill_distance_matrix() writes the GPU result through the one
 *        DistanceMatrix whichever storage holds it; this test exercises both via
 *        the Metal strategy and confirms the numbers match the CPU reference.
 *
 * @date 2026-04-12
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <dtwc.hpp>
#include <algorithms/fast_clara.hpp>

#include "../support/scratch_directory.hpp"

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_METAL
TEST_CASE("Metal mmap path skipped", "[metal][mmap]")
{
  SKIP("DTWC_HAS_METAL not defined");
}
#else

namespace {
std::vector<std::vector<double>> gen(size_t N, size_t L, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<std::vector<double>> s(N);
  for (auto &v : s) {
    v.resize(L);
    for (auto &x : v) x = d(rng);
  }
  return s;
}

dtwc::Problem make_problem(size_t N, size_t L, unsigned seed)
{
  auto vecs = gen(N, L, seed);
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = "s" + std::to_string(i);
  dtwc::Data data{std::move(vecs), std::move(names)};
  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.band = -1;
  return prob;
}
} // namespace

TEST_CASE("Metal strategy via Problem::fill_distance_matrix (dense)", "[metal][dispatch]")
{
  const size_t N = 6;
  const size_t L = 64;

  auto prob_cpu = make_problem(N, L, 999);
  prob_cpu.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_cpu.fill_distance_matrix();

  auto prob_gpu = make_problem(N, L, 999);
  prob_gpu.distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  prob_gpu.fill_distance_matrix();

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      CAPTURE(i, j);
      double cpu_val = prob_cpu.dist_by_ind(int(i), int(j));
      double gpu_val = prob_gpu.dist_by_ind(int(i), int(j));
      REQUIRE_THAT(gpu_val,
                   WithinRel(cpu_val, 1e-4) || WithinAbs(cpu_val, 1e-3));
    }
  }
}

TEST_CASE("Metal strategy via Problem::fill_distance_matrix (mmap)", "[metal][mmap]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in");
#else
  const size_t N = 6;
  const size_t L = 64;

  const dtwc::test_support::ScratchDirectory scratch{ "metal_mmap" };
  const auto &tmpdir = scratch.path;

  auto prob_cpu = make_problem(N, L, 777);
  prob_cpu.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_cpu.fill_distance_matrix();

  auto prob_gpu = make_problem(N, L, 777);
  prob_gpu.set_output_folder(tmpdir);
  prob_gpu.distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  // A cache left by an earlier run would reopen filled and skip the GPU.
  std::filesystem::remove(tmpdir / "metal_mmap_distmat.bin");
  prob_gpu.use_mmap_distance_matrix(tmpdir / "metal_mmap_distmat.bin");
  prob_gpu.fill_distance_matrix();

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      CAPTURE(i, j);
      double cpu_val = prob_cpu.dist_by_ind(int(i), int(j));
      double gpu_val = prob_gpu.dist_by_ind(int(i), int(j));
      REQUIRE_THAT(gpu_val,
                   WithinRel(cpu_val, 1e-4) || WithinAbs(cpu_val, 1e-3));
    }
  }
#endif
}

// FX-1: a squared-L2 cache is filled by the Problem's Metal route with the
// squared-L2 kernel (it used to be refused as external-fill-only), and matches
// the CPU squared-L2 kernels, banded and full, on variable-length series.
TEST_CASE("Metal squared-L2 cache via Problem::fill_distance_matrix", "[metal][mmap][fx1]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in");
#else
  const dtwc::test_support::ScratchDirectory scratch{ "metal_sql2_mmap" };
  const auto &tmpdir = scratch.path;

  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < 6; ++i) series.push_back(gen(1, 60 + i, 31 + i).front());

  for (const int band : { -1, 8 }) {
    CAPTURE(band);
    const auto cache = tmpdir / ("metal_sql2_band" + std::to_string(band) + ".bin");
    std::filesystem::remove(cache);

    dtwc::Problem prob("metal_sql2");
    prob.set_data(dtwc::Data{ std::vector<std::vector<double>>(series),
                              { "s0", "s1", "s2", "s3", "s4", "s5" } });
    prob.set_band(band);
    prob.set_distance_strategy(dtwc::DistanceMatrixStrategy::Metal);
    prob.use_mmap_distance_matrix(cache, dtwc::core::MetricType::SquaredL2);
    prob.fill_distance_matrix();

    for (size_t i = 0; i < series.size(); ++i) {
      for (size_t j = i + 1; j < series.size(); ++j) {
        CAPTURE(i, j);
        const double oracle = band < 0
          ? dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                    dtwc::core::MetricType::SquaredL2)
          : dtwc::dtwBanded<double>(series[i], series[j], band, -1.0,
                                    dtwc::core::MetricType::SquaredL2);
        REQUIRE_THAT(prob.dist_by_ind(int(i), int(j)),
                     WithinRel(oracle, 1e-4) || WithinAbs(oracle, 1e-3));
      }
    }
  }
#endif
}

// IF-2 S2: the metric is the Problem's (set_metric), so a dense Metal fill
// computes squared L2 without a cache, within the FP32 band above.
TEST_CASE("set_metric: the Metal fill computes squared L2", "[metal][metric][if2]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < 6; ++i) series.push_back(gen(1, 60 + i, 41 + i).front());
  for (const int band : { -1, 8 }) {
    CAPTURE(band);
    dtwc::Problem prob("metal_dense_sql2");
    prob.set_data(dtwc::Data{ std::vector<std::vector<double>>(series),
                              { "s0", "s1", "s2", "s3", "s4", "s5" } });
    prob.set_band(band);
    prob.set_device(dtwc::Device::GPU);
    REQUIRE(prob.distance_strategy == dtwc::DistanceMatrixStrategy::Metal);
    prob.set_metric(dtwc::core::MetricType::SquaredL2);
    prob.fill_distance_matrix();
    for (size_t i = 0; i < series.size(); ++i)
      for (size_t j = i + 1; j < series.size(); ++j) {
        CAPTURE(i, j);
        const double oracle = dtwc::distance::dtw<double>(
          series[i], series[j], band, dtwc::core::MetricType::SquaredL2);
        const double l1 = dtwc::distance::dtw<double>(series[i], series[j], band);
        REQUIRE_FALSE(std::abs(oracle - l1) <= 1e-3 * std::abs(l1)); // discriminates
        REQUIRE_THAT(prob.dist_by_ind(int(i), int(j)),
                     WithinRel(oracle, 1e-4) || WithinAbs(oracle, 1e-3));
      }
  }
}

// IF-2 S2: FastCLARA's samples take the parent's device. An in-memory sample is
// a view of the parent's series, which the GPU fill refuses before any pair; a
// sample covering every series is FastPAM on the parent, which runs on the GPU.
TEST_CASE("FastCLARA on a GPU device: a view sample is refused, a full sample runs",
          "[metal][fast_clara][if2]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (int i = 0; i < 12; ++i) {
    const double base = i < 6 ? 0.0 : 20.0;
    series.push_back({ base + 0.1 * i, base + 1.0, base + 0.5, base - 0.2 * i });
    names.push_back("s" + std::to_string(i));
  }
  dtwc::Problem prob("metal_clara");
  prob.set_data(dtwc::Data{ std::move(series), std::move(names) });
  prob.set_device(dtwc::Device::GPU);

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 2;
  opts.n_samples = 2;
  opts.sample_size = 8;
  REQUIRE_THROWS_MATCHES(dtwc::algorithms::fast_clara(prob, opts), dtwc::DeviceError,
                         Catch::Matchers::MessageMatches(
                           Catch::Matchers::ContainsSubstring("non-owning view")));

  opts.sample_size = 12;
  const auto result = dtwc::algorithms::fast_clara(prob, opts);
  REQUIRE(result.labels.size() == 12);
  CHECK(result.labels[0] != result.labels[11]);
  CHECK(prob.is_distance_matrix_filled());
}

#endif // DTWC_HAS_METAL
