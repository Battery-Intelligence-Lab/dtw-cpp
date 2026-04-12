/**
 * @file test_metal_mmap.cpp
 * @brief Verify the Metal backend writes correctly into both
 *        DenseDistanceMatrix and the memory-mapped distance matrix.
 *
 *        The Problem::fillDistanceMatrix() dispatch uses visit_distmat to
 *        route into either storage; this test exercises both paths via the
 *        Metal strategy and confirms the numbers match the CPU reference.
 *
 * @date 2026-04-12
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>

#include <filesystem>
#include <random>
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

TEST_CASE("Metal strategy via Problem::fillDistanceMatrix (dense)", "[metal][dispatch]")
{
  const size_t N = 6;
  const size_t L = 64;

  auto prob_cpu = make_problem(N, L, 999);
  prob_cpu.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_cpu.fillDistanceMatrix();

  auto prob_gpu = make_problem(N, L, 999);
  prob_gpu.distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  prob_gpu.fillDistanceMatrix();

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      CAPTURE(i, j);
      double cpu_val = prob_cpu.distByInd(int(i), int(j));
      double gpu_val = prob_gpu.distByInd(int(i), int(j));
      REQUIRE_THAT(gpu_val,
                   WithinRel(cpu_val, 1e-4) || WithinAbs(cpu_val, 1e-3));
    }
  }
}

TEST_CASE("Metal strategy via Problem::fillDistanceMatrix (mmap)", "[metal][mmap]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in");
#else
  const size_t N = 6;
  const size_t L = 64;

  auto tmpdir = std::filesystem::temp_directory_path() / "dtwc_metal_mmap_test";
  std::filesystem::create_directories(tmpdir);

  auto prob_cpu = make_problem(N, L, 777);
  prob_cpu.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_cpu.fillDistanceMatrix();

  auto prob_gpu = make_problem(N, L, 777);
  prob_gpu.output_folder = tmpdir;
  prob_gpu.use_mmap_distance_matrix(tmpdir / "metal_mmap_distmat.bin");
  prob_gpu.distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  prob_gpu.fillDistanceMatrix();

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      CAPTURE(i, j);
      double cpu_val = prob_cpu.distByInd(int(i), int(j));
      double gpu_val = prob_gpu.distByInd(int(i), int(j));
      REQUIRE_THAT(gpu_val,
                   WithinRel(cpu_val, 1e-4) || WithinAbs(cpu_val, 1e-3));
    }
  }
#endif
}

#endif // DTWC_HAS_METAL
