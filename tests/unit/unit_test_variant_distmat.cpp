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

#include <filesystem>

using namespace dtwc;
namespace fs = std::filesystem;

TEST_CASE("Problem uses DenseDistanceMatrix by default for small N", "[variant][distmat]")
{
  DataLoader dl("data/dummy");
  Problem prob("test_variant", dl);
  REQUIRE(prob.size() == 25);

  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());

  double d = prob.distByInd(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(d == prob.distByInd(1, 0)); // symmetry
}

#ifdef DTWC_HAS_MMAP
TEST_CASE("Problem uses MmapDistanceMatrix when forced", "[variant][distmat][mmap]")
{
  auto cache_path = fs::temp_directory_path() / "dtwc_test" / "variant_mmap.dtwcache";
  fs::create_directories(cache_path.parent_path());
  if (fs::exists(cache_path)) fs::remove(cache_path);

  DataLoader dl("data/dummy");
  Problem prob("test_mmap", dl);

  // Force mmap mode
  prob.use_mmap_distance_matrix(cache_path);

  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());

  double d = prob.distByInd(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(d == prob.distByInd(1, 0));

  REQUIRE(fs::exists(cache_path));
  REQUIRE(fs::file_size(cache_path) > 0);

  // Cleanup
  fs::remove_all(cache_path.parent_path());
}

TEST_CASE("MmapDistanceMatrix warmstart via Problem", "[variant][distmat][mmap]")
{
  auto cache_path = fs::temp_directory_path() / "dtwc_test" / "warmstart_prob.dtwcache";
  fs::create_directories(cache_path.parent_path());
  if (fs::exists(cache_path)) fs::remove(cache_path);

  double d01_original;

  // First run: fill distance matrix
  {
    DataLoader dl("data/dummy");
    Problem prob("test_warmstart", dl);
    prob.use_mmap_distance_matrix(cache_path);
    prob.fillDistanceMatrix();
    d01_original = prob.distByInd(0, 1);
  }

  // Second run: reopen - distances should persist
  {
    DataLoader dl("data/dummy");
    Problem prob("test_warmstart", dl);
    prob.use_mmap_distance_matrix(cache_path);
    REQUIRE(prob.isDistanceMatrixFilled());
    REQUIRE(prob.distByInd(0, 1) == d01_original);
  }

  // Cleanup
  fs::remove_all(cache_path.parent_path());
}
#endif
