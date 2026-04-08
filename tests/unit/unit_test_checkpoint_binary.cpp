/**
 * @file unit_test_checkpoint_binary.cpp
 * @brief Tests for binary checkpoint save/load functionality.
 *
 * Verifies:
 *  - Round-trip: save then load produces identical ClusteringResult
 *  - Load returns false for non-existent file
 *  - Magic/version validation rejects corrupt files
 *
 * @author Volkan Kumtepeli
 * @date 08 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInitCheckpointBin {
  TestDataInitCheckpointBin() { dtwc::settings::paths::setDataPath(DTWC_TEST_DATA_DIR); }
} test_data_init_checkpoint_bin_;

using namespace dtwc;
namespace fs = std::filesystem;

namespace {

/// Build a Problem from the dummy dataset with N series.
Problem make_problem_bin(int N_data)
{
  dtwc::DataLoader dl{ settings::paths::data / "dummy", N_data };
  dl.startColumn(1).startRow(1);
  dtwc::Problem prob{ "ckpt_bin_test", dl };
  return prob;
}

/// Helper to create a unique temporary directory for each test.
fs::path make_temp_dir_bin(const std::string &suffix)
{
  auto dir = fs::temp_directory_path() / ("dtwc_test_ckpt_bin_" + suffix);
  if (fs::exists(dir))
    fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

/// Clean up a temporary directory.
void cleanup_dir_bin(const fs::path &dir)
{
  if (fs::exists(dir))
    fs::remove_all(dir);
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// 1. Round-trip: save then load produces identical ClusteringResult
// ---------------------------------------------------------------------------
TEST_CASE("Binary checkpoint save and load", "[checkpoint][binary]")
{
  auto dir = make_temp_dir_bin("roundtrip");

  constexpr int N = 8;
  auto prob = make_problem_bin(N);
  prob.fillDistanceMatrix();

  // Run clustering to get state
  auto result = dtwc::fast_pam(prob, 3, 100);

  SECTION("Save and reload produces identical state")
  {
    auto ckpt_path = dir / "state.bin";
    save_binary_checkpoint(result, ckpt_path);
    REQUIRE(fs::exists(ckpt_path));

    core::ClusteringResult loaded;
    REQUIRE(load_binary_checkpoint(loaded, ckpt_path));

    REQUIRE(loaded.labels == result.labels);
    REQUIRE(loaded.medoid_indices == result.medoid_indices);
    REQUIRE(loaded.total_cost == result.total_cost);
    REQUIRE(loaded.iterations == result.iterations);
    REQUIRE(loaded.converged == result.converged);
  }

  SECTION("Load returns false for non-existent file")
  {
    core::ClusteringResult loaded;
    REQUIRE_FALSE(load_binary_checkpoint(loaded, dir / "nonexistent.bin"));
  }

  SECTION("Load rejects file with bad magic bytes")
  {
    auto bad_path = dir / "bad_magic.bin";
    {
      std::ofstream out(bad_path, std::ios::binary);
      const char bad_magic[] = "XXXX";
      out.write(bad_magic, 4);
      // Write enough bytes to avoid short-read
      char zeros[28] = {};
      out.write(zeros, sizeof(zeros));
    }

    core::ClusteringResult loaded;
    REQUIRE_FALSE(load_binary_checkpoint(loaded, bad_path));
  }

  // Cleanup
  cleanup_dir_bin(dir);
}
