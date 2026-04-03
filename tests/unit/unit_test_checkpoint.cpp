/**
 * @file unit_test_checkpoint.cpp
 * @brief Tests for checkpoint save/load functionality.
 *
 * Verifies:
 *  - Round-trip: save then load produces identical distance matrix
 *  - Partial checkpoint: only some pairs computed
 *  - Metadata round-trips correctly
 *  - Missing directory is created automatically
 *  - Dimension mismatch is detected
 *  - Missing checkpoint returns false
 *  - Fully computed matrix sets filled flag
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../test_util.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInitCheckpoint {
  TestDataInitCheckpoint() { dtwc::settings::paths::setDataPath(DTWC_TEST_DATA_DIR); }
} test_data_init_checkpoint_;

using Catch::Matchers::WithinAbs;
using namespace dtwc;
namespace fs = std::filesystem;

namespace {

/// Build a Problem from the dummy dataset with N series.
Problem make_problem(int N_data)
{
  dtwc::DataLoader dl{ settings::paths::dataPath / "dummy", N_data };
  dl.startColumn(1).startRow(1);
  dtwc::Problem prob{ "checkpoint_test", dl };
  return prob;
}

/// Helper to create a unique temporary directory for each test.
std::string make_temp_dir(const std::string &suffix)
{
  auto dir = fs::temp_directory_path() / ("dtwc_test_ckpt_" + suffix);
  // Clean up if left over from previous test run
  if (fs::exists(dir))
    fs::remove_all(dir);
  return dir.string();
}

/// Clean up a temporary directory.
void cleanup_dir(const std::string &dir)
{
  if (fs::exists(dir))
    fs::remove_all(dir);
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// 1. Round-trip: save then load produces identical distance matrix
// ---------------------------------------------------------------------------
TEST_CASE("Checkpoint round-trip preserves full distance matrix", "[checkpoint]")
{
  constexpr int N = 8;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());

  auto ckpt_dir = make_temp_dir("roundtrip");

  // Save
  REQUIRE_NOTHROW(save_checkpoint(prob, ckpt_dir));

  // Verify files exist
  REQUIRE(fs::exists(fs::path(ckpt_dir) / "distances.csv"));
  REQUIRE(fs::exists(fs::path(ckpt_dir) / "metadata.txt"));

  // Load into a fresh problem with same data
  auto prob2 = make_problem(N);
  REQUIRE_FALSE(prob2.isDistanceMatrixFilled());

  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE(loaded);
  REQUIRE(prob2.isDistanceMatrixFilled());

  // Compare all entries
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(prob2.distByInd(i, j), WithinAbs(prob.distByInd(i, j), 1e-10));
    }
  }

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 2. Partial checkpoint: only some pairs computed
// ---------------------------------------------------------------------------
TEST_CASE("Checkpoint saves and loads partial distance matrix", "[checkpoint]")
{
  constexpr int N = 6;
  auto prob = make_problem(N);

  // Compute only a few pairs (not all)
  prob.distByInd(0, 1);
  prob.distByInd(0, 2);
  prob.distByInd(2, 3);
  REQUIRE_FALSE(prob.isDistanceMatrixFilled());

  auto ckpt_dir = make_temp_dir("partial");

  save_checkpoint(prob, ckpt_dir);

  // Load into fresh problem
  auto prob2 = make_problem(N);
  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE(loaded);
  REQUIRE_FALSE(prob2.isDistanceMatrixFilled()); // Not all pairs computed

  // Verify the computed pairs match
  REQUIRE_THAT(prob2.distByInd(0, 1), WithinAbs(prob.distByInd(0, 1), 1e-10));
  REQUIRE_THAT(prob2.distByInd(0, 2), WithinAbs(prob.distByInd(0, 2), 1e-10));
  REQUIRE_THAT(prob2.distByInd(2, 3), WithinAbs(prob.distByInd(2, 3), 1e-10));
  // Symmetry preserved
  REQUIRE_THAT(prob2.distByInd(1, 0), WithinAbs(prob.distByInd(1, 0), 1e-10));

  // Uncomputed pair should still be NaN in the raw matrix
  REQUIRE_FALSE(prob2.distance_matrix().is_computed(4, 5));

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 3. Metadata round-trips correctly
// ---------------------------------------------------------------------------
TEST_CASE("Checkpoint metadata file contains expected fields", "[checkpoint]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  auto ckpt_dir = make_temp_dir("metadata");
  save_checkpoint(prob, ckpt_dir);

  // Read metadata file and verify contents
  std::string content;
  {
    std::ifstream meta_file(fs::path(ckpt_dir) / "metadata.txt");
    REQUIRE(meta_file.good());
    content.assign((std::istreambuf_iterator<char>(meta_file)),
                    std::istreambuf_iterator<char>());
  } // meta_file closed here

  // Check key fields are present
  REQUIRE(content.find("n=5") != std::string::npos);
  REQUIRE(content.find("band=") != std::string::npos);
  REQUIRE(content.find("variant=Standard") != std::string::npos);
  REQUIRE(content.find("pairs_computed=") != std::string::npos);
  REQUIRE(content.find("timestamp=") != std::string::npos);

  // pairs_computed should be N*N = 25 (all computed including diagonal)
  REQUIRE(content.find("pairs_computed=25") != std::string::npos);

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 4. Missing checkpoint directory creates it
// ---------------------------------------------------------------------------
TEST_CASE("save_checkpoint creates directory if it does not exist", "[checkpoint]")
{
  constexpr int N = 3;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  auto ckpt_dir = make_temp_dir("create_dir");
  // Ensure it does not exist
  if (fs::exists(ckpt_dir))
    fs::remove_all(ckpt_dir);
  REQUIRE_FALSE(fs::exists(ckpt_dir));

  REQUIRE_NOTHROW(save_checkpoint(prob, ckpt_dir));
  REQUIRE(fs::exists(ckpt_dir));
  REQUIRE(fs::exists(fs::path(ckpt_dir) / "distances.csv"));
  REQUIRE(fs::exists(fs::path(ckpt_dir) / "metadata.txt"));

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 5. Dimension mismatch is detected
// ---------------------------------------------------------------------------
TEST_CASE("load_checkpoint rejects dimension mismatch", "[checkpoint]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  auto ckpt_dir = make_temp_dir("mismatch");
  save_checkpoint(prob, ckpt_dir);

  // Load into a problem with different N
  auto prob2 = make_problem(3); // 3 != 5
  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE_FALSE(loaded);

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 6. Missing checkpoint returns false
// ---------------------------------------------------------------------------
TEST_CASE("load_checkpoint returns false for nonexistent path", "[checkpoint]")
{
  constexpr int N = 3;
  auto prob = make_problem(N);

  bool loaded = load_checkpoint(prob, "/nonexistent/path/that/does/not/exist");
  REQUIRE_FALSE(loaded);
}


// ---------------------------------------------------------------------------
// 7. Overwriting an existing checkpoint works
// ---------------------------------------------------------------------------
TEST_CASE("save_checkpoint overwrites existing checkpoint", "[checkpoint]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  auto ckpt_dir = make_temp_dir("overwrite");

  // First save with partial data
  prob.distByInd(0, 1);
  save_checkpoint(prob, ckpt_dir);

  // Now fill fully and save again
  prob.fillDistanceMatrix();
  save_checkpoint(prob, ckpt_dir);

  // Load and verify it's the full matrix
  auto prob2 = make_problem(N);
  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE(loaded);
  REQUIRE(prob2.isDistanceMatrixFilled());

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 8. DenseDistanceMatrix count_computed and all_computed
// ---------------------------------------------------------------------------
TEST_CASE("DenseDistanceMatrix count_computed and all_computed", "[checkpoint][distance_matrix]")
{
  core::DenseDistanceMatrix dm(4);

  // Initially all NaN
  REQUIRE(dm.count_computed() == 0);
  REQUIRE_FALSE(dm.all_computed());

  // Set a few entries (set is symmetric, so sets 2 entries per call)
  dm.set(0, 0, 0.0);
  dm.set(1, 1, 0.0);
  dm.set(0, 1, 1.5);  // sets (0,1) and (1,0)

  // 0,0 + 1,1 + 0,1 + 1,0 = 4 entries
  REQUIRE(dm.count_computed() == 4);
  REQUIRE_FALSE(dm.all_computed());

  // Fill the rest
  dm.set(2, 2, 0.0);
  dm.set(3, 3, 0.0);
  dm.set(0, 2, 2.0);
  dm.set(0, 3, 3.0);
  dm.set(1, 2, 4.0);
  dm.set(1, 3, 5.0);
  dm.set(2, 3, 6.0);

  REQUIRE(dm.count_computed() == 16); // 4*4
  REQUIRE(dm.all_computed());
}
