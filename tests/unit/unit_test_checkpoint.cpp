/**
 * @file unit_test_checkpoint.cpp
 * @brief Tests for checkpoint save/load functionality.
 *
 * Verifies:
 *  - Round-trip: save then load produces identical distance matrix
 *  - Partial checkpoint: only some pairs computed
 *  - V2 manifest metadata is published correctly
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
  TestDataInitCheckpoint() { dtwc::settings::paths::set_data_path(DTWC_TEST_DATA_DIR); }
} test_data_init_checkpoint_;

using Catch::Matchers::WithinAbs;
using namespace dtwc;
namespace fs = std::filesystem;

namespace {

/// Build a Problem from the dummy dataset with N series.
Problem make_problem(int N_data)
{
  dtwc::DataLoader dl{ settings::paths::data / "dummy", N_data };
  dl.start_column(1).start_row(1);
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

/// Resolve the immutable payload selected by the strict CURRENT file.
fs::path active_checkpoint_payload(const std::string &dir)
{
  const fs::path root(dir);
  std::ifstream current(root / "CURRENT", std::ios::binary);
  REQUIRE(current.good());
  std::string generation;
  REQUIRE(static_cast<bool>(std::getline(current, generation)));
  REQUIRE(generation.size() == 64);
  REQUIRE(current.peek() == std::char_traits<char>::eof());
  return root / "generations" / generation;
}


/// pairs_computed recorded in the manifest of the active generation.
std::size_t manifest_pairs_computed(const std::string &dir)
{
  std::ifstream manifest(active_checkpoint_payload(dir) / "metadata.txt",
                         std::ios::binary);
  REQUIRE(manifest.good());
  const std::string key = "pairs_computed=";
  std::string line;
  while (std::getline(manifest, line))
    if (line.rfind(key, 0) == 0)
      return static_cast<std::size_t>(std::stoull(line.substr(key.size())));
  FAIL("checkpoint manifest has no pairs_computed key");
  return 0;
}


/// Number of immutable generations published under a checkpoint root.
std::size_t count_generations(const std::string &dir)
{
  const fs::path generations = fs::path(dir) / "generations";
  if (!fs::exists(generations)) return 0;
  std::size_t total = 0;
  for (const auto &entry : fs::directory_iterator(generations))
    if (entry.is_directory()) ++total;
  return total;
}

/// Bit-exact comparison of two resident dense distance matrices.
void require_identical_matrices(const Problem &lhs, const Problem &rhs)
{
  const auto &a = lhs.dense_distance_matrix();
  const auto &b = rhs.dense_distance_matrix();
  REQUIRE(a.size() == b.size());
  REQUIRE(a.packed_count() == b.packed_count());
  for (std::size_t k = 0; k < a.packed_count(); ++k)
    REQUIRE(a.raw()[k] == b.raw()[k]);
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// 1. Round-trip: save then load produces identical distance matrix
// ---------------------------------------------------------------------------
TEST_CASE("Checkpoint round-trip preserves full distance matrix", "[checkpoint]")
{
  constexpr int N = 8;
  auto prob = make_problem(N);
  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());

  auto ckpt_dir = make_temp_dir("roundtrip");

  // Save
  REQUIRE_NOTHROW(save_checkpoint(prob, ckpt_dir));

  // Verify files exist
  const fs::path payload = active_checkpoint_payload(ckpt_dir);
  REQUIRE(fs::exists(payload / "distances.csv"));
  REQUIRE(fs::exists(payload / "metadata.txt"));

  // Load into a fresh problem with same data
  auto prob2 = make_problem(N);
  REQUIRE_FALSE(prob2.is_distance_matrix_filled());

  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE(loaded);
  REQUIRE(prob2.is_distance_matrix_filled());

  // Compare all entries
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(prob2.dist_by_ind(i, j), WithinAbs(prob.dist_by_ind(i, j), 1e-10));
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
  prob.dist_by_ind(0, 1);
  prob.dist_by_ind(0, 2);
  prob.dist_by_ind(2, 3);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());

  auto ckpt_dir = make_temp_dir("partial");

  save_checkpoint(prob, ckpt_dir);

  // Load into fresh problem
  auto prob2 = make_problem(N);
  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE(loaded);
  REQUIRE_FALSE(prob2.is_distance_matrix_filled()); // Not all pairs computed

  // Verify the computed pairs match
  REQUIRE_THAT(prob2.dist_by_ind(0, 1), WithinAbs(prob.dist_by_ind(0, 1), 1e-10));
  REQUIRE_THAT(prob2.dist_by_ind(0, 2), WithinAbs(prob.dist_by_ind(0, 2), 1e-10));
  REQUIRE_THAT(prob2.dist_by_ind(2, 3), WithinAbs(prob.dist_by_ind(2, 3), 1e-10));
  // Symmetry preserved
  REQUIRE_THAT(prob2.dist_by_ind(1, 0), WithinAbs(prob.dist_by_ind(1, 0), 1e-10));

  // Uncomputed pair should still be NaN in the raw matrix
  REQUIRE_FALSE(prob2.dense_distance_matrix().is_computed(4, 5));

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 3. Metadata round-trips correctly
// ---------------------------------------------------------------------------
TEST_CASE("Checkpoint metadata file contains expected fields", "[checkpoint]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);
  prob.fill_distance_matrix();

  auto ckpt_dir = make_temp_dir("metadata");
  save_checkpoint(prob, ckpt_dir);

  // Read the active v2 manifest and verify its canonical fields.
  std::string content;
  {
    std::ifstream meta_file(active_checkpoint_payload(ckpt_dir) / "metadata.txt");
    REQUIRE(meta_file.good());
    content.assign((std::istreambuf_iterator<char>(meta_file)),
                    std::istreambuf_iterator<char>());
  } // meta_file closed here

  // Check identity/integrity fields as well as the logical payload shape.
  REQUIRE(content.find("format=dtwc-dense-checkpoint\n") != std::string::npos);
  REQUIRE(content.find("version=2\n") != std::string::npos);
  REQUIRE(content.find("n=5") != std::string::npos);
  REQUIRE(content.find("pairs_computed=") != std::string::npos);
  REQUIRE(content.find("timestamp=") != std::string::npos);
  REQUIRE(content.find("identity_sha256=") != std::string::npos);
  REQUIRE(content.find("payload_sha256=") != std::string::npos);

  // pairs_computed = N*(N+1)/2 = 15 (packed triangular, all computed)
  REQUIRE(content.find("pairs_computed=15") != std::string::npos);

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 4. Missing checkpoint directory creates it
// ---------------------------------------------------------------------------
TEST_CASE("save_checkpoint creates directory if it does not exist", "[checkpoint]")
{
  constexpr int N = 3;
  auto prob = make_problem(N);
  prob.fill_distance_matrix();

  auto ckpt_dir = make_temp_dir("create_dir");
  // Ensure it does not exist
  if (fs::exists(ckpt_dir))
    fs::remove_all(ckpt_dir);
  REQUIRE_FALSE(fs::exists(ckpt_dir));

  REQUIRE_NOTHROW(save_checkpoint(prob, ckpt_dir));
  REQUIRE(fs::exists(ckpt_dir));
  const fs::path payload = active_checkpoint_payload(ckpt_dir);
  REQUIRE(fs::exists(payload / "distances.csv"));
  REQUIRE(fs::exists(payload / "metadata.txt"));

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 5. Dimension mismatch is detected
// ---------------------------------------------------------------------------
TEST_CASE("load_checkpoint rejects dimension mismatch", "[checkpoint]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);
  prob.fill_distance_matrix();

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
  prob.dist_by_ind(0, 1);
  save_checkpoint(prob, ckpt_dir);

  // Now fill fully and save again
  prob.fill_distance_matrix();
  save_checkpoint(prob, ckpt_dir);

  // The superseded partial generation is gone: one directory, one generation.
  REQUIRE(count_generations(ckpt_dir) == 1);
  REQUIRE(manifest_pairs_computed(ckpt_dir) == N * (N + 1) / 2);

  // Load and verify it's the full matrix
  auto prob2 = make_problem(N);
  bool loaded = load_checkpoint(prob2, ckpt_dir);
  REQUIRE(loaded);
  REQUIRE(prob2.is_distance_matrix_filled());

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

  // Set a few entries (packed triangular: each set() marks 1 unique entry)
  dm.set(0, 0, 0.0);
  dm.set(1, 1, 0.0);
  dm.set(0, 1, 1.5);  // maps to tri(1,0) — single entry

  // 3 unique packed entries: (0,0), (1,1), tri(1,0)
  REQUIRE(dm.count_computed() == 3);
  REQUIRE_FALSE(dm.all_computed());

  // Fill the rest — N=4 has N*(N+1)/2 = 10 packed entries
  dm.set(2, 2, 0.0);
  dm.set(3, 3, 0.0);
  dm.set(0, 2, 2.0);
  dm.set(0, 3, 3.0);
  dm.set(1, 2, 4.0);
  dm.set(1, 3, 5.0);
  dm.set(2, 3, 6.0);

  REQUIRE(dm.count_computed() == 10); // N*(N+1)/2
  REQUIRE(dm.all_computed());
}


// ---------------------------------------------------------------------------
// 9. Automatic mid-fill checkpointing: one generation per row block
// ---------------------------------------------------------------------------
TEST_CASE("Automatic checkpointing retains exactly one generation",
          "[checkpoint][fill]")
{
  constexpr int N = 5;
  auto reference = make_problem(N);
  reference.fill_distance_matrix();

  auto ckpt_dir = make_temp_dir("auto_interval1");
  auto prob = make_problem(N);
  prob.checkpoint.enabled = true;
  prob.checkpoint.save_interval = 1;
  prob.checkpoint.directory = ckpt_dir;
  REQUIRE_NOTHROW(prob.fill_distance_matrix());
  REQUIRE(prob.is_distance_matrix_filled());

  // Five blocks each publish a generation, and each publication supersedes the
  // previous one, so the directory holds exactly one: the complete matrix.
  REQUIRE(count_generations(ckpt_dir) == 1);
  REQUIRE(manifest_pairs_computed(ckpt_dir) == N * (N + 1) / 2);

  auto restored = make_problem(N);
  REQUIRE(load_checkpoint(restored, ckpt_dir));
  REQUIRE(restored.is_distance_matrix_filled());
  require_identical_matrices(restored, reference);

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 10. Crash-resume: a resumed fill recomputes only the missing cells
// ---------------------------------------------------------------------------
TEST_CASE("Resumed automatic fill only computes uncomputed cells",
          "[checkpoint][fill]")
{
  constexpr int N = 6;
  auto reference = make_problem(N);
  reference.fill_distance_matrix();

  // Stand-in for a crash after the first two rows: rows 0 and 1 are marked
  // computed with sentinel values no DTW kernel can produce.
  auto crashed = make_problem(N);
  {
    auto &matrix = crashed.dense_distance_matrix();
    matrix.resize(N); // allocation is deferred until the first fill
    for (int i = 0; i < 2; ++i)
      for (int j = i + 1; j < N; ++j)
        matrix.set(i, j, 900.0 + 10.0 * i + j);
  }
  auto partial_dir = make_temp_dir("resume_partial");
  save_checkpoint(crashed, partial_dir);
  // Mid-fill evidence survives pruning: the surviving generation is the partial
  // one, and its manifest counts exactly the cells computed so far.
  REQUIRE(count_generations(partial_dir) == 1);
  REQUIRE(manifest_pairs_computed(partial_dir) == 9);

  auto resumed = make_problem(N);
  REQUIRE(load_checkpoint(resumed, partial_dir));
  // 5 entries from row 0 plus 4 from row 1; the diagonal is still uncomputed.
  REQUIRE(resumed.dense_distance_matrix().count_computed() == 9);

  auto resume_dir = make_temp_dir("resume_fill");
  resumed.checkpoint.enabled = true;
  resumed.checkpoint.save_interval = 2;
  resumed.checkpoint.directory = resume_dir;
  resumed.fill_distance_matrix();

  REQUIRE(resumed.is_distance_matrix_filled());
  REQUIRE(resumed.dense_distance_matrix().count_computed() == N * (N + 1) / 2);
  REQUIRE(count_generations(resume_dir) == 1);
  REQUIRE(manifest_pairs_computed(resume_dir) == N * (N + 1) / 2);

  // Poisoned cells survive untouched: the resumed fill did not recompute them.
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < N; ++j)
      REQUIRE(resumed.dist_by_ind(i, j) == 900.0 + 10.0 * i + j);

  // Every previously missing cell is the exact brute-force value.
  for (int i = 2; i < N; ++i)
    for (int j = i; j < N; ++j)
      REQUIRE(resumed.dist_by_ind(i, j) == reference.dist_by_ind(i, j));

  cleanup_dir(partial_dir);
  cleanup_dir(resume_dir);
}


// ---------------------------------------------------------------------------
// 11. Invalid automatic-checkpoint settings fail before any work
// ---------------------------------------------------------------------------
TEST_CASE("Automatic checkpointing rejects a non-positive save interval",
          "[checkpoint][fill]")
{
  auto ckpt_dir = make_temp_dir("auto_interval0");
  auto prob = make_problem(4);
  prob.checkpoint.enabled = true;
  prob.checkpoint.save_interval = 0;
  prob.checkpoint.directory = ckpt_dir;

  REQUIRE_THROWS_AS(prob.fill_distance_matrix(), InvalidInput);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  REQUIRE_FALSE(fs::exists(ckpt_dir));
}


// ---------------------------------------------------------------------------
// 12. Disabled checkpointing writes nothing
// ---------------------------------------------------------------------------
TEST_CASE("Disabled checkpointing creates no checkpoint directory",
          "[checkpoint][fill]")
{
  auto ckpt_dir = make_temp_dir("auto_disabled");
  auto prob = make_problem(5);
  prob.checkpoint.directory = ckpt_dir;
  REQUIRE_FALSE(prob.checkpoint.enabled); // default

  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE_FALSE(fs::exists(ckpt_dir));
}


// ---------------------------------------------------------------------------
// 13. Mapped distance storage rejects automatic checkpointing
// ---------------------------------------------------------------------------
#ifdef DTWC_HAS_MMAP
TEST_CASE("Automatic checkpointing rejects mmap distance storage",
          "[checkpoint][fill]")
{
  auto scratch = make_temp_dir("auto_mmap");
  fs::create_directories(scratch);
  auto ckpt_dir = make_temp_dir("auto_mmap_ckpt");

  auto prob = make_problem(5);
  prob.use_mmap_distance_matrix(fs::path(scratch) / "distmat.dtwcache");
  prob.checkpoint.enabled = true;
  prob.checkpoint.save_interval = 2;
  prob.checkpoint.directory = ckpt_dir;

  REQUIRE_THROWS_AS(prob.fill_distance_matrix(), InvalidInput);
  REQUIRE_FALSE(fs::exists(ckpt_dir));

  cleanup_dir(scratch);
}
#endif


// ---------------------------------------------------------------------------
// 14. Pruned + automatic checkpointing downgrades to the exact row schedule
// ---------------------------------------------------------------------------
TEST_CASE("Automatic checkpointing downgrades Pruned to BruteForce",
          "[checkpoint][fill]")
{
  constexpr int N = 8;
  auto reference = make_problem(N);
  reference.distance_strategy = DistanceMatrixStrategy::BruteForce;
  reference.fill_distance_matrix();

  auto ckpt_dir = make_temp_dir("auto_pruned");
  auto prob = make_problem(N);
  prob.distance_strategy = DistanceMatrixStrategy::Pruned;
  prob.checkpoint.enabled = true;
  prob.checkpoint.save_interval = 3;
  prob.checkpoint.directory = ckpt_dir;
  REQUIRE_NOTHROW(prob.fill_distance_matrix());

  REQUIRE(prob.is_distance_matrix_filled());
  REQUIRE(count_generations(ckpt_dir) == 1);
  require_identical_matrices(prob, reference);

  cleanup_dir(ckpt_dir);
}


// ---------------------------------------------------------------------------
// 15. Automatic checkpointing rejects an empty directory before any work
// ---------------------------------------------------------------------------
TEST_CASE("Automatic checkpointing rejects an empty directory",
          "[checkpoint][fill]")
{
  auto prob = make_problem(4);
  prob.checkpoint.enabled = true;
  prob.checkpoint.save_interval = 2;
  prob.checkpoint.directory.clear();

  REQUIRE_THROWS_AS(prob.fill_distance_matrix(), InvalidInput);
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  REQUIRE(prob.dense_distance_matrix().count_computed() == 0);
}


// ---------------------------------------------------------------------------
// 16. A throwing automatic save keeps the cells the completed blocks computed
// ---------------------------------------------------------------------------
TEST_CASE("A failing automatic save preserves computed distances",
          "[checkpoint][fill]")
{
  constexpr int N = 6;
  auto reference = make_problem(N);
  reference.fill_distance_matrix();

  // A regular file where the checkpoint root belongs: the first save fails in
  // the filesystem, after its row block has already been computed.
  auto scratch = make_temp_dir("save_throws");
  fs::create_directories(scratch);
  const fs::path blocking_file = fs::path(scratch) / "not_a_directory";
  {
    std::ofstream blocker(blocking_file);
    blocker << "occupied";
  }
  REQUIRE(fs::is_regular_file(blocking_file));

  auto prob = make_problem(N);
  {
    // Row 0 is poisoned with values no DTW kernel can produce, so any later
    // recomputation of those cells is visible.
    auto &matrix = prob.dense_distance_matrix();
    matrix.resize(N);
    for (int j = 1; j < N; ++j)
      matrix.set(0, j, 900.0 + j);
  }
  prob.checkpoint.enabled = true;
  prob.checkpoint.save_interval = 2;
  prob.checkpoint.directory = blocking_file.string();

  REQUIRE_THROWS(prob.fill_distance_matrix());
  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  // Diagonal (6) + poisoned row 0 (5) + genuine row 1 (4) from the first block.
  REQUIRE(prob.dense_distance_matrix().count_computed() == 15);

  // A subsequent disabled fill completes and touches only uncomputed cells.
  prob.checkpoint.enabled = false;
  REQUIRE_NOTHROW(prob.fill_distance_matrix());
  REQUIRE(prob.is_distance_matrix_filled());
  for (int j = 1; j < N; ++j)
    REQUIRE(prob.dist_by_ind(0, j) == 900.0 + j);
  for (int i = 1; i < N; ++i)
    for (int j = i; j < N; ++j)
      REQUIRE(prob.dist_by_ind(i, j) == reference.dist_by_ind(i, j));

  cleanup_dir(scratch);
}
