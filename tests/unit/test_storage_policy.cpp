/**
 * @file test_storage_policy.cpp
 * @brief StoragePolicy::Auto routing, hpc metadata-only load, and mmap/heap
 *        round-trip equality (Task 1.4).
 *
 * @details Each test pins a LIVE public entry point (LESSONS: tests drive real
 *          paths, never a dead sibling):
 *   (a) drives DataLoader::load_stored() with StoragePolicy::Auto — footprint
 *       routing to the mmap-backed store when the threshold is injected low.
 *   (b) drives DataLoader::load() under device='hpc' (dtwc::env()) — the
 *       metadata-only path; asserts the bulk reader is never invoked and that
 *       local series access throws ("data not resident locally").
 *   (c) drives DataLoader::load() (heap) vs DataLoader::load_stored()
 *       (StoragePolicy::Mmap) — pairwise DTW distances must be digit-identical.
 *
 * Uses the existing 6x6 fixture data/test/nonUnimodular_1_Nc_2.csv (6 series of
 * length 6) — no new test data added. mmap tests skip cleanly when llfio is not
 * compiled in, matching the existing DTWC_HAS_MMAP skip pattern (test_metal_mmap.cpp).
 *
 * @date 2026-07-07
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

using namespace dtwc;
namespace fs = std::filesystem;

namespace {

// The batch CSV fixture: 6 rows (series) x 6 numeric fields — small, real, existing.
fs::path fixture_path()
{
  return fs::path(DTWC_TEST_DATA_DIR) / "test" / "nonUnimodular_1_Nc_2.csv";
}

fs::path make_scratch_dir(const std::string &tag)
{
  const auto dir = fs::temp_directory_path() / ("dtwc_storage_policy_" + tag);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

} // namespace

// ===========================================================================
// (a) StoragePolicy::Auto with an artificially low threshold routes to mmap.
//     LIVE path: DataLoader::load_stored() (Auto branch).
// ===========================================================================
TEST_CASE("StoragePolicy::Auto routes to mmap when footprint exceeds threshold", "[storage][mmap]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (llfio); StoragePolicy::Auto->mmap unavailable");
#else
  dtwc::env().set_device("cpu"); // ensure a local device (not hpc) for storage routing

  const auto scratch = make_scratch_dir("auto");

  // REGISTERED EXPECTATION (before the run): the fixture footprint is
  // 6 series x 6 values x 8 bytes = 288 B. With ram_limit injected to 1 byte,
  // 288 > 1, so StoragePolicy::Auto MUST spill to the mmap-backed store
  // (is_mmap()==true, surfaced as a view Data), never keep it on heap.
  DataLoader dl{ fixture_path() };
  dl.verbosity(0)
    .storage_policy(core::StoragePolicy::Auto)
    .ram_limit(1)
    .mmap_cache_path(scratch / "auto_store.dtws");

  LoadedData loaded = dl.load_stored();

  REQUIRE(loaded.is_mmap());        // routed to the mmap-backed store
  REQUIRE(loaded.data.is_view());   // mmap route surfaces as a view (span machinery preserved)
  REQUIRE(loaded.data.size() > 0);

  fs::remove_all(scratch);
#endif
}

// ===========================================================================
// (b) device='hpc' load is metadata-only: touches NO bulk reader, and local
//     series access throws. LIVE path: DataLoader::load() under dtwc::env() hpc.
// ===========================================================================
TEST_CASE("device='hpc' load is metadata-only and never invokes the bulk reader", "[storage][hpc]")
{
  // Point the global Env at a scratch .env with valid keys + an injected passing
  // auth probe so device='hpc' selection succeeds offline (mirrors the success
  // path in test_env_device.cpp). This makes DataLoader::load() take the hpc gate.
  const auto dir = make_scratch_dir("hpc_env");
  {
    std::ofstream out(dir / ".env", std::ios::binary);
    out << "SLURM_HOST=arc-login.arc.ox.ac.uk\n"
           "SLURM_USER=abcd1234\n"
           "SLURM_REMOTE_BASE=/data/coml-battery/dtwc-runs\n";
  }
  dtwc::env().set_env_file_dir(dir);
  dtwc::env().set_hpc_auth_probe([](const std::string &, const std::string &) { return true; });
  dtwc::env().set_device("hpc");
  REQUIRE(dtwc::env().device() == dtwc::Device::HPC);

  DataLoader dl{ fixture_path() };
  dl.verbosity(0);

  // REGISTERED EXPECTATION (before the run): the metadata-only load reads
  // shapes/counts/names WITHOUT materialising payload, so the bulk-reader counter
  // must remain 0; local series() access must throw ("data not resident locally").
  DataLoader::reset_bulk_read_count();
  Data meta = dl.load();
  REQUIRE(DataLoader::bulk_read_count() == 0);   // bulk reader never touched
  REQUIRE(meta.is_metadata_only());
  REQUIRE(meta.size() > 0);
  REQUIRE_THROWS_AS(meta.series(0), std::runtime_error); // not resident locally

  // Contrast: a normal cpu load DOES invoke the bulk reader (counter increments),
  // proving the counter measures the live payload path — not a dead symbol. The
  // metadata shapes/counts must equal the full load's (correct, not stubbed).
  dtwc::env().set_device("cpu");
  DataLoader::reset_bulk_read_count();
  Data full = dl.load();
  REQUIRE(DataLoader::bulk_read_count() >= 1);
  REQUIRE(full.size() == meta.size());
  for (std::size_t i = 0; i < full.size(); ++i)
    REQUIRE(full.series_flat_size(i) == meta.series_flat_size(i));

  dtwc::env().set_device("cpu"); // restore device for any later test in this binary
  fs::remove_all(dir);
}

// ===========================================================================
// (c) mmap-backed store vs heap: pairwise DTW distances are DIGIT-IDENTICAL.
//     LIVE paths: DataLoader::load() (heap) and DataLoader::load_stored()
//     (StoragePolicy::Mmap) feeding dtwc::distance::dtw().
// ===========================================================================
TEST_CASE("mmap vs heap load yields digit-identical DTW distances", "[storage][mmap]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (llfio); mmap vs heap round-trip unavailable");
#else
  dtwc::env().set_device("cpu");
  const auto scratch = make_scratch_dir("roundtrip");

  DataLoader dl_heap{ fixture_path() };
  dl_heap.verbosity(0);
  Data heap = dl_heap.load();

  DataLoader dl_mmap{ fixture_path() };
  dl_mmap.verbosity(0)
    .storage_policy(core::StoragePolicy::Mmap)
    .mmap_cache_path(scratch / "roundtrip_store.dtws");
  LoadedData mm = dl_mmap.load_stored();

  REQUIRE(mm.is_mmap());
  REQUIRE(mm.data.size() == heap.size());

  // REGISTERED EXPECTATION (before the run): the mmap store is a byte copy of the
  // heap doubles (MmapDataStore ELEM_SIZE=8, no transform) and dtwc::distance::dtw
  // reads identical spans, so EVERY pairwise DTW distance is EXACTLY equal
  // (digit-identical: ==, not approximate). Any inequality => storage corrupted data.
  const std::size_t n = heap.size();
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double d_heap = dtwc::distance::dtw<double>(heap.series(i), heap.series(j));
      const double d_mmap = dtwc::distance::dtw<double>(mm.data.series(i), mm.data.series(j));
      CAPTURE(i, j, d_heap, d_mmap);
      REQUIRE(d_heap == d_mmap); // digit-identical
    }
  }

  fs::remove_all(scratch);
#endif
}

// ===========================================================================
// (d) Pruned is a dense-only implementation. Auto/explicit strategy routing
//     must never enter it after a Problem has selected mapped matrix storage.
// ===========================================================================
TEST_CASE("Pruned strategy routing fills mmap storage without dense access",
          "[storage][mmap][pruned][m42]")
{
  const auto scratch = make_scratch_dir("m42_pruned_route");

  auto make_problem = [](size_t n, core::DTWVariant variant,
                         DistanceMatrixStrategy strategy) {
    std::vector<std::vector<double>> series;
    std::vector<std::string> names;
    for (size_t i = 0; i < n; ++i) {
      const double x = static_cast<double>(i);
      series.push_back({x / 6.0, static_cast<double>((i * 3) % 17),
                        4.0 + x / 9.0, static_cast<double>((i * i) % 19)});
      names.push_back("mmap-route-" + std::to_string(i));
    }
    Problem prob("m42_mmap_route");
    prob.set_data(Data(std::move(series), std::move(names)));
    prob.set_band(0);
    auto params = prob.variant_params;
    params.variant = variant;
    params.adtw_penalty = 0.75;
    prob.set_variant(params);
    prob.set_distance_strategy(strategy);
    return prob;
  };

#ifndef DTWC_HAS_MMAP
  auto prob = make_problem(64, core::DTWVariant::Standard,
                           DistanceMatrixStrategy::Auto);
  REQUIRE_THROWS_WITH(
    prob.use_mmap_distance_matrix(scratch / "unsupported.dtwm"),
    "MmapDistanceMatrix: this build has no memory-mapped support "
    "(rebuild with -DDTWC_ENABLE_LLFIO=ON / llfio available).");
#else
  auto run = [&](size_t n, core::DTWVariant variant,
                 DistanceMatrixStrategy strategy, const std::string &tag) {
    auto prob = make_problem(n, variant, strategy);
    const auto cache = scratch / (tag + ".dtwm");
    prob.use_mmap_distance_matrix(cache);
    prob.fill_distance_matrix();
    REQUIRE(prob.is_distance_matrix_filled());
    const auto &matrix = std::get<core::MmapDistanceMatrix>(prob.distance_matrix());
    REQUIRE(matrix.size() == n);
    for (size_t i = 0; i < n; ++i) {
      REQUIRE(matrix.get(i, i) == 0.0);
      for (size_t j = i + 1; j < n; ++j) {
        const double expected = variant == core::DTWVariant::ADTW
          ? dtwc::adtwBanded<double>(prob.series(i), prob.series(j), 0, 0.75)
          : dtwc::dtwBanded<double>(prob.series(i), prob.series(j), 0);
        REQUIRE(matrix.get(i, j) == expected);
      }
    }
  };

  run(63, core::DTWVariant::Standard, DistanceMatrixStrategy::Auto,
      "standard-auto-63");
  run(64, core::DTWVariant::Standard, DistanceMatrixStrategy::Auto,
      "standard-auto-64");
  run(64, core::DTWVariant::ADTW, DistanceMatrixStrategy::Auto,
      "adtw-auto-64");
  run(64, core::DTWVariant::Standard, DistanceMatrixStrategy::Pruned,
      "standard-explicit-64");
#endif

  fs::remove_all(scratch);
}
