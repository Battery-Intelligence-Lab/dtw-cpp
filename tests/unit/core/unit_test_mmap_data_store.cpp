/**
 * @file unit_test_mmap_data_store.cpp
 * @brief Unit tests for MmapDataStore class (memory-mapped time series cache).
 *
 * @date 08 Apr 2026
 */

#include <dtwc.hpp>
#include <core/mmap_data_store.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace dtwc;
using namespace dtwc::core;
namespace fs = std::filesystem;

#ifndef DTWC_HAS_MMAP

TEST_CASE("MmapDataStore tests require LLFIO", "[mmap][data]")
{
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
}

#else

static fs::path temp_store(const std::string &name)
{
  auto p = fs::temp_directory_path() / "dtwc_test" / name;
  fs::create_directories(p.parent_path());
  if (fs::exists(p)) fs::remove(p);
  return p;
}

static Data make_test_data(int N, int L, unsigned seed = 42)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < N; ++i) {
    std::vector<double> s(static_cast<size_t>(L));
    for (auto &v : s) v = dist(rng);
    vecs.push_back(std::move(s));
    names.push_back("s" + std::to_string(i));
  }
  return Data(std::move(vecs), std::move(names));
}

TEST_CASE("MmapDataStore create and access", "[mmap][data]")
{
  auto path = temp_store("data_basic.cache");
  auto data = make_test_data(100, 25);

  SECTION("Create from Data, verify series count and lengths")
  {
    auto store = MmapDataStore::create(path, data);
    REQUIRE(store.size() == 100);
    REQUIRE(store.ndim() == 1);
    for (size_t i = 0; i < 100; ++i)
      REQUIRE(store.series_length(i) == 25);
  }

  SECTION("Series data matches original")
  {
    auto store = MmapDataStore::create(path, data);
    for (size_t i = 0; i < store.size(); ++i) {
      const double *ptr = store.series_data(i);
      REQUIRE(ptr != nullptr);
      for (size_t j = 0; j < 25; ++j)
        REQUIRE(ptr[j] == data.p_vec[i][j]);
    }
  }

  SECTION("File persists and can be reopened")
  {
    double first_val;
    {
      auto store = MmapDataStore::create(path, data);
      first_val = store.series_data(0)[0];
    }
    // Reopen
    auto store = MmapDataStore::open(path);
    REQUIRE(store.size() == 100);
    REQUIRE(store.series_data(0)[0] == first_val);
  }

  if (fs::exists(path)) fs::remove(path);
}

TEST_CASE("MmapDataStore variable length series", "[mmap][data]")
{
  auto path = temp_store("data_varlen.cache");

  // Create data with different lengths
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0 },
    { 4.0, 5.0 },
    { 6.0, 7.0, 8.0, 9.0, 10.0 }
  };
  std::vector<std::string> names = { "short", "shorter", "long" };
  Data data(std::move(vecs), std::move(names));

  auto store = MmapDataStore::create(path, data);
  REQUIRE(store.size() == 3);
  REQUIRE(store.series_length(0) == 3);
  REQUIRE(store.series_length(1) == 2);
  REQUIRE(store.series_length(2) == 5);

  REQUIRE(store.series_data(0)[0] == 1.0);
  REQUIRE(store.series_data(0)[2] == 3.0);
  REQUIRE(store.series_data(1)[0] == 4.0);
  REQUIRE(store.series_data(2)[4] == 10.0);

  if (fs::exists(path)) fs::remove(path);
}

TEST_CASE("MmapDataStore multivariate", "[mmap][data]")
{
  auto path = temp_store("data_mv.cache");

  // 2 series, 3 timesteps each, ndim=2 (interleaved: [t0_f0, t0_f1, t1_f0, ...])
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 },  // 3 timesteps x 2 features
    { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }
  };
  std::vector<std::string> names = { "a", "b" };
  Data data(std::move(vecs), std::move(names), 2);

  auto store = MmapDataStore::create(path, data);
  REQUIRE(store.size() == 2);
  REQUIRE(store.ndim() == 2);
  REQUIRE(store.series_length(0) == 3);    // 6 values / 2 ndim = 3 timesteps
  REQUIRE(store.series_flat_size(0) == 6);

  REQUIRE(store.series_data(0)[0] == 1.0);
  REQUIRE(store.series_data(0)[5] == 6.0);
  REQUIRE(store.series_data(1)[0] == 7.0);

  if (fs::exists(path)) fs::remove(path);
}

TEST_CASE("MmapDataStore edge cases", "[mmap][data]")
{
  SECTION("N=0")
  {
    auto path = temp_store("data_n0.cache");
    Data data;
    auto store = MmapDataStore::create(path, data);
    REQUIRE(store.size() == 0);
    if (fs::exists(path)) fs::remove(path);
  }

  SECTION("N=1, L=1")
  {
    auto path = temp_store("data_n1.cache");
    std::vector<std::vector<double>> vecs = { { 42.0 } };
    std::vector<std::string> names = { "x" };
    Data data(std::move(vecs), std::move(names));
    auto store = MmapDataStore::create(path, data);
    REQUIRE(store.size() == 1);
    REQUIRE(store.series_data(0)[0] == 42.0);
    if (fs::exists(path)) fs::remove(path);
  }

  SECTION("Open nonexistent file throws")
  {
    REQUIRE_THROWS(MmapDataStore::open("nonexistent_file.cache"));
  }
}

// Regression for audit CRITICAL #5 (mmap_data_store.hpp open()).
// Before the fix, open() validated only the offset-table END and the sentinel
// offsets_[N]; interior offsets_[0..N-1] came straight from the (attacker-controllable)
// file with NO monotonicity/bounds check. A patched interior offset makes
// series_flat_size(i) = (offsets_[i+1]-offsets_[i])/8 underflow to a huge size_t and
// series_data(i) point out of bounds. The unfixed open() does NOT throw here (the header
// CRC covers only bytes 0-27, not the offset table, so the patch is invisible to it, and
// the sentinel/data-size check still passes because offsets_[N] is unchanged). The fix
// validates the whole offset table: first offset 0, monotone, 8-aligned, in bounds.
TEST_CASE("MmapDataStore open rejects out-of-bounds interior offset", "[mmap][data][security]")
{
  auto path = temp_store("data_bad_offset.cache");

  std::vector<std::vector<double>> vecs = { { 1, 2, 3 }, { 4, 5 }, { 6, 7, 8, 9, 10 } };
  std::vector<std::string> names = { "a", "b", "c" };
  Data data(std::move(vecs), std::move(names));
  {
    auto store = MmapDataStore::create(path, data);
    store.sync();
  } // close/unmap so the file is flushed for the raw patch below

  // Offset table starts at byte 64 (HEADER_SIZE); entry i is at 64 + i*8.
  // offsets = [0, 24, 40, 80]; total data = 80 bytes. Patch interior offset[1] to a
  // value far beyond the data section: OOB and non-monotone (offset[1] > offset[2]).
  const std::streamoff offset1_pos = 64 + 1 * static_cast<std::streamoff>(sizeof(uint64_t));
  const uint64_t bad_offset = uint64_t{ 1 } << 40; // >> total data bytes (80)
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(offset1_pos);
    f.write(reinterpret_cast<const char *>(&bad_offset), sizeof(bad_offset));
  }

  REQUIRE_THROWS_AS(MmapDataStore::open(path), std::runtime_error);

  if (fs::exists(path)) fs::remove(path);
}

TEST_CASE("MmapDataStore large N=5000", "[mmap][data]")
{
  auto path = temp_store("data_large.cache");
  auto data = make_test_data(5000, 25);

  {
    auto store = MmapDataStore::create(path, data);
    REQUIRE(store.size() == 5000);
    // Spot check last series
    for (size_t j = 0; j < 25; ++j)
      REQUIRE(store.series_data(4999)[j] == data.p_vec[4999][j]);
  }

  // Reopen
  {
    auto store = MmapDataStore::open(path);
    REQUIRE(store.size() == 5000);
    REQUIRE(store.series_length(4999) == 25);
  }

  if (fs::exists(path)) fs::remove(path);
}

#endif // DTWC_HAS_MMAP
