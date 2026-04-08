# Mmap Distance Matrix + Variant Integration + Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory-mapped distance matrix for large-N problems, integrate via `std::variant` into `Problem`, and extend checkpoint system for crash-resume.

**Architecture:** `MmapDistanceMatrix` wraps llfio's `mapped_file_handle` with the exact same API as `DenseDistanceMatrix` (get/set/is_computed/size etc.). `Problem` holds `std::variant<DenseDistanceMatrix, MmapDistanceMatrix>` and resolves once at algorithm entry — hot loops receive concrete types via template. Checkpoint extends existing CSV system with binary state (medoids + labels + cost) alongside the mmap cache.

**Tech Stack:** C++17, llfio (via CPM), Catch2, CMake/CPM

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `cmake/Dependencies.cmake` | Add llfio via CPM | Modify |
| `dtwc/CMakeLists.txt` | Add new source to target | Modify |
| `dtwc/core/mmap_distance_matrix.hpp` | MmapDistanceMatrix class | Create |
| `dtwc/core/distance_matrix.hpp` | Add `tri_index` as free function (shared) | Modify |
| `dtwc/Problem.hpp` | `std::variant` distMat, template dispatch | Modify |
| `dtwc/Problem.cpp` | Auto mmap threshold, variant dispatch | Modify |
| `dtwc/checkpoint.hpp` | Binary checkpoint API | Modify |
| `dtwc/checkpoint.cpp` | Binary checkpoint impl | Modify |
| `dtwc/dtwc_cl.cpp` | `--restart` flag, `--mmap-threshold` | Modify |
| `tests/unit/core/unit_test_mmap_distance_matrix.cpp` | MmapDistanceMatrix tests | Create |
| `tests/unit/unit_test_variant_distmat.cpp` | Variant integration tests | Create |
| `tests/unit/unit_test_checkpoint_binary.cpp` | Binary checkpoint tests | Create |

---

### Task 1: Add llfio dependency via CPM

**Files:**
- Modify: `cmake/Dependencies.cmake`
- Modify: `dtwc/CMakeLists.txt`

- [ ] **Step 1: Add llfio to Dependencies.cmake**

Add after the yaml-cpp block (~line 114):

```cmake
# llfio — memory-mapped I/O for large distance matrices (optional)
option(DTWC_ENABLE_MMAP "Enable memory-mapped distance matrix support via llfio" ON)
if(DTWC_ENABLE_MMAP)
  if(NOT TARGET llfio::hl)
    CPMAddPackage(
      NAME llfio
      GITHUB_REPOSITORY ned14/llfio
      GIT_TAG develop
      DOWNLOAD_ONLY YES
    )
    if(llfio_ADDED)
      add_subdirectory(${llfio_SOURCE_DIR} ${llfio_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
  endif()

  if(TARGET llfio_hl)
    message(STATUS "  llfio:    YES (memory-mapped distance matrix enabled)")
    set(DTWC_HAS_MMAP TRUE)
  else()
    message(WARNING "  llfio:    NOT FOUND — memory-mapped distance matrix disabled")
    set(DTWC_HAS_MMAP FALSE)
  endif()
endif()
```

- [ ] **Step 2: Wire llfio into the dtwc target**

In `dtwc/CMakeLists.txt`, add after existing optional dep blocks:

```cmake
if(DTWC_HAS_MMAP)
  target_link_libraries(dtwc PUBLIC llfio_hl)
  target_compile_definitions(dtwc PUBLIC DTWC_HAS_MMAP)
endif()
```

- [ ] **Step 3: Verify build**

Run: `cmake --preset clang-win && cmake --build build --parallel 8`
Expected: Clean build, llfio downloaded and detected.

- [ ] **Step 4: Commit**

```bash
git add cmake/Dependencies.cmake dtwc/CMakeLists.txt
git commit -m "feat: add llfio dependency for memory-mapped distance matrix"
```

---

### Task 2: Extract shared `tri_index` and write MmapDistanceMatrix tests (RED)

**Files:**
- Modify: `dtwc/core/distance_matrix.hpp` (make `tri_index` a free function)
- Create: `dtwc/core/mmap_distance_matrix.hpp` (minimal stub)
- Create: `tests/unit/core/unit_test_mmap_distance_matrix.cpp`

- [ ] **Step 1: Extract `tri_index` as a shared free function**

In `dtwc/core/distance_matrix.hpp`, add before the class:

```cpp
/// Packed lower-triangular index: (i,j) -> offset into N*(N+1)/2 array.
/// Symmetric: tri_index(i,j) == tri_index(j,i).
inline size_t tri_index(size_t i, size_t j)
{
  if (i < j) std::swap(i, j);
  return i * (i + 1) / 2 + j;
}

inline size_t packed_size(size_t n) { return n * (n + 1) / 2; }
```

Then change `DenseDistanceMatrix`'s private `tri_index` and `packed_size` to use the free functions:
- Remove `static size_t tri_index(...)` and `static size_t packed_size(...)` from the class
- Update all internal calls (they already use unqualified names, so the free functions will be found via ADL/namespace lookup)

- [ ] **Step 2: Create minimal MmapDistanceMatrix stub**

Create `dtwc/core/mmap_distance_matrix.hpp` — just enough to fail compilation in tests:

```cpp
#pragma once

#ifdef DTWC_HAS_MMAP

#include <cstddef>
#include <filesystem>
#include <string>

namespace dtwc::core {

class MmapDistanceMatrix {
public:
  MmapDistanceMatrix() = default;
  explicit MmapDistanceMatrix(const std::filesystem::path &cache_path, size_t n);

  // Open existing cache file (warmstart)
  static MmapDistanceMatrix open(const std::filesystem::path &cache_path);

  double get(size_t i, size_t j) const;
  void set(size_t i, size_t j, double v);
  bool is_computed(size_t i, size_t j) const;
  size_t size() const;
  double max() const;
  size_t count_computed() const;
  bool all_computed() const;
  double *raw();
  const double *raw() const;
  size_t packed_count() const;
  void sync();
};

} // namespace dtwc::core

#endif // DTWC_HAS_MMAP
```

- [ ] **Step 3: Write failing tests**

Create `tests/unit/core/unit_test_mmap_distance_matrix.cpp`:

```cpp
#include <dtwc/core/mmap_distance_matrix.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>

#ifdef DTWC_HAS_MMAP

namespace fs = std::filesystem;
using dtwc::core::MmapDistanceMatrix;

static fs::path temp_cache(const std::string &name)
{
  auto p = fs::temp_directory_path() / "dtwc_test" / name;
  fs::create_directories(p.parent_path());
  if (fs::exists(p)) fs::remove(p);
  return p;
}

TEST_CASE("MmapDistanceMatrix basic operations", "[mmap][distmat]")
{
  auto path = temp_cache("basic.dtwcache");
  constexpr size_t N = 10;

  SECTION("Create new matrix — all entries uncomputed")
  {
    MmapDistanceMatrix dm(path, N);
    REQUIRE(dm.size() == N);
    REQUIRE(dm.packed_count() == N * (N + 1) / 2);
    REQUIRE(dm.count_computed() == 0);
    REQUIRE_FALSE(dm.all_computed());

    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        REQUIRE_FALSE(dm.is_computed(i, j));
  }

  SECTION("Set and get — symmetry")
  {
    MmapDistanceMatrix dm(path, N);
    dm.set(3, 7, 42.0);
    REQUIRE(dm.get(3, 7) == 42.0);
    REQUIRE(dm.get(7, 3) == 42.0);
    REQUIRE(dm.is_computed(3, 7));
    REQUIRE(dm.is_computed(7, 3));
  }

  SECTION("Set diagonal")
  {
    MmapDistanceMatrix dm(path, N);
    dm.set(0, 0, 0.0);
    REQUIRE(dm.get(0, 0) == 0.0);
    REQUIRE(dm.is_computed(0, 0));
  }

  SECTION("max()")
  {
    MmapDistanceMatrix dm(path, N);
    dm.set(0, 1, 5.0);
    dm.set(2, 3, 10.0);
    dm.set(4, 5, 3.0);
    REQUIRE(dm.max() == 10.0);
  }

  SECTION("count_computed and all_computed")
  {
    MmapDistanceMatrix dm(path, N);
    REQUIRE(dm.count_computed() == 0);

    // Fill all entries
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, static_cast<double>(i + j));

    REQUIRE(dm.count_computed() == N * (N + 1) / 2);
    REQUIRE(dm.all_computed());
  }
}

TEST_CASE("MmapDistanceMatrix persistence (warmstart)", "[mmap][distmat]")
{
  auto path = temp_cache("warmstart.dtwcache");
  constexpr size_t N = 5;

  // Create, write some entries, let destructor flush
  {
    MmapDistanceMatrix dm(path, N);
    dm.set(0, 0, 0.0);
    dm.set(1, 0, 1.5);
    dm.set(2, 3, 7.7);
    dm.sync();
  }

  // Reopen — values must persist
  {
    auto dm = MmapDistanceMatrix::open(path);
    REQUIRE(dm.size() == N);
    REQUIRE(dm.get(0, 0) == 0.0);
    REQUIRE(dm.get(1, 0) == 1.5);
    REQUIRE(dm.get(0, 1) == 1.5);  // symmetry
    REQUIRE(dm.get(2, 3) == 7.7);
    REQUIRE(dm.is_computed(2, 3));
    REQUIRE_FALSE(dm.is_computed(4, 4));  // not set
    REQUIRE(dm.count_computed() == 3);
  }

  fs::remove(path);
}

TEST_CASE("MmapDistanceMatrix N=0 and N=1", "[mmap][distmat]")
{
  SECTION("N=0")
  {
    auto path = temp_cache("n0.dtwcache");
    MmapDistanceMatrix dm(path, 0);
    REQUIRE(dm.size() == 0);
    REQUIRE(dm.packed_count() == 0);
    REQUIRE(dm.all_computed());
    fs::remove(path);
  }

  SECTION("N=1")
  {
    auto path = temp_cache("n1.dtwcache");
    MmapDistanceMatrix dm(path, 1);
    REQUIRE(dm.size() == 1);
    REQUIRE(dm.packed_count() == 1);
    dm.set(0, 0, 0.0);
    REQUIRE(dm.all_computed());
    fs::remove(path);
  }
}

TEST_CASE("MmapDistanceMatrix large N=1000", "[mmap][distmat]")
{
  auto path = temp_cache("large.dtwcache");
  constexpr size_t N = 1000;

  {
    MmapDistanceMatrix dm(path, N);
    // Write diagonal + a few off-diagonal
    for (size_t i = 0; i < N; ++i)
      dm.set(i, i, 0.0);
    dm.set(999, 0, 123.456);
    dm.sync();
  }

  {
    auto dm = MmapDistanceMatrix::open(path);
    REQUIRE(dm.size() == N);
    REQUIRE(dm.get(0, 0) == 0.0);
    REQUIRE(dm.get(999, 999) == 0.0);
    REQUIRE(dm.get(999, 0) == 123.456);
    REQUIRE(dm.get(0, 999) == 123.456);
    REQUIRE(dm.count_computed() == N + 1);  // diagonal + one off-diag
  }

  fs::remove(path);
}

#else // !DTWC_HAS_MMAP

TEST_CASE("MmapDistanceMatrix skipped — llfio not available", "[mmap][distmat]")
{
  WARN("DTWC_HAS_MMAP not defined — skipping mmap tests");
}

#endif
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cmake --build build --parallel 8 && ctest --test-dir build -R mmap -C Release -v`
Expected: Linker errors (stub has no implementation) or test failures.

- [ ] **Step 5: Commit (RED phase)**

```bash
git add dtwc/core/distance_matrix.hpp dtwc/core/mmap_distance_matrix.hpp tests/unit/core/unit_test_mmap_distance_matrix.cpp
git commit -m "test(RED): add MmapDistanceMatrix tests and stub"
```

---

### Task 3: Implement MmapDistanceMatrix with llfio (GREEN)

**Files:**
- Modify: `dtwc/core/mmap_distance_matrix.hpp` (full implementation)

- [ ] **Step 1: Implement the full class**

Replace the stub in `dtwc/core/mmap_distance_matrix.hpp`:

```cpp
#pragma once

#ifdef DTWC_HAS_MMAP

#include "distance_matrix.hpp" // for tri_index, packed_size

#include <llfio/v2.0/llfio.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace dtwc::core {

namespace llfio = LLFIO_V2_NAMESPACE;

/**
 * @brief Memory-mapped symmetric distance matrix with packed triangular storage.
 *
 * Binary file layout (32-byte header + packed doubles):
 *   bytes 0-3:   magic "DTWM"
 *   bytes 4-5:   version (uint16_t = 1)
 *   bytes 6-9:   endian marker (uint32_t = 0x01020304)
 *   bytes 10:    elem_size (uint8_t = 8 for double)
 *   bytes 11:    reserved (0)
 *   bytes 12-19: N (uint64_t, number of series)
 *   bytes 20-23: header CRC32 (of bytes 0-19)
 *   bytes 24-31: reserved (0, padding to 32-byte alignment)
 *   bytes 32+:   double[N*(N+1)/2], NaN = uncomputed
 *
 * Same API as DenseDistanceMatrix. Thread-safety: same contract (disjoint pair writes).
 */
class MmapDistanceMatrix {
  static constexpr size_t HEADER_SIZE = 32;
  static constexpr char MAGIC[4] = {'D', 'T', 'W', 'M'};
  static constexpr uint16_t VERSION = 1;
  static constexpr uint32_t ENDIAN_MARKER = 0x01020304;

  llfio::mapped_file_handle mfh_;
  double *data_ = nullptr;
  size_t n_ = 0;

  static size_t file_size(size_t n) { return HEADER_SIZE + packed_size(n) * sizeof(double); }

  void write_header()
  {
    auto *base = reinterpret_cast<uint8_t *>(mfh_.address());
    std::memcpy(base, MAGIC, 4);
    uint16_t ver = VERSION;
    std::memcpy(base + 4, &ver, 2);
    uint32_t endian = ENDIAN_MARKER;
    std::memcpy(base + 6, &endian, 4);
    base[10] = 8; // sizeof(double)
    base[11] = 0; // reserved
    uint64_t nn = static_cast<uint64_t>(n_);
    std::memcpy(base + 12, &nn, 8);
    // CRC32 of bytes 0-19 (header fields before CRC)
    uint32_t crc = crc32(base, 20);
    std::memcpy(base + 20, &crc, 4);
    // bytes 24-31 reserved, zero
    std::memset(base + 24, 0, 8);
  }

  void validate_header() const
  {
    auto *base = reinterpret_cast<const uint8_t *>(mfh_.address());
    if (std::memcmp(base, MAGIC, 4) != 0)
      throw std::runtime_error("MmapDistanceMatrix: invalid magic in cache file");

    uint16_t ver;
    std::memcpy(&ver, base + 4, 2);
    if (ver != VERSION)
      throw std::runtime_error("MmapDistanceMatrix: unsupported version " + std::to_string(ver));

    uint32_t endian;
    std::memcpy(&endian, base + 6, 4);
    if (endian != ENDIAN_MARKER)
      throw std::runtime_error("MmapDistanceMatrix: endian mismatch — file created on different architecture");

    if (base[10] != 8)
      throw std::runtime_error("MmapDistanceMatrix: elem_size " + std::to_string(base[10]) + " != 8 (double)");

    uint32_t stored_crc, computed_crc;
    std::memcpy(&stored_crc, base + 20, 4);
    computed_crc = crc32(base, 20);
    if (stored_crc != computed_crc)
      throw std::runtime_error("MmapDistanceMatrix: header CRC mismatch — file corrupted");
  }

  /// Simple CRC32 (IEEE polynomial, no table — header is only 20 bytes).
  static uint32_t crc32(const uint8_t *data, size_t len)
  {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
      crc ^= data[i];
      for (int j = 0; j < 8; ++j)
        crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
  }

  void init_data_ptr()
  {
    data_ = reinterpret_cast<double *>(
      reinterpret_cast<uint8_t *>(mfh_.address()) + HEADER_SIZE);
  }

public:
  MmapDistanceMatrix() = default;

  /// Create a new cache file of size N, NaN-filled.
  explicit MmapDistanceMatrix(const std::filesystem::path &cache_path, size_t n)
    : n_(n)
  {
    auto sz = file_size(n);
    auto result = llfio::mapped_file_handle::mapped_file(
      sz, cache_path, llfio::file_handle::mode::write,
      llfio::file_handle::creation::if_needed,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);
    if (!result)
      throw std::runtime_error("MmapDistanceMatrix: cannot create " + cache_path.string()
                               + ": " + result.error().message().c_str());
    mfh_ = std::move(result.value());
    mfh_.truncate(sz).value();
    mfh_.update_map().value();

    init_data_ptr();

    // Fill data region with NaN
    auto packed = packed_size(n);
    for (size_t i = 0; i < packed; ++i)
      data_[i] = std::numeric_limits<double>::quiet_NaN();

    write_header();
  }

  /// Open existing cache file (warmstart).
  static MmapDistanceMatrix open(const std::filesystem::path &cache_path)
  {
    MmapDistanceMatrix m;
    auto result = llfio::mapped_file_handle::mapped_file(
      0, cache_path, llfio::file_handle::mode::write,
      llfio::file_handle::creation::open_existing,
      llfio::file_handle::caching::all,
      llfio::file_handle::flag::none);
    if (!result)
      throw std::runtime_error("MmapDistanceMatrix::open: cannot open " + cache_path.string()
                               + ": " + result.error().message().c_str());
    m.mfh_ = std::move(result.value());
    m.mfh_.update_map().value();

    m.validate_header();

    auto *base = reinterpret_cast<const uint8_t *>(m.mfh_.address());
    uint64_t nn;
    std::memcpy(&nn, base + 12, 8);
    m.n_ = static_cast<size_t>(nn);
    m.init_data_ptr();

    // Validate file size
    auto expected = file_size(m.n_);
    if (m.mfh_.maximum_extent().value() < expected)
      throw std::runtime_error("MmapDistanceMatrix::open: file too small for N=" + std::to_string(m.n_));

    return m;
  }

  double get(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return data_[tri_index(i, j)];
  }

  void set(size_t i, size_t j, double v)
  {
    assert(i < n_ && j < n_ && !std::isnan(v));
    data_[tri_index(i, j)] = v;
  }

  bool is_computed(size_t i, size_t j) const
  {
    assert(i < n_ && j < n_);
    return !std::isnan(data_[tri_index(i, j)]);
  }

  size_t size() const { return n_; }

  double max() const
  {
    double result = -std::numeric_limits<double>::infinity();
    auto packed = packed_size(n_);
    for (size_t i = 0; i < packed; ++i)
      if (!std::isnan(data_[i]) && data_[i] > result)
        result = data_[i];
    return std::isfinite(result) ? result : 0.0;
  }

  size_t count_computed() const
  {
    size_t count = 0;
    auto packed = packed_size(n_);
    for (size_t i = 0; i < packed; ++i)
      if (!std::isnan(data_[i])) ++count;
    return count;
  }

  bool all_computed() const
  {
    auto packed = packed_size(n_);
    for (size_t i = 0; i < packed; ++i)
      if (std::isnan(data_[i])) return false;
    return true;
  }

  double *raw() { return data_; }
  const double *raw() const { return data_; }
  size_t packed_count() const { return packed_size(n_); }

  /// Flush mapped pages to disk.
  void sync()
  {
    if (mfh_.is_valid())
      mfh_.barrier({}, llfio::mapped_file_handle::barrier_kind::nowait_data_only).value();
  }
};

} // namespace dtwc::core

#endif // DTWC_HAS_MMAP
```

- [ ] **Step 2: Build and run mmap tests**

Run: `cmake --build build --parallel 8 && ctest --test-dir build -R mmap -C Release --output-on-failure`
Expected: All mmap tests PASS.

- [ ] **Step 3: Run full test suite**

Run: `ctest --test-dir build -C Release -j8`
Expected: 64+ tests pass (63 existing + mmap tests).

- [ ] **Step 4: Commit (GREEN phase)**

```bash
git add dtwc/core/mmap_distance_matrix.hpp dtwc/core/distance_matrix.hpp
git commit -m "feat: implement MmapDistanceMatrix with llfio backend"
```

---

### Task 4: Write variant integration tests (RED)

**Files:**
- Create: `tests/unit/unit_test_variant_distmat.cpp`

- [ ] **Step 1: Write failing tests for variant-based Problem**

```cpp
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

  // Fill distance matrix — should use dense (in-memory) for N=25
  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());

  // distByInd should work
  double d = prob.distByInd(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(d == prob.distByInd(1, 0));  // symmetry
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

  // Cache file should exist on disk
  REQUIRE(fs::exists(cache_path));
  REQUIRE(fs::file_size(cache_path) > 0);

  fs::remove(cache_path);
}

TEST_CASE("MmapDistanceMatrix warmstart via Problem", "[variant][distmat][mmap]")
{
  auto cache_path = fs::temp_directory_path() / "dtwc_test" / "warmstart_prob.dtwcache";
  fs::create_directories(cache_path.parent_path());
  if (fs::exists(cache_path)) fs::remove(cache_path);

  double d01_original;

  // First run: fill distance matrix, save to cache
  {
    DataLoader dl("data/dummy");
    Problem prob("test_warmstart", dl);
    prob.use_mmap_distance_matrix(cache_path);
    prob.fillDistanceMatrix();
    d01_original = prob.distByInd(0, 1);
  }

  // Second run: open existing cache, distances should persist
  {
    DataLoader dl("data/dummy");
    Problem prob("test_warmstart", dl);
    prob.use_mmap_distance_matrix(cache_path);  // reopens existing
    REQUIRE(prob.isDistanceMatrixFilled());
    REQUIRE(prob.distByInd(0, 1) == d01_original);
  }

  fs::remove(cache_path);
}
#endif
```

- [ ] **Step 2: Run to verify compile/link failure**

Run: `cmake --build build --parallel 8`
Expected: Fails — `use_mmap_distance_matrix()` doesn't exist yet.

- [ ] **Step 3: Commit (RED)**

```bash
git add tests/unit/unit_test_variant_distmat.cpp
git commit -m "test(RED): add variant distance matrix integration tests"
```

---

### Task 5: Integrate MmapDistanceMatrix into Problem via std::variant (GREEN)

**Files:**
- Modify: `dtwc/Problem.hpp`
- Modify: `dtwc/Problem.cpp`

- [ ] **Step 1: Add variant type alias and new members to Problem.hpp**

Add include at top:
```cpp
#ifdef DTWC_HAS_MMAP
#include "core/mmap_distance_matrix.hpp"
#endif
#include <variant>
```

Change the type alias and member:
```cpp
#ifdef DTWC_HAS_MMAP
  using distMat_t = std::variant<core::DenseDistanceMatrix, core::MmapDistanceMatrix>;
#else
  using distMat_t = core::DenseDistanceMatrix;
#endif
```

Keep private `distMat` member as `distMat_t`.

Add public method:
```cpp
#ifdef DTWC_HAS_MMAP
  /// Force mmap-backed distance matrix. Call before fillDistanceMatrix().
  void use_mmap_distance_matrix(const std::filesystem::path &cache_path);
#endif
```

Add private helper for dispatch:
```cpp
  /// Visit the distance matrix variant (avoids std::visit in hot paths).
  template <typename F>
  decltype(auto) visit_distmat(F &&f) {
#ifdef DTWC_HAS_MMAP
    return std::visit(std::forward<F>(f), distMat);
#else
    return f(distMat);
#endif
  }
  template <typename F>
  decltype(auto) visit_distmat(F &&f) const {
#ifdef DTWC_HAS_MMAP
    return std::visit(std::forward<F>(f), distMat);
#else
    return f(distMat);
#endif
  }
```

- [ ] **Step 2: Update all distMat access in Problem.hpp/cpp to use visit_distmat**

Every place that calls `distMat.xxx()` needs to become `visit_distmat([&](auto &m) { return m.xxx(); })`.

Key locations in `Problem.hpp`:
- `maxDistance()`: `return visit_distmat([](auto &m) { return m.max(); });`
- `isDistanceMatrixFilled()`: `return visit_distmat([](auto &m) { return m.size() > 0 && m.all_computed(); });`
- `distance_matrix()`: These accessors need adjustment — return `const distMat_t&` (variant)

Key locations in `Problem.cpp`:
- `distByInd()`: lazy alloc + compute via variant
- `fillDistanceMatrix_BruteForce()`: resize + parallel fill via variant
- `fillDistanceMatrix()`: dispatch

- [ ] **Step 3: Implement `use_mmap_distance_matrix()` in Problem.cpp**

```cpp
#ifdef DTWC_HAS_MMAP
void Problem::use_mmap_distance_matrix(const std::filesystem::path &cache_path)
{
  const size_t N = data.size();
  if (std::filesystem::exists(cache_path)) {
    distMat = core::MmapDistanceMatrix::open(cache_path);
    auto &m = std::get<core::MmapDistanceMatrix>(distMat);
    if (m.size() != N)
      throw std::runtime_error("Mmap cache size " + std::to_string(m.size())
                               + " != data size " + std::to_string(N));
  } else {
    distMat = core::MmapDistanceMatrix(cache_path, N);
  }
}
#endif
```

- [ ] **Step 4: Build and run variant tests**

Run: `cmake --build build --parallel 8 && ctest --test-dir build -R variant -C Release --output-on-failure`
Expected: All variant tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `ctest --test-dir build -C Release -j8`
Expected: All tests pass (existing + mmap + variant).

- [ ] **Step 6: Commit (GREEN)**

```bash
git add dtwc/Problem.hpp dtwc/Problem.cpp
git commit -m "feat: integrate MmapDistanceMatrix into Problem via std::variant"
```

---

### Task 6: Write binary checkpoint tests (RED)

**Files:**
- Create: `tests/unit/unit_test_checkpoint_binary.cpp`

- [ ] **Step 1: Write failing tests**

```cpp
#include <dtwc.hpp>
#include <dtwc/checkpoint.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <vector>

using namespace dtwc;
namespace fs = std::filesystem;

TEST_CASE("Binary checkpoint save and load", "[checkpoint][binary]")
{
  auto dir = fs::temp_directory_path() / "dtwc_test" / "checkpoint_bin";
  fs::create_directories(dir);

  DataLoader dl("data/dummy");
  Problem prob("ckpt_test", dl);
  prob.fillDistanceMatrix();

  // Run a clustering to get state
  auto result = dtwc::fast_pam(prob, 3, 100);
  prob.set_numberOfClusters(3);
  prob.clusters_ind = result.labels;
  prob.centroids_ind = result.medoid_indices;

  SECTION("Save and reload produces identical state")
  {
    save_binary_checkpoint(prob, result, dir / "state.bin");
    REQUIRE(fs::exists(dir / "state.bin"));

    core::ClusteringResult loaded_result;
    REQUIRE(load_binary_checkpoint(prob, loaded_result, dir / "state.bin"));

    REQUIRE(loaded_result.labels == result.labels);
    REQUIRE(loaded_result.medoid_indices == result.medoid_indices);
    REQUIRE(loaded_result.total_cost == result.total_cost);
    REQUIRE(loaded_result.iterations == result.iterations);
    REQUIRE(loaded_result.converged == result.converged);
  }

  SECTION("Load returns false for non-existent file")
  {
    core::ClusteringResult loaded_result;
    REQUIRE_FALSE(load_binary_checkpoint(prob, loaded_result, dir / "nonexistent.bin"));
  }

  // Cleanup
  fs::remove_all(dir);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build --parallel 8`
Expected: Fails — `save_binary_checkpoint`/`load_binary_checkpoint` don't exist.

- [ ] **Step 3: Commit (RED)**

```bash
git add tests/unit/unit_test_checkpoint_binary.cpp
git commit -m "test(RED): add binary checkpoint tests"
```

---

### Task 7: Implement binary checkpoint (GREEN)

**Files:**
- Modify: `dtwc/checkpoint.hpp`
- Modify: `dtwc/checkpoint.cpp`

- [ ] **Step 1: Add binary checkpoint API to checkpoint.hpp**

```cpp
/// Save clustering state to binary file (medoids, labels, cost, iteration, converged).
void save_binary_checkpoint(const Problem &prob, const core::ClusteringResult &result,
                            const std::filesystem::path &path);

/// Load clustering state from binary file. Returns false if file doesn't exist.
bool load_binary_checkpoint(const Problem &prob, core::ClusteringResult &result,
                            const std::filesystem::path &path);
```

- [ ] **Step 2: Implement in checkpoint.cpp**

Binary format:
```
bytes 0-3:   magic "DCKP" (4 bytes)
bytes 4-5:   version uint16 = 1
bytes 6-7:   reserved
bytes 8-11:  k (int32)
bytes 12-15: N (int32)
bytes 16-19: iterations (int32)
bytes 20:    converged (uint8)
bytes 21-23: reserved
bytes 24-31: total_cost (double)
bytes 32+:   medoid_indices (k * int32)
then:        labels (N * int32)
```

Implementation:
```cpp
void save_binary_checkpoint(const Problem &prob, const core::ClusteringResult &result,
                            const std::filesystem::path &path)
{
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot create checkpoint file: " + path.string());

  // Header
  out.write("DCKP", 4);
  uint16_t ver = 1; out.write(reinterpret_cast<const char*>(&ver), 2);
  uint16_t reserved = 0; out.write(reinterpret_cast<const char*>(&reserved), 2);

  int32_t k = static_cast<int32_t>(result.medoid_indices.size());
  int32_t N = static_cast<int32_t>(result.labels.size());
  int32_t iters = static_cast<int32_t>(result.iterations);
  uint8_t conv = result.converged ? 1 : 0;

  out.write(reinterpret_cast<const char*>(&k), 4);
  out.write(reinterpret_cast<const char*>(&N), 4);
  out.write(reinterpret_cast<const char*>(&iters), 4);
  out.write(reinterpret_cast<const char*>(&conv), 1);
  char pad[3] = {0,0,0}; out.write(pad, 3);

  double cost = result.total_cost;
  out.write(reinterpret_cast<const char*>(&cost), 8);

  // Data
  for (int idx : result.medoid_indices) {
    int32_t v = static_cast<int32_t>(idx);
    out.write(reinterpret_cast<const char*>(&v), 4);
  }
  for (int idx : result.labels) {
    int32_t v = static_cast<int32_t>(idx);
    out.write(reinterpret_cast<const char*>(&v), 4);
  }
}

bool load_binary_checkpoint(const Problem &prob, core::ClusteringResult &result,
                            const std::filesystem::path &path)
{
  if (!fs::exists(path)) return false;

  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  char magic[4];
  in.read(magic, 4);
  if (std::memcmp(magic, "DCKP", 4) != 0)
    throw std::runtime_error("Invalid checkpoint magic in " + path.string());

  uint16_t ver; in.read(reinterpret_cast<char*>(&ver), 2);
  if (ver != 1) throw std::runtime_error("Unsupported checkpoint version " + std::to_string(ver));

  uint16_t reserved; in.read(reinterpret_cast<char*>(&reserved), 2);

  int32_t k, N, iters;
  uint8_t conv;
  in.read(reinterpret_cast<char*>(&k), 4);
  in.read(reinterpret_cast<char*>(&N), 4);
  in.read(reinterpret_cast<char*>(&iters), 4);
  in.read(reinterpret_cast<char*>(&conv), 1);
  char pad[3]; in.read(pad, 3);

  double cost;
  in.read(reinterpret_cast<char*>(&cost), 8);

  result.medoid_indices.resize(k);
  for (int32_t i = 0; i < k; ++i) {
    int32_t v; in.read(reinterpret_cast<char*>(&v), 4);
    result.medoid_indices[i] = v;
  }

  result.labels.resize(N);
  for (int32_t i = 0; i < N; ++i) {
    int32_t v; in.read(reinterpret_cast<char*>(&v), 4);
    result.labels[i] = v;
  }

  result.total_cost = cost;
  result.iterations = iters;
  result.converged = (conv != 0);

  return true;
}
```

- [ ] **Step 3: Build and run checkpoint tests**

Run: `cmake --build build --parallel 8 && ctest --test-dir build -R checkpoint_binary -C Release --output-on-failure`
Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run: `ctest --test-dir build -C Release -j8`
Expected: All tests pass.

- [ ] **Step 5: Commit (GREEN)**

```bash
git add dtwc/checkpoint.hpp dtwc/checkpoint.cpp
git commit -m "feat: add binary checkpoint save/load for clustering state"
```

---

### Task 8: Add `--restart` and `--mmap-threshold` CLI flags

**Files:**
- Modify: `dtwc/dtwc_cl.cpp`

- [ ] **Step 1: Add CLI options**

After existing clustering options (~line 153):
```cpp
// Mmap / checkpoint options
bool restart = false;
size_t mmap_threshold = 50000;
app.add_flag("--restart", restart, "Resume from checkpoint (distance matrix cache + clustering state)");
app.add_option("--mmap-threshold", mmap_threshold, "N above which to use memory-mapped distance matrix (0=always, -1=never)")
    ->check(CLI::NonNegativeNumber);
```

- [ ] **Step 2: Wire into Problem after data load**

After the auto method selection block:
```cpp
#ifdef DTWC_HAS_MMAP
  // Mmap distance matrix for large N
  if (mmap_threshold > 0 && prob.size() >= mmap_threshold) {
    auto cache_path = output_dir / (prob_name + "_distmat.cache");
    prob.use_mmap_distance_matrix(cache_path);
    if (verbose)
      std::cout << "Using memory-mapped distance matrix: " << cache_path << "\n";
  }
#endif

  // Restart from checkpoint
  if (restart) {
    auto ckpt_path = output_dir / (prob_name + "_checkpoint.bin");
    dtwc::core::ClusteringResult ckpt_result;
    if (dtwc::load_binary_checkpoint(prob, ckpt_result, ckpt_path)) {
      if (verbose)
        std::cout << "Loaded checkpoint: " << ckpt_result.iterations
                  << " iterations, cost=" << ckpt_result.total_cost << "\n";
      // TODO: pass ckpt_result as warm start to algorithms (Step 5 future work)
    }
  }
```

- [ ] **Step 3: Build and test**

Run: `cmake --build build --parallel 8 && ctest --test-dir build -C Release -j8`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add dtwc/dtwc_cl.cpp
git commit -m "feat: add --restart and --mmap-threshold CLI flags"
```

---

### Task 9: Update CHANGELOG and final verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add entries under DTWC v2.0.0**

Under `## Performance / API`:
```
* Added `MmapDistanceMatrix` — memory-mapped distance matrix via llfio for large-N problems (N>50K). Supports warmstart: reopen existing cache to resume interrupted computation.
```

Under `## CLI`:
```
* Added `--mmap-threshold` to control when memory-mapped distance matrix activates (default 50K).
* Added `--restart` to resume from binary checkpoint (distance matrix cache + clustering state).
```

Under `## Build system`:
```
* Added llfio dependency (optional, `DTWC_ENABLE_MMAP=ON` by default) for cross-platform memory-mapped I/O.
```

- [ ] **Step 2: Full verification**

Run all tests: `ctest --test-dir build -C Release -j8`
Run CLI: `./build/bin/dtwc_cl -k 3 -i data/dummy -v --mmap-threshold 0 -o /tmp/dtwc_mmap_test`
Verify cache file created: `ls -la /tmp/dtwc_mmap_test/*.cache`

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for mmap distance matrix and binary checkpoint"
```

---

## Verification Summary

After all tasks:
1. `ctest --test-dir build -C Release -j8` — all tests pass
2. `./build/bin/dtwc_cl -k 3 -i data/dummy -v` — auto-selects pam, uses dense matrix (N=25 < 50K)
3. `./build/bin/dtwc_cl -k 3 -i data/dummy -v --mmap-threshold 0` — forces mmap, creates cache file
4. Two runs with `--mmap-threshold 0` reuse the same cache (warmstart — second run skips distance computation)
5. `--restart` loads binary checkpoint if present
