# Plan: Implement Steps 1+2 — Count-Before-Load + Auto Method Selection

## Context

From the mmap/auto-select plan (`zippy-crafting-piglet.md`), Steps 1 and 2 are independent of mmap and require no new dependencies. Step 1 adds `DataLoader::count()` to determine N without loading data. Step 2 adds `--method auto` (new default) that picks pam vs clara based on N.

The dummy test data has 25 CSV files in `data/dummy/`. Existing tests: 63 pass, 2 CUDA skip.

## Step 1: `DataLoader::count()`

**File**: [DataLoader.hpp](dtwc/DataLoader.hpp)

Add a `count()` method after `load()` (after line 125):

```cpp
/**
 * @brief Count series without loading data.
 * @return Number of series that would be loaded.
 */
size_t count() const
{
  if (fs::is_directory(data_path)) {
    size_t n = 0;
    for ([[maybe_unused]] const auto &entry : fs::directory_iterator(data_path)) {
      ++n;
      if (Ndata >= 0 && static_cast<int>(n) >= Ndata) break;
    }
    return n;
  }
  // Batch file: count lines after skipping start_row, respecting Ndata
  std::ifstream in(data_path, std::ios_base::in);
  if (!in.good())
    throw std::runtime_error("DataLoader::count: cannot open " + data_path.string());
  
  std::string line;
  int line_no = 0;
  size_t n_rows = 0;
  while (std::getline(in, line)) {
    if (line_no++ < start_row) continue;
    ++n_rows;
    if (Ndata >= 0 && static_cast<int>(n_rows) >= Ndata) break;
  }
  return n_rows;
}
```

Key design choices:
- `const` method — doesn't modify DataLoader state
- Directory mode: iterates `fs::directory_iterator` without reading file contents
- Batch file mode: counts lines (must read file, but doesn't parse CSV columns)
- Respects `Ndata` limit and `start_row` skip, matching `load()` semantics exactly
- ~25 lines, no new includes needed (`<fstream>` already via `fileOperations.hpp`)

**Needs `#include <fstream>` added** to DataLoader.hpp since count() uses `std::ifstream` directly (currently it comes transitively through fileOperations.hpp, but explicit is better).

## Step 2: Auto Method Selection

**File**: [dtwc_cl.cpp](dtwc/dtwc_cl.cpp)

### 2a. Change default method (line 107)
```cpp
std::string method = "auto";  // was "pam"
```

### 2b. Add "auto" to CheckedTransformer map (line 118-121)
```cpp
{"auto", "auto"}, {"pam", "pam"}, {"clara", "clara"}, ...
```

### 2c. Update help string (line 116)
```
"Clustering method: auto, pam, clara, kmedoids, mip, hierarchical"
```

### 2d. Auto-resolve after data load (after line 348)
Insert between the "Data loaded" message and DTW configuration:

```cpp
if (method == "auto") {
  const size_t N = prob.size();
  method = (N <= 5000) ? "pam" : "clara";
  if (verbose)
    std::cout << "Auto-selected method: " << method << " (N=" << N << ")\n";
}
```

### 2e. CLARA sample_size auto-scaling for large N (before clara_opts construction, ~line 497)
When auto-selected CLARA with large N, scale sample size:

```cpp
if (sample_size < 0 && prob.size() > 50000)
  sample_size = static_cast<int>(std::max(40 + 2 * n_clusters,
    static_cast<int>(std::sqrt(static_cast<double>(prob.size())) * n_clusters)));
```

## Step 3: Tests

**File**: [unit_test_DataLoader.cpp](tests/unit/unit_test_DataLoader.cpp) — add `count()` tests

Add new SECTION entries to the existing test case:

1. **Directory count**: `DataLoader("data/dummy").count()` == 25
2. **Directory count with Ndata limit**: `DataLoader("data/dummy", 5).count()` == 5
3. **Batch file count**: count lines in a known batch file
4. **Count matches load**: verify `dl.count() == dl.load().size()` for dummy data

No separate CLI test for auto-select needed — the stress test (`stress_test_cli.sh`) already tests `--method` options, and the existing integration tests exercise pam/clara paths. The auto-select is a simple string comparison that routes to already-tested code paths.

## Step 4: CHANGELOG

Add under `## Unreleased`:
- `DataLoader::count()` — count series without loading data
- `--method auto` (new default) — auto-selects pam (N≤5000) or clara (N>5000)

## Files Modified

| File | Change |
|------|--------|
| `dtwc/DataLoader.hpp` | Add `count()` method, add `#include <fstream>` |
| `dtwc/dtwc_cl.cpp` | Default "auto", add to map, auto-resolve logic, CLARA scaling |
| `tests/unit/unit_test_DataLoader.cpp` | Add count() test sections |
| `CHANGELOG.md` | Document new features |

## Verification

1. Build: `cmake --build build --parallel 8`
2. Run all tests: `ctest --test-dir build -C Release -j8` — expect 63+ pass
3. Manual check: `./build/bin/Release/dtwc_cl -k 3 -i data/dummy` should print "Auto-selected method: pam (N=25)"
4. Manual check: `./build/bin/Release/dtwc_cl -k 3 -m pam -i data/dummy` should NOT print auto-select message
