# CMake Build System Cleanup & Activation Plan

## Context

The project uses cmake infrastructure from cpp-best-practices/cmake_template, but the key macros (`dtwc_setup_options`, `dtwc_local_options`, `dtwc_global_options`) are **defined but never called**. This means:

- **No compiler warnings** applied to `project_warnings` (empty INTERFACE library)
- **No sanitizers** (ASan/UBSan infrastructure exists but disconnected)
- **No static analyzers** (clang-tidy/cppcheck infrastructure exists but disconnected)
- **No IPO/LTO** despite infrastructure
- **No ccache** despite infrastructure
- **Missing cmake files** referenced by ProjectOptions.cmake (`Hardening.cmake`, `Linker.cmake`)
- **MSVC C4849 warnings** from `#pragma omp simd reduction(+:sum)` — MSVC doesn't support reduction on simd directives

The goal is to wire up the existing infrastructure so all checks are active and the library builds cleanly with zero warnings.

---

## Implementation Plan

### Step 1: Wire up ProjectOptions in root CMakeLists.txt

**File:** `CMakeLists.txt`

Replace the manually-created empty targets (lines 174-178) with calls to the ProjectOptions macros:

```cmake
# BEFORE (current):
add_library(project_warnings INTERFACE)
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

# AFTER:
include(cmake/ProjectOptions.cmake)
dtwc_setup_options()
dtwc_global_options()
dtwc_local_options()
# dtwc_local_options() creates dtwc_warnings and dtwc_options targets
# Alias them so existing code doesn't need changing:
add_library(project_warnings ALIAS dtwc_warnings)
add_library(project_options ALIAS dtwc_options)
```

### Step 2: Fix missing cmake files

**2a. Remove Hardening.cmake reference** — The file doesn't exist and we don't need it. Comment out lines 186-198 in `cmake/ProjectOptions.cmake` (the `dtwc_enable_hardening` call).

**2b. Remove Linker.cmake reference** — The file doesn't exist. Comment out lines 133-136 in `cmake/ProjectOptions.cmake` (the `configure_linker` call), OR create a minimal stub.

### Step 3: Fix MSVC OpenMP `#pragma omp simd reduction` warnings

**Files:**
- `dtwc/core/lower_bound_impl.hpp:113`
- `dtwc/core/z_normalize.hpp:30,36`

MSVC doesn't support `reduction()` on `simd` directives. Guard with compiler check:

```cpp
#if defined(_MSC_VER)
  #pragma omp simd
#else
  #pragma omp simd reduction(+:sum)
#endif
```

Note: On MSVC, the reduction is still correct because `#pragma omp simd` without reduction in a sequential context just vectorizes — the reduction variable is thread-local in the loop. The warning C4849 says the clause is ignored, not that the result is wrong.

### Step 4: Suppress Eigen internal warnings

Eigen headers generate warnings under strict `-Wall -Wextra -Wconversion` etc. These should be suppressed since we don't control Eigen's code. In `cmake/Dependencies.cmake`, mark Eigen as SYSTEM:

```cmake
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${Eigen_SOURCE_DIR}"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Eigen_SOURCE_DIR}")
```

Or use `SYSTEM` keyword in CPMAddPackage.

### Step 5: Fix any new warnings from strict warning flags

Once warnings-as-errors is enabled, compile errors will surface. Common ones to expect:
- `-Wconversion` / `-Wsign-conversion`: `size_t` ↔ `int` casts in DTW code
- `-Wshadow`: variable shadowing in nested scopes
- `-Wold-style-cast`: C-style casts
- These need to be fixed one by one in the source code.

**Strategy:** First build with warnings-as-errors OFF but all warnings ON to see the full list. Then fix them. Then enable warnings-as-errors.

### Step 6: Remove dead cmake files

Delete cmake files that are truly unused and not part of the active infrastructure:
- `cmake/Doxygen.cmake` — if Doxygen is handled elsewhere (check docs workflow)
- `cmake/SystemLink.cmake` — never included
- `cmake/LibFuzzer.cmake` — never included
- `cmake/PackageProject.cmake` — never included (but check if it's useful for install)

Keep: `cmake/PreventInSourceBuilds.cmake` — uncomment it in root CMakeLists.txt.

### Step 7: Enable PreventInSourceBuilds

**File:** `CMakeLists.txt:170`

Uncomment:
```cmake
include(cmake/PreventInSourceBuilds.cmake)
```

---

## Critical Files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Wire up ProjectOptions, alias targets, uncomment PreventInSourceBuilds |
| `cmake/ProjectOptions.cmake` | Remove/guard Hardening.cmake and Linker.cmake references |
| `cmake/Dependencies.cmake` | Mark Eigen as SYSTEM include |
| `dtwc/core/lower_bound_impl.hpp` | Guard `#pragma omp simd reduction` for MSVC |
| `dtwc/core/z_normalize.hpp` | Same MSVC guard |
| Various `.cpp`/`.hpp` files | Fix warnings surfaced by strict flags |

## Verification

1. Build with MSVC (Windows): `cmake -B build -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON && cmake --build build --config Release`
2. Run tests: `ctest --test-dir build --build-config Release -j4`
3. Check zero warnings in build output
4. Build with GCC (if available via WSL): verify sanitizers and warning flags apply
