# Plan: C++20 Upgrade + std::span DTW Interface

## Context

Phase 3c of the mmap architecture. `dtw_fn_t` currently uses `const vector<double>&` — incompatible with mmap-backed data. Upgrading to C++20 gives `std::span<const double>`, constructible from both `vector<double>` and `{double*, size_t}`.

**This is a semver-major breaking change.** External code calling DTW functions without explicit template args will need `<double>`. All consumers must compile with C++20.

## Adversarial Review Findings (addressed)

| Finding | Severity | Resolution |
|---------|----------|------------|
| Template deduction: `span<const T>` non-deduced from `vector<T>` | CRITICAL | Lambdas pass spans (deduction works span→span). External callers need explicit `<double>` (they already do). |
| Cascading scope: ~25+ functions need span, not just 17 | CRITICAL | Full audit done. All functions reachable from dtw_fn_ lambdas listed below. |
| CUDA headers with `<span>` | CRITICAL | Upgrade `CMAKE_CUDA_STANDARD` to 20 (CUDA 12.6 supports it). |
| GCC 10 / Clang 12 fragile C++20 | IMPORTANT | Drop from CI matrix. Require GCC 11+, Clang 14+. |
| cmake `cxx_std_17` in multiple places | IMPORTANT | Update all: ProjectOptions.cmake, MATLAB bindings, examples. |
| DDTW derivative_transform output needs vector | IMPORTANT | Input param → span, output param stays vector. |
| `has_missing`, `interpolate_linear` need span | IMPORTANT | Change to accept span (body uses `[i]` and `.size()` — both work). |
| Mixing C++20/C++17 translation units | IMPORTANT | All public headers get `<span>`, consumers must use C++20. |
| soft_dtw body uses `[i]` and `.size()` | MINOR | Both work on span. No body changes needed. |

## Task 1: C++20 Upgrade (build system)

**Files to modify:**

| File | Line | Change |
|------|------|--------|
| `cmake/StandardProjectSettings.cmake` | 26 | `CMAKE_CXX_STANDARD 17` → `20` |
| `CMakeLists.txt` | 131 | `CMAKE_CUDA_STANDARD 17` → `20` |
| `cmake/ProjectOptions.cmake` | ~63 | `cxx_std_17` → `cxx_std_20` |
| `bindings/matlab/CMakeLists.txt` | ~25 | `cxx_std_17` → `cxx_std_20` |
| `examples/cpp/example_project/CMakeLists.txt` | ~6 | `cxx_std_17` → `cxx_std_20` |

**CI matrix** (`.github/workflows/ubuntu-unit.yml`):
- Drop GCC 10: remove from matrix
- Drop Clang 12, 13: remove from matrix
- Keep: GCC 11, 12; Clang 14, 15, 16, 17

Build and verify all tests pass.

## Task 2: Change dtw_fn_t to std::span

**File:** `dtwc/Problem.hpp`

```cpp
#include <span>

// Line 84, FROM:
using dtw_fn_t = std::function<data_t(const std::vector<data_t> &, const std::vector<data_t> &)>;

// TO:
using dtw_fn_t = std::function<data_t(std::span<const data_t>, std::span<const data_t>)>;
```

**Call sites** (Problem.cpp:375, 431): `dtw_fn_(p_vec(i), p_vec(j))` — `p_vec(i)` returns `vector<data_t>&` which implicitly converts to `span<const data_t>`. **No change needed at call sites.**

## Task 3: Change all functions reachable from dtw_fn_ lambdas

The 12 lambdas in `rebind_dtw_fn()` receive `const auto &x` (deduced as `span<const data_t>`). Every function they call must accept span.

**Full list of functions to change** (`const vector<data_t>&` → `std::span<const data_t>`):

### warping.hpp (3 functions)
- `dtwFull(const vector<data_t>&, ...)` → `dtwFull(span<const data_t>, ...)`
- `dtwFull_L(const vector<data_t>&, ...)` → `dtwFull_L(span<const data_t>, ...)`
- `dtwBanded(const vector<data_t>&, ...)` → `dtwBanded(span<const data_t>, ...)`

### warping_adtw.hpp (2 functions)
- `adtwFull_L` → span
- `adtwBanded` → span

### warping_ddtw.hpp (3 changes)
- `ddtwBanded(const vector&, const vector&, ...)` → `ddtwBanded(span, span, ...)`
  - Body creates derivative vectors — must construct from span: `vector<data_t> xv(x.begin(), x.end())`
- `ddtwFull_L` → same pattern
- `derivative_transform_inplace(const vector<data_t> &x, vector<data_t> &dx)` → input param span: `derivative_transform_inplace(span<const data_t> x, vector<data_t> &dx)`. Output stays vector (needs `.resize()`).

### warping_wdtw.hpp (4 functions)
- `wdtwFull` (2 overloads) → span for x, y AND weights
- `wdtwBanded` (2 overloads) → span for x, y AND weights

### warping_missing.hpp (3 functions)
- `dtwMissing_L`, `dtwMissing`, `dtwMissing_banded` → span
- Note: `dtwMissing_banded` has `std::tie(x,y)` pattern — works with span (both are lvalue refs, ternary types match)

### warping_missing_arow.hpp (3 functions)
- `dtwAROW_L`, `dtwAROW`, `dtwAROW_banded` → span

### soft_dtw.hpp (2 functions)
- `soft_dtw(const vector<T>&, ...)` → `soft_dtw(span<const T>, ...)`
  - Body uses `x.size()` and `x[i]` — both work on span
- `soft_dtw_gradient` → span for inputs, returns `vector<T>` (unchanged)

### missing_utils.hpp (2 functions)
- `has_missing(const vector<T>&)` → `has_missing(span<const T>)`
- `interpolate_linear(const vector<T>&)` → `interpolate_linear(span<const T>)` (returns vector)

### lower_bounds.hpp / lower_bound_impl.hpp
- Already use pointer+length signatures. **No changes needed.**

### pruned_distance_matrix.cpp
- Calls `prob.p_vec(i)` which returns `vector<data_t>&`. These are passed to `compute_summary`, `compute_envelope`, `lb_keogh_symmetric` — all accept pointers. **No changes needed** (vector→pointer conversion still works).

**Total: ~22 function signature changes across 8 header files.**

Each change follows the same pattern — the function body calls `.data()`, `.size()`, or `[i]` which all work identically on span.

Add `#include <span>` to each modified header.

## Task 4: Fix DDTW span→vector bridge

In `warping_ddtw.hpp`, `ddtwBanded` needs to create vectors from span input (derivative transform needs mutable vectors):

```cpp
template <typename data_t>
data_t ddtwBanded(std::span<const data_t> x, std::span<const data_t> y, int band) {
  std::vector<data_t> dx, dy;
  derivative_transform_inplace(x, dx);  // x is span — reads via x[i]
  derivative_transform_inplace(y, dy);
  return dtwBanded<data_t>(std::span<const data_t>(dx), std::span<const data_t>(dy), band);
}
```

Same for `ddtwFull_L`.

## Task 5: CLARA — keep working (no behavior change yet)

`fast_clara.cpp:144`:
```cpp
sub_vecs.push_back(prob.p_vec(idx));  // p_vec returns vector&, copy into sub_vecs
```

This still works — `p_vec(i)` returns `vector<data_t>&`, which can be copied into a `vector<vector<data_t>>`. **No change needed.** The zero-copy view optimization comes in Phase 4.

## Task 6: Build, test, verify

1. Build: `cmake --build build --parallel 8`
2. All 67 tests pass
3. Benchmarks: no regression
4. CLI: `./build/bin/dtwc_cl -k 3 -i data/dummy -v` works
5. Commit

## Verification

- All 67+ tests pass (tests use explicit `<data_t>` template args — unaffected)
- Benchmarks: `bench_mmap_access` — verify no perf regression
- Python bindings: use explicit `<double>` — unaffected
- MATLAB bindings: use explicit `<double>` — unaffected
- New capability: `dtw_fn_` can accept mmap-backed data via span

## What NOT to do

- Don't change `p_vec(i)` return type yet — Phase 4
- Don't change `Data::p_vec` storage — Phase 4
- Don't add Eigen::Map to interfaces — use internally only (Phase 4)
- Don't change pointer+length overloads — they're already optimal
- Don't keep both vector AND span overloads — just change vector→span. External callers use explicit template args.
