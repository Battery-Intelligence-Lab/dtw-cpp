# DTWC++ Transformation Plan v3 (Post-Adversarial Review)

## Execution Plan: Phase 0 Parallel Worktree Decomposition

### Context

Phase 0 fixes 24 bugs across ~15 files. These are decomposed into 8 independent work units that can execute in parallel worktrees. Each unit is independently mergeable. An additional test-writing unit runs first to establish test coverage before modifications.

### E2E Test Recipe

```bash
cd build_dir
cmake .. -DCMAKE_BUILD_TYPE=Debug -DDTWC_BUILD_TESTING=ON
cmake --build . --parallel 4
ctest -j4 -C Debug --output-on-failure
```

No UI or server verification needed — unit tests are sufficient for Phase 0 correctness fixes.

### Work Units

#### Unit 0: PRE — Write Tests Before Modifications
**Files:** `tests/unit/unit_test_warping_phase0.cpp`, `tests/unit/unit_test_Problem_phase0.cpp`, `tests/unit/unit_test_scores_phase0.cpp`
**Task:** Write NEW Catch2 test files that will verify Phase 0 fixes. Tests should initially FAIL on the current code (proving the bugs exist), then PASS after fixes. Tests:
- `throw 1` produces non-std::exception (test that `writeMedoids` throws `std::runtime_error`)
- `dtwBanded` with `float` default produces different precision than `double`
- `fillDistanceMatrix` symmetry and non-negativity properties
- `calculateMedoids` thread safety (call from two threads)
- DTW `.at()` vs `[]` gives same results (regression)
- `Data::size()` returns correct type for large datasets

#### Unit 1: Core C++ Bug Fixes (warping.hpp + settings.hpp)
**Files:** `dtwc/warping.hpp`, `dtwc/settings.hpp`
**Changes:**
- warping.hpp:109-110 — replace `.at(i)` with `[i]` and `.at(j)` with `[j]`
- warping.hpp:144 — change `template <typename data_t = float>` to `template <typename data_t = double>`
- warping.hpp:155-156 — replace full matrix `C.resize(m_long, m_short)` with rolling buffer `std::vector<data_t>` of width `2*band+1` for banded DTW
- settings.hpp:35 — change `static std::mt19937` to `inline std::mt19937`

#### Unit 2: Problem.cpp Bug Fixes
**Files:** `dtwc/Problem.cpp`, `dtwc/Problem.hpp`, `dtwc/Data.hpp`
**Changes:**
- Problem.cpp:136-143 — replace N^2 linearized iteration with triangular nested loop using `size_t`
- Problem.cpp:235,250 — change `static` vectors to local variables in `calculateMedoids`
- Problem.hpp:136 — rename `cluster_by_kMedoidsPAM` to `cluster_by_kMedoidsLloyd` (and all call sites in Problem.cpp)
- Data.hpp:37 — change `static_cast<int>` to return `size_t` (or `int64_t`)
- Problem_IO.cpp:42 — change `throw 1` to `throw std::runtime_error(...)`

#### Unit 3: Scores Fix (DBI)
**Files:** `dtwc/scores.cpp`, `dtwc/scores.hpp`
**Changes:**
- scores.cpp:99-153 — fix commented-out DBI: replace `dist(medoid_i, medoid_i)` with average within-cluster scatter `S_i = mean(d(x, m_i))`. Uncomment with correct formula.

#### Unit 4: Examples and Benchmark Fixes
**Files:** `examples/MIP_single.cpp`, `examples/MIP_multiple.cpp`, `benchmark/UCR_dtwc.cpp`
**Changes:**
- MIP_single.cpp:39 — change `dtwc::settings::band` to `prob.band`
- MIP_multiple.cpp:41 — change `dtwc::settings::band` to `prob.band`
- UCR_dtwc.cpp:26,29,61 — replace `settings::root_folder` with `settings::paths::dataPath` parent
- UCR_dtwc.cpp:127,136 — change `settings::band` to `prob.band`
- UCR_dtwc.cpp:31 — change `throw 11` to `throw std::runtime_error(...)`

#### Unit 5: CMake Fixes
**Files:** `CMakeLists.txt`, `cmake/StandardProjectSettings.cmake`, `cmake/Dependencies.cmake`, `tests/CMakeLists.txt`
**Changes:**
- CMakeLists.txt:16-18 — fix `option()` syntax to include help strings
- CMakeLists.txt:20-21 — change `set(DTWC_ENABLE_GUROBI ON)` to `option()`
- CMakeLists.txt:35 — remove unconditional `enable_testing()`
- CMakeLists.txt:66 — remove duplicate `enable_testing()` (keep `include(CTest)`)
- CMakeLists.txt:40-62 — wrap executables in `if(PROJECT_IS_TOP_LEVEL)`
- StandardProjectSettings.cmake:4 — fix message to say "Release" not "RelWithDebInfo"
- StandardProjectSettings.cmake:61-64 — change `CMAKE_CURRENT_SOURCE_DIR` to `CMAKE_BINARY_DIR`
- Dependencies.cmake:53 — pin Armadillo to specific tag (not branch)
- tests/CMakeLists.txt:1 — add `CONFIGURE_DEPENDS` to GLOB_RECURSE

#### Unit 6: CI Workflow Fixes
**Files:** `.github/workflows/ubuntu-unit.yml`, `.github/workflows/windows-unit.yml`, `.github/workflows/macos-unit.yml`
**Changes:**
- Uncomment `- main` in push branch triggers in all 3 files
- Add ASan+UBSan CI job to ubuntu-unit.yml (new matrix entry with `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"`)

#### Unit 7: Python Packaging Fixes
**Files:** `pyproject.toml`, `setup.py` (DELETE), `.python-version`
**Changes:**
- pyproject.toml:7 — change `requires-python` to `">=3.9"`
- pyproject.toml:10 — change numpy requirement to `">=1.21"`
- pyproject.toml:12 — change pybind11 requirement to `">=2.11"`
- pyproject.toml:4 — replace placeholder description
- pyproject.toml: add VERSION file reference or set version to `"1.0.0"`
- DELETE setup.py entirely (conflicts with pyproject.toml)
- Create `VERSION` file at repo root with content `1.0.0`

#### Unit 8: Style Guide Updates
**Files:** `.claude/cpp-style.md`, `.claude/python-style.md`
**Changes:**
- cpp-style.md: update naming conventions to enforce snake_case for new functions (already partially done)
- cpp-style.md: add "No virtual dispatch in hot paths" performance rule
- cpp-style.md: add "Template on constraint only, not metrics" rule
- cpp-style.md: add ScratchMatrix pattern documentation
- python-style.md:7 — change minimum Python to 3.9
- python-style.md: update scikit-learn API guidance per adversarial review (no check_estimator, medoid_indices_ not cluster_centers_)
- python-style.md: add variable-length series input guidance (accept both ndarray and list-of-arrays)

#### Unit 9: Baseline Benchmarks (Before Performance Changes)

**Files:** `benchmarks/bench_dtw_baseline.cpp`, `benchmarks/CMakeLists.txt` (new), `cmake/Dependencies.cmake` (add Google Benchmark)

**Task:** Write baseline microbenchmarks capturing current performance BEFORE any optimization. These numbers are the reference for measuring improvement. Use Google Benchmark (BSD-3, via CPM). Benchmarks:

- `BM_dtwFull_scalar` — full DTW for series lengths 100, 500, 1000, 4000, 8000
- `BM_dtwFull_L_scalar` — linear-space DTW same lengths
- `BM_dtwBanded_scalar` — banded DTW with band=10, 50, 100 for length 1000, 8000
- `BM_fillDistanceMatrix` — fill N x N distance matrix for N=25 (dummy data), N=100
- `BM_calculateMedoids` — medoid calculation for N=25, k=3
- `BM_assignClusters` — cluster assignment for N=25, k=3

Store results as JSON in `benchmarks/baselines/` for future comparison. Add `DTWC_BUILD_BENCHMARK` CMake option to gate benchmark builds.

### Worker Instructions Template

Each worker receives the overall goal, its specific unit task, the e2e test recipe, and the standard post-implementation checklist (simplify, test, commit, PR).

---

**Target version:** v2.0.0 | **C++ standard:** C++17 minimum, C++23 preferred | **License:** BSD-3

## Context

DTWC++ is a published JOSS paper library (BSD-3, Oxford Battery Intelligence Lab) for DTW-based time series clustering. Target scale: **100 million time series x 8K samples**. Five adversarial expert agents reviewed v2 of this plan. This v3 incorporates all critical and high-severity findings.

### Adversarial Review Summary (Critical Findings Incorporated)

| ID | Finding | Expert | Status |
|----|---------|--------|--------|
| C1 | "PAM" is actually Lloyd iteration, not PAM | Scientific | **Fixed in Phase 0** |
| C2 | `dtwBanded` allocates 512MB/thread for 8K series; rolling buffer needs 808 bytes | HPC | **Fixed in Phase 0** |
| C3 | Soft-DTW via autodiff is fundamentally wrong (`std::min` gives hard subgradients) | Scientific | **Fixed in Phase 10** |
| C4 | DBI implementation is mathematically wrong (`dist(medoid, medoid)` = 0 always) | Scientific | **Fixed in Phase 0** |
| C5 | `pip install` will fail (Armadillo needs system LAPACK/BLAS, broken version pins) | Bindings | **Fixed in Phase 4** |
| C6 | No roofline analysis; DTW is memory-bound (0.125 FLOP/byte), SIMD won't help without fixing memory access | HPC | **Fixed in Phase 2** |
| H1 | `IDistanceMatrix` virtual dispatch: 300ms overhead per PAM iteration at N=10K | C++ Architect | **Fixed in Phase 1** |
| H2 | `DTWPolicy` template explosion: 54+ instantiations for negligible gain | C++ Architect | **Fixed in Phase 1** |
| H3 | LB_Keogh invalid for cosine/Huber metrics | Scientific | **Fixed in Phase 2** |
| H4 | WDTW/ADTW need recurrence changes, DDTW needs preprocessing — not metric swaps | Scientific | **Fixed in Phase 10** |
| H5 | Problem class over-decomposition (6 classes from 1 with tight coupling) | C++ Architect | **Fixed in Phase 1** |
| H6 | Armadillo isolation unrealistic without `ScratchMatrix<T>` design | C++ Architect | **Fixed in Phase 1** |
| H7 | Variable-length series + NumPy is unresolved | Bindings | **Fixed in Phase 4** |
| H8 | Z-normalization not mentioned (standard practice for DTW clustering) | HPC | **Fixed in Phase 1** |
| H9 | FastPAM not considered (O(k) speedup, Schubert & Rousseeuw 2021) | Scientific+HPC | **Fixed in Phase 1** |
| H10 | CMake `option()` syntax broken, source tree pollution, `main` CI trigger commented out | DevOps | **Fixed in Phase 0** |

### Non-Negotiable Rules (Updated)

1. **Documentation must reflect all mathematics** — every algorithm must have its formulation in `docs/`.
2. **Test-driven development with benchmarks** — independent agent writes tests/benchmarks. No performance claim without measurement.
3. **All citations tracked** — every algorithm must have verified citations in `.claude/CITATIONS.md`.
4. **Metric x lower-bound compatibility matrix** — LB_Keogh/LB_Kim must be validated per metric. Disable for unvalidated combinations.
5. **No silent single-core fallback** — warn on stderr if parallelism is unavailable.
6. **No custom binary formats** — all outputs in standard formats (HDF5, CSV) readable by Python/MATLAB/R.

---

## Phase 0: Critical Fixes (PR 1-2)

### 0A: Bugs and Correctness

| # | Bug | File | Fix |
|---|-----|------|-----|
| 0.1 | `throw 1;` bare integer throw | Problem_IO.cpp:42 | `throw std::runtime_error(...)` |
| 0.2 | `static std::mt19937` per-TU ODR issue | settings.hpp:35 | Change to `inline`. Long-term: `RNGManager` with deterministic child seeding. |
| 0.3 | `.at()` bounds-checked access in hot loop | warping.hpp:109-110 | Replace with `operator[]` |
| 0.4 | **`dtwBanded` allocates full N x N matrix** even for banded DTW. 8K x 8K = 512MB/thread. | warping.hpp:149,155-156 | **Replace with rolling buffer of width `2*band+1`.** For band=50: 808 bytes vs 512MB. This is the single highest-impact fix. |
| 0.5 | `fillDistanceMatrix` int overflow at N>46K | Problem.cpp:137 | Use `size_t`. Also: **audit all `int` sizes** — `Data::size()` returns `int` (overflows at 2.1B), `clusters_ind` is `vector<int>`. Change to `size_t` or `int64_t` throughout. |
| 0.6 | `dtwBanded` template defaults to `float` | warping.hpp:144 | Change default to `double` to match `data_t` |
| 0.7 | `static` vectors in `calculateMedoids` (data race) | Problem.cpp:235 | Make local. Also fix `clusterCosts` on line 250 (same issue). |
| 0.8 | `fillDistanceMatrix` iterates N^2 with N^2/2 no-ops | Problem.cpp:136-143 | Use triangular iteration: nested `for i in 0..N, for j in i+1..N` |
| 0.9 | DBI implementation wrong (`dist(medoid, medoid)` = 0) | scores.cpp:119 (commented) | Fix to use average within-cluster scatter: `S_i = mean(d(x, m_i) for x in cluster_i)` |
| 0.10 | Broken examples (`settings::band`) | examples/ | Use `prob.band` |
| 0.11 | Broken benchmark (`settings::root_folder`) | benchmark/ | Use `settings::paths::dataPath` |
| 0.12 | **Rename `cluster_by_kMedoidsPAM` to `cluster_by_kMedoidsLloyd`** | Problem.hpp/cpp | Current code does alternating assign+update-within-cluster (Lloyd iteration), NOT PAM's SWAP phase. Mislabeling is scientifically misleading. |

### 0B: CMake Fixes

| # | Bug | Fix |
|---|-----|-----|
| 0.13 | `option(DTWC_BUILD_EXAMPLES OFF)` — `OFF` is description, not default | `option(DTWC_BUILD_EXAMPLES "Build examples" OFF)` |
| 0.14 | `set(DTWC_ENABLE_GUROBI ON)` overrides user flags | Change to `option()` |
| 0.15 | `CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE` pollutes source tree | Output to `${CMAKE_BINARY_DIR}/bin` |
| 0.16 | `enable_testing()` called unconditionally + duplicated | Guard with `if(DTWC_BUILD_TESTING)` |
| 0.17 | StandardProjectSettings says "RelWithDebInfo" but sets Release | Fix message |
| 0.18 | `main` branch CI trigger commented out | Uncomment in all 3 workflow files |
| 0.19 | Executables built unconditionally | Guard with `if(PROJECT_IS_TOP_LEVEL)` |
| 0.20 | No sanitizer CI | Add ASan+UBSan job (GCC + Clang) |
| 0.21 | Armadillo pinned to branch `15.2.x` (not reproducible) | Pin to specific tag |
| 0.22 | `pyproject.toml` requires Python >=3.13 and numpy>=2.4.2 (doesn't exist) | Fix to `>=3.9` and `>=1.21` |
| 0.23 | Delete `setup.py` (conflicts with `pyproject.toml`) | Delete, use scikit-build-core only |
| 0.24 | Version mismatch (CMake 1.0.0, pyproject 0.1.0, setup.py 1.0.0) | Single `VERSION` file at repo root, read by CMake and pyproject.toml |

### 0C: Naming Convention

Establish and enforce before writing new code:
- **Types:** `PascalCase` (`TimeSeries`, `ClusteringResult`)
- **Functions/methods:** `snake_case` (`fill_distance_matrix`, `dtw_banded`)
- **Member variables:** `snake_case` with trailing underscore for private (`band_`, `dist_mat_`)
- **Namespaces:** `snake_case` (`dtwc`, `dtwc::core`)
- **Constants/enums:** `PascalCase` values (`ConstraintType::SakoeChibaBand`)
- No `I`-prefix for interfaces (not Java)

Apply incrementally via clang-tidy rename checks. Don't block other work.

---

## Phase 1: Core Architecture Refactor

### 1.1 Directory Restructure

Same as v2 plan (core/, algorithms/, io/, mip/, types/, enums/).

### 1.2 ScratchMatrix<T> (Replaces Armadillo in Hot Paths)

The experts identified that `mdspan` is a *view*, not an *owner*. Armadillo removal requires owning the memory. Solution: a minimal 20-line class:

```cpp
// core/scratch_matrix.hpp
template <typename T>
class ScratchMatrix {
    std::vector<T> data_;
    size_t rows_, cols_;
public:
    void resize(size_t r, size_t c) { rows_ = r; cols_ = c; data_.resize(r * c); }
    T& operator()(size_t i, size_t j) { return data_[i * cols_ + j]; } // row-major
    const T& operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }
    T* raw() { return data_.data(); }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    void fill(T val) { std::fill(data_.begin(), data_.end(), val); }
};
```

**Row-major layout** (unlike Armadillo's column-major) so that the DTW inner loop `for i in row` accesses contiguous memory. `thread_local ScratchMatrix<double>` replaces `thread_local arma::Mat`.

For the banded variant: a **rolling buffer** of width `2*band+1` replaces the full matrix:

```cpp
// thread_local std::vector<double> band_buffer((2*band+1) * sizeof(double));
// Access: band_buffer[i_local * (2*band+1) + j_local]
```

### 1.3 TimeSeriesView (Simplified per Expert Feedback)

```cpp
template <typename T = double>
struct TimeSeriesView {
    const T* data;
    size_t length;
    // No name, no alignment hints — those belong at container/allocator level
};
```

Name metadata stays in `Data`. Alignment is the allocator's concern (`aligned_allocator<T, 64>` for owning storage). SIMD dispatch checks alignment once per batch, not per view.

### 1.4 DTW Function Design (Simplified per Expert Feedback)

**Template on constraint type only** (2-3 variants). Pass metric as runtime callable. Rationale from C++ architect: metric dispatch overhead is 0.003% of DTW cost — templates are unjustified.

```cpp
// core/dtw.hpp
template <ConstraintType Constraint = ConstraintType::None, typename T = double, typename Metric>
T dtw(const T* x, size_t nx, const T* y, size_t ny,
      int band, const Metric& metric, double early_abandon = -1.0);

// Convenience with default L1
template <ConstraintType Constraint = ConstraintType::None, typename T = double>
T dtw(const T* x, size_t nx, const T* y, size_t ny, int band = -1);

// Runtime dispatch for bindings (switches on constraint only — 3 branches max)
double dtw_runtime(const double* x, size_t nx, const double* y, size_t ny,
                   const DTWOptions& opts);
```

This gives **6 instantiations max** (3 constraints x 2 scalar types) instead of 54+.

### 1.5 Distance Matrix (No Virtual Dispatch)

Per C++ architect: virtual `get(i,j)` costs 300ms per PAM iteration at N=10K. Use template parameter or flat array.

```cpp
// core/distance_matrix.hpp

// Dense: flat double* array with N*i + j indexing. Simple, fast, inlineable.
class DenseDistanceMatrix {
    std::vector<double> data_;
    size_t n_;
public:
    double get(size_t i, size_t j) const { return data_[i * n_ + j]; }
    void set(size_t i, size_t j, double v) { data_[i * n_ + j] = v; data_[j * n_ + i] = v; }
    // ...
};

// Clustering algorithms are templated on DistMatrix type (CRTP/template param)
template <typename DistMatrix>
ClusteringResult pam_swap(DistMatrix& dm, int k, const PAMOptions& opts);
```

Strategy selection happens at construction time (which `DistMatrix` to create), not at element access.

### 1.6 Problem Class (Minimal Decomposition)

Per C++ architect: 6 classes is over-engineering. The pieces are tightly coupled.

- **`Problem`** — keep as main coordinator. Rename methods per naming convention.
- **`ClusteringResult`** — extract as pure data struct: `{labels, medoid_indices, costs, iteration_history}`.
- **Clustering algorithms as free functions:** `ClusteringResult lloyd_iteration(Problem&, opts)`, `ClusteringResult pam_swap(Problem&, opts)`, `ClusteringResult fast_pam(Problem&, opts)`.
- **I/O as free functions:** `void write_clusters(const Problem&, const ClusteringResult&, path)`.

Three logical units instead of six classes.

### 1.7 Implement True PAM SWAP + FastPAM

The current "PAM" is Lloyd iteration (assign then update medoid within cluster). True PAM considers swapping any medoid with any non-medoid globally.

Implement **FastPAM** (Schubert & Rousseeuw 2021): O(k) speedup over naive PAM SWAP. Maintain nearest and second-nearest medoid for each point, only recompute when medoids change.

Keep Lloyd iteration as `lloyd_kmedoids()` (fast but lower quality). Offer `fast_pam()` as default. User chooses via options.

### 1.8 Z-Normalization

Add as preprocessing option. Standard practice for DTW clustering (UCR Suite recommends it). Without it, DTW is dominated by amplitude differences.

```cpp
template <typename T>
void z_normalize(T* series, size_t n); // in-place: subtract mean, divide by std

struct DTWOptions {
    // ...
    bool z_normalize = false;  // apply z-normalization before DTW
};
```

### 1.9 Metric x Lower-Bound Compatibility Matrix

LB_Keogh is **only valid for L1 and squared L2**. Invalid for cosine (defined between vectors, not scalars) and Huber (piecewise, not a squared norm).

```cpp
// core/lower_bounds.hpp
template <typename Metric>
constexpr bool lb_keogh_valid_for = false;

template <> constexpr bool lb_keogh_valid_for<L1Metric> = true;
template <> constexpr bool lb_keogh_valid_for<SquaredL2Metric> = true;
// Cosine, Huber: deliberately NOT listed — LB pruning disabled for these
```

At compile time, `if constexpr (lb_keogh_valid_for<Metric>)` enables/disables pruning. At runtime, the `DTWOptions` check validates `use_lb_keogh` against `metric`.

---

## Phase 2: Performance — Memory First, Then SIMD

### 2.0 Roofline Analysis (Do This FIRST)

The HPC expert's analysis: DTW inner loop does 5 FLOPs per 40 bytes = **0.125 FLOP/byte**. This is extremely low — DTW is **memory-bound**, not compute-bound.

**Implication:** SIMD won't help if memory access is the bottleneck. Fix memory first:

| Optimization | Expected Impact | Phase |
|---|---|---|
| Rolling buffer for banded DTW (808B vs 512MB) | **>100x** for 8K series | Phase 0 |
| Row-major `ScratchMatrix` (contiguous inner loop) | ~2x (cache line utilization) | Phase 1 |
| Envelope precomputation (O(N*L) once, amortized O(N^2) times) | 40-70% pair pruning for all-pairs | Phase 2 |
| Multi-pair SIMD (after memory is fixed) | 2-3x AVX2, 4-5x AVX-512 (realistic, not 4x/8x) | Phase 2 |

### 2.1 UCR Suite Optimizations

Same as v2 but with corrected claims:

- **LB_Kim** + **LB_Keogh** + **Early Abandoning** + **Cumulative LB residual**
- **Pruning effectiveness for all-pairs distance matrix:** 40-70% (NOT 90-99% — that figure is for subsequence search with a single query, per HPC expert)
- **Envelope precomputation cost:** 2 x N x L x 8 bytes. For 10K series of 8K length: 1.28 GB. Must be documented and budgeted.
- **LB_Keogh only enabled for L1 and SquaredL2** (see compatibility matrix above)

### 2.2 SIMD DTW (After Memory Fixes)

Multi-pair SIMD with corrected expectations:

- **AVX2 doubles:** Realistic speedup **2-3x** (not 4x). The min-of-three bottleneck and memory access limit gains.
- **AVX-512 doubles:** Realistic **4-5x** (not 8x).
- **Same-length padding overhead:** ~25% compute waste for heterogeneous-length datasets. Document this tradeoff.
- **Anti-diagonal wavefront** is actually better for single large pairs (8K x 8K has 8K independent cells per anti-diagonal). Implement both strategies.
- **SIMD data layout:** Consider AOSOA (Array of Structures of Arrays) for multi-pair: `x0[0],x1[0],x2[0],x3[0],x0[1],...` to avoid gather loads.

### 2.3 Microbenchmark Framework

Same as v2, but: **do NOT gate CI on benchmark regressions.** Shared GitHub Actions runners have 20-50% variance. Report benchmarks as PR comments. Use relative benchmarks within the same run (e.g., "SIMD is 3x faster than scalar").

---

## Phase 3: GPU/CUDA Support

### 3.1 Corrected Memory Analysis

The GPU tiling math from v2 omitted the DP cost matrix tile:

| Resource | Size | Fits 48KB shared? |
|---|---|---|
| Two 512-element float series tiles | 4 KB | Yes |
| 512 x 512 float DP tile | **1 MB** | **NO** |
| Two 32-element float series tiles | 256 B | Yes |
| 32 x 32 float DP tile | 4 KB | **Yes** |

**Correct tile size: 32x32** (not 512x512). This means 8K/32 = 250 tiles per dimension = 62,500 tile-pairs per DTW, with inter-tile sequential dependencies (anti-diagonal wavefront at tile level).

### 3.2 Corrected Kernel Design

- **One thread block per pair** has poor occupancy for small N (<1000 pairs). Add a fallback: multiple pairs per block for small N.
- **Banded DTW on GPU:** Warp divergence when band boundaries don't align with warp boundaries. Sort threads or use warp-aligned band chunks.
- **GPU memory:** Pre-allocate pools via `cudaMallocAsync`. Use pinned host memory (`cudaMallocHost`) for transfers. Explicit CUDA streams for compute/transfer overlap.
- **CUDA C++ standard:** Set to C++17 (NVCC doesn't fully support C++20/23). Strict firewall: `.cu` files receive raw pointers, never Armadillo/mdspan types.
- **macOS + CUDA:** Hard incompatibility (CUDA removed from macOS since CUDA 11). Document: CUDA is Linux/Windows only.
- **CUDA redistribution:** Users install CUDA Toolkit. We `find_package(CUDAToolkit)`. `cudart` is redistributable for wheels.

---

## Phase 4: Python Bindings — pybind11 (Revised)

### 4.1 Stay with pybind11 (Expert Recommendation)

Per bindings expert: The existing skills file (670 lines of pybind11 patterns) is valuable institutional knowledge. nanobind's performance advantages (compile speed, binary size) are irrelevant — binding dispatch is nanoseconds vs milliseconds of DTW computation. Smaller community means harder debugging.

**Decision:** Keep pybind11. If GPU tensor interop (`nb::ndarray<T, nb::device::cuda>`) is needed later, add a thin nanobind GPU extension module alongside.

### 4.2 Variable-Length Series Input (Critical Gap)

Accept **both** rectangular and ragged inputs:

```python
def fit(self, X):
    """
    X: numpy ndarray of shape (n_samples, n_timesteps) — equal-length series
       OR list of 1D numpy arrays — variable-length series
       OR pandas DataFrame — rows are series
    """
```

C++ side: `vector<vector<double>>` for variable-length, `double*` + stride for rectangular (zero-copy via buffer protocol).

### 4.3 scikit-learn API (Corrected)

Per bindings expert: Do NOT try to pass `check_estimator()`. k-medoids DTW has fundamental mismatches:
- `predict()` computes DTW to all medoids on-the-fly (O(k*L^2) per new series)
- `cluster_centers_` is misleading (medoids are actual data points, not means)

Instead: inherit `BaseEstimator` + `ClusterMixin` (get `set_params`/`get_params` for free). Provide:
- `labels_`, `medoid_indices_`, `medoids_`, `inertia_`
- `predict()` with clear docstring warning about on-the-fly computation
- Skip `transform()` (distance matrix shape N x N doesn't match sklearn's N x k expectation)

### 4.4 Packaging: cibuildwheel + Armadillo Strategy

The critical blocker: Armadillo needs LAPACK/BLAS system packages.

**Strategy for wheels:**
- **Linux:** Build in `manylinux_2_28` container with statically linked OpenBLAS + Armadillo
- **macOS:** Link against Accelerate framework (system-provided, no extra dep)
- **Windows:** Bundle OpenBLAS DLLs in wheel

```yaml
# .github/workflows/wheels.yml using cibuildwheel
- uses: pypa/cibuildwheel@v2
  env:
    CIBW_BEFORE_ALL_LINUX: yum install -y openblas-devel
```

Delete `setup.py`. Fix `pyproject.toml` to `requires-python = ">=3.9"`.

### 4.5 Binding Consistency Enforcement

Per bindings expert: 4 manually-synchronized codebases will diverge. Add a CI check:
- Script parses C++ public API headers (via libclang or regex)
- Verifies every public method has corresponding binding in `py_main.cpp`
- Warns on missing MATLAB bindings (non-blocking)

---

## Phase 5: Scaling — FastCLARA + MPI

### 5.1 CLARA Sample Size (Corrected Analysis)

Per scientific expert: The v2 plan's "10K-50K" sample size is a hand-wave with no formal guarantee.

Theory (Papadimitriou & Christodoulakis 2006): O(k²/ε²) samples for (1+ε)-approximation. For k=100, ε=0.1: need ~1M samples. For k=10, ε=0.1: need ~10K.

**Decision:** Default sample size = `max(40 + 2*k, 5*k*k)` (Kaufman & Rousseeuw heuristic for small k, formal bound for large k). Allow user override. Document that no formal approximation guarantee exists.

### 5.2 FastCLARA (Schubert & Rousseeuw 2021)

Use FastPAM (Phase 1.7) inside each CLARA sample. Number of samples R = 5 (default, user-configurable).

### 5.3 Assignment Phase Performance

After finding medoids, assigning 100M series to nearest medoid requires 100M x k DTW computations. For k=10, 8K banded series (band=50): ~0.2 hours on GPU. **Banded DTW is required** for the assignment phase at this scale. Document this.

### 5.4 MPI Architecture

Same as v2 but with OpenMP (not Taskflow) for intra-node parallelism.

---

## Phase 6: Checkpointing

Same as v2 (HDF5 primary, CSV fallback). No changes from expert review.

---

## Phase 7: Missing Data (Corrected Strategy)

### Primary: DTW-AROW (Not Zero-Cost)

Per scientific expert: Zero-cost DTW + normalization corrects magnitude but NOT routing bias — the path preferentially routes through missing regions. DTW-AROW's one-to-one constraint for missing positions prevents this exploitation.

**Decision:** Implement DTW-AROW as primary strategy. Offer zero-cost as fast approximation with documented bias warning.

```cpp
enum class MissingStrategy {
    Error,          // default — throw on NaN (backward compatible)
    AROW,           // DTW-AROW: one-to-one alignment for missing positions (recommended)
    ZeroCost,       // fast approximation: zero cost for missing, biased
    ZeroCostNorm,   // zero cost + path-length normalization (less biased, still imperfect)
    Interpolate,    // preprocess: linear interpolation before DTW
};
```

---

## Phase 8: I/O — HDF5 + Parquet

Same as v2. Use `find_package(HDF5)` (NOT CPM — HDF5's build system is too complex for CPM). Arrow/Parquet also via `find_package()`.

---

## Phase 9: MATLAB Bindings

Same as v2 but: evaluate C++ MEX API (`mex.hpp`) alongside legacy C API, given MathWorks' deprecation trajectory. CI: weekly/release-only (MATLAB license required).

---

## Phase 10: DTW Variants (Corrected Architecture)

### Variants Requiring Recurrence Changes (NOT Metric Swaps)

Per scientific expert: The metric abstraction `T operator()(T a, T b)` cannot implement these:

| Variant | What Changes | Architecture Need |
|---|---|---|
| **WDTW** | Weight `w(|i-j|)` multiplies cost per cell | Recurrence modification: `C(i,j) = w(|i-j|) * d(x[i],y[j]) + min(...)` |
| **ADTW** | Penalty for non-diagonal steps | Recurrence modification: add penalty to horizontal/vertical moves |
| **Soft-DTW** | `softmin_gamma` replaces `std::min` | Entirely separate algorithm with log-sum-exp stabilization |

### Variants Requiring Preprocessing (NOT Metric Swaps)

| Variant | Preprocessing |
|---|---|
| **DDTW** | Compute derivative series `x'[i] = ((x[i]-x[i-1]) + (x[i+1]-x[i-1])/2) / 2`, then apply standard DTW on x' |
| **Shape-DTW** | Extract shape descriptors, then apply DTW |

### Soft-DTW: Proper Implementation

**Do NOT use autodiff.** `std::min` gives hard subgradients (0 almost everywhere). Soft-DTW requires:

1. Replace `min(a,b,c)` with `softmin_gamma(a,b,c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))`
2. Log-sum-exp trick for numerical stability: `softmin = -gamma * (max_val + log(sum exp((max_val - a_i) / gamma)))`
3. Explicit backward pass from Cuturi & Blondel (2017) or forward-mode AD through the softmin (NOT through std::min)
4. Gamma parameter selection (default: 1.0, user-configurable)

```cpp
// core/soft_dtw.hpp — separate algorithm, NOT a metric substitution
template <typename T>
T soft_dtw(const T* x, size_t nx, const T* y, size_t ny,
           T gamma = 1.0, int band = -1);

template <typename T>
void soft_dtw_gradient(const T* x, size_t nx, const T* y, size_t ny,
                       T gamma, T* grad_x_out);
```

### Transform Pipeline

To support DDTW and Shape-DTW cleanly:

```cpp
// core/transforms.hpp
template <typename T>
std::vector<T> derivative_transform(const T* series, size_t n);

// Usage: DTW on transformed series
auto dx = derivative_transform(x, nx);
auto dy = derivative_transform(y, ny);
double d = dtw(dx.data(), dx.size(), dy.data(), dy.size(), band);
```

---

## Phase 11: Build System + Parallel Backend

### CMake Modernization

- Add `install()` rules and target export (`dtwc::dtwc`) — **do this in Phase 0**, not Phase 11
- Single `VERSION` file at repo root
- `find_package(dtwc)` via generated config
- Feature summary at configure time
- Drop GCC 10-11 and Clang 12-14 from CI if targeting C++23 features
- Use `find_package()` for heavy deps (HDF5, Arrow). CPM only for header-only/light deps.
- Add `CPM_SOURCE_CACHE` with CI caching for reproducibility

### Parallel Backend: Keep OpenMP, Add Verification

Per HPC expert: Taskflow is oversold for this workload. DTW distance matrix is a flat parallel-for — OpenMP's `schedule(dynamic)` is ideal. Taskflow's advantage (complex DAGs) is irrelevant here.

**Decision:** Keep OpenMP as primary. Fix the `omp_set_num_threads` global state problem (use `num_threads(N)` clause on pragma instead). Add Taskflow as optional alternative only if complex task graphs emerge (e.g., pipelined GPU+CPU computation).

**Parallelism verification:**

```cpp
// At startup or via --selftest
int measured = measure_actual_threads(); // run small parallel task, count unique thread IDs
if (measured == 1 && requested > 1)
    std::cerr << "WARNING: Only 1 thread active. Check OMP_NUM_THREADS or CPU affinity.\n";
```

### Self-Test Command

`dtwc selftest` — verify build config, measure actual parallelism, run numerical correctness checks (DTW identity/symmetry/non-negativity, LB <= DTW, PAM on reference dataset, GPU parity if available).

### C++23 Features

Same as v2 with kokkos/mdspan for C++17 backport. **Do NOT use `std::expected` for DTW return type** (per architect: forces callers to unwrap O(N^2) times for rare errors).

---

## Phase 12: Testing (Every Phase)

### Missing Tests to Add Immediately

- Clustering (Lloyd iteration, FastPAM, MIP)
- Silhouette, corrected DBI, CH index
- K-means++ initialization
- Distance matrix fill (correctness + symmetry)
- Z-normalization
- Early abandoning correctness

### Test Categories

1. **Unit** (Catch2), **Property** (symmetry, identity, LB <= DTW), **Integration**, **Regression**, **Performance** (Google Benchmark), **Cross-platform** CI, **Binding** (pytest + faulthandler + ASan), **MATLAB** (weekly)

---

## Dependency Table (Updated)

| Dependency | License | Via | Status |
|---|---|---|---|
| Armadillo | Apache-2 | CPM (pin tag) | Existing — isolate from core via ScratchMatrix |
| Catch2 | BSL-1 | CPM | Existing |
| HiGHS | MIT | CPM | Existing (make truly optional) |
| CLI11 | BSD-3 | CPM | Existing |
| rapidcsv | BSD-3 | CPM | Existing |
| Google Benchmark | Apache-2 | CPM | Add Phase 2 |
| kokkos/mdspan | Apache-2 | CPM | Add Phase 1 (header-only) |
| toml++ | MIT | CPM | Add Phase 11 (header-only) |
| pybind11 | BSD-3 | CPM | Existing (fix bindings) |
| scikit-build-core | Apache-2 | pip | Add Phase 4 |
| HDF5 | BSD-like | **find_package** (NOT CPM) | Add Phase 6 (optional) |
| Apache Arrow | Apache-2 | **find_package** (NOT CPM) | Add Phase 8 (optional) |
| CUDA Toolkit | NVIDIA EULA | **find_package** | Add Phase 3 (optional, users install) |

---

## Execution Order

```
Phase 0 (bugs + CMake + rename PAM) ─────────────────────────
    |
Phase 1 (architecture: ScratchMatrix, FastPAM, z-norm) ─────
    |                    |              |           |
Phase 2 (roofline first, Phase 4       Phase 7     Phase 11
    then SIMD+pruning)   (Python+cibw)  (missing)   (CMake+CLI)
    |
Phase 3 (GPU/CUDA)
    |
Phase 5 (FastCLARA + MPI)
    |
Phase 6 (checkpoint) + Phase 8 (HDF5/Parquet)
    |
Phase 9 (MATLAB) + Phase 10 (DTW variants + Soft-DTW)
```

---

## Citations to Add (Corrections from Scientific Expert)

**Fix in CITATIONS.md:**
- Jain (2018): correct title to "Semi-Metrification of the Dynamic Time Warping Distance"
- Yurtman et al. (2023): correct co-authors to Soenen, Meert, Blockeel

**Add missing:**
- Schubert & Rousseeuw (2021), "Fast and eager k-medoids clustering", JMLR 22(1), 4653-4688 — **FastPAM**
- Rousseeuw (1987), "Silhouettes: A graphical aid...", J. Comput. Appl. Math. 20, 53-65 — **Silhouette**
- Keogh & Pazzani (2001), "Derivative Dynamic Time Warping", SDM — **DDTW**
- Jeong et al. (2011), "Weighted dynamic time warping", Pattern Recognition 44(9), 2231-2240 — **WDTW**
- Cuturi & Blondel (2017), "Soft-DTW", ICML, PMLR 70, 894-903 — **Soft-DTW**
- Lemire (2009), "Faster retrieval with a two-pass dynamic-time-warping lower bound" — **LB_Improved**
