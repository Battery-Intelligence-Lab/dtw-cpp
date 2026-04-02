# Execution Plan: MIP Solver Improvements (Tasks 1-3)

## Context

Four waves of algorithm and DTW variant work are complete (1A/1B/2A/2B). The MIP solver
path (`Method::MIP`) is functional but unoptimized — no warm start, excessive
`NumericFocus`, no solver settings exposed to users. The detailed design is in
`.claude/superpowers/plans/2026-04-02-mip-solver-improvements.md`. This plan covers
executing Tasks 1-3 from that document. Task 4 (Benders) and Task 5 (odd-cycle cuts)
are deferred.

**Goals:**
1. Add FastPAM warm start to both HiGHS and Gurobi MIP solvers
2. Tune Gurobi parameters (NumericFocus, MIPFocus, branching priority)
3. Add `MIPSettings` struct and expose solver settings in CLI + TOML config
4. Add yaml-cpp as optional dependency for YAML config support
5. Add tests, update CHANGELOG

---

## Adversarial Review Findings (incorporated)

Two adversarial passes identified critical issues in the original design plan. All are
addressed below. Key corrections:

- **HighsSolution API:** Aggregate init `{start_values, {}}` is wrong — struct starts
  with `bool value_valid`. Must construct properly and set `value_valid = true`.
- **CLI11 TOML sections:** `[solver]` section requires a CLI11 subcommand, not flat options.
  Decision: use flat top-level TOML keys with `mip-` prefix for simplicity.
- **yaml-cpp CMake:** Must link in root `CMakeLists.txt` (not `Dependencies.cmake`) because
  `dtwc_cl` target doesn't exist inside `dtwc_setup_dependencies()`. Need alias guard and
  `NOT TARGET` guard for CPM consistency.
- **Variable indexing:** Gurobi uses `medoid + point * Nb`, HiGHS uses `medoid * Nb + point`.
  Both have diagonal at `i * (Nb + 1)`. Warm start code must use the correct convention per solver.
- **Field name:** `ClusteringResult.medoid_indices` (not `.medoids`).

---

## Step 1: Add `MIPSettings` struct

**File:** `dtwc/Problem.hpp`

Add before the `Problem` class definition:

```cpp
/// MIP solver tuning parameters.
struct MIPSettings {
  double mip_gap = 1e-5;
  int time_limit_sec = -1;       // -1 = unlimited
  bool warm_start = true;         // Run FastPAM first and feed as MIP start
  int numeric_focus = 1;          // Gurobi NumericFocus (0-3)
  int mip_focus = 2;              // Gurobi MIPFocus (0=balanced, 2=optimal)
  bool verbose_solver = false;    // Show solver log output
};
```

Add `MIPSettings mip_settings;` as a public member of `Problem` (near line 90, alongside
other settings like `verbose`, `distance_strategy`).

---

## Step 2: MIP Warm Start + Tuning

### `dtwc/mip/mip_Gurobi.cpp`

Add `#include "../algorithms/fast_pam.hpp"` at the top (NOT in mip.hpp — keep header clean).

After `model.setObjective(obj, GRB_MINIMIZE)` (current line 69), before `model.optimize()`:

**Warm start:**
```cpp
if (prob.mip_settings.warm_start) {
  auto pam_result = fast_pam(prob, static_cast<int>(Nc));
  // Distance matrix already filled by fillDistanceMatrix() above — no double computation.

  for (size_t idx = 0; idx < Nb * Nb; ++idx)
    w[idx].set(GRB_DoubleAttr_Start, 0.0);

  for (int med : pam_result.medoid_indices)
    w[static_cast<size_t>(med) * (Nb + 1)].set(GRB_DoubleAttr_Start, 1.0);

  // Gurobi indexing: A[i,j] at flat index i + j * Nb (column-major)
  for (size_t j = 0; j < Nb; ++j) {
    int med = pam_result.medoid_indices[pam_result.labels[j]];
    w[static_cast<size_t>(med) + j * Nb].set(GRB_DoubleAttr_Start, 1.0);
  }
}
```

**Tuning (replaces hardcoded NumericFocus=3):**
```cpp
model.set(GRB_IntParam_NumericFocus, prob.mip_settings.numeric_focus);
model.set(GRB_IntParam_MIPFocus, prob.mip_settings.mip_focus);
model.set(GRB_DoubleParam_MIPGap, prob.mip_settings.mip_gap);
if (prob.mip_settings.time_limit_sec > 0)
  model.set(GRB_DoubleParam_TimeLimit, static_cast<double>(prob.mip_settings.time_limit_sec));
if (!prob.mip_settings.verbose_solver)
  model.set(GRB_IntParam_OutputFlag, 0);
```

**Branching priority on diagonals (after variable creation, line 38):**
```cpp
for (size_t i = 0; i < Nb; ++i)
  w[i * (Nb + 1)].set(GRB_IntAttr_BranchPriority, 100);
```

### `dtwc/mip/mip_Highs.cpp`

Add `#include "../algorithms/fast_pam.hpp"` at the top.

After `highs.passModel(model)` (current line 156), before `highs.run()`:

**Warm start (correct HighsSolution API):**
```cpp
if (prob.mip_settings.warm_start) {
  auto pam_result = fast_pam(prob, static_cast<int>(Nc));

  HighsSolution sol;
  sol.col_value.resize(Nvar, 0.0);
  sol.value_valid = true;

  for (int med : pam_result.medoid_indices)
    sol.col_value[static_cast<size_t>(med) * (Nb + 1)] = 1.0;

  // HiGHS indexing: A[i,j] at flat index i * Nb + j (row-major)
  for (size_t j = 0; j < Nb; ++j) {
    int med = pam_result.medoid_indices[pam_result.labels[j]];
    sol.col_value[static_cast<size_t>(med) * Nb + j] = 1.0;
  }

  highs.setSolution(sol);
}
```

**Tuning:**
```cpp
highs.setOptionValue("mip_rel_gap", prob.mip_settings.mip_gap);
if (prob.mip_settings.time_limit_sec > 0)
  highs.setOptionValue("time_limit", static_cast<double>(prob.mip_settings.time_limit_sec));
if (!prob.mip_settings.verbose_solver)
  highs.setOptionValue("output_flag", false);
```

Also suppress the hardcoded `std::cout << "HiGS is being called!"` — gate it behind
`prob.mip_settings.verbose_solver || prob.verbose`.

---

## Step 3: Expose Settings in CLI + TOML Config

**File:** `dtwc/dtwc_cl.cpp`

CLI11 TOML sections (`[solver]`) require subcommands. To keep things simple,
use flat top-level keys. **kebab-case everywhere** — CLI flags, TOML keys, and
YAML keys all use the same names (`mip-gap`, `time-limit`, etc.). This means
config file keys match CLI flags exactly with zero mapping.

After the existing `--solver` option block (line 172), add:

```cpp
// MIP solver settings — kebab-case matches CLI flags, TOML keys, and YAML keys
double mip_gap = 1e-5;
int time_limit = -1;
bool no_warm_start = false;
int numeric_focus = 1;
int mip_focus = 2;
bool verbose_solver = false;

app.add_option("--mip-gap", mip_gap, "MIP optimality gap tolerance (default: 1e-5)");
app.add_option("--time-limit", time_limit, "MIP solver time limit in seconds (-1 = unlimited)");
app.add_flag("--no-warm-start", no_warm_start, "Disable FastPAM warm start for MIP");
app.add_option("--numeric-focus", numeric_focus, "Gurobi NumericFocus (0-3, default: 1)");
app.add_option("--mip-focus", mip_focus, "Gurobi MIPFocus (0-3, default: 2)");
app.add_flag("--verbose-solver", verbose_solver, "Show MIP solver log output");
```

After `CLI11_PARSE`, wire into prob:

```cpp
prob.mip_settings.mip_gap = mip_gap;
prob.mip_settings.time_limit_sec = time_limit;
prob.mip_settings.warm_start = !no_warm_start;
prob.mip_settings.numeric_focus = numeric_focus;
prob.mip_settings.mip_focus = mip_focus;
prob.mip_settings.verbose_solver = verbose_solver;
```

**Naming convention:** kebab-case everywhere. Config keys = CLI flags.

**TOML** (`examples/config.toml`):
```toml
input = "data/scooters.csv"
output = "results/"
clusters = 5
method = "mip"
solver = "highs"
mip-gap = 1e-5
time-limit = 300
numeric-focus = 1
mip-focus = 2
verbose-solver = false
verbose = true
```

**YAML** (`examples/config.yaml`):
```yaml
input: data/scooters.csv
output: results/
clusters: 5
method: mip
solver: highs
mip-gap: 1.0e-5
time-limit: 300
numeric-focus: 1
mip-focus: 2
verbose-solver: false
verbose: true
```

Identical keys across CLI, TOML, and YAML. No mapping needed.

---

## Step 4: YAML Config Support (Optional Dependency)

### `CMakeLists.txt` (root)

Add option near other optional deps:
```cmake
option(DTWC_ENABLE_YAML "Enable YAML configuration file support via yaml-cpp" OFF)
```

Default OFF — TOML covers all settings; YAML is a convenience for users who prefer it.

### `cmake/Dependencies.cmake`

Add inside `dtwc_setup_dependencies()`, after CLI11 block:
```cmake
if(DTWC_ENABLE_YAML AND NOT TARGET yaml-cpp)
  CPMAddPackage(
    NAME yaml-cpp
    URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz"
    OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF"
    SYSTEM YES
  )
  # Ensure namespaced alias exists (CPM subdirectory may not create it)
  if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
    add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
  endif()
  if(NOT TARGET yaml-cpp)
    message(WARNING "yaml-cpp not found -- YAML config support disabled")
    set(DTWC_ENABLE_YAML OFF PARENT_SCOPE)
  endif()
endif()
```

### `CMakeLists.txt` (root, inside `PROJECT_IS_TOP_LEVEL` block, after `add_executable(dtwc_cl)`)

```cmake
if(DTWC_ENABLE_YAML AND TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(dtwc_cl PRIVATE yaml-cpp::yaml-cpp)
  target_compile_definitions(dtwc_cl PRIVATE DTWC_HAS_YAML)
endif()
```

### `dtwc/dtwc_cl.cpp`

Replace `app.set_config("--config", "", "Read TOML configuration file")` with:

```cpp
std::string config_path;
app.add_option("--config", config_path, "Configuration file (TOML or YAML)");
```

Before `CLI11_PARSE`, add TOML config support conditionally:
```cpp
// If config path is .toml, use CLI11 built-in parser
// CLI11_PARSE will handle TOML automatically if set_config was used
// We handle YAML separately after parse
```

After `CLI11_PARSE`, add YAML loading:
```cpp
if (!config_path.empty()) {
  auto ext = fs::path(config_path).extension().string();
  if (ext == ".yaml" || ext == ".yml") {
#ifdef DTWC_HAS_YAML
    load_yaml_config(config_path, /* references to all option variables */);
    // CLI flags override: re-parse CLI args after YAML loading
    // (CLI11 already parsed, so CLI values take precedence by default —
    //  only set YAML values for options that weren't explicitly provided on CLI)
#else
    std::cerr << "Error: YAML config requires building with -DDTWC_ENABLE_YAML=ON\n";
    return EXIT_FAILURE;
#endif
  } else {
    // Re-parse with TOML: use CLI11's built-in config reader
    // (This path handles .toml and .ini files)
    app.set_config("--config", config_path, "Read configuration file");
    // Note: must re-parse or handle differently — see implementation detail
  }
}
```

Implementation detail: CLI11's `set_config` must be called before `CLI11_PARSE`. The
cleanest approach is to keep `app.set_config("--config")` for TOML (pre-parse), and add
a separate `--yaml-config` option for YAML (post-parse), or detect the extension in a
two-pass approach. The implementor should choose the simplest working approach.

---

## Step 5: Tests

**File:** `tests/unit/unit_test_mip.cpp` (new — auto-registered by CMake glob)

Use synthetic data (N=8-10 short series) — NOT `data/dummy` which is too large.

```cpp
#include <dtwc.hpp>
#include <dtwc/algorithms/fast_pam.hpp>
#include <catch2/catch_test_macros.hpp>

// Helper: build a small Problem with N synthetic series of length L
static dtwc::Problem make_small_problem(int N, int L) { ... }

#ifdef DTWC_ENABLE_HIGHS
TEST_CASE("MIP HiGHS: warm start produces valid result", "[mip][highs]") {
  auto prob = make_small_problem(8, 20);
  prob.set_numberOfClusters(2);
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;
  prob.cluster();

  REQUIRE(prob.centroids_ind.size() == 2);
  REQUIRE(prob.clusters_ind.size() == 8);
  // Every point assigned to a valid cluster
  for (auto c : prob.clusters_ind)
    REQUIRE((c >= 0 && c < 2));
}

TEST_CASE("MIP HiGHS: cold start matches warm start", "[mip][highs]") {
  auto prob1 = make_small_problem(8, 20);
  prob1.set_numberOfClusters(2);
  prob1.mip_settings.warm_start = false;
  prob1.set_solver(dtwc::Solver::HiGHS);
  prob1.method = dtwc::Method::MIP;
  prob1.cluster();
  double cold_cost = prob1.findTotalCost();

  auto prob2 = make_small_problem(8, 20);
  prob2.set_numberOfClusters(2);
  prob2.mip_settings.warm_start = true;
  prob2.set_solver(dtwc::Solver::HiGHS);
  prob2.method = dtwc::Method::MIP;
  prob2.cluster();
  double warm_cost = prob2.findTotalCost();

  // Both should find the same optimal (or warm should be equal/better)
  REQUIRE(warm_cost <= cold_cost + 1e-6);
}
#endif

#ifdef DTWC_ENABLE_GUROBI
// Equivalent tests for Gurobi
#endif
```

---

## Step 6: Example Config + CHANGELOG

**Create** `examples/config.toml` with all documented settings.
**Create** `examples/config.yaml` with equivalent YAML format.

**CHANGELOG.md** (Unreleased section):
```
### Added
- MIP warm start: `--method mip` now runs FastPAM first and feeds the solution
  as a MIP start, reducing solve time significantly.
- MIP solver settings in CLI/TOML: `--mip-gap`, `--time-limit`, `--no-warm-start`,
  `--numeric-focus`, `--mip-focus`, `--verbose-solver`.
- Optional YAML config support (`--config config.yaml`) via yaml-cpp.

### Changed
- Gurobi NumericFocus reduced 3 -> 1, added MIPFocus=2 and branching priority
  on medoid selection variables A[i,i].
- MIP solver output suppressed by default (use `--verbose-solver`).
```

---

## Files Summary

| File | Action | Change |
|------|--------|--------|
| `dtwc/Problem.hpp` | Modify | Add `MIPSettings` struct + member |
| `dtwc/mip/mip_Gurobi.cpp` | Modify | Warm start, tuning, branching priority, output suppression |
| `dtwc/mip/mip_Highs.cpp` | Modify | Warm start (correct HighsSolution API), tuning, output suppression |
| `dtwc/dtwc_cl.cpp` | Modify | CLI solver options, YAML dispatch |
| `CMakeLists.txt` | Modify | YAML option, yaml-cpp link for dtwc_cl |
| `cmake/Dependencies.cmake` | Modify | yaml-cpp CPMAddPackage with guards |
| `CHANGELOG.md` | Modify | Document changes |
| `tests/unit/unit_test_mip.cpp` | Create | Warm start + settings tests |
| `examples/config.toml` | Create | Example TOML config |
| `examples/config.yaml` | Create | Example YAML config |

## Key Reusable Code

| Symbol | Location | Usage |
|--------|----------|-------|
| `fast_pam(Problem&, int, int)` | `dtwc/algorithms/fast_pam.hpp:44` | Warm start |
| `ClusteringResult.medoid_indices` | `dtwc/core/clustering_result.hpp:20` | Medoid indices |
| `ClusteringResult.labels` | `dtwc/core/clustering_result.hpp:19` | Point assignments |
| `extract_mip_solution()` | `dtwc/mip/mip_Highs.cpp:30` | Reuse pattern |

## Verification

1. **Build:** `cmake -S . -B build -DDTWC_BUILD_TESTING=ON && cmake --build build --config Release -j`
2. **Build with YAML:** Add `-DDTWC_ENABLE_YAML=ON`
3. **Build without YAML:** `-DDTWC_ENABLE_YAML=OFF` (must still compile)
4. **Tests:** `ctest --test-dir build --build-config Release -R mip`
5. **Regression:** `ctest --test-dir build --build-config Release` (all existing tests pass)
6. **CLI smoke:** `./build/bin/dtwc_cl -i data/dummy -k 3 -m mip -v --verbose-solver`
