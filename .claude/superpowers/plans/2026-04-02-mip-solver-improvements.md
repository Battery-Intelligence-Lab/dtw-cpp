# MIP Solver Improvements — Implementation Plan

**Date:** 2026-04-02
**Status:** Draft
**Branch:** Claude
**Background:** `.claude/UNIMODULAR.md` (TU analysis of the p-median LP relaxation)

---

## Motivation

The MIP path (`Method::MIP`) in DTWC++ solves exact k-medoids via the Balinski p-median
formulation. The current implementation is functional but leaves significant performance
on the table. The UNIMODULAR.md analysis identified that:

1. The constraint matrix is NOT totally unimodular for p >= 3 (odd-cycle obstruction).
2. Once medoid variables A[i,i] are fixed, the remaining assignment is a **transportation
   problem** with a TU constraint matrix — the LP gives integer assignments for free.
3. Fractional LP solutions are half-integral, caused by odd cycles among facilities.

These structural insights lead to four concrete improvements, ordered by impact/effort.

---

## Current Code

| File | Lines | What it does |
|------|-------|-------------|
| `dtwc/mip/mip.hpp` | 17 | Forward declarations for `MIP_clustering_byGurobi`, `MIP_clustering_byHiGHS` |
| `dtwc/mip/mip_Gurobi.cpp` | 99 | Gurobi MIP: builds model, sets `NumericFocus=3`, solves, extracts solution |
| `dtwc/mip/mip_Highs.cpp` | 189 | HiGHS MIP: builds model, sets integrality on all vars, solves, extracts solution |
| `dtwc/Problem.cpp:437-447` | 11 | `cluster_by_MIP()` dispatches to Gurobi or HiGHS |
| `dtwc/Problem.hpp:67` | 1 | `Solver mipSolver` member |
| `dtwc/algorithms/fast_pam.hpp` | 44 | `fast_pam(Problem&, int, int)` returns `ClusteringResult` |

**Key observations about the current code:**

- Neither solver receives a warm start (MIP start) from the heuristic solution.
- `fillDistanceMatrix()` is called inside the MIP function — the distance matrix is ready.
- `fast_pam()` returns `ClusteringResult` with `.medoids` (vector of indices) and `.labels`.
- The Gurobi code uses `NumericFocus = 3` (maximum), which is overkill for 0/1/-1 constraint
  coefficients and adds 1.5-3x overhead.
- Variable indexing: flat `i + j * Nb` in Gurobi (column-major), `i + j * Nb` in HiGHS.
  Diagonal A[i,i] is at flat index `i * (Nb + 1)`.
- Neither solver uses branching priorities, symmetry breaking, or cutting planes.

---

## Task 1: MIP Warm Start from FastPAM

**Impact: HIGH.  Effort: LOW (~30 lines total).**

The PAM heuristic already runs fast. Feeding its solution as a MIP start gives the solver
a tight upper bound immediately, which prunes the branch-and-bound tree dramatically.

### Files to modify

- `dtwc/mip/mip_Gurobi.cpp`
- `dtwc/mip/mip_Highs.cpp`
- `dtwc/mip/mip.hpp` (add parameter or use Problem's existing medoid info)

### Implementation

Run `fast_pam()` before solving the MIP. Pass its medoid set and assignment as a warm start.

**Gurobi** (insert after line 69 `model.setObjective(...)`, before `model.optimize()`):

```cpp
// --- MIP warm start from FastPAM ---
{
  auto pam_result = fast_pam(prob, static_cast<int>(Nc));

  // Set all variables to 0 first
  for (size_t idx = 0; idx < Nb * Nb; ++idx)
    w[idx].set(GRB_DoubleAttr_Start, 0.0);

  // Set diagonal (medoid) variables
  for (int med : pam_result.medoids)
    w[static_cast<size_t>(med) * (Nb + 1)].set(GRB_DoubleAttr_Start, 1.0);

  // Set assignment variables: A[medoid, point] = 1
  for (size_t j = 0; j < Nb; ++j) {
    int med = pam_result.medoids[pam_result.labels[j]];
    w[static_cast<size_t>(med) + j * Nb].set(GRB_DoubleAttr_Start, 1.0);
  }
}
```

**HiGHS** (insert after `highs.passModel(model)`, before `highs.run()`):

```cpp
// --- MIP warm start from FastPAM ---
{
  auto pam_result = fast_pam(prob, static_cast<int>(Nc));

  std::vector<double> start_values(Nvar, 0.0);

  for (int med : pam_result.medoids)
    start_values[static_cast<size_t>(med) * (Nb + 1)] = 1.0;

  for (size_t j = 0; j < Nb; ++j) {
    int med = pam_result.medoids[pam_result.labels[j]];
    start_values[static_cast<size_t>(med) * Nb + j] = 1.0;
  }

  highs.setSolution(HighsSolution{start_values, {}});
}
```

### Tests

- [ ] Gurobi MIP with warm start produces the same or better objective than without.
- [ ] HiGHS MIP with warm start produces the same or better objective than without.
- [ ] Warm start solution is feasible (no solver warnings about infeasible MIP start).
- [ ] Existing `unit_test_Problem` MIP tests still pass.

---

## Task 2: Gurobi Parameter Tuning

**Impact: MEDIUM.  Effort: LOW (~5 lines).**

### Files to modify

- `dtwc/mip/mip_Gurobi.cpp`

### Changes

Replace lines 71-72:

```cpp
// Current:
model.set(GRB_IntParam_NumericFocus, 3); // Much numerics
model.set(GRB_DoubleParam_MIPGap, 1e-5);

// New:
model.set(GRB_IntParam_NumericFocus, 1);  // Sufficient for 0/1/-1 constraints
model.set(GRB_IntParam_MIPFocus, 2);      // Focus on proving optimality (tight LP bound)
model.set(GRB_DoubleParam_MIPGap, 1e-5);
```

### Branching priority on diagonal (facility) variables

Add after variable creation (after line 38):

```cpp
// Branch on medoid selection first — once A[i,i] is fixed,
// assignment is a transportation problem (TU) and LP-integral.
for (size_t i = 0; i < Nb; ++i)
  w[i * (Nb + 1)].set(GRB_IntAttr_BranchPriority, 100);
```

**Why this works:** The UNIMODULAR.md analysis proves that fixed medoid variables yield a
TU assignment subproblem. By branching on the N diagonal variables first (instead of all
N^2), the solver effectively solves a problem with N binary variables plus a free assignment.

### Tests

- [ ] Same optimal objective as before (or better, with faster solve time).
- [ ] Solver log shows branching on diagonal variables (verify with `model.set(GRB_IntParam_OutputFlag, 1)`).

---

## Task 3: TOML + YAML Config with MIP/Solver Settings

**Impact: MEDIUM.  Effort: MEDIUM (~100-120 lines).**

The CLI already has `--config` via CLI11's built-in TOML parser and `--solver highs|gurobi`.
This task: (1) adds YAML as an alternative config format via yaml-cpp, (2) exposes
MIP-relevant solver settings in both config formats and CLI flags.

**Both formats supported:** TOML via CLI11 built-in (zero deps), YAML via yaml-cpp (optional).
Detection by file extension: `.toml` uses CLI11, `.yaml`/`.yml` uses yaml-cpp. If yaml-cpp
is not available (`-DDTWC_ENABLE_YAML=OFF`), YAML files produce a clear error message.

### Files to modify

- `cmake/Dependencies.cmake` — add yaml-cpp via CPM (gated on `DTWC_ENABLE_YAML`)
- `CMakeLists.txt` — add `DTWC_ENABLE_YAML` option (default ON), link yaml-cpp
- `dtwc/dtwc_cl.cpp` — YAML loader, solver option group, config dispatch by extension
- `dtwc/Problem.hpp` — add `MIPSettings` struct

### Dependency: yaml-cpp (optional)

```cmake
# In cmake/Dependencies.cmake:
if(DTWC_ENABLE_YAML)
  CPMAddPackage(
    NAME yaml-cpp
    URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz"
    OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF"
             "YAML_CPP_BUILD_CONTRIB OFF"
  )
  if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(dtwc_cl PRIVATE yaml-cpp::yaml-cpp)
    target_compile_definitions(dtwc_cl PRIVATE DTWC_HAS_YAML)
  endif()
endif()
```

### Config file examples (TOML and YAML, equivalent)

**TOML** (`config.toml`):
```toml
input = "data/scooters.csv"
output = "results/"
name = "scooter_clusters"
verbose = true
clusters = 5
method = "pam"
band = 50
metric = "l1"
variant = "standard"
max-iter = 100
device = "cpu"
precision = "auto"

[solver]
name = "highs"
mip-gap = 1e-5
time-limit = 300
warm-start = true
numeric-focus = 1
mip-focus = 2
benders = "auto"
verbose = false
```

**YAML** (`config.yaml`):
```yaml
input: data/scooters.csv
output: results/
name: scooter_clusters
verbose: true

clustering:
  clusters: 5
  method: pam
  band: 50
  metric: l1
  variant: standard
  max_iter: 100

solver:
  name: highs
  mip_gap: 1.0e-5
  time_limit: 300
  warm_start: true
  numeric_focus: 1
  mip_focus: 2
  benders: auto
  verbose: false

device: cpu
precision: auto
```

### Config loading dispatch

```cpp
// In dtwc_cl.cpp main(), after CLI11_PARSE:
std::string config_path;
app.add_option("--config", config_path, "Configuration file (TOML or YAML)");

// After parse:
if (!config_path.empty()) {
  auto ext = fs::path(config_path).extension().string();
  if (ext == ".yaml" || ext == ".yml") {
#ifdef DTWC_HAS_YAML
    load_yaml_config(config_path, /* option references */);
#else
    throw std::runtime_error("YAML support not compiled. Use TOML or rebuild with -DDTWC_ENABLE_YAML=ON");
#endif
  }
  // else: CLI11 handles TOML natively via set_config()
}
```

### CLI flags (override any config file)

```
dtwc_cl --config config.yaml                           # YAML config
dtwc_cl --config config.toml                           # TOML config
dtwc_cl -m mip --mip-gap 0.01 --time-limit 60         # override specific settings
dtwc_cl -m mip --no-warm-start                         # disable warm start
dtwc_cl -m mip --benders on                            # force Benders
```

### MIPSettings struct (in `Problem.hpp`)

```cpp
struct MIPSettings {
  double mip_gap = 1e-5;
  int time_limit_sec = -1;       // -1 = unlimited
  bool warm_start = true;         // Run FastPAM and feed as MIP start
  int numeric_focus = 1;          // Gurobi NumericFocus (0-3)
  int mip_focus = 2;              // Gurobi MIPFocus (0=balanced, 2=optimal)
  std::string benders = "auto";   // "auto", "on", "off"
  bool verbose_solver = false;
};
```

### CLI flags (also settable in YAML)

```
--mip-gap <float>           MIP optimality gap tolerance (default: 1e-5)
--time-limit <int>          Solver time limit in seconds
--no-warm-start             Disable FastPAM warm start for MIP
--numeric-focus <int>       Gurobi NumericFocus (0-3)
--mip-focus <int>           Gurobi MIPFocus (0-3)
--benders <auto|on|off>     Benders decomposition mode
```

### Pass settings through to solvers

Both `MIP_clustering_byGurobi` and `MIP_clustering_byHiGHS` gain a `const MIPSettings&`
parameter (or read from `prob.mip_settings`):

```cpp
// In mip_Gurobi.cpp:
model.set(GRB_IntParam_NumericFocus, prob.mip_settings.numeric_focus);
model.set(GRB_IntParam_MIPFocus, prob.mip_settings.mip_focus);
model.set(GRB_DoubleParam_MIPGap, prob.mip_settings.mip_gap);
if (prob.mip_settings.time_limit_sec > 0)
  model.set(GRB_DoubleParam_TimeLimit, prob.mip_settings.time_limit_sec);
if (!prob.mip_settings.verbose_solver)
  model.set(GRB_IntParam_OutputFlag, 0);

// In mip_Highs.cpp:
highs.setOptionValue("mip_rel_gap", prob.mip_settings.mip_gap);
if (prob.mip_settings.time_limit_sec > 0)
  highs.setOptionValue("time_limit", static_cast<double>(prob.mip_settings.time_limit_sec));
if (!prob.mip_settings.verbose_solver)
  highs.setOptionValue("output_flag", false);
```

### Tests

- [ ] TOML config with `[solver]` section parses correctly.
- [ ] YAML config with `solver:` section parses correctly (when yaml-cpp available).
- [ ] CLI flags override config file values (both TOML and YAML).
- [ ] Default values match current behaviour (except NumericFocus 3->1).
- [ ] `--no-warm-start` disables PAM warm start.
- [ ] Build without yaml-cpp (`-DDTWC_ENABLE_YAML=OFF`) works; YAML config gives clear error.
- [ ] Unknown solver settings are rejected with a clear error.
- [ ] Provide example `config.toml` and `config.yaml` in `examples/` directory.

---

## Task 4: Benders Decomposition for Large N

**Impact: HIGH.  Effort: HIGH (~200-300 lines new file).**

For N > 200, the compact N^2-variable MIP becomes impractical. Benders decomposition
exploits the same TU insight: the master problem selects medoids (N binary variables),
the subproblem assigns points (always integral, O(Nk)).

### Files to create

- `dtwc/mip/benders.hpp`
- `dtwc/mip/benders.cpp`

### Files to modify

- `dtwc/mip/mip.hpp` — add `MIP_clustering_byBenders` declaration
- `dtwc/Problem.cpp` — dispatch to Benders when N > threshold (or new solver enum value)

### Algorithm

```
Input: distance matrix D[N x N], k, PAM solution (medoids, labels, cost)
Output: optimal medoid set and assignment

Master problem:
  min  theta
  s.t. sum_i y_i = k              (exactly k medoids)
       y_i in {0,1}               (N binary variables)
       theta >= 0
       + Benders optimality cuts   (added iteratively)

Subproblem (given fixed y*):
  For each point j, assign to nearest open medoid:
    sigma(j) = argmin_{i : y*_i = 1} D[i,j]
  Compute exact cost:
    Z = sum_j D[sigma(j), j]

If Z <= theta* + eps:
  Current solution is optimal. Stop.
Else:
  Add Benders optimality cut:
    theta >= Z - sum_{i in S} delta_i * (1 - y_i)
  where S = current medoid set,
        delta_i = sum_{j assigned to i} (D[i,j] - D[second_nearest(j), j])
  (delta_i is the cost increase if medoid i is removed)
  Re-solve master. Repeat.
```

### Implementation notes

- Warm-start the master with the PAM medoid set.
- The subproblem is pure computation (O(Nk) assignment), NOT an LP solve.
- Benders cuts can be added as lazy constraints in HiGHS (`kCallbackMipDefineLazyConstraints`)
  or Gurobi (`GRBCallback::addLazy()`), allowing the solver to add them during B&B.
- Alternative: iterative cut-and-solve loop if lazy callback integration is complex.
- Expected: converges in 5-20 iterations for typical clustering instances.

### HiGHS lazy constraint callback skeleton

```cpp
// Register callback:
highs.setCallback(benders_callback, &callback_data);
highs.setOptionValue("mip_allow_restart", false);  // Required for lazy constraints

// Callback function:
HighsCallbackFunctionType benders_callback = [](int type, const char*, const HighsCallbackDataOut* data_out,
                                                  HighsCallbackDataIn* data_in, void* user_data) {
  if (type != kCallbackMipDefineLazyConstraints) return;

  auto* ctx = static_cast<BendersContext*>(user_data);
  const double* y = data_out->mip_solution;  // Current integer solution

  // Solve subproblem: assign each point to nearest open medoid
  double actual_cost = 0.0;
  // ... compute cost, generate cut if violated ...

  // Add cut:
  data_in->user_cut_indices = ...;
  data_in->user_cut_values = ...;
  data_in->user_cut_rhs = ...;
};
```

### Gurobi callback skeleton

```cpp
class BendersCallback : public GRBCallback {
  // ...
protected:
  void callback() override {
    if (where == GRB_CB_MIPSOL) {
      // Get current integer solution
      std::vector<double> y(N);
      for (size_t i = 0; i < N; ++i)
        y[i] = getSolution(y_vars[i]);

      // Solve subproblem, add lazy cut if violated
      double actual_cost = compute_assignment_cost(y, D, N, k);
      if (actual_cost > theta_val + eps) {
        GRBLinExpr cut = /* ... */;
        addLazy(cut >= /* rhs */);
      }
    }
  }
};
```

### Decision: when to use Benders

| N | Strategy |
|---|----------|
| N <= 200 | Compact MIP (current, with warm start + tuning from Tasks 1-2) |
| N > 200 | Benders decomposition |

The threshold can be made configurable or determined empirically.

### Tests

- [ ] Benders produces the same optimal objective as compact MIP on small instances (N=20, 50).
- [ ] Benders solves N=500 within 60 seconds (compact MIP may timeout).
- [ ] PAM warm start is used in the master.
- [ ] Iteration count is logged.
- [ ] Solution is feasible and matches expected assignment.

---

## Task 4: Odd-Cycle Cutting Planes (Optional, Advanced)

**Impact: MEDIUM.  Effort: HIGH (~150-200 lines).**

The UNIMODULAR.md analysis proves that LP fractionality arises from odd cycles among
facilities. {0, 1/2}-Chvatal-Gomory cuts (Caprara & Fischetti, 1996) directly target
these odd cycles.

### When to use

This is most useful in the compact MIP (Task 2), not Benders (Task 3). If Benders
handles large N, cutting planes help for the medium-N regime (50 < N < 200) where
compact MIP is used but the LP is fractional.

### Separation algorithm

1. Solve LP relaxation.
2. Build auxiliary graph: nodes = facilities, edge (i,j) with weight based on
   fractionality of A[i,j].
3. Find minimum-weight odd cycle (shortest-path in a bipartite expansion of the graph).
4. Add the odd-cycle inequality:
   `sum_{(i,j) in C} A[i,j] <= (|C| - 1) / 2`
5. Re-solve LP with the cut.

### Implementation approach

- Add as user cuts in the solver callback (same framework as Benders lazy cuts).
- Alternatively, add as a cut-and-solve loop before falling back to B&B.
- Polynomial-time separation via shortest odd-cycle detection (Grötschel et al.).

### Decision: defer until Tasks 1-3 are done

This is the most complex improvement and has diminishing returns if Benders handles
large N. Implement only if medium-N instances (50-200) show persistent fractionality
that slows compact MIP despite Tasks 1-2.

### Tests

- [ ] Cuts are valid (do not exclude any integer solution).
- [ ] LP bound improves after adding cuts.
- [ ] No regression on instances that are already LP-integral.

---

## Execution Order

```
Task 1: MIP warm start from FastPAM
  → immediate benefit, no architectural risk
  → prerequisite for all other tasks (warm start should always be on)

Task 2: Gurobi parameter tuning + branching priority
  → quick win, ~5 lines
  → can be done in parallel with Task 1

Task 3: Benders decomposition
  → the big-N story, unlocks N > 200
  → depends on Task 1 (warm start for master)
  → most implementation effort

Task 4: Odd-cycle cuts (optional)
  → only if medium-N compact MIP is still slow after Tasks 1-2
  → most theoretical complexity
```

---

## Verification Plan

### Correctness

- All existing MIP tests pass unchanged.
- Warm-started MIP gives same optimal objective as cold-start (within tolerance).
- Benders gives same optimal objective as compact MIP on shared test instances.
- Solutions are feasible: k medoids, every point assigned to exactly one open medoid.

### Performance

Compare solve time (wall clock) on:

| Instance | N | k | Expected improvement |
|----------|---|---|---------------------|
| Small | 20 | 3 | Warm start: 2-5x faster (mostly B&B pruning) |
| Medium | 100 | 5 | Warm start + branching: 5-20x faster |
| Large | 500 | 10 | Benders: feasible (compact MIP likely timeouts) |
| Very large | 2000 | 20 | Benders: the only viable exact path |

Report: solve time, iteration count (Benders), LP bound, final objective, gap.

---

## References

- UNIMODULAR.md Sections 5-6 (solving strategies, implementation details, code snippets)
- Schubert & Rousseeuw (2021) — FastPAM used for warm start
- Duran-Mateluna, Ales, & Elloumi (2023) — Benders for p-median at 238K+ points
- Caprara & Fischetti (1996) — {0, 1/2}-CG cuts for half-integral LP solutions
- Baiou & Barahona (2009, 2011) — odd-cycle characterization of LP fractionality
