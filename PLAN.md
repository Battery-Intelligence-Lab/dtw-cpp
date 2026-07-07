# DTWC++ 2.0 — Deep Refactor & Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status: DRAFT v0.1 — Phase 0 is final; Phases 1–7 are being filled in from research reports (`.claude/reports/*-2026-07-06.md`) as they land. Do not start Phase ≥1 tasks until this header says FINAL for that phase.**

**Goal:** Ship DTWC++ 2.0: a top-down redesigned, cross-language-consistent (C++/Python/MATLAB), device-selectable (cpu/gpu/hpc), zero-overhead, cross-platform DTW clustering library that is the fastest available for large-N workloads — with all 2026-06-01 audit criticals fixed, automated packaging (wheels + MEX + executables) for all platforms, and an improved exact solver.

**Architecture:** Layered redesign around a single environment/session object with PyTorch-style device selection; compile-time cost-function injection in C++ with pre-compiled variant dispatch in bindings; SLURM HPC backend behind the same API via `.env` credentials; correctness fixes land first so every later phase builds on trusted numerics.

**Tech Stack:** C++20 (C++23 preferred where toolchains allow), CMake ≥3.26, **nanobind** (existing binding — `python/_dtwcpp_core.cpp`, STABLE_ABI Py≥3.12), MATLAB MEX, OpenMP (+ CUDA, Metal, MPI, HiGHS/Gurobi optional), scikit-build-core + cibuildwheel, Hugo docs.

## Global Constraints (apply to every task)

- Core builds with NO optional deps (OpenMP, HiGHS, CUDA, Metal, MPI all optional). Never make an optional dep required.
- No runtime dependence on repo-relative paths.
- Every task: tests added/updated, `CHANGELOG.md` (Unreleased) updated, lint clean.
- C++20 minimum; no naked `new`/`delete` in core; buffer > thread_local >> heap allocation in hot paths.
- No over-engineering: minimal edits, every changed line traces to the plan.
- Python tooling via `uv` only, never pip.
- No silent fallbacks anywhere: a requested capability (parallelism, GPU, metric, device) that cannot be delivered must error or warn loudly — never quietly degrade. This is a 2.0 design principle (user instruction, TODO.md items 3, 6, and High-bug list).
- Performance regressions gate merges: benchmark suite must not regress >2% on the recorded baseline (baseline recorded in Phase 0 Task 0.0 below). **Caveat (user, 2026-07-06): this machine runs multiple workloads in parallel — local wall-clock benchmarks are ADVISORY ONLY (record numbers + "machine under load" flag, never pass/fail on them). Hard perf gates run only on a quiet dedicated machine or CI. The 31.2 ms anchor itself carries this caveat.**
- Data files are read-only; never write outside the git project root.
- Registered pass/fail bands go into the test/bench script BEFORE the run; verdicts printed against them.

## Phase structure & dependency graph

```text
Phase 0  Correctness triage (audit criticals + highs)     — no dependencies; START NOW
Phase 1  Core C++ API redesign (Env/device/lazy-load)     — after Phase 0
Phase 2  Cross-language parity (Python, MATLAB, CLI)      — after Phase 1 API freeze
Phase 3  Parallelism & GPU out-of-the-box + introspection — overlaps Phase 1 (backend), finishes after Phase 2 (binding surface)
Phase 4  Solver upgrade (LP/network/branching)            — independent math track; integrates after Phase 1
Phase 5  Speed: SIMD, memory layout, new kernels, LB cascade — after Phase 0 baseline; informed by literature report
Phase 6  Packaging & release automation (wheels/MEX/exe)  — after Phase 2; PyPI publish LAST, gated on full green
Phase 7  Documentation website                            — after Phases 1–6 API stabilises
```

Parallelisable: Phase 0 tasks are mutually independent (per-file). Phase 4 research/design runs alongside Phases 1–3. Phase 5 kernel work independent of Phase 2.

## Execution protocol (orchestrator contract)

- Implementation agents: **Opus 4.8, effort=xhigh**, one agent per task, fresh context each. **Max 4 concurrent agents** (user rule 2026-07-07 — 5-hour session limit; run fan-outs as sequential batches of ≤4, and carry completed batches' results forward as constants so interrupted runs never redo finished work).
- Every agent brief must include: exact task text from this file, the Global Constraints block, and pointers to `.claude/summaries/handoff-2026-06-01-adversarial-audit.md` (for Phase 0) or the relevant `.claude/reports/*.md`.
- Each task ends: build + run the named test gate + report delta vs recorded baseline. A task that cannot meet its gate reports FAILED with numbers — no silent partial completion.
- After each phase: one adversarial review agent checks the phase's diff against this plan; findings triaged before next phase starts.
- Orchestrator updates the checkboxes and the per-phase status lines in this file after verifying each agent's evidence.

---

## Phase 0 — Correctness triage [CLOSED 2026-07-07]

**Status: CLOSED.** 13 task commits (`a992183..4052699`) + 7 remediation commits (`ffb7a8d..76cd5bf`, R1–R7). Remediation gate PASS: rebuild exit 0; ctest verbatim "100% tests passed, 0 tests failed out of 76" (76/0/6; ≥ prior 68-pass floor; skips unchanged: CUDA×2, Metal×3, IO×1 — documented env skips). Blocker #1 CLOSED: live `detail::dispatch_mv_metric` has real `case L2: return fn(MVL2Dist{})` (warping.hpp:264), test drives public `dtwBanded_mv`/`dtwFull_L_mv`, run verbatim "All tests passed (6 assertions in 1 test case)". LLFIO=OFF configure re-run by gate: exit 0. MEX runtime verify (R6, MATLAB R2024b `-batch`): verbatim "19 Passed, 0 Failed, 0 Incomplete", MATLAB_EXIT=0, no crash, `exist('dtwc_mex')==3` (live MEX, not skip branch) — guards all correct, no dtwc_mex.cpp change needed. Remediation wave findings (all closed):

- R1 (BLOCKER): task 0.6 L2 fix landed on dead `core::dispatch_mv_metric` (zero call sites); live `detail::dispatch_mv_metric` (warping.hpp:240) still L2→L1; CHANGELOG claim false.
- R2: Metal KVN/pruning `num_pairs` int32 sites remained (metal_dtw.mm:440,460,551,911,951,1481,1944 — plan's 0.2 file list dropped the audit's KVN site); MSL decode untested.
- R3: llfio still hard-required (`mip/CMakeLists.txt:20` unconditional link) despite new option.
- R4: never-scoped audit Highs — scores `Nc<2` guard, algos `int N` truncation, 0.14 local-dispatch spy tests for mip/kmedoids/hierarchical.
- R5: never-scoped MPI bundle (atomic_min comment, MPI_IN_PLACE, `int(N*N)` overflow; partition imbalance deferred to Phase 5 as perf).
- R6: MEX validation test executed for REAL in local MATLAB R2024b (never run before).
- R7: stale `nn_dist` comment in `core/pruned_distance_matrix.cpp:409-412` (R5 diagnosed: code correct — both `nn_dist[i]`/`nn_dist[j]` updated via atomic CAS by design per :446-449; comment claims i-only).

**OPEN residuals surfaced by remediation (not closed by R1–R7):**

- Metal NxN main-path `pair_offset` still int32 (`constant int& pair_offset [[buffer(8)]]` in dtw_wavefront/:84, dtw_wavefront_global/:204, dtw_banded_row/:329, dtw_regtile_w4/:820, dtw_regtile_w8/:838, regtile body :759; host cast :1666 + setBytes :1686). Unpruned NxN with N≳65536 wraps negative → decode garbage/OOB. Same Critical class as 0.2; mechanical widen (5 signatures + 1 param + 2 host lines) but touches 0.2's converted main-kernel ABI — own follow-up task. Also pre-existing by 0.2's design: int32 `pair_indices`/`active_pairs` buffer chain caps pruning survivor indices at 2^31.
- `DTWC_ENABLE_LLFIO=OFF` full BUILD still fails: `Problem.hpp:24`, `core/mmap_distance_matrix.hpp:43`, `core/mmap_data_store.hpp:45` include `<llfio/...>` unconditionally. R3 closed the mip link/define level (configure+generate exit 0 verified); header guards on `DTWC_HAS_MMAP` belong to the core/header owner → fold into Phase 1 (Task 1.5 type/layout pass or a dedicated task).
- MPI triangular-partition load imbalance: perf-only, TODO comment in `mpi_distance_matrix.cpp:93-101` → Phase 5.
- Benign relaxed read of `nn_dist[j]` racing the CAS write (`pruned_distance_matrix.cpp:413/:258`) — documented tradeoff, awareness only.

Verified-closed by review: 0.3, 0.4(code), 0.5(HiGHS), 0.6-SoftDTW, 0.7, 0.8, 0.9, 0.10, 0.11, 0.13, 0.14(validation half). Deferred by design: `default_data_t` → Task 1.5. GPU-runtime verification (0.1 CUDA, 0.2 Metal) needs H100/macOS CI — inspection-only locally, correct by reading per reviewer.

### Original task specs (kept for reference)

All items are CONFIRMED findings from the 2026-06-01 60-agent audit, re-verified against source 2026-07-06 (`.claude/TODO.md`). Full proposed patches: `.claude/summaries/handoff-2026-06-01-adversarial-audit.md`. Each task is independent; run in parallel. Every fix needs a regression test that FAILS before the fix and PASSES after (record both runs).

### Task 0.0: Record the performance + test baseline

**Files:** Create: `.claude/baselines/2026-07-06-phase0.md`
- [ ] Build current `main`-equivalent state (branch `Claude`) with default options; run full test suite; record pass/fail counts and names of any failing tests verbatim.
- [ ] Run the existing benchmark suite (see `benchmarks/`) on 3 representative UCR datasets; record wall-times.
- [ ] Commit the baseline file. All later "no regression" claims diff against THIS artifact.

### Task 0.1: CUDA wavefront drops anti-diagonal cells for max_L > 2048

**Files:** Modify: `cuda/cuda_dtw.cu:280` region. Test: CUDA test dir.
- Root cause: `MAX_SI=8` × blockDim 256 = 2048-cell cap; longer anti-diagonals silently truncated → wrong DTW. 3-buffer path is correct.
- [ ] Write failing test: series length 4096, compare wavefront kernel vs CPU oracle (non-degenerate series — random walk, NOT constant/symmetric); registered band: max |Δ| ≤ 1e-9 (f64).
- [ ] Fix: grid-stride loop over anti-diagonal cells (or route max_L>2048 to the 3-buffer path with a loud dispatch log). Keep occupancy; no shared-mem overflow.
- [ ] Verify test passes on GPU runner; commit.

### Task 0.2: Metal decode_pair FP32 sqrt + int32 num_pairs overflow

**Files:** Modify: `metal/metal_dtw.mm:57` (decode), `:102,220,341,761` (num_pairs).
- [ ] Failing test: pair decode for N=8192 (fp32 sqrt wrong) and N=50000 (int32 overflow) — pure host-side decode test, no GPU needed.
- [ ] Fix: FP64 decode on host (match MPI implementation — that copy is the correct one per audit), `size_t`/`int64_t` num_pairs.
- [ ] SSOT: extract ONE `decode_pair` into a shared header used by CUDA/Metal/MPI (audit found 3 divergent copies). All three call sites updated; digit-identical decode test across all three.
- [ ] Commit.

### Task 0.3: mmap overflow + offset validation

**Files:** Modify: `dtwc/mmap_distance_matrix.hpp:120`, `dtwc/mmap_data_store.hpp:229` (paths per audit; confirm on read).
- [ ] Failing tests: (a) crafted `n` where `n*(n+1)/2` wraps → must throw, not pass truncation check; (b) file with interior offset pointing OOB → must throw.
- [ ] Fix: checked multiplication (`__builtin_mul_overflow` / manual guard), validate every offset monotone + in-bounds at load.
- [ ] Commit.

### Task 0.4: MEX input validation

**Files:** Modify: `bindings/matlab/dtwc_mex.cpp`.
- [ ] Add `mxIsDouble`/`mxIsComplex`/dimension guards on every entry point; error via `mexErrMsgIdAndTxt`, never NULL-deref.
- [ ] Test: MATLAB-side (or mock) test feeding int32/complex/empty arrays → clean error message, no crash.
- [ ] Commit.

### Task 0.5: MIP status handling — assert() → real error path

**Files:** Modify: `dtwc/mip/mip_Highs.cpp:199`, `dtwc/mip/mip_Gurobi.cpp:120`.
- [ ] Failing test: infeasible/limited solve (e.g. iteration limit 0) → must throw `dtwc::SolverError` (or existing error type) with solver status text, never return empty centroids.
- [ ] Fix: replace `assert` with status check + throw; Gurobi catch path same treatment.
- [ ] Commit.

### Task 0.6: CPU dispatch — SoftDTW fallthrough + L2-computes-L1

**Files:** Modify: `dtwc/core/dtw.cpp:56`, `dtwc/core/dtw_cost.hpp:83,92`.
- OPEN question resolved for the plan: `MetricType::L2` gets a true L2 implementation (elementwise sqrt of squared sums for multivariate; |diff| is already correct for scalar univariate — document that univariate L1≡L2 pointwise, so the fix is multivariate-only + an explicit `case`). If maintainer later declares alias intent, the explicit case still stands.
- [ ] Failing tests: (a) SoftDTW variant request on CPU path → must either compute SoftDTW or throw NotImplemented — never silently return Standard-L1; gamma≤0 → throw. (b) Multivariate L2 metric on a hand-computed 2×3 example (register exact expected value in the test BEFORE implementing).
- [ ] Fix dispatchers; delete dead `core::dispatch_metric` (audit: zero call sites) in the same commit.
- [ ] Commit.

### Task 0.7: CUDA int32 index overflow N > 46341

**Files:** Modify: `cuda/cuda_dtw.cu:202` (`result_matrix[si*N+sj]`), `:73` (`row_start`).
- [ ] Fix: `size_t`/`long long` indexing (adjacent-series math already `long long`). Test: compile-time static_assert on index type + (if GPU with enough RAM unavailable) a host-side index-arithmetic unit test reproducing the wrap at N=46342.
- [ ] Commit.

### Task 0.8: I/O reader hardening (Arrow IPC + Parquet)

**Files:** Modify: `dtwc/io/arrow_ipc_reader.hpp`, `dtwc/io/parquet_reader.hpp`.
- [ ] Failing tests: Float32 file read as `DoubleArray` → must convert or throw with type name; list offsets OOB → throw; `ndim=0` → throw (currently div-by-zero).
- [ ] Fix: explicit type check + cast path, offset bounds validation, ndim≥1 guard.
- [ ] Commit.

### Task 0.9: CLI argument handling

**Files:** Modify: CLI main (locate `--metric`, `--device` parsing).
- [ ] Failing tests: `--metric` on CPU path must take effect (currently CUDA-only) or error "unsupported on cpu"; `cuda:abc` → clean error not `std::terminate`; `--device CUDA:0` case-insensitive; unknown device → error, NOT silent CPU fallback.
- [ ] Fix parser; commit.

### Task 0.10: TimeSeries::view() drops ndim

**Files:** Modify: `dtwc/core/time_series.hpp:66`.
- [ ] Failing test: multivariate series (ndim=3) → view → round-trip → ndim must survive.
- [ ] Fix: carry ndim in the view constructor. Commit.

### Task 0.11: fast_clara seed + serial-assign fixes

**Files:** Modify: `dtwc/algorithms/fast_clara.cpp` (locate exact lines on read).
- [ ] Fix: single RNG type (`mt19937_64`) both paths — registered test: same seed → identical medoids RAM vs chunked on a 200-series synthetic set; add OpenMP to in-RAM assign loop.
- [ ] Commit.

### Task 0.12: Build supply-chain pinning

**Files:** Modify: `cmake/Dependencies.cmake`, CI workflow with codecov step.
- [ ] Pin `llfio` to a specific SHA (pick current develop HEAD, record it + date in a comment; flag OPEN for maintainer to bless); make it optional not REQUIRED (audit: violates optional-deps rule) — core must configure without llfio.
- [ ] Add `URL_HASH` to every CPM URL tarball; pin quickcpplib clone to a SHA.
- [ ] Replace codecov bash-uploader `curl <()` with the official codecov-action pinned by SHA.
- [ ] Gate: fresh configure+build on a clean tree with network, then repeat with `CPM_SOURCE_CACHE` offline → identical. Commit.

### Task 0.14: Python `_api.cluster()` ignores `method` — always FastPAM

**Files:** Modify: `python/dtwcpp/_api.py:154-166`.
- [ ] Failing test: `cluster(..., method="mip")` (and each other documented method) must dispatch to that method or raise `ValueError("unknown method: ...")` — currently every value silently runs FastPAM locally.
- [ ] Fix dispatch; commit.

### Task 0.13: Dead-code removals

**Files:** Delete/modify: `dtwc/types/types_util.hpp` (`is_integer`/`is_zero`/`is_one`), README/benchmark references to `DTWC_ENABLE_SIMD` (unbuildable — Phase 5 will re-introduce properly).
- [ ] Grep-verify zero call sites before each removal; build + full test suite green after. Commit.

### Phase 0 gate (all tasks)

- [ ] Full test suite: every pre-existing pass still passes; every new test passes; report delta vs Task 0.0 baseline ("N passing → M: +list").
- [ ] Benchmarks within 2% of baseline.
- [ ] Adversarial review agent re-reads the audit handoff and confirms each Critical/High is closed with a test, or lists what remains.
- [ ] CHANGELOG.md Unreleased updated with one line per fix.

---

## Phase 1 — Core C++ API redesign [FINAL pending 1.1 design review]

Basis: `.claude/reports/api-surface-2026-07-06.md`. Current shape: `DataLoader` (CSV/TSV builder) → `Data` (4 modes, public `p_vec` wrong in 3 of them) → `Problem` god-object (data + distMat variant + naked public config fields + clustering state + CSV writers) ← free algorithms returning `ClusteringResult`; results NOT written back into Problem in C++ (but auto-wired in Python/MATLAB — behavioral divergence).

**Design decisions (fixed now; 1.1 may refine details, not reverse these):**

- **Two-tier API, identical in all three languages (CasADi model).**
  - *Tier 1 (high-level):* `dtwc.device(name)` global setter, `dtwc.load(path_or_array, ...)`, `dtwc.cluster(data, k, method, ...) -> Result`, `Result` with `labels`, `medoids`, `score(name)`, `save(dir)`, `plot()` (plot: Python/MATLAB only; C++ writes plottable CSV). Python's existing 2026-06-30 `device()/load()/cluster()` flow is the seed — C++/MATLAB conform to it.
  - *Tier 2 (advanced):* `Problem` retained, cleaned: snake_case methods, PascalCase classes everywhere; config via setters that keep invariants (no naked public fields that de-sync — e.g. `variant_params` write must rebind `dtw_fn_`); algorithms ALWAYS write results back into Problem (kills divergence #4).
- **One name per concept.** Canonical: `set_n_clusters`. All duplicates (`set_numberOfClusters`, `set_number_of_clusters`) become deprecated shims in C++ (`[[deprecated]]`), removed from bindings entirely (2.0 is the break point). Full rename table produced in Task 1.1 (covers `maxIter`→`max_iter`, `N_repetition`→`n_repetitions`, `scores::daviesBouldinIndex`→`scores::davies_bouldin`, etc.).
- **One precision story.** `data_t = double` is THE default everywhere: template default `T=float` (settings.hpp:29) → `double`; CLI `--dtype` default float32 (dtwc_cl.cpp:148) → float64; `storage.hpp:21` comment fixed. Float32 stays as explicit opt-in (`Precision::Float32`, `--dtype f32`). Distances are ALWAYS accumulated in double regardless of storage (existing contract, now documented).
- **Env/device core.** `dtwc::Env` owns device (`cpu`/`gpu`/`hpc`), precision, thread policy; language-level `device()` setters delegate to it. `hpc` reads SLURM credentials from `.env`: missing file / missing key / auth failure each produce a specific actionable message (name the missing key, print an example `.env` block, name the host it tried) then clean error return — never a stack trace, never silent local fallback.
- **Lazy loading & big-data policy.** `device=hpc`: metadata-only local load (shapes/counts/names), bulk data streamed to cluster at submit. Local devices: `StoragePolicy` auto mode — mmap-backed store when estimated footprint > threshold (default 50% free RAM, overridable). Existing view-mode spans (48× CLARA win) preserved.
- **Zero-overhead cost injection.** Metric/cost = template parameter via the existing `detail` functor pattern; every shipped variant explicitly instantiated once in the library; runtime selection = ONE switch at the call boundary, never in the hot loop. New user cost functions: C++ header template injection (zero cost); bindings select among precompiled instantiations.
- **Error taxonomy.** `dtwc::Error` base + `InvalidInput`, `SolverError`, `DeviceError`, `IOError`. No `assert`-as-validation, no `exit()` in library code. Bindings translate to native exceptions/`mexErrMsgIdAndTxt`.

### Task 1.1: API contract document (freeze artifact)

**Files:** Create: `docs/api-contract-2.0.md`. Consumes: api-surface report. Produces: exact class/method/param/default table for every public symbol in all 3 languages + full 1.x→2.0 rename table + deprecation list.
- [ ] Write contract implementing the decisions above; every Tier-1/Tier-2 signature spelled out per language (C++/Python/MATLAB columns must be textually alignable).
- [ ] Adversarial review agent checks contract against: the 5 load-bearing constraints (report §constraints — labels-CSV contract, CLI/TOML de-facto API, input formats, checkpoint triple, perf/precision contracts), every TODO.md backlog item, and the Global Constraints. Findings fixed before freeze.
- [ ] Mark contract FROZEN in its header; Phase 2 implements against it verbatim. Commit.

### Task 1.2: Error taxonomy

**Files:** Create: `dtwc/error.hpp`. Modify: every `assert`-as-validation / `exit()` site in dtwc/ (enumerate by grep in-task).
- [ ] Tests: each error type constructible + message content; MIP status path (Task 0.5) migrates onto `SolverError`.
- [ ] Commit.

### Task 1.3: `dtwc::Env` + device registry

**Files:** Create: `dtwc/env.hpp`, `dtwc/env.cpp`. Modify: `dtwc/settings.hpp`, CLI wiring. Consumes: 1.2 errors. Produces: `Env::set_device(std::string_view)`, `Env::device() -> Device`, `Env::threads()`, singleton `dtwc::env()`.
- [ ] Tests: unknown device name → `DeviceError` listing valid names; `gpu` on non-GPU build → `DeviceError` with build-flag hint (NOT silent cpu); `.env` errors per the design decision (3 cases: no file, missing key, bad host) — each message asserted verbatim in test.
- [ ] Commit.

### Task 1.4: Storage policy + lazy load

**Files:** Modify: `dtwc/Data.hpp`, `dtwc/storage.hpp`, loaders. Produces: `StoragePolicy::Auto` + threshold config; hpc metadata-only load path.
- [ ] Tests: synthetic large-N estimate triggers mmap route (threshold injected low); hpc device load touches only headers (instrument reader call counts); round-trip equality mmap vs heap on a real small dataset (digit-identical).
- [ ] Commit.

### Task 1.5: Precision unification (`data_t`/`--dtype`/template defaults → double)

**Files:** Modify: `dtwc/settings.hpp:29`, `dtwc/dtwc_cl.cpp:148`, `dtwc/storage.hpp:21` comment, any `default_data_t` user.
- [ ] Registered check BEFORE change: run one banded-DTW test at f32 and f64, record both values; after change, f64 result must be digit-identical to the pre-change f64 run (proves default flip, not numeric change).
- [ ] Commit.

### Task 1.6: Problem cleanup + rename shims + result write-back

**Files:** Modify: `dtwc/Problem.hpp/.cpp`, `dtwc/scores.hpp`, algorithm entry points.
- [ ] Apply rename table from 1.1 with `[[deprecated("use set_n_clusters")]]` shims; setters guard invariants (`variant_params` rebinds `dtw_fn_`); `fast_pam/fast_clara/clarans/hierarchical` write labels/medoids/k back into Problem (matching what Python/MATLAB wrappers do today — then delete the wrapper-side auto-wiring in Phase 2).
- [ ] Tests: deprecated shim still works + emits warning; `silhouette(prob)` works in pure C++ after `fast_pam` without manual wiring; `variant_params` write → `dtw_fn_` rebound (behavioral test on a known distance).
- [ ] Full suite + bench vs baseline (Global Constraint ≤2%). Commit.

## Phase 2 — Cross-language parity [FINAL pending 1.1 contract]

Implements `docs/api-contract-2.0.md` verbatim. Known gaps to close (api-surface report top-10):

### Task 2.1: Python parity

**Files:** Modify: `python/src/_dtwcpp_core.cpp` (nanobind), `python/dtwcpp/_api.py`, `python/dtwcpp/__init__.py`.
- [ ] Bind missing: `Problem.set_solver`, `MIPSettings.benders`/`max_benders_iter`, `output_folder`, `storage_policy`, `lb_strategy`, `cuda_settings`, `use_mmap_distance_matrix`, f32/view data modes, `ndim` (multivariate) in load path.
- [ ] Unify distance-matrix access per contract (one name, zero-copy where safe); fix mixed list-vs-ndarray arg types (all distance functions take ndarray zero-copy).
- [ ] Delete wrapper-side result auto-wiring (now in C++, Task 1.6). Add `Metric` param (MATLAB has it, Python lacks it).
- [ ] Parity test: enumerate contract symbols via introspection → assert all present with exact names/defaults.
- [ ] Commit.

### Task 2.2: MATLAB parity

**Files:** Modify: `bindings/matlab/dtwc_mex.cpp`, `bindings/matlab/*.m` (Problem.m, DTWClustering.m, +dtwc/ namespace as needed).
- [ ] Add: ragged input (cell arrays), series names, `ndim` multivariate, MIP surface (MIPSettings + set_solver + benders), `device` param, checkpointing controls, CUDA dispatch — the "MATLAB Phase 2" backlog folded in.
- [ ] Rename to contract names (snake_case methods; keep MATLAB 1-based conversion at MEX boundary only).
- [ ] Parity test: MATLAB script asserting contract symbols (run in CI when MEX CI lands in Phase 6; locally via user's MATLAB meanwhile — mark test skippable-with-loud-notice).
- [ ] Commit.

### Task 2.3: CLI conformance

**Files:** Modify: `dtwc/dtwc_cl.cpp`, TOML config schema.
- [ ] CLI flags/TOML keys renamed to contract vocabulary WITH old names accepted + deprecation warning (CLI is a de-facto API for `cluster_generic.slurm` + `_hpc.build_dtwc_command` — those two callers updated in the same commit).
- [ ] Commit.

### Task 2.4: Cross-language conformance fixture

**Files:** Create: `tests/conformance/` (TOML fixture + small recorded dataset + runner per language).
- [ ] One fixture: load recorded dataset → banded DTW → fast_pam k=3, fixed seed → labels + medoids + 3 scores. Run from C++, Python, MATLAB (MATLAB skippable, loud), CLI. Assert digit-identical labels/medoids and scores equal to 1e-12 rel.
- [ ] This fixture is the permanent parity gate — wire into CI. Commit.

## Phase 3 — Parallelism & GPU out-of-the-box [DRAFT — build-state report landed; tasks being finalised]

Current state (`.claude/reports/build-state-2026-07-06.md`): OpenMP is the ONLY backend (no std::execution/TBB anywhere). Known silent-fallback holes to close:
1. OpenMP-missing is only a configure-time `message(WARNING)` (`dtwc/CMakeLists.txt:118-123`) — build/wheel succeeds serial.
2. Runtime warning lives only in `get_max_threads()` (`parallelisation.hpp:33-47`); fast_pam/fast_clara/pruned parallel regions never call it.
3. **GPU→CPU fallbacks are `if (verbose)`-gated — silent by default** (`Problem.cpp:404,450,470,477,493`).
4. macOS-Intel cross-built wheels likely serial (arm64 runner installs arm64-only libomp) [inferred — verify one CI log].
5. MSVC OpenMP rides `project_options` INTERFACE (`/openmp:experimental`); consumers that drop `project_options` silently serialise (`dtwc/CMakeLists.txt:113`, `python/CMakeLists.txt:33-40`).

Plan:
- Keep OpenMP as the backend (only one wired everywhere; std::execution was the user's earlier pain point). Make it FAIL the build when absent unless `-DDTWC_ALLOW_SEQUENTIAL=ON`; runtime `omp_get_max_threads()==1` on multi-core hardware → loud warning at Env construction, not buried in one helper.
- Un-gate GPU→CPU fallback messages from `verbose` — always warn (no-silent-fallback global constraint).
- CI gate: built wheels must assert `OPENMP_AVAILABLE==True` on all platforms + `CIBW_TEST_COMMAND` import test.
- Introspection: building blocks already bound (`system_info`/`cuda_device_info`/`metal_available`/`openmp_max_threads` in `_dtwcpp_core.cpp:813-939`; `dtwcpp.check_system()`; MATLAB `dtwc_mex('system_check')`). Build `dtwc.test.parallelisation()` / `dtwc.test.gpu()` on top: structured pass/fail return (not prints), proof-of-engagement (threads actually spawn; GPU kernel actually executes and validates against CPU oracle), same output schema in all 3 languages. Add Metal to Python `check_system` (bound but not reported).
- CUDA H100 items from TODO backlog fold in here (arch-aware dispatch, k-vs-all kernel wiring, multi-stream).

## Phase 4 — Solver upgrade: "LR-core" [FINAL — from solver-math report]

Source: `.claude/reports/solver-math-2026-07-06.md` (full derivations; move to `docs/` in Phase 7 — `.claude/reports/` is gitignored). Math verdict, verified by independent re-derivation + enumeration:

- p-median matrix TU for N≤2 only; the TU substructure that matters: **Cardinality+Linking rows are TU for ALL N** (Ghouila-Houri proof in report) ⇒ Lagrangian dual of assignment-dualized problem equals the LP bound (Geoffrion) **without forming the N²-column LP**. All hardness lives in the k-subset choice of y (N binaries); x is an O(Nk) scan once y is fixed.
- FALSIFIED (registered bands, scipy/HiGHS vertex LPs + brute-force IP oracle): half-integrality of fractional vertices (§3.3); "LP integral 80–90%" as a polytope property — it is data-regime-dependent (clustered non-metric D: 250/250 integral; uniform D: 26% at N=20, max gap 13.7%). User's "almost unimodular" observation = his data lives in the integral regime.
- Previous attempts (2023, removed in `f7064b3`) all failed by solving the x-space LP explicitly (dense/sparse tableau simplex, Gomory, OSQP/ADMM). Live 2026 `benders.cpp` is directionally right but re-solves the master MIP per round; its "O(pk)" cut generation is actually O(N²).
- **P4 guard:** odd-cycle cuts alone cannot close instances (cardinality row breaks Baiou–Barahona) — the TODO "odd-cycle cutting planes" item is DEPRIORITISED; do not over-invest.

### Task 4.1: Lagrangian root solver

**Files:** Create: `dtwc/mip/lagrangian_root.{hpp,cpp}`. Consumes: distance matrix (packed/mmap), FastPAM UB (Task 5.1 or current fast_pam). Produces: `LagrangianRoot::solve(D, k) -> {lower_bound, best_ub, multipliers, gap}`.
- [ ] Inner problem per facility: `ρ_i = Σ_j min(0, D_ij − μ_j)`, pick k best; subgradient + Polyak steps off the FastPAM upper bound; primal repair each major iteration.
- [ ] Leading-order cost = T·N² memory traffic (one stream of D per iter). Registered bands (from report): **P1** — root certifies FastPAM optimal (gap ≤0.1%) on ≥90% of UCR datasets (FALSIFIED if <70%); **P3** — iteration time within 2× of bytes(D)/STREAM bandwidth; N=10⁴,k=20 exact ≤10 min single node.
- [ ] Oracle: brute-force IP on N≤14 (non-degenerate random D, NOT uniform/symmetric); match Gurobi/HiGHS optimum 1e-6 rel on N≤2000 where they prove optimality.

### Task 4.2: Reduced-cost fixing

**Files:** Create: `dtwc/mip/reduced_cost_fixing.{hpp,cpp}` (Beasley-style, driven by 4.1 multipliers).
- [ ] Registered band **P2**: when root gap ≤1%, fixing eliminates ≥80% of candidate medoids.
- [ ] Test: every fixed-out medoid verified absent from the brute-force optimum on N≤14 instances.

### Task 4.3: Core Benders with y-only branching

**Files:** Modify: `dtwc/mip/benders.cpp` + solver glue. Consumes: 4.1 bound + 4.2 core.
- [ ] One lazy-cut tree instead of re-solving master per round (HiGHS lazy-callback support: OPEN — investigate in-task; if absent, branch-and-bound loop owned by us with HiGHS LP nodes); branch on fractional y only; cut generation cost stated honestly (O(N²) today — reduce on the fixed core).
- [ ] Gate: matches Gurobi/HiGHS proven optima 1e-6 rel on ALL such instances; beats their wall-time at N≥2000 **or the phase is recorded FALSIFIED and the default path stays Gurobi/HiGHS** (LR root still ships as a bound/certificate tool either way).

### Task 4.4: Wire into API + `method="mip"` upgrade

- [ ] `Method::LRCore` (name per contract 1.1); `mip` keeps meaning solver-backed exact; docs state regime of validity (dense D in RAM/mmap; N·N doubles is the budget).
- [ ] CHANGELOG + LESSONS.md entry: killed ideas from 2023 attempts + falsified claims (keep killed ideas killed).

## Phase 5 — Speed & algorithms program [FINAL — ranked from literature report]

Source: `.claude/reports/literature-2026-07-06.md` (citations in `.claude/CITATIONS.md`). Every task: register the benchmark band in the bench script BEFORE the run; a candidate that misses its band is recorded FALSIFIED and dropped — no rescue-tuning past 2 attempts. DTW is memory-bound (0.125 FLOP/byte): memory layout before SIMD, always.

Priority order (impact ÷ effort):

### Task 5.1: FasterPAM + LAB init + FasterCLARA

**Files:** Modify: `dtwc/algorithms/fast_pam.cpp` (replace O(N²k) swap — audit High), `fast_clara.cpp`.
- [ ] Implement Schubert & Rousseeuw FasterPAM swap (O(1)-per-medoid update; *Information Systems* 101:101804, 2021 — verified 458×/1191× at k=100/200) on the existing cached/mmap matrix; LAB init; CLARA upgrade.
- [ ] Registered bands: (a) identical or better final objective vs current fast_pam on 10 UCR sets, fixed seeds; (b) swap-phase wall-time ≥10× faster at N=5000,k=50.
- [ ] Note: BanditPAM++ deliberately REJECTED — its win is avoiding distance computes; ours are O(1) cached [literature report].

### Task 5.2: Lower-bound cascade upgrade

**Files:** Modify: `dtwc/lb/` (or wherever Lemire envelope lives — locate in-task).
- [ ] Cascade: LB_KimFL O(1) → LB_Keogh (existing) → LB_Webb (always ≥ Keogh tightness; Webb & Petitjean PR 2021). LB_Enhanced for wide bands (Tan SDM 2019).
- [ ] Registered band: ≥25% fewer full DTW calls on matrix build, 5 UCR sets, band=10%; digit-identical distance matrix (exact LBs only).

### Task 5.3: TADPole-style matrix-build pruning

**Files:** Create: `dtwc/algorithms/pruned_matrix_build.{hpp,cpp}`.
- [ ] Reuse per-series envelopes + Euclidean UBs across all pairs (Begum KDD 2015); provably identical results clause is the test: labels digit-identical, ≥50% DTW calls pruned (paper claims ~94% — register 50% as pass floor).
- [ ] Explicit anti-goal: NO Elkan/triangle-inequality pruning — DTW is not a metric [literature report].

### Task 5.4: EAPruned kernel restructure

**Files:** Modify: core DTW kernels (`dtwc/core/`), reference MonashTS/tempo implementation.
- [ ] Fuse pruning + early-abandon per Herrmann & Webb DMKD 2021 (claims 2.88× vs classic EA). Registered band: ≥1.5× kernel-level on UCR mix, digit-identical distances.

### Task 5.5: MSM + TWE distances (quality lever — falsifies DTW-only)

**Files:** Create: `dtwc/core/msm.hpp`, `dtwc/core/twe.hpp`; wire into variant enum + bindings + contract.
- [ ] Two O(n²) DP kernels in the existing kernel framework (KAIS 2024: MSM best clustering distance, DTW ≈ Euclidean for k-medoids quality). Oracle: reference values from aeon on 20 series pairs (non-degenerate), 1e-10 rel.
- [ ] Bench: matrix-build time within 1.3× of DTW-same-band (same DP structure).

### Task 5.6: Multivariate independent DTW + TC-DTW bounds

**Files:** Modify: multivariate kernel dispatch; add channel-wise-sum mode alongside dependent mode.
- [ ] Independent-DTW (Shokoohi-Yekta DMKD 2017 — neither mode dominates, both needed); TC-DTW LB tightening if time (arXiv:2101.07731).
- [ ] Oracle: hand-computed 2-channel example + aeon cross-check.

### Task 5.7: Arrow C Data / PyCapsule zero-copy ingest

**Files:** Modify: Python load path (`_dtwcpp_core.cpp`), vendor nanoarrow (two files, no dep).
- [ ] Consume `__arrow_c_array__` from polars/DuckDB/pyarrow/pandas zero-copy. Test: polars large_list → cluster without pyarrow installed.

### Task 5.8: OneBatchPAM (100M-tier scaling)

**Files:** Create: `dtwc/algorithms/one_batch_pam.cpp`.
- [ ] O(mn) dissimilarities, m=O(log n) (AAAI 2025, arXiv:2501.19285). Registered band: objective within 5% of FasterPAM on N=50k synthetic while computing ≤10% of the matrix. If band missed → FALSIFIED, document, keep CLARA as the big-N path.

### Task 5.9: DBA + soft-DTW barycenter k-means (parity feature)

**Files:** Create: `dtwc/algorithms/barycenter.{hpp,cpp}`; k-means-DTW driver.
- [ ] SSG-preferred (Schultz & Jain 2018) + soft-DTW barycenter (Cuturi & Blondel 2017 — gradient already in-repo). Positioned as ecosystem parity, NOT a quality win (k-medoids beats barycentric k-means in the 2024 evaluation — say so in docs).

### Task 5.10: sklearn estimator wrapper + conda-forge (CPU)

**Files:** Create: `python/dtwcpp/sklearn.py`; conda-forge feedstock (separate repo, Phase 6 coordinates).
- [ ] `DTWCKMedoids(BaseEstimator, ClusterMixin)` with `__sklearn_tags__` (sklearn ≥1.6), `metric="precomputed"` both directions. Fills the dead scikit-learn-extra KMedoids vacuum [literature report].

### Task 5.11: Profiler pass + OpenMP schedule sweep + SIMD prototype (measure-first gate)

- [ ] Cache-stat profile (VTune/perf) of the two hot kernels on the H100 host or local; OpenMP `schedule(dynamic,1|16|guided)` sweep (TODO backlog); THEN decide Highway inter-pair SIMD prototype — no peer-reviewed CPU SIMD-DTW win exists [UNVERIFIED in literature]; prototype only if profile shows compute-bound sections. `DTWC_ENABLE_SIMD` references stay deleted (Task 0.13) unless this task lands a real implementation.

### Explicit rejections (killed ideas — do not reopen without overturning evidence)

- FastDTW: verified trap ("much slower than exact DTW" — Wu & Keogh TKDE 2022).
- BanditPAM/++: dominated by FasterPAM on precomputed matrices (see 5.1).
- Elkan/triangle pruning on DTW: invalid, DTW not a metric.
- ONNX export: no sensible story (custom ops kill portability). R/Julia bindings: deferred, native competitors saturate.

## Phase 6 — Packaging & release [DRAFT — build-state report landed; tasks being finalised]

Current state (`.claude/reports/build-state-2026-07-06.md`): wheel CI ALREADY EXISTS — cibuildwheel on 3 OS incl. macOS x86_64+arm64, sdist, **PyPI OIDC trusted publishing wired on `v*` tags** (`python-wheels.yml:74-103`). Gaps to close:

- cibuildwheel pinned at v2.21 → **cp314 silently skipped** — bump pin.
- **No `CIBW_TEST_COMMAND`** — wheels are never imported in CI. Add import + smoke-cluster test + `OPENMP_AVAILABLE==True` assert per platform.
- **No MEX CI at all** (zero matlab/mex hits in `.github/`); distribution today = committed `dtwc_mex.mexw64` binary. Add MEX build job per platform (needs `matlab-actions/setup-matlab` or header-only MEX shim build — decide in-task); resolves the tracked-binary question (Decision log).
- **No executable distribution** — only MEX + python module have install() rules. Add CPack/release workflow producing `dtwc_main`/`dtwc_cl`/`dtwc-convert` archives per platform on `v*` tags.
- Version SSOT: `VERSION` file vs `pyproject.toml:7` duplicated — single source.
- Wheels ship no MIP solver (HiGHS OFF in `pyproject.toml:60`) — decide: bundle HiGHS in wheels (size cost) or document CLI/C++ as the MIP path. Leaning bundle-HiGHS (it is the free solver; MIP-in-Python is a 2.0 headline feature). Gurobi stays external always.
- linux-aarch64 wheel: add if runner available, else document.
- CI push triggers: branch `Claude` runs almost nothing (only cuda-mpi-detect) — add `Claude` to unit-test workflow triggers for the duration of this refactor, remove before release.
- PyPI publish ONLY after Phases 0–5 gates green + end-to-end run per platform. Release: DTWC++ 2.0 (`v2.0.0` tag drives everything).
- `device="hpc"` end-to-end on Oxford ARC stays BLOCKED (needs a real cluster run — user action); release notes mark HPC as beta until verified.

## Phase 7 — Documentation [DRAFT]

- Hugo site: new API contract pages (all 3 languages), device-selection guide, `.env` HPC setup guide, data-conversion page (TODO backlog), Mermaid architecture diagram, migration guide 1.x → 2.0.

---

## Decision log

- 2026-07-06 `default_data_t` → `double` (audit High; OPEN question resolved in favour of precision; Float32 stays as explicit opt-in). Owner: Phase 1.
- 2026-07-06 `MetricType::L2` becomes a real L2 (multivariate); univariate unchanged (pointwise L1≡L2). Owner: Task 0.6.
- 2026-07-06 No silent fallback principle promoted to Global Constraint.
- OPEN: llfio pin SHA needs maintainer blessing (Task 0.12 picks a candidate, flags it).
- OPEN: MEX binary in git (`dtwc_mex.mexw64`) — user deferred 2026-07-06; revisit at Phase 6.
- 2026-07-06 (user): local benchmarks unreliable — machine shared with parallel workloads. All perf verdicts this session advisory; hard gates on quiet machine/CI only. Phase 0 gate adjusted mid-flight (bench demoted from FAIL-able band to advisory record).

## Progress log

- 2026-07-06: Plan v0.1 drafted. Phase 0 FINAL. Research agents dispatched: api-surface, build-state, solver-math (Fable max), literature.
- 2026-07-06: All 4 research reports landed (`.claude/reports/*-2026-07-06.md`). Phases 1–6 filled to FINAL (1.1 contract review + baseline numbers pending). Plan v0.9.
- 2026-07-06: Awaiting Task 0.0 baseline agent; Phase 0 fix wave (Opus 4.8 xhigh) launches immediately after — fixers must not run while baseline builds/tests.
- 2026-07-06: Baseline GREEN (65/0/5, clang 21.1.8; bench anchors ADVISORY — machine shared). Fix wave launched; bench demoted to advisory mid-flight per user (parallel workloads).
- 2026-07-07 (user): **MATLAB IS INSTALLED locally** (R2024b + R2025b, `C:\Program Files\MATLAB\`). MATLAB-side tests are runnable here via `matlab -batch` — Task 0.4's MEX validation test and Phase 2.2 parity tests get real local verification, not skip-only. Post-gate step added: build MEX + run the 0.4 test in MATLAB.
- 2026-07-07: Session limit killed 9/12 fixers + gate mid-wave. Completed & kept: 0.5 (MIP status→throw), 0.10 (ndim view — fix added `ndim` field + `timesteps()`, larger than planned one-liner, justified), 0.13 (dead code; `isFractional` found dead but out-of-scope, left in). Partial edits from dead agents reverted (`git checkout`); wave resumed from cache.
