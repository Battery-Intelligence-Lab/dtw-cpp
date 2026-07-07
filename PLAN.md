# DTWC++ 2.0 — Deep Refactor & Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status: v1.0 GRAND PLAN (2026-07-07, written by Fable for the Opus 4.8 orchestrator).**
> Phases 0–2 DONE. Phase 3 wave A DONE, wave B IN FLIGHT at handoff — resume/verify it FIRST (see §Orchestrator handoff).
> Phases 4–7 FINAL and executable as written. Read §Execution protocol + §Orchestrator handoff before dispatching any agent.

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

### Orchestrator handoff (Fable → Opus 4.8, 2026-07-07)

Fable wrote this plan and orchestrated Phases 0–3A. An Opus 4.8 orchestrator continues from here. Contract unchanged: orchestrator plans/verifies/commits docs; Opus-xhigh implementation agents write the code; PLAN.md is orchestrator-owned (agents never stage it).

**Task 3.6 DONE 2026-07-07 (commit `6df3c80`, Opus 4.8).** The RuntimeSingleThread silent-fallback hole (review H1) is closed: every compute path funnels through `get_max_threads()` → the shared process-once emitter `dtwc::warn_if_single_threaded()`; the high-level Python compute function calls it directly. One `std::call_once` guard shared with the Env ctor (CLI warns once, not twice). Duplicated `parallelisation.hpp` string removed → SSOT in env.cpp. Verified: (a) `test_runtime_loudness_compute` PASS; (b) Python subprocess H1 repro (`OMP_NUM_THREADS=1`, no `device()`) warns + pytest PASS; (c) live `dtwc_cl` warns EXACTLY once (count=1, SSOT-identical); (d) floors — ctest "0 failed out of 86" (80 non-skip), pytest 391, MATLAB 25/25 (19 input-validation + 6 test_api, serial-MEX parallelisation honesty preserved).

**FIRST ACTION — Phase 4 (LR-core solver).** Phase 3 fully closed. Start the sequential 4.1→4.4 chain (spec below). Wave B DONE (`ad34b6a`, `ecc522c`); nothing to resume.

**Test floors (regression bands — every later gate holds these or explains the delta with names). Re-recorded at Phase 3 close:**

- ctest (`build/baseline-2026-07-06`, clang 21.1.8 + Ninja + Release, Gurobi+HiGHS ON, CUDA/Metal OFF): "0 tests failed out of 86", 80 non-skip; 6 documented skips (test_cuda_correctness, test_cuda_lb_keogh, test_io_readers, test_metal_correctness, test_metal_lb_keogh, test_metal_mmap). (Was 85/79 before Task 3.6 added `test_runtime_loudness_compute`.)
- CUDA (`build/cuda-verify`, MSVC host, RTX 4000 Ada): `ctest -R cuda` → "0 tests failed out of 2" — these two suites now RUN here, never skip-only. Run log: `.claude/baselines/2026-07-07-cuda-first-runtime-verification.md`.
- pytest: "391 passed, 10 skipped" (fresh `.pyd` verified per the rebuild recipe below; was 390 before Task 3.6 added `test_single_thread_warning.py`).
- MATLAB `-batch` (gate used R2025b; R2024b also installed): validation 19/19, parity 20/20, test_api 6/6, conformance 1/1. **addpath ORDER matters** — add the fresh `build/mex-verify/bin` LAST so it prepends ahead of the stale in-git `bindings/matlab/dtwc_mex.mexw64` (wrong order → 0xc0000005 crash from the April binary; verify with `which('dtwc_mex','-all')` before running). `tests/matlab/test_dtwc.m` 12/14 is PRE-EXISTING (fix = Task 6.0c), not a gate failure. Serial MEX must report `parallelisation() pass=0` with a reason naming the sequential build — honest, not faked availability.
- Conformance (permanent parity gate): labels/medoids digit-identical across all 4 routes; silhouette 0.96894972764334841, DB 0.038333333333333337, dunn 11.5 vs `tests/conformance/conformance_reference.txt` (≤1e-12 rel).

**Workflow pattern (proven over 4 runs incl. 3 session-limit kills + clean resumes):** one Workflow-tool script per wave — impl agents (≤4 concurrent, disjoint file ownership, REPORT_SCHEMA `{taskId,status,filesTouched,testsAdded,changelogLines,notes}`) → one gate agent (GATE_SCHEMA `{verdict,testsPassed,testsFailed,testsSkipped,details}`; quotes decisive outputs VERBATIM; registered bands written into the script BEFORE launch) → repair loop ≤3 rounds on gate FAIL → one commit agent per task (conventional message + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer; never stages PLAN.md) → per phase, one adversarial review agent (REVIEW_SCHEMA `{verdict: CLEAN|FINDINGS, findings[{severity,file,claim,evidence}], notes}`) whose findings the orchestrator triages before the next phase. Prior wave scripts to crib from live in `C:\Users\engs2321\.claude\projects\C--D-git-dtw-cpp\b3179291-e4e2-417f-929b-4f8596c78042\workflows\scripts\`.

**Proven recipes / gotchas:**

- Build dirs: `build/baseline-2026-07-06` = canonical clang gate dir. `build/cfg-normal|cfg-noomp|cfg-seq` = 3.1 configure-matrix scratch. `build/mex-verify` = MEX, deliberately SERIAL (needs `-DDTWC_ALLOW_SEQUENTIAL=ON` since 3.1; MEX OpenMP decision = Task 6.2). `build/cuda-verify` = CUDA with MSVC host (nvcc 13.0 rejects clang 21; RTX 4000 Ada sm_89 is LOCAL — CUDA tests run live here, never inspection-only).
- Python extension rebuild (from-scratch pip wheel BLOCKED by MSB3491 → Task 6.5): configure `build/cfg-gate-normal` with `-DDTWC_BUILD_PYTHON=ON -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_HIGHS=OFF` + venv Python; build target `_dtwcpp_core`; copy fresh `.pyd` + `libomp.dll` into venv site-packages; verify import + one NEW symbol before pytest (stale-.pyd false-greens are real).
- `dtwc::Env` constructs only after CLI arg validation — runtime-loudness checks need a real `--input`; `--help` proves nothing.
- clangd diagnostics on mex.h / nanobind includes = stale-PCH IDE noise; the arbiter is always the gate's fresh full rebuild.
- Benchmarks ADVISORY ONLY on this machine (parallel workloads); hard perf verdicts need a quiet machine or CI. uv only, never pip. Max 4 concurrent agents (5-hr session limit). Never write outside the git root; data read-only. PyPI publish only on explicit user go.

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

## Phase 1 — Core C++ API redesign [DONE 2026-07-07]

**1.1 DONE:** `docs/api-contract-2.0.md` FROZEN (commit `c8c74bd`). Adversarial review round 1 FAIL (missing `set_method` row; MATLAB `cluster()`/`refresh_distance_matrix()` mislabeled live vs [new]) → fixed → re-review PASS.
**1.2 DONE:** `dtwc/error.hpp` taxonomy landed (commit `1f4d985`): soft_dtw.hpp assert→InvalidInput, MIP Gurobi/HiGHS status→SolverError (3 sites each). Gate PASS: ctest verbatim "100% tests passed, 0 tests failed out of 77" (71 ran+passed, 6 documented env skips; Release/NDEBUG = the config where old asserts were no-ops).
**Wave 2 DONE (1.5 → 1.3∥1.4∥1.6), commits `442676a` (1.5), `95b8d10` (1.3), `4778377` (1.4), `59bf4de` (1.6 + CHANGELOG).** Gate PASS: full rebuild exit 0 (165/165), ctest verbatim "100% tests passed, 0 tests failed out of 80" (74 ran+passed, +3 new suites test_env_device/test_problem_api_2_0/test_storage_policy, 6 documented skips, zero regressions). 1.5 registered digit-identity held: three f64 fingerprints identical to 20 digits pre/post default flip (82.15998622421159325, 127.22830559900998537, 49.45496556469289828; 240 assertions both runs). env.cpp no-silent-fallback read-verified (GPU-on-CPU throw at :233, .env failures throw before device change, HPC set only after full success). 13 Problem shims + 5 scores shims `[[deprecated]]` forwarding. Corrections vs plan text: CLI `--dtype` default was at dtwc_cl.cpp:226 (not :148); storage.hpp is `dtwc/core/storage.hpp`. Bench: advisory-only, skipped (machine under load) — run before release.

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

## Phase 2 — Cross-language parity [DONE 2026-07-07]

**Status (2026-07-07):** All four tasks done, committed `83e7668` (2.1 Python), `c4ba175` (2.2 MATLAB), `b6fe07f` (2.3 CLI/TOML), `0ff0e9b` (2.4 conformance fixture + CHANGELOG). Run `wf_d2e27165-0c8`.

- **Gate PASS** [confirmed — gate transcript + orchestrator re-run]: full rebuild "EXIT_BUILD=0"; decisive ctest verbatim "100% tests passed, 0 tests failed out of 81" (75 non-skip + 6 documented skips; one intermittent 0xc0000409 in unit_test_clustering_algorithms arbitered pre-existing/unchanged-by-Phase-2, passed 3/3 isolated + clean full re-run); pytest verbatim "387 passed, 10 skipped" against freshness-verified .pyd (mtime 11:50:52 > source 11:24:54, all 2.1 symbols present); MATLAB R2024b real runs: validation floor 19/19, new parity 20/20, conformance 1/1.
- **Conformance fixture (2.4)** — the permanent parity gate: `tests/conformance/` fixture, k=3 seed=29, all four routes (C++/Python/CLI/MATLAB) digit-identical labels/medoids, scores ≤1e-12 rel vs `conformance_reference.txt` (silhouette 0.96894972764334841, DB 0.038333333333333337, dunn 11.5). Orchestrator independently re-ran cpp_conformance + unit_test_cli_args post-workflow: both Passed.
- **CLI deprecation smoke**: `--clusters` → exit 0 + stderr "[dtwc] warning: '--clusters' is deprecated, use '--n-clusters' instead"; `--n-clusters` → silent. SSOT mapping in `cli_renames()`.
- **OPEN residuals surfaced by Phase 2:**
  - `tests/matlab/test_dtwc.m` 12/14 — test_clustering_medoids + test_clustering_fit_predict assume obsolete columns=series orientation; PRE-EXISTING (April MEX binary digit-identical 12/14, same two names). Own task: fix stale assertions to rows=series.
  - From-scratch wheel rebuild (`pip --no-build-isolation`) blocked by llfio→quickcpplib→outcome MSBuild superbuild "error MSB3491"; pre-existing env issue, clang/ninja core build green. Fold into Phase 6 packaging.
  - `bindings/matlab/dtwc_mex.mexw64` stale April binary in git shadows fresh MEX if addpath mis-ordered (caused real 0xc0000005 in gate harness until precedence fixed) — strengthens Phase 6 MEX-binary-in-git removal case.
  - unit_test_clustering_algorithms: repo-relative path dependency (".\data\dummy") + rare 0xc0000409 flake; separate owner.

Implements `docs/api-contract-2.0.md` verbatim. Known gaps to close (api-surface report top-10):

### Task 2.1: Python parity

**Files:** Modify: `python/src/_dtwcpp_core.cpp` (nanobind), `python/dtwcpp/_api.py`, `python/dtwcpp/__init__.py`.
- [x] Bind missing: `Problem.set_solver`, `MIPSettings.benders`/`max_benders_iter`, `output_folder`, `storage_policy`, `lb_strategy`, `cuda_settings`, `use_mmap_distance_matrix`, f32/view data modes, `ndim` (multivariate) in load path.
- [x] Unify distance-matrix access per contract (one name, zero-copy where safe); fix mixed list-vs-ndarray arg types (all distance functions take ndarray zero-copy).
- [x] Delete wrapper-side result auto-wiring (now in C++, Task 1.6). Add `Metric` param (MATLAB has it, Python lacks it).
- [x] Parity test: enumerate contract symbols via introspection → assert all present with exact names/defaults.
- [x] Commit.

### Task 2.2: MATLAB parity

**Files:** Modify: `bindings/matlab/dtwc_mex.cpp`, `bindings/matlab/*.m` (Problem.m, DTWClustering.m, +dtwc/ namespace as needed).
- [x] Add: ragged input (cell arrays), series names, `ndim` multivariate, MIP surface (MIPSettings + set_solver + benders), `device` param, checkpointing controls, CUDA dispatch — the "MATLAB Phase 2" backlog folded in.
- [x] Rename to contract names (snake_case methods; keep MATLAB 1-based conversion at MEX boundary only).
- [x] Parity test: MATLAB script asserting contract symbols (run in CI when MEX CI lands in Phase 6; locally via user's MATLAB meanwhile — mark test skippable-with-loud-notice).
- [x] Commit.

### Task 2.3: CLI conformance

**Files:** Modify: `dtwc/dtwc_cl.cpp`, TOML config schema.
- [x] CLI flags/TOML keys renamed to contract vocabulary WITH old names accepted + deprecation warning (CLI is a de-facto API for `cluster_generic.slurm` + `_hpc.build_dtwc_command` — those two callers updated in the same commit).
- [x] Commit.

### Task 2.4: Cross-language conformance fixture

**Files:** Create: `tests/conformance/` (TOML fixture + small recorded dataset + runner per language).
- [x] One fixture: load recorded dataset → banded DTW → fast_pam k=3, fixed seed → labels + medoids + 3 scores. Run from C++, Python, MATLAB (MATLAB skippable, loud), CLI. Assert digit-identical labels/medoids and scores equal to 1e-12 rel.
- [x] This fixture is the permanent parity gate — wire into CI. Commit.

## Phase 3 — Parallelism & GPU out-of-the-box [CLOSED 2026-07-07 — all tasks incl. Task 3.6 DONE]

**Wave A status (2026-07-07):** 3.1/3.2/3.4 done, committed `c2f8e99` (build: OpenMP FATAL_ERROR + opt-out), `3f4c755` (feat: Env sequential warning + always-on GPU fallback warnings), `65df555` (ci: CIBW wheel gate). Run `wf_d43062ea-0a6` (one session-limit kill; gate re-run on resume).

- **Gate PASS** [confirmed]: configure matrix re-run by gate — (a) exit 0 OpenMP found; (b) disable-OpenMP exit 1, error names `DTWC_ALLOW_SEQUENTIAL`; (c) opt-out exit 0 + loud warning. ctest verbatim "100% tests passed, 0 tests failed out of 84" (78 non-skip = 75 floor + 3 new; 6 documented skips; no clustering flake). pytest "387 passed, 10 skipped" on fresh clang-built .pyd. MATLAB floors 19/19, 20/20, 1/1 (test_dtwc 12/14 pre-existing, quoted).
- **Orchestrator live check** [confirmed]: `OMP_NUM_THREADS=1 dtwc_cl --input ... --device cpu` → stderr "[DTWC++ WARNING] OpenMP is available but only 1 thread is usable — DTWC++ is running SINGLE-THREADED." byte-identical to env.cpp SSOT.
- **Known-unverified**: MSVC `/openmp:experimental` direct-attach branch not exercisable locally (clang toolchain) — needs one MSVC CI run. 3.4's CIBW gate is NEEDS-CI-RUN (advisory until next push).
- **Residuals surfaced by wave A:**
  - `build/mex-verify` is deliberately SERIAL (OpenMP disabled at Phase 2 to avoid MATLAB/libomp clash; now requires the explicit `-DDTWC_ALLOW_SEQUENTIAL=ON` opt-out). MATLAB users currently get single-threaded DTW — loud (3.2 warning + DTWC_SEQUENTIAL_BUILD) but slow. Decide MEX OpenMP strategy (MATLAB-bundled libomp? iomp?) in Phase 6.
  - `dtwc_cl` "--input is required (via CLI or YAML config)" message says YAML where TOML is primary — cosmetic, fold into any later CLI touch (Task 6.0e).

**Wave B status (2026-07-07):** 3.3 + 3.5 done, committed `ad34b6a` (3.3 dtwc.test API), `ecc522c` (3.5 CUDA verification). Run `wf_abeeee64-25f`. Gate PASS [confirmed — gate re-ran every suite itself; orchestrator independently re-ran CUDA ctest + Python live check]:

- Baseline ctest verbatim "100% tests passed, 0 tests failed out of 85" (79 non-skip = 78 wave-A floor + test_test_api; 6 documented skips; no clustering flake).
- **CUDA first-ever runtime verification on the local RTX 4000 Ada** — `build/cuda-verify` `ctest -R cuda` "0 tests failed out of 2"; binaries verbatim "All tests passed (7312 assertions in 55 test cases)" + "All tests passed (688 assertions in 8 test cases)". Gate also built test_test_api CUDA-ON: `gpu available=1 backend=cuda device_name="NVIDIA RTX 4000 Ada Generation (compute 8.9, 20474 MB)" validated=1 pass=1` (real kernel vs FP64 CPU oracle). Run log artifact: `.claude/baselines/2026-07-07-cuda-first-runtime-verification.md`.
- Python: pytest "390 passed, 10 skipped" on fresh `.pyd`; live `dtwcpp.test.parallelisation()` = `{available: True, max_threads: 24, threads_engaged: 24, pass: True}`; `gpu()` on the CUDA-OFF baseline honestly returns `available: False` with an actionable reason string [confirmed — orchestrator re-ran both].
- MATLAB (R2025b, serial MEX, fresh-MEX precedence verified): 19/19, 20/20, new test_test_api 6/6, conformance 1/1; `parallelisation()` honestly reports `pass=0` naming the sequential build. test_dtwc 12/14 pre-existing (non-gating → Task 6.0c).
- Gate env note: MSYS printf backslash-escaping corrupted the vcvars path in the CUDA build batch; fixed via heredoc+CRLF (recipe gotcha for future CUDA builds).

**Phase 3 adversarial review — triage (Fable, 2026-07-07; full findings in run journal):**

- **H1 CONFIRMED → Task 3.6 (blocks Phase 4).** RuntimeSingleThread silent fallback SURVIVES for Python compute paths and direct C++ `Problem` usage: the 3.2 warning fires only from the `Env` constructor (env.cpp:234), but `Problem.cpp` has zero `env()` references and the nanobind compute bindings (`compute_distance_matrix` :718, `fill_distance_matrix` :670, `cluster` :690) never construct it; `get_max_threads()` (parallelisation.hpp) warns only in its no-OpenMP `#else` branch, not when OpenMP is capped to 1 thread. Scenario: `OMP_NUM_THREADS=1` (common on SLURM/containers) + `import dtwcpp; dtwcpp.cluster(X,k)` without calling `dtwcpp.device()` → single-threaded, zero stderr. Falsifies the earlier "front-ends run through the singleton dtwc::env()" claim (verified only on dtwc_cl, over-generalised).
- **M1 UPHELD → fixed by orchestrator (docs commit).** `ecc522c` over-claimed: its code delta is one `#include <numeric>` line; the CUDA arch list predates Phase 3 (blame 3550bb9 2026-04-09, 367a9b4 2026-04-04); verification numbers had no committed artifact. Fixed: CHANGELOG reworded (no new dispatch code; deliverable = build recipe + verification event) + verbatim run-log artifact committed at `.claude/baselines/2026-07-07-cuda-first-runtime-verification.md`.
- **L1 UPHELD → folded into Task 6.1.** Proof-of-engagement self-disables on 1-vCPU/thread-capped runners (`test_test_api.cpp:51` guards the `threads_engaged >= 2` REQUIRE behind `max_threads >= 2`); the CIBW gate asserts only the compile-time `OPENMP_AVAILABLE` flag. 6.1's wheel-gate upgrade must assert runtime engagement conditioned on runner CPU count.
- **Review CLEAN on:** DTWC_SEQUENTIAL_BUILD ODR safety (PUBLIC on dtwc++, single-TU test_api.hpp inclusion), project_options alias accuracy, hole-5 fix compile-time regression guard, gpu() oracle non-degeneracy (3 distinct series, shape+max-abs-error checks — not exploitable by zero buffers), all 5 un-gated GPU fallback warnings.

Current state (`.claude/reports/build-state-2026-07-06.md`): OpenMP is the ONLY backend (no std::execution/TBB anywhere). Known silent-fallback holes to close:
1. OpenMP-missing is only a configure-time `message(WARNING)` (`dtwc/CMakeLists.txt:118-123`) — build/wheel succeeds serial.
2. Runtime warning lives only in `get_max_threads()` (`parallelisation.hpp:33-47`); fast_pam/fast_clara/pruned parallel regions never call it.
3. **GPU→CPU fallbacks are `if (verbose)`-gated — silent by default** (`Problem.cpp:404,450,470,477,493`).
4. macOS-Intel cross-built wheels likely serial (arm64 runner installs arm64-only libomp) [inferred — verify one CI log].
5. MSVC OpenMP rides `project_options` INTERFACE (`/openmp:experimental`); consumers that drop `project_options` silently serialise (`dtwc/CMakeLists.txt:113`, `python/CMakeLists.txt:33-40`).

**NEW FACT (2026-07-07, orchestrator-verified):** local machine has nvcc 13.0 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`) + NVIDIA RTX 4000 Ada Generation (sm_89). The 2026-06-01 "GPU fixes need a separate CUDA box" constraint is STALE — CUDA can be compiled and runtime-verified locally. Metal still needs macOS CI.

**Design decision (logged):** OpenMP stays the only CPU backend. Absence becomes a configure-time FATAL_ERROR unless `-DDTWC_ALLOW_SEQUENTIAL=ON` is given explicitly. This strengthens "optional dependency" to "explicit opt-out" — the runbook's "core must build without them" holds via the flag; silent-serial wheels/builds die. (TODO item 6; no-silent-fallback global constraint.)

Batching: **wave A** = 3.1 ∥ 3.2 ∥ 3.4 (disjoint files) → gate → commits; **wave B** = 3.3 ∥ 3.5 (needs A's CMake + Problem.cpp state) → gate → commits → phase adversarial review.

### Task 3.1: Build-time parallelism guarantee [wave A]

**Files:** Modify: `dtwc/CMakeLists.txt`, `python/CMakeLists.txt`, root `CMakeLists.txt`/cmake modules as needed.
- [x] OpenMP not found → `FATAL_ERROR` naming the escape hatch `-DDTWC_ALLOW_SEQUENTIAL=ON`; with the flag → configure succeeds, loud warning, `DTWC_SEQUENTIAL_BUILD` compile definition set.
- [x] Close hole 5: link `OpenMP::OpenMP_CXX` (or MSVC flags) on the dtwc targets directly, not only via `project_options` INTERFACE.
- [x] Registered configure matrix (run all three, quote verbatim): (a) normal configure exit 0; (b) `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` exit ≠0 with error text containing `DTWC_ALLOW_SEQUENTIAL`; (c) both flags exit 0 + warning. Full rebuild + ctest floor (75 non-skip, 0 fail) unchanged.

### Task 3.2: Runtime loudness — Env thread warning + un-gated GPU fallbacks [wave A]

**Files:** Modify: `dtwc/parallelisation.hpp`, `dtwc/Problem.cpp` (:404,450,470,477,493), `dtwc/env.cpp`/`env.hpp`.
- [x] Env construction: effective threads==1 while `hardware_concurrency()>1` (or `DTWC_SEQUENTIAL_BUILD`) → one loud stderr warning, exact string documented and asserted byte-for-byte in a test.
- [x] The 5 `if (verbose)`-gated GPU→CPU fallback messages → ALWAYS emitted to stderr (verbose adds detail only, never gates the warning).
- [x] Tests: stderr-capture test for both warning classes; ctest floor unchanged.

### Task 3.3: `dtwc.test` introspection API — same schema in C++/Python/MATLAB [wave B]

**Files:** Create: `dtwc/test_api.hpp` (header-only — avoids CMake source-list edits). Modify: `python/src/_dtwcpp_core.cpp` + `python/dtwcpp/` (test namespace), `bindings/matlab/dtwc_mex.cpp` + wrapper `.m`. Tests per language.
- [x] `parallelisation()`: run a real OMP region, collect distinct thread ids → `{available, max_threads, threads_engaged, pass}`. Proof-of-engagement: distinct ids ≥2 on multicore, not a flag read.
- [x] `gpu()`: if CUDA/Metal compiled in, execute a tiny kernel and validate vs CPU oracle (≤1e-12) → `{available, backend, device_name, validated, pass}`; else `{available:false, reason:"..."}` — never throws, never silent, reason names what is missing.
- [x] Same field names in all three languages (C++ `dtwc::test::*`, Python `dtwcpp.test.*`, MATLAB `dtwc_mex('test_parallelisation')` + wrapper). Add Metal to Python `check_system` report (bound but unreported).
- [x] Registered (local, baseline CUDA-OFF build): parallelisation `pass=true, threads_engaged>=2`; gpu `available=false` with non-empty reason. (Real-GPU validation of the same API happens in 3.5's CUDA build.)

### Task 3.4: CI wheel parallelism gate [wave A]

**Files:** Modify: `.github/workflows/*.yml`, `pyproject.toml` (cibuildwheel).
- [x] `CIBW_TEST_COMMAND`: import test + assert OpenMP available via the EXISTING bound introspection (`dtwcpp.check_system()`/`OPENMP_AVAILABLE`) — do NOT depend on 3.3's new API (wave order). Phase 6 upgrades this to `dtwcpp.test.parallelisation()`.
- [x] Hole 4 (macOS-Intel serial wheels): fix cross-arch libomp install, or explicitly drop x86_64-macOS wheels with a loud release-notes line — decide from the CI config evidence, document which.
- [x] Gate (local): every changed YAML parses (`python -c "yaml.safe_load"` exit 0); logic reviewed against band. Runtime CI confirmation = NEEDS-CI-RUN, recorded advisory (no push from agents).

### Task 3.5: CUDA enablement + first-ever runtime verification (local RTX 4000 Ada) [wave B]

**Files:** Modify: `dtwc/cuda/**` (`.cu`/`.hpp`), GPU dispatch sites in `Problem.cpp` (after 3.2), CUDA CMake as needed. New scratch build dir: `build/cuda-verify`.
- [x] Configure `build/cuda-verify` with `-DDTWC_ENABLE_CUDA=ON` (nvcc 13.0 needs MSVC host on Windows — try MSVC generator/clang-cl host; ≤3 distinct configure strategies, then report FAILED with verbatim errors).
- [x] Run `test_cuda_correctness` + `test_cuda_lb_keogh` on the real GPU (permanently skipped until now) — first runtime verification of Phase 0 CUDA audit fixes (wavefront max_L>2048 double-buffer, int64 pair indexing). Registered: both suites RUN (not skipped), 0 failed, tolerances per the tests' own bands.
- [x] TODO backlog: arch-aware dispatch (sm_89 local + sm_90/H100 in fat binary or PTX), k-vs-all kernel wiring, multi-stream. Correctness runtime-verified on RTX 4000; H100 PERF claims stay ADVISORY until an ARC run.
- [x] Baseline build (CUDA OFF) ctest floor unchanged — CUDA work must not perturb CPU paths.

### Task 3.6: Close the RuntimeSingleThread silent fallback on compute entry points [review H1 — DONE 2026-07-07, commit `6df3c80`]

**Files modified:** `dtwc/parallelisation.hpp` (both branches of `get_max_threads()` now call the shared emitter; duplicated hard-coded string deleted), `dtwc/env.hpp`/`env.cpp` (new free function `dtwc::warn_if_single_threaded()`; `Env::warn_if_sequential()` delegates to it; the once_flag hoisted to a file-scope guard shared by both paths), `python/src/_dtwcpp_core.cpp` (explicit call at the top of `compute_distance_matrix`). New tests: `tests/unit/test_runtime_loudness_compute.cpp`, `tests/python/test_single_thread_warning.py`.

- [x] Shared process-once emitter in env.cpp (single `std::call_once` guard, SSOT strings, no duplication). `get_max_threads()` calls it in BOTH branches; the Python compute free function calls it directly (deterministic coverage of the unpruned raw-pragma branch). Design note: routing through `get_max_threads()` covers all chunked paths (Problem fill, pruned matrix, fast_pam swap, mpi) because they reach it via `omp_chunk_size()`.
- [x] Design intent unchanged from 3.2: deliberate `OMP_NUM_THREADS=1` still warns.
- [x] Registered bands ALL MET: (a) `test_runtime_loudness_compute` — `fill_distance_matrix` with OMP forced to 1, warning EXACTLY once byte-identical to SSOT, no repeat on a second compute call; (b) `test_single_thread_warning.py` — subprocess `OMP_NUM_THREADS=1` + `compute_distance_matrix` with NO `device()` call warns on stderr; (c) live `dtwc_cl --input … --device cpu` under `OMP_NUM_THREADS=1` → warning count == 1 (Env ctor + compute share the guard); (d) floors — ctest "0 failed out of 86" (80 non-skip), pytest 391, MATLAB 25/25 (19 input-validation + 6 test_api; serial-MEX `parallelisation() pass=0` honesty preserved).
- [x] CHANGELOG updated; committed `fix: warn on single-thread OpenMP from all compute entry points (Phase 3 review H1)`.

## Phase 4 — Solver upgrade: "LR-core" [FINAL — from solver-math report]

Source: `.claude/reports/solver-math-2026-07-06.md` (full derivations; move to `docs/` in Phase 7 — `.claude/reports/` is gitignored). Math verdict, verified by independent re-derivation + enumeration:

- p-median matrix TU for N≤2 only; the TU substructure that matters: **Cardinality+Linking rows are TU for ALL N** (Ghouila-Houri proof in report) ⇒ Lagrangian dual of assignment-dualized problem equals the LP bound (Geoffrion) **without forming the N²-column LP**. All hardness lives in the k-subset choice of y (N binaries); x is an O(Nk) scan once y is fixed.
- FALSIFIED (registered bands, scipy/HiGHS vertex LPs + brute-force IP oracle): half-integrality of fractional vertices (§3.3); "LP integral 80–90%" as a polytope property — it is data-regime-dependent (clustered non-metric D: 250/250 integral; uniform D: 26% at N=20, max gap 13.7%). User's "almost unimodular" observation = his data lives in the integral regime.
- Previous attempts (2023, removed in `f7064b3`) all failed by solving the x-space LP explicitly (dense/sparse tableau simplex, Gomory, OSQP/ADMM). Live 2026 `benders.cpp` is directionally right but re-solves the master MIP per round; its "O(pk)" cut generation is actually O(N²).
- **P4 guard:** odd-cycle cuts alone cannot close instances (cardinality row breaks Baiou–Barahona) — the TODO "odd-cycle cutting planes" item is DEPRIORITISED; do not over-invest.

**Batching (Opus orchestrator):** STRICTLY SEQUENTIAL — 4.1 → 4.2 → 4.3 → 4.4; each consumes the previous, and this is math-heavy work: one Opus-xhigh agent per task with its own gate, no fan-out. Validate every oracle on a NON-degenerate case FIRST (random non-symmetric D — uniform/symmetric instances let sign/factor errors hide; CLAUDE.md §4). A FALSIFIED band (esp. 4.3's wall-time clause) is a deliverable, not a failure — record the numbers, keep the LR root as bound/certificate tool, move on. Note 4.1 prefers the FasterPAM UB from 5.1: if Phase 5 hasn't run yet, use current fast_pam and say so in the report — do NOT block on Phase 5.

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

**Batching (Opus orchestrator, ≤4 concurrent, disjoint file ownership):** wave 1 = 5.1 ∥ 5.2 ∥ 5.5 ∥ 5.7 → gate; wave 2 = 5.3 ∥ 5.4 ∥ 5.10 → gate (5.3 consumes 5.2's envelopes; 5.4 owns `dtwc/core/` kernels ALONE in its wave); wave 3 = 5.6 ∥ 5.8 ∥ 5.9 → gate (5.6 touches kernel dispatch only after 5.4 settles); 5.11 LAST and alone (its whole point is measure-first). ALL speedup bands are ADVISORY on this machine (parallel workloads — record numbers + "machine under load" flag); the digit-identity/correctness halves of each band remain HARD gates. Hard perf verdicts only on a quiet machine or CI.

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

## Phase 6 — Packaging & release [FINAL 2026-07-07]

Current state (`.claude/reports/build-state-2026-07-06.md` + Phase 3 additions): wheel CI ALREADY EXISTS — cibuildwheel on 3 OS, sdist, **PyPI OIDC trusted publishing wired on `v*` tags** (`python-wheels.yml:74-103`); 3.4 added the CIBW OpenMP gate (NEEDS-CI-RUN) and dropped macOS x86_64 wheels (arm64-only libomp on the runner — documented in CHANGELOG).

**Batching:** 6.0 first (residual burn-down; its sub-tasks are independent — one wave of ≤4, then the remainder) → 6.1 ∥ 6.2 ∥ 6.3 ∥ 6.4 → 6.5 ∥ 6.6 → phase gate → adversarial review. CI-run verification needs a push: agents NEVER push — the orchestrator asks the user to push/trigger, then reads the Actions log and quotes it verbatim.

### Task 6.0: Residual burn-down (tracked debt — close each or formally defer with owner+reason in the Decision log)

- [ ] **a. Metal NxN `pair_offset` int32 → wider type**: kernels dtw_wavefront/:84, dtw_wavefront_global/:204, dtw_banded_row/:329, dtw_regtile_w4/:820, dtw_regtile_w8/:838, regtile body :759; host cast :1666 + setBytes :1686 (all `metal/metal_dtw.mm`). N≳65536 wraps negative today. No macOS locally: compile-inspection + host-side index-arithmetic unit test reproducing the wrap; flag NEEDS-MACOS-CI. Also document the pre-existing int32 `pair_indices`/`active_pairs` 2^31 cap (0.2 design).
- [ ] **b. `DTWC_ENABLE_LLFIO=OFF` full BUILD**: guard `Problem.hpp:24`, `core/mmap_distance_matrix.hpp:43`, `core/mmap_data_store.hpp:45` behind `DTWC_HAS_MMAP`. Registered: LLFIO=OFF configure+BUILD exit 0; mmap-dependent suites skip LOUDLY (named skip reason); baseline-config floors unchanged.
- [ ] **c. `tests/matlab/test_dtwc.m` stale assertions**: test_clustering_medoids + test_clustering_fit_predict assume obsolete columns=series → fix to rows=series. Registered: 14/14 in local MATLAB.
- [ ] **d. unit_test_clustering_algorithms**: replace repo-relative `".\data\dummy"` with a path resolved from the test binary location or a configure-time definition (Global Constraint: no repo-relative runtime paths). Then chase the intermittent 0xc0000409: 20 repeats; if it persists, minimise and open a dedicated task — do NOT paper over with retries.
- [ ] **e. `dtwc_cl` required-arg message**: "via CLI or YAML config" → name TOML as primary, e.g. "via CLI or config file (TOML; YAML if built with DTWC_ENABLE_YAML)". One-line + test string update.
- [ ] **f. Supply chain**: pin yaml-cpp to a tag/SHA; record the candidate llfio SHA + date (maintainer blessing stays OPEN in Decision log — do not silently self-bless).

### Task 6.1: Wheel pipeline hardening

**Files:** `.github/workflows/python-wheels.yml`, `pyproject.toml`.

- [ ] Bump the cibuildwheel pin (v2.21 silently skips cp314); enumerate the resulting python-tag matrix in the report.
- [ ] Upgrade `CIBW_TEST_COMMAND` to the 3.3 API with the review-L1 fix built in: assert `r=dtwcpp.test.parallelisation(); r['available']` always, and `r['threads_engaged']>=2` whenever `os.cpu_count()>=2` (runner-CPU-conditioned so 1-vCPU runners can't vacuously green a runtime-serialisation regression) + a smoke-cluster on the conformance mini-fixture asserting the reference labels.
- [ ] **HiGHS-in-wheels DECISION** (leaning bundle — MIP-in-Python is a 2.0 headline; Gurobi stays external always): implement, measure wheel size; if any wheel exceeds ~100 MB, reconsider and document the choice either way in the Decision log.
- [ ] linux-aarch64 wheel if a hosted runner exists; else one release-notes line. Carry the 3.4 macOS-x86_64 drop through consistently (classifiers, docs).
- [ ] Registered: one full wheel-CI matrix run green INCLUDING the test command on every platform — requires user-triggered push; quote the Actions log verbatim.

### Task 6.2: MEX packaging + OpenMP strategy + binary-in-git removal

**Files:** `bindings/matlab/`, MEX CMake target, new `.github/workflows/` MEX job.

- [ ] **DECIDE MEX OpenMP** (today `build/mex-verify` is deliberately SERIAL — Phase 2 MATLAB/libomp clash). Candidates, evidence-based in `build/mex-verify-omp`: (i) link MATLAB's own bundled iomp/libomp (locate under `matlabroot`, link that exact library), (ii) statically link LLVM libomp, (iii) stay serial + loud warning. Run the full MATLAB suite + `dtwc_mex('test_parallelisation')` per candidate in local MATLAB. Registered: chosen build 19/19 + 20/20 + 6/6 + 1/1 AND `threads_engaged>=2` — or a documented serial verdict with the 3.2 warning quoted verbatim.
- [ ] MEX CI: `matlab-actions/setup-matlab` build+test job per platform. If licensing blocks CI, document the local-verify protocol as the release gate instead (exact commands).
- [ ] Remove stale committed `bindings/matlab/dtwc_mex.mexw64` from git (Decision-log OPEN → needs explicit user sign-off in the PR; the stale April binary shadowed fresh MEX and caused a real 0xc0000005 in the Phase 2 gate). Distribution becomes CI artifacts / release archives.

### Task 6.3: Executable distribution

**Files:** CPack config + release workflow (new or extended).

- [ ] CPack zip/tgz of every install()-ed executable (enumerate in-task — at minimum `dtwc_cl`) per platform on `v*` tags, with LICENSE + minimal README. Registered: unpack into a bare temp dir OUTSIDE the repo, run with an absolute `--input` → works (proves Global Constraint 1, no repo-relative paths).

### Task 6.4: Version SSOT

- [ ] Single source: root `VERSION` file feeding CMake AND `pyproject.toml` (scikit-build-core dynamic metadata). `dtwcpp.__version__`, `dtwc_cl --version`, MEX version string all read the same value; one test asserts all four match. Set `2.0.0rc1`.

### Task 6.5: MSB3491 wheel-rebuild blocker

- [ ] Root-cause the llfio→quickcpplib→outcome MSVC superbuild failure (verbatim "error MSB3491"; pre-existing — blocks `pip wheel --no-build-isolation` from scratch on this machine while clang/Ninja core builds stay green). Ranked candidates: (i) pin/patch quickcpplib, (ii) wheels built llfio-free (mmap OFF in wheels — FIRST measure what that costs: the 48× CLARA view-mode path; if wheels lose mmap, docs must say so loudly), (iii) prebuilt llfio. Reproduce → fix → registered: a from-scratch wheel build exits 0 locally (`uv build` or the pip equivalent).

### Task 6.6: CI trigger hygiene + release dry-run

- [ ] Add branch `Claude` to unit-test workflow triggers (temporary — leave a `TODO(release): remove` comment).
- [ ] TestPyPI dry-run of the complete publish path on a `v2.0.0rc1` tag. REAL PyPI publish: ONLY after Phases 0–6 gates green + end-to-end smoke per platform + **explicit user go** (user rule: publish only when 100% sure).

### Phase 6 gate

- [ ] Every 6.0 sub-task closed or formally deferred (owner + reason in Decision log).
- [ ] Local floors hold (§Orchestrator handoff values, updated for any new suites).
- [ ] One full CI run green: wheel matrix + test commands, MEX job (or documented local protocol), release archives produced and tempdir-tested.
- [ ] Adversarial review agent on the phase diff. Hunt list: version skew between the four version strings; wheels missing modules (`dtwcpp.test`, sklearn wrapper if 5.10 landed); CI-green-but-vacuous test commands (exit-code swallowing, unconditional `|| true`, asserts that can't fail on the runner); packaging paths violating Constraint 1; HiGHS licence/attribution files if bundled.

## Phase 7 — Documentation website [FINAL 2026-07-07]

Prereq: Phase 6 version SSOT + the frozen contract. **Batching:** 7.1 alone (scaffold) → 7.2 ∥ 7.3 ∥ 7.4 ∥ 7.5 → 7.6 docs gate.

### Task 7.1: Site scaffold

- [ ] FIRST inspect what exists (`docs/`, any Doxygen config, gh-pages branch, the plan header says Hugo — but decide from repo evidence, log the decision): Hugo vs MkDocs-Material, CI deploy to GitHub Pages on main. Landing page: what/why + 60-second quickstart shown side-by-side in all 3 languages (same fixture as conformance — the snippets are then provably runnable).

### Task 7.2: API reference

- [ ] Generate the 3-language API pages FROM `docs/api-contract-2.0.md` (it is the SSOT — never hand-duplicate signatures): Tier-1 (`device`/`load`/`cluster`/`Result`) and Tier-2 (`Problem`) pages with C++/Python/MATLAB tabs textually aligned. Doxygen for deep C++ reference only if cheap; the contract pages are the primary surface.

### Task 7.3: Guides

- [ ] Device selection (cpu/gpu/hpc) incl. `.env` HPC setup — document the three exact error messages (missing file / missing key / bad host) VERBATIM from env.cpp; `dtwc.test.parallelisation()`/`gpu()` usage page; the no-silent-fallback design note (why configure fails without OpenMP, the `-DDTWC_ALLOW_SEQUENTIAL=ON` escape hatch, what the single-thread warning means); data formats + conversion (CSV/TSV/Arrow/Parquet, converter tool); solver guide (HiGHS default, Gurobi optional, LR-core if Phase 4 landed — with its regime of validity and the FALSIFIED-claims honesty from the report); Mermaid architecture diagram.
- [ ] Migration guide 1.x → 2.0: full rename table from contract 1.1, deprecation list + removal schedule, behavioural changes (double default, C++ result write-back, loud warnings, OpenMP requirement).

### Task 7.4: Math documentation

- [ ] Move the solver derivations OUT of gitignored `.claude/reports/solver-math-2026-07-06.md` into `docs/math/lr-core.md`: full re-derivable derivation (TU substructure proof, Geoffrion equality, the falsified claims WITH their registered bands — killed ideas stay visibly killed). CLAUDE.md rule: every analytic result of lasting value gets a full derivation in docs the reader can re-derive from scratch.

### Task 7.5: Benchmarks page

- [ ] UCR results (128 datasets, 4 architectures, H100 14.2×): publish ONLY numbers from quiet-machine/CI runs with methodology + hardware stated per table; local advisory numbers stay OUT (Global Constraint / user caveat 2026-07-06).

### Task 7.6: Docs gate

- [ ] Adversarial review agent diffs every documented signature/flag/error-string against the code (contract drift, stale messages, dead flags). Link checker green. All three quickstarts EXECUTED verbatim (copy-paste → run → output matches the page).

## Release — v2.0.0 (after Phases 3.6–7 closed)

- [ ] Re-record perf numbers on a QUIET machine (this is where advisory becomes real) + fresh UCR spot-bench vs the Phase 0 baseline artifact.
- [ ] CHANGELOG.md: collapse Unreleased → `2.0.0` with date; link the migration guide.
- [ ] Tag `v2.0.0` → wheels + MEX + archives; TestPyPI-verified path → PyPI on **explicit user go**.
- [ ] Release notes: HPC device marked BETA (blocked on a real Oxford ARC end-to-end run — user action); Metal marked build-verified/runtime-unverified unless macOS CI landed; MSVC `/openmp:experimental` branch verified by then or called out.
- [ ] Post-release: 2.1 milestone with everything formally deferred (Decision log is the source).

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
- 2026-07-07 (evening): **Fable → Opus 4.8 orchestrator handoff. Grand plan v1.0.** Phase 3 CLOSED (wave B gate PASS; commits `ad34b6a`/`ecc522c`; CUDA runtime-verified on local RTX 4000 — first ever). Review triaged: H1 → Task 3.6 (FIRST ACTION), M1 → fixed (CHANGELOG + `.claude/baselines/2026-07-07-cuda-first-runtime-verification.md`), L1 → folded into 6.1. Phases 6–7 finalised; §Orchestrator handoff added with floors, workflow pattern, proven recipes. Remaining sequence: 3.6 → Phase 4 (sequential) → Phase 5 (3 waves + 5.11) → Phase 6 → Phase 7 → Release.
- 2026-07-07 (night, Opus 4.8): **Task 3.6 DONE** (commit `6df3c80`) — RuntimeSingleThread silent-fallback (review H1) closed. Shared process-once emitter `warn_if_single_threaded()` in env.cpp; `get_max_threads()` calls it (covers all `omp_chunk_size`-reached compute paths); Python `compute_distance_matrix` calls it directly. One `std::call_once` guard shared with the Env ctor. All 4 registered bands verified live (C++ + Python-subprocess + `dtwc_cl`-once + floors ctest 86/pytest 391/MATLAB 25). Floors re-recorded. **Phase 3 fully closed; Phase 4 (LR-core) is next.**
