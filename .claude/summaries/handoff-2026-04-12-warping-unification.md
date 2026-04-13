---
name: handoff-2026-04-12-warping-unification
description: Session handoff — warping header family unified via Policy-Based Design (Phases 1+2). 1.5–2.8× perf improvement, 3 silent dispatch bugs fixed. Plus check-code-quality skill + audit.
type: project
---

# Session Handoff — 2026-04-12 (evening) — Warping Unification + Code-Quality Skill

Branch: **Claude**. Worked on top of commit `0f846fa`. All work in this session was committed by the user as `eb7b572 commit redundancy improvement` (which bundles Phase 1 + GPU parity + CSV-leak fix + GPU-bench context injection). Phase 2 additions + the check-code-quality skill are **uncommitted** in the working tree as of handoff time.

## Goals (session intent)

1. Continue the deferred GPU-parity items from the prior session (CUDA/Metal LB_Keogh, `LowerBoundStrategy` enum, base `DistMatOptions`/`Result`, `dispatch_gpu_backend<T>`, etc.).
2. Fix a stray-CSV-in-repo-root bug the user flagged.
3. Blank hostnames from benchmark JSONs; keep CPU/GPU specs.
4. User then pivoted to a **redundancy audit**: "why do we have `warping_adtw.hpp`, `warping_ddtw.hpp`, `warping_missing_arow.hpp` etc. — silly design." Go to plan mode and ultrathink a better design.
5. User asked to verify we did not compromise performance.
6. User asked for a `check-code-quality` skill (12 categories), adversarial-hardened, then used to audit the codebase.

## Accomplishments

### 1. GPU parity + configuration (pre-redundancy-audit work, now committed in `eb7b572`)

- **`MetalDistMatResult::lb_time_sec`** — parity with CUDA; timing threaded through the LB pre-pass in `metal_dtw.mm`.
- **`dtwc::metal::compute_lb_keogh_metal(series, band)`** — standalone LB_Keogh entry at [dtwc/metal/metal_dtw.hpp:85-102](../../dtwc/metal/metal_dtw.hpp#L85-L102). Mirrors `cuda::compute_lb_keogh_cuda`. 4 new tests in [tests/unit/test_metal_lb_keogh.cpp](../../tests/unit/test_metal_lb_keogh.cpp).
- **`dtwc::LowerBoundStrategy` enum** (`Auto`/`None`/`Kim`/`Keogh`/`KimKeogh`) at [dtwc/enums/LowerBoundStrategy.hpp](../../dtwc/enums/LowerBoundStrategy.hpp); threaded through `Problem::lb_strategy` and `fill_distance_matrix_pruned`. `None` short-circuits to BruteForce.
- **`dtwc::KernelOverride` enum** (`Auto`/`Wavefront`/`WavefrontGlobal`/`BandedRow`/`RegTile`) + `max_length_hint` field added to both `CUDADistMatOptions` and `MetalDistMatOptions`. Metal wires the override into its dispatch; `kernel_used` now reflects the actual pipeline (was previously re-derived post-hoc, which silently broke under override).
- **`dtwc::gpu::DistMatOptionsBase` / `DistMatResultBase`** at [dtwc/core/gpu_dtw_common.hpp](../../dtwc/core/gpu_dtw_common.hpp) — CUDA and Metal option/result structs now inherit common fields.
- **`dispatch_gpu_backend` lambda in `Problem::fillDistanceMatrix`** — collapses the parallel CUDA/Metal case blocks into one generic post-processor.

### 2. Stray-CSV leak fixed

Tests `unit_test_fast_pam.cpp` and `unit_test_fast_clara.cpp` called `setResultsPath(".")`, dumping CSVs into whatever CWD the test ran from (including the repo root). Routed both to `std::filesystem::temp_directory_path() / "dtwc_fast_pam_test"` etc. at [tests/unit/algorithms/unit_test_fast_pam.cpp:29-33](../../tests/unit/algorithms/unit_test_fast_pam.cpp#L29-L33), [unit_test_fast_clara.cpp:31-37](../../tests/unit/algorithms/unit_test_fast_clara.cpp#L31-L37). Deleted 6 stray CSVs from repo root.

### 3. Hostname stripping + GPU specs in benchmarks

- 13 JSONs under `benchmarks/results/mac_m2max/` had `host_name` blanked via `sed -i '' -E 's/"host_name": "[^"]*"/"host_name": ""/'`.
- **`bench_metal_dtw` / `bench_cuda_dtw`** now define a custom `main()` that injects `gpu_backend`, `gpu_device_info`, `gpu_available` into Google Benchmark's JSON context via `benchmark::AddCustomContext`. [benchmarks/bench_metal_dtw.cpp:400-418](../../benchmarks/bench_metal_dtw.cpp#L400-L418), [benchmarks/bench_cuda_dtw.cpp:370-388](../../benchmarks/bench_cuda_dtw.cpp#L370-L388). CMake drops `benchmark_main` for these targets.
- **Wrapper at [scripts/run_bench.sh](../../scripts/run_bench.sh)** — runs a benchmark, auto-generates a timestamped JSON under `benchmarks/results/_autorun/`, strips `host_name` post-run. Handles BSD (macOS) + GNU sed. End-to-end verified: `host_name: ''` and `gpu_device_info: Apple M2 Max (registryID=0x100000446, max_working_set=77.76 GB)` both present.
- **Memory saved** at `~/.claude/projects/-Users-engs2321-Desktop-git-dtw-cpp/memory/feedback_json_hostnames.md` — "strip hostnames, keep CPU + GPU specs."

### 4. Warping unification — Policy-Based Design (Phases 1 + 2)

**The big structural win.** Entered plan mode, spawned one Explore agent for the audit, wrote plan to `.claude/plans/federated-seeking-summit.md`, user approved.

Three axes of DTW variation — **pointwise cost, cell recurrence, window shape** — are now three orthogonal policies:

- **New [dtwc/core/dtw_kernel.hpp](../../dtwc/core/dtw_kernel.hpp)** — one templated family: `dtw_kernel_full<T, Cost, Cell>`, `dtw_kernel_linear<...>`, `dtw_kernel_banded<...>`. Cell policies: `StandardCell` (min-of-3 + cost), `ADTWCell<T>{penalty}` (penalty on horizontal/vertical steps).
- **New [dtwc/core/dtw_cost.hpp](../../dtwc/core/dtw_cost.hpp)** — Cost functors, all `noexcept` single-method structs: `SpanL1Cost<T>`, `SpanSquaredL2Cost<T>`, `SpanWeightedL1Cost<T>` (WDTW), `SpanNanAwareL1Cost<T>` (ZeroCost missing) + MV counterparts for all four. `dispatch_metric()` / `dispatch_mv_metric()` moved here from `warping.hpp` `detail::`.
- **[dtwc/warping.hpp](../../dtwc/warping.hpp), [warping_adtw.hpp](../../dtwc/warping_adtw.hpp), [warping_wdtw.hpp](../../dtwc/warping_wdtw.hpp), [warping_missing.hpp](../../dtwc/warping_missing.hpp)** — all now thin public-API wrappers that build a Cost + Cell and call the shared kernel. `detail::_impl` functions in warping.hpp are preserved as ~10-line shims forwarding to the kernel, so `warping_missing_arow.hpp` (not migrated in Phase 2 — see Deferred) still compiles unchanged.
- **[dtwc/warping_ddtw.hpp](../../dtwc/warping_ddtw.hpp)** — untouched; already delegated to `dtwBanded` after derivative preprocessing. Good pre-existing design.

**Design pattern: Policy-Based Design** (Alexandrescu, *Modern C++ Design*; Iglberger, *C++ Software Design*, ch. on Strategy via templates). Compile-time Strategy — zero runtime indirection. **Explicitly rejected** alternatives: virtual functions (indirection in hot loop), `std::variant` (runtime dispatch), CRTP (adds inheritance; flat policy structs simpler). C++20 `concept` constraints are the only open refinement (better error messages).

### 5. Silent dispatch bugs fixed during the unification

1. **`dtwc::core::dtw_runtime()`** silently ignored `opts.variant_params.variant` (always ran Standard DTW). Fixed at [dtwc/core/dtw.cpp](../../dtwc/core/dtw.cpp). Regression test in [tests/unit/unit_test_mv_variants.cpp](../../tests/unit/unit_test_mv_variants.cpp) (`dtw_runtime honours DTWVariant::ADTW/WDTW`).
2. **`adtwBanded_mv` / `wdtwBanded_mv` / `dtwMissing_banded_mv`** fell back to unbanded MV (documented TODOs at `warping_adtw.hpp:344`, `warping_wdtw.hpp:478`, `warping_missing.hpp:324`). Now use the real banded MV kernel. Regression tests in [unit_test_mv_variants.cpp](../../tests/unit/unit_test_mv_variants.cpp) and [unit_test_mv_missing.cpp](../../tests/unit/unit_test_mv_missing.cpp) with a shifted-peak pattern confirm tight bands produce strictly greater distances than unbanded.

### 6. Performance: measured A/B/C (NOT a regression — a speedup)

Ran `git stash` isolated to warping files + new kernel/cost headers, rebuilt, benched, restored. Three runs per state, min_time 1.0 s, 3 repetitions.

| Benchmark | Pre-refactor | Phase 1 unified | Speedup |
|-----------|--------------|-----------------|---------|
| `BM_dtwFull_L/1000` | 2967 μs | **1924 μs** | **1.54×** |
| `BM_dtwBanded/1000/50` | 258 μs | **146 μs** | **1.77×** |
| `BM_wdtwBanded_g/1000/50` | 450 μs | **159 μs** | **2.83×** |

Phase 2 (warping_missing migration) is within noise of Phase 1 on these benchmarks — expected, since Phase 2 only touched the missing-data path.

**Why faster, not slower:** old per-variant files had duplicated structural bookkeeping — e.g. WDTW's banded had a ~50-line divergent `if (low == 0)` block. The unified kernel collapses all cases to one uniform code path. The sentinel-based boundary handling (`std::min({maxValue, maxValue, left}) + cost` via `StandardCell`) constant-folds under `-O2`. Template Cost/Cell are pass-by-value 8–24-byte structs kept in registers — no indirection in the inner loop.

Artefacts: [/tmp/state_A2.json](/tmp/state_A2.json), [/tmp/state_B2.json](/tmp/state_B2.json), [/tmp/state_C.json](/tmp/state_C.json) (these are /tmp — not committed; rerun via the Bash instructions below if needed).

### 7. `check-code-quality` skill

**Flow:** user requested an independent expert agent to write a skill, then adversarial review, then apply improvements, then use it on the codebase. Executed all four steps.

- **Skill file**: [.claude/skills/check-code-quality.md](../../.claude/skills/check-code-quality.md) (344 lines after hardening).
- **Covers**: 5 philosophical domains (Turner best practices / Iglberger design patterns / FFmpeg perf / Rust-zealot safety / MISRA defense-industry) × 12 concrete checks (A-L) + 1 informational hardening section (M). Each check ships runnable `rg`/`bash` commands and explicit PASS/WARN/FAIL thresholds.
- **Adversarial review (agent #2) found 16 issues**: 3 P0 (wrong source paths `dtwc/src/**` — does not exist; ripgrep backrefs need `--pcre2`; CSV check missed `results/` subdir), 7 P1 (host_name writer coverage, heap-in-loop awk depth counter broken on nested braces, >5-arg regex didn't exempt Options structs, coverage check used broken empty-regex grep, `fd` not checked for availability, build target silenced errors, scope-arg was prose-only), 6 P2. **All P0 + P1 fixed** in the final skill.
- **Machine-parseable output**: final JSON trailer with `verdict` + per-section statuses for downstream piping.
- **Enforced the user's project-specific invariants as HARD rules in section L**: host_name in JSON → FAIL, CSV in repo root → FAIL, naked `new`/`delete` in core → FAIL, `find_package(... REQUIRED)` on optional deps → FAIL.

### 8. Audit results — Verdict: **WARN**

Skill executor agent ran the whole skill. No hard-rule breaches, but two coverage gaps:

| Section | Status | Key finding |
|---------|--------|-------------|
| A Tests | PASS | 70/70 pass (2 CUDA skipped correctly on macOS) |
| **B Coverage** | **WARN** | ~61% of public `dtwc::` symbols (56 of 92) not referenced in tests. Genuine gaps: `MIP_clustering_byGurobi/Benders/HiGHS`, `Problem::cluster_by_MIP`, `Problem::readDistanceMatrix`, `Problem::printClusters`/`printDistanceMatrix`, `Problem::assignClusters`, several `Data` ctors |
| C Perf docs | PASS | GFLOPS + roofline documented; `benchmarks/plot_benchmarks.py:205-239` has a roofline routine; JSON artefacts present |
| D Hot loops | PASS | Thread_local scratch confirmed. `clarans.cpp:73-83` allocates in outer loop but amortised (should get a `// amortised` comment) |
| E Duplication | PASS | Remaining matches are header comment banners |
| **F API robustness** | **WARN** | `dtwc/fileOperations.hpp:121,161` `load_folder`/`load_batch_file` have 6 positional args. `dtwc/algorithms/fast_pam.cpp:46` `compute_nearest_and_second` has 7 args |
| G Citations | PASS | CITATIONS.md complete (Sakoe-Chiba 1978, Keogh 2005, Cuturi 2017, Yurtman 2023, Jeong 2011, Marteau 2009, Lemire 2009, Rakthanmanon 2012) |
| H Reinventing | PASS | No hand-rolled Matrix/Logger/Optional |
| I License | PASS | All optional deps gated; Gurobi commercial-isolated |
| **J Wrapper parity** | **WARN** | `uv run pytest tests/integration/test_cross_language.py` tried to rebuild the Python wheel from source (llfio CMake error), could not measure the Python/MATLAB ≤10% parity claim |
| K External bench | PASS | README claims 12× vs aeon, 1.7× vs dtaidistance, 42× end-to-end; `benchmarks/bench_cross_library.py` ships the measurement |
| L Project invariants | PASS | All hard rules green (host_name, repo-root CSV, user-home paths, optional-deps-optional, naked new/delete, Unreleased section) |
| M Hardening (info) | — | `.clang-tidy` absent, reproducible-build flags (`SOURCE_DATE_EPOCH`, `-ffile-prefix-map`) absent. Sanitizers + `-Werror` opt-in via `DTWC_DEV_MODE`. 8 CI workflows. `compile_commands.json` fresh |

## Decisions (with rationale)

1. **Policy-Based Design over Strategy-with-virtuals or std::variant dispatch.** Alexandrescu/Iglberger canon; zero-overhead inlining for hot loops. Measured 1.54–2.83× speedup confirms the compile-time approach was correct.
2. **Phase scope: Standard + ADTW + WDTW + DDTW + ZeroCost-missing.** AROW has a genuinely different recurrence (`C(i,j) = C(i-1,j-1)` when either value is missing) that does not fit the current `Cell::combine(diag, up, left, cost, ...)` contract without extending it. Deferred to Phase 3.
3. **Keep `detail::_impl` shims in `warping.hpp`.** `warping_missing_arow.hpp` still routes through them; removing them would cascade into AROW migration, expanding scope beyond what the plan approved.
4. **Legacy `detail::MissingMVL1Dist` etc. retained in `warping_missing.hpp`.** Tests (`unit_test_mv_missing.cpp`) call them directly as `(a, b, ndim)` functors. Kept as compat shims alongside the new index-based `SpanMVNanAware*` functors in `core/dtw_cost.hpp`.
5. **Skill is read-only.** `allowed-tools` excludes `Write`/`Edit` so the skill is provably safe to invoke; downstream actions (applying the skill's findings) are a separate step.
6. **Skill's `host_name` check targets writers, not just committed artefacts.** Reviewer's P1 fix — auditing artefacts alone would miss the next `gethostname()` call that would regenerate them.

## Open questions / known issues

1. **Phase 3 not started.** Three deferred items (in increasing complexity):
   - Fold `warping_missing_arow.hpp` — needs an extended Cell contract (e.g. `Cell::combine(diag, up, left, cost, row, col, is_missing_pair)` or a dedicated AROW kernel).
   - Fold `soft_dtw.hpp` — needs `SoftCell{gamma}` with log-sum-exp; soft-DTW also has a gradient function that uses a full matrix, so the scratch buffer ownership differs.
   - Shrink `Problem::rebind_dtw_fn()`'s 130-line switch via a visitor pattern that maps `{DTWVariantParams, MissingStrategy, ndim}` to a kernel + policy bundle.
2. **Audit WARN items not yet addressed:**
   - **B. Coverage**: the MIP path (`MIP_clustering_byBenders/byGurobi/byHiGHS`, `cluster_by_MIP`) has no integration test. Would need a small test that runs MIP on a tiny dataset — HiGHS-only path since Gurobi is commercial.
   - **F. API robustness**: wrap `load_folder` / `load_batch_file` positional args in a `LoadOptions` struct.
   - **J. Wrapper parity**: install the Python wheel into the test env (`uv pip install -e .`) so `tests/integration/test_cross_language.py` can actually run.
3. **Hardening gaps** (informational, from section M): no `.clang-tidy`, no reproducible-build flags. These are not strictly required but are senior-eng hygiene.
4. **`benchmarks/results/_autorun/`** contains local bench artefacts from this session. User may want them committed or gitignored — currently untracked.

## Files changed (uncommitted as of handoff)

```
M CHANGELOG.md                                          (+16 net)
M dtwc/core/dtw_cost.hpp                               (+68 NaN-aware cost functors)
M dtwc/warping_missing.hpp                             (rewrite: 329 -> 301 LOC; uses core kernel)
M tests/unit/unit_test_mv_missing.cpp                  (+17 banded MV regression test)
?? .claude/skills/check-code-quality.md                (344 lines)
?? .claude/summaries/handoff-2026-04-12-warping-unification.md  (this file)
```

Everything from Phase 1 + GPU parity + CSV fix + JSON host_name work is already committed in `eb7b572`.

## How to continue next session

1. **Commit Phase 2 + the skill**: `git add dtwc/core/dtw_cost.hpp dtwc/warping_missing.hpp tests/unit/unit_test_mv_missing.cpp CHANGELOG.md .claude/skills/check-code-quality.md .claude/summaries/handoff-2026-04-12-warping-unification.md && git commit -m "..."`.
2. **Run the skill** on your own: it's at `.claude/skills/check-code-quality.md`. Skills in that directory aren't auto-exposed to the `Skill:` tool — either spawn a general-purpose agent with the file as its brief (see the audit run in this session for the prompt), or read it and run the commands manually.
3. **Top-priority Phase 3 pick:** shrink `Problem::rebind_dtw_fn()` (Phase 3 item 3) — lowest risk, highest immediate payoff. The other two Phase 3 items (AROW + SoftDTW) each need design work on extending the Cell contract.
4. **Audit-driven follow-ups:** add a MIP integration test (B/F combined), wrap `load_folder` args in LoadOptions, fix the Python wheel build so `test_cross_language.py` runs.
5. **Perf verification**: `./scripts/run_bench.sh ./build/bin/bench_dtw_baseline --benchmark_filter=BM_dtwBanded --benchmark_repetitions=3` — output lands under `benchmarks/results/_autorun/` with host_name pre-stripped.

## Ultrathink nuggets for next session

- The 1.54–2.83× speedup was not in the plan — it fell out of collapsing divergent branches in the WDTW/ADTW banded paths. **Generalising a specialised loop was faster than the specialised loops.** Counter-intuitive; the plan assumed parity at best.
- The adversarial agent caught 3 P0 bugs the expert missed. **Two-agent sequential review (expert → adversarial) was strictly better than one round.** Worth keeping as a pattern for any non-trivial skill / code generation in this repo.
- The audit agent found 2 genuine gaps the main-thread investigator would have missed (Python wheel install, LoadOptions pattern on load_folder). Running a skill is cheaper than hand-auditing — worth repeating.
