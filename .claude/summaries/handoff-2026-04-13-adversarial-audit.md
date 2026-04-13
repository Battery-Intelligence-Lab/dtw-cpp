---
name: handoff-2026-04-13-adversarial-audit
description: Session handoff — adversarial audit on Windows, code-quality skill hardened, key design/runtime drifts documented, next simplification targets prioritised.
type: project
---

# Session Handoff — 2026-04-13 — Adversarial Audit + Skill Hardening

Branch: **Claude**. Working tree at session end: modified `.claude/LESSONS.md`, `.claude/skills/check-code-quality.md`, plus this handoff file.

## Session intent

User asked to:
1. Read the recent handoff summaries and `.claude/TODO.md`.
2. Perform an adversarial audit of the repository as it exists now.
3. Find redundancies / dedup opportunities.
4. Improve the local skills, especially `check-code-quality`, so future audits miss less.
5. Write a handoff focused on next steps, design cleanup, simplification, performance, and portability.

## What was verified

### Prior handoffs / TODO re-read

Re-read:
- `.claude/TODO.md`
- `.claude/summaries/handoff-2026-04-12-warping-unification.md`
- `.claude/summaries/handoff-2026-04-13-phase3-complete.md`
- `.claude/summaries/handoff-2026-04-13-phase4-wheel-fix.md`

Cross-checked those notes against the current tree instead of trusting them.

### Build / test state on this Windows tree

Release build:
- `cmake --build build --config Release -j 8` succeeds.

Release tests:
- `ctest --test-dir build -C Release --output-on-failure -j 4` is **not green** on this machine.
- Parallel run failed with two crash-style failures:
  - `test_fast_pam_adversarial`
  - `unit_test_clustering_algorithms`
- Serial rerun showed:
  - `unit_test_clustering_algorithms` passes with `-j 1` (parallel-only failure mode / shared-state suspicion).
  - `test_fast_pam_adversarial` still fails when targeted directly; the sub-case `"Adversarial: Better than random medoid selection"` exits immediately after seeding.

This is the most important "current truth" update: the repo is not presently in a clean Windows-tested state despite earlier handoffs being green on macOS / other contexts.

## Audit findings

### P0 / P1-level blockers

1. **Windows test health is overstated.**
   - `ctest --test-dir build -C Release --output-on-failure -j 4` failed.
   - `test_fast_pam_adversarial` is not just a flaky parallel issue; it fails even when isolated.
   - `unit_test_clustering_algorithms` appears to have a parallel-only failure mode.

2. **User-facing docs claim YAML/CLI precedence that the code explicitly contradicts.**
   - Docs: `docs/content/getting-started/configuration.md:147` says CLI flags take precedence over YAML.
   - Code: `dtwc/dtwc_cl.cpp:299` has a TODO stating YAML currently overrides CLI whenever the key exists.
   - This is a real behaviour/documentation mismatch, not a cosmetic note.

3. **A test that looks like FastPAM coverage is actually exercising the legacy Lloyd path.**
   - `tests/unit/adversarial/test_fast_pam_adversarial.cpp:57` sets `prob.method = Method::Kmedoids`.
   - `tests/unit/algorithms/unit_test_fast_pam.cpp:113` already documents that `cluster_by_kMedoidsLloyd()` is the old Lloyd path and separately tests real `fast_pam(...)`.
   - Result: the adversarial test name creates false confidence about FastPAM correctness coverage.

### Structural / design findings

4. **`.claude/design.md` is stale after the Phase 3/4 kernel unification.**
   - It still says "Separate functions per variant" and "Soft-DTW is a separate algorithm" (`.claude/design.md:15-19`).
   - The shipped architecture now routes Standard / ADTW / WDTW / DDTW / Soft-DTW / AROW / missing-data paths through `dtw_kernel_*` + policy cells/costs (`dtwc/core/dtw_kernel.hpp`, `dtwc/core/dtw_dispatch.cpp`).
   - This is an internal-doc drift risk: future refactors could be argued from a design doc that no longer describes the code.

5. **`Problem` still mixes pure clustering, reporting, and file I/O side effects.**
   - `Problem` has a `verbose` flag (`dtwc/Problem.hpp:141`), but `cluster_by_kMedoidsLloyd()` prints unconditionally (`dtwc/Problem.cpp:635`, `:680`, `:704`) and writes result files (`dtwc/Problem.cpp:657`, `:705`; `dtwc/Problem_IO.cpp:34`, `:184`).
   - Tests compensate by redirecting outputs to temp paths rather than the API being clean by construction.
   - This is a design simplification target: pure algorithm result first, optional reporting second.

6. **Algorithm dispatch is fragmented across multiple naming systems.**
   - Core enum: `dtwc/enums/Method.hpp:13-15` only has `Kmedoids` and `MIP`.
   - CLI has separate string-level dispatch for `pam`, `clara`, `kmedoids`, `hierarchical` (`dtwc/dtwc_cl.cpp:733-846`).
   - Python / MATLAB expose `fast_pam` / `fast_clara` directly as free functions rather than going through the same method abstraction.
   - This is not a correctness bug today, but it increases naming drift and duplicated test/docs burden.

7. **String/enum parsing is duplicated in multiple places.**
   - CLI has repeated `CheckedTransformer` maps plus a second YAML alias-normalisation block (`dtwc/dtwc_cl.cpp:151-273`, `:369-390`).
   - MATLAB has its own parsers for missing strategy, distance strategy, linkage, and DTW variant (`bindings/matlab/dtwc_mex.cpp:225-246`, `:364-377`).
   - Python mirrors the same enum surface manually (`python/src/_dtwcpp_core.cpp:51-95`).
   - This is a good dedup target: centralise enum/string mapping in core, then reuse everywhere.

## Skill changes made

Updated `.claude/skills/check-code-quality.md` to make future audits materially stricter and more accurate:

1. **Removed the hardcoded repo path.**
   - Uses `git rev-parse --show-toplevel` with a `pwd -P` fallback.

2. **Made the build/test step multi-config aware.**
   - Detects `CMAKE_CONFIGURATION_TYPES` and uses `--config Release` / `ctest -C Release` when needed.

3. **Added serial reruns of failed tests.**
   - The skill now distinguishes:
     - deterministic failing/crashing tests → `FAIL`
     - parallel-only failures / flakes → `WARN`

4. **Made coverage probing less Linux-specific.**
   - Looks for both Unix archive output and Windows `.lib` output.
   - Uses `nm` / `llvm-nm` detection instead of assuming one fixed toolchain.

5. **Added a new verdict-bearing section for truthfulness / drift.**
   - Catches:
     - docs vs code contradiction
     - misleading algorithm/test naming
     - stale internal architecture docs

6. **Clarified the skill's write policy.**
   - No tracked-source edits.
   - Build/test output is allowed only in the configured build directory or temp.

Also updated `.claude/LESSONS.md` with two durable lessons:
- rerun test failures serially after the first parallel pass
- never let a test name imply coverage of algorithm X when it actually calls algorithm Y

## Recommended next steps

Priority order:

1. **Fix `test_fast_pam_adversarial` first.**
   - Reproduce with the single failing case: `"Adversarial: Better than random medoid selection"`.
   - Most likely root areas:
     - legacy Lloyd k-medoids path, not FastPAM
     - empty-cluster / duplicate-medoid handling
     - side effects inside `Problem::cluster_by_kMedoidsLloyd()`

2. **Resolve the YAML precedence bug before editing docs again.**
   - Either:
     - implement true CLI-wins precedence using `app["--flag"]->count()`, or
     - explicitly document current YAML-wins behaviour and stop claiming otherwise.
   - The code TODO already points at the right fix surface.

3. **Rename or rewrite `test_fast_pam_adversarial.cpp`.**
   - Option A: rename it to legacy/k-medoids/Lloyd terminology.
   - Option B: keep the file name and switch it to actual `fast_pam(...)`.
   - Best outcome: do both categories explicitly, with separate files.

4. **Split pure clustering from reporting side effects.**
   - Legacy Lloyd path should return a `ClusteringResult`-style object and only write CSVs / print via an explicit reporting layer.
   - This simplifies tests, wrappers, CLI integration, and future method unification.

5. **Create one shared enum/string conversion layer.**
   - Suggested target: `dtwc/core/enum_strings.hpp` or similar.
   - Feed it into:
     - CLI transformers / YAML normalisation
     - MATLAB MEX parsers
     - any future docs generation / help text

6. **Unify algorithm naming at the API boundary.**
   - Consider introducing a richer algorithm enum or dispatcher that can represent:
     - FastPAM
     - legacy Lloyd k-medoids
     - FastCLARA
     - CLARANS
     - hierarchical
     - MIP
   - Then keep `Problem.method`, CLI, docs, and bindings aligned to one vocabulary.

## Design simplifications that should preserve performance / portability

1. **Do not undo the unified DTW kernel family.**
   - The current policy-based `dtw_kernel_*` setup is the correct simplification: less code, fewer dispatch bugs, no measured perf loss.

2. **Simplify at the orchestration layer, not the hot path.**
   - Good targets:
     - string parsing / enum conversions
     - algorithm naming and dispatch
     - reporting / file I/O
     - docs/truthfulness checks
   - Poor targets:
     - reintroducing per-variant recurrence loops
     - virtualising the inner DTW path

3. **Treat `Problem` as the next architectural pressure point.**
   - It currently acts as:
     - algorithm state holder
     - I/O/reporting surface
     - GPU dispatch orchestrator
     - MIP entry point
     - legacy clustering engine
   - Splitting those responsibilities is the biggest clarity win still available without touching kernel performance.

## Suggested continuation prompt

If continuing next session, the best follow-on is:

> Reproduce and fix the Windows failure in `test_fast_pam_adversarial`, then rename/split the misleading FastPAM vs Lloyd tests so adversarial coverage lines up with the real implementation under test.

