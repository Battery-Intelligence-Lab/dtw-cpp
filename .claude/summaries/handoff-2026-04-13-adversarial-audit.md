---
name: handoff-2026-04-13-adversarial-audit
description: Session handoff - adversarial audit on Windows, code-quality skill hardened, runtime/design drift documented, and cross-language docs/API parity work staged with local Hugo+Doxygen output.
type: project
---

# Session Handoff - 2026-04-13 - Adversarial Audit + Skill Hardening

Branch: **Claude**. Working tree at session end: modified `dtwc/Problem.cpp`, `tests/unit/adversarial/test_fast_pam_adversarial.cpp`, `tests/unit/unit_test_clustering_algorithms.cpp`, docs pages/config, additive distance facades for C++/Python/MATLAB, plus local `.claude` notes and skills.

## Session intent

User asked to:
1. Read the recent handoff summaries and `.claude/TODO.md`.
2. Perform an adversarial audit of the repository as it exists now.
3. Find redundancies and dedup opportunities.
4. Improve the local skills, especially `check-code-quality`, so future audits miss less.
5. Write a handoff focused on next steps, design cleanup, simplification, performance, and portability.
6. Continue by compiling the Hugo docs site including API docs, checking interface divergence across C++, Python, and MATLAB, and pushing the docs toward a more consistent flow.

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
- `ctest --test-dir build -C Release --output-on-failure -j 4` is green on this machine: `70/70` tests passed, with the expected 5 CUDA/Metal tests skipped.
- There was a real Windows failure earlier in the session:
  - `test_fast_pam_adversarial`
  - `unit_test_clustering_algorithms`
- Root cause: `Problem::assignClusters()` could run in parallel while `distByInd()` was still lazily materialising distance entries. Different tasks could race on the same packed `(i,j)` / `(j,i)` slot.
- Fix applied:
  - `dtwc/Problem.cpp`: force serial `assignClusters()` when the distance matrix is not fully materialised yet; keep the normal parallel path once `fillDistanceMatrix()` has completed.
  - `tests/unit/unit_test_clustering_algorithms.cpp`: prefill the matrix before calling `assignClusters()`.
  - `tests/unit/adversarial/test_fast_pam_adversarial.cpp`: prefill the matrix before the random-medoid comparison path.

The important current truth is that the repository is green on this Windows Release tree, but the earlier failure exposed a real thread-safety hole in the lazy distance path.

### Docs / API build state on this Windows tree

- Local prerequisites installed during the session:
  - Hugo already present
  - installed `GoLang.Go` so `hugo mod get -u` works locally
  - installed `DimitriVanHeesch.Doxygen`
  - installed `Graphviz.Graphviz`
- Local docs build now works:
  - `docs/`: `hugo mod get -u`
  - `docs/`: `hugo --minify --baseURL "https://battery-intelligence-lab.github.io/dtw-cpp/"`
  - repo root: `doxygen docs/Doxyfile`
- Generated Doxygen HTML was staged into `docs/public/Doxygen` to match the CI workflow layout.
- Important machine-safety fix:
  - `NUM_PROC_THREADS` was already `1`
  - `DOT_NUM_THREADS` was `0`, which let Graphviz use all available cores
  - updated `docs/Doxyfile` to `DOT_NUM_THREADS = 2` so future local API builds do not freeze the machine
- Doxygen now indexes:
  - C++ core headers
  - checked-in Python wrapper sources
  - MATLAB package sources
- Graph pages confirmed in the generated output:
  - `classdtwc_1_1_problem__coll__graph.svg`
  - `classdtwcpp_1_1__clustering_1_1_d_t_w_clustering__coll__graph.svg`
  - `namespacedtwc_1_1distance.html`
  - `namespacedtwcpp_1_1distance.html`

## Audit findings

### P0 / P1-level blockers

1. **The lazy distance path had a real parallel race.**
   - It is fixed in this session, and the full Windows Release suite is now green.
   - The important lesson is architectural: any future caller that expects `assignClusters()` to compute distances on demand inside a parallel region is relying on unsafe behaviour.
   - The safe contract is now explicit: compute lazily only in serial, or call `fillDistanceMatrix()` first.

2. **User-facing docs claim YAML/CLI precedence that the code explicitly contradicts.**
   - Docs: `docs/content/getting-started/configuration.md:147` says CLI flags take precedence over YAML.
   - Code: `dtwc/dtwc_cl.cpp:299` has a TODO stating YAML currently overrides CLI whenever the key exists.
   - This is a real behaviour/documentation mismatch, not a cosmetic note.

3. **A test that looks like FastPAM coverage is actually exercising the legacy Lloyd path.**
   - `tests/unit/adversarial/test_fast_pam_adversarial.cpp:57` sets `prob.method = Method::Kmedoids`.
   - `tests/unit/algorithms/unit_test_fast_pam.cpp:113` already documents that `cluster_by_kMedoidsLloyd()` is the old Lloyd path and separately tests real `fast_pam(...)`.
   - Result: the adversarial test name creates false confidence about FastPAM correctness coverage.

### Structural / design findings

4. **`.claude/design.md` was stale after the Phase 3/4 kernel unification.**
   - At the start of this session it still said "Separate functions per variant" and "Soft-DTW is a separate algorithm".
   - The shipped architecture now routes Standard / ADTW / WDTW / DDTW / Soft-DTW / AROW / missing-data paths through `dtw_kernel_*` plus policy cells/costs (`dtwc/core/dtw_kernel.hpp`, `dtwc/core/dtw_dispatch.cpp`).
   - This was an internal-doc drift risk: future refactors could be argued from a design doc that no longer described the code.

5. **`Problem` still mixes pure clustering, reporting, and file I/O side effects.**
   - `Problem` has a `verbose` flag (`dtwc/Problem.hpp:141`), but `cluster_by_kMedoidsLloyd()` prints unconditionally (`dtwc/Problem.cpp:635`, `:680`, `:704`) and writes result files (`dtwc/Problem.cpp:657`, `:705`; `dtwc/Problem_IO.cpp:34`, `:184`).
   - Tests compensate by redirecting outputs to temp paths rather than the API being clean by construction.
   - This is a design simplification target: pure algorithm result first, optional reporting second.

6. **Algorithm dispatch is fragmented across multiple naming systems.**
   - Core enum: `dtwc/enums/Method.hpp:13-15` only has `Kmedoids` and `MIP`.
   - CLI has separate string-level dispatch for `pam`, `clara`, `kmedoids`, `hierarchical` (`dtwc/dtwc_cl.cpp:733-846`).
   - Python and MATLAB expose `fast_pam` / `fast_clara` directly as free functions rather than going through the same method abstraction.
   - This is not a correctness bug today, but it increases naming drift and duplicated test/docs burden.

7. **String/enum parsing is duplicated in multiple places.**
   - CLI has repeated `CheckedTransformer` maps plus a second YAML alias-normalisation block (`dtwc/dtwc_cl.cpp:151-273`, `:369-390`).
   - MATLAB has its own parsers for missing strategy, distance strategy, linkage, and DTW variant (`bindings/matlab/dtwc_mex.cpp:225-246`, `:364-377`).
   - Python mirrors the same enum surface manually (`python/src/_dtwcpp_core.cpp:51-95`).
   - This is a good dedup target: centralise enum/string mapping in core, then reuse everywhere.

8. **The cross-language public surface has drifted even where functionality is broadly similar.**
   - CLI is config-first and already handles YAML/TOML.
   - C++ is `Problem`-centric but examples had drifted toward older `DataLoader` + Lloyd flows.
   - Python exposed many pairwise distances at the package root (`dtwcpp.dtw_distance`, `soft_dtw_distance`, etc.).
   - MATLAB similarly favored root package functions (`dtwc.dtw_distance`, etc.).
   - Result: examples made the bindings feel more different than the actual implementation warrants.

9. **The repository lacked one obvious cross-language "distance namespace" story.**
   - This session added and standardized facades:
     - C++: `dtwc::distance::*`
     - Python: `dtwcpp.distance.*`
     - MATLAB: `dtwc.distance.*`
   - A generic dispatcher entry point was also added in Python/MATLAB:
     - `dtwcpp.distance.dtw(..., variant="soft_dtw", ...)`
     - `dtwc.distance.dtw(..., 'Variant', 'soft_dtw', ...)`
   - Python and MATLAB root-level distance aliases were then removed as part of an intentional breaking change.

10. **Doxygen/Hugo output needed a more explicit parity map instead of raw API dumps alone.**
   - Added `docs/content/api/interface-parity.md`.
   - The page now:
     - states the intended shared flow
     - links to the generated namespace/class pages
     - embeds a cross-language interface-flow diagram
     - embeds the generated Doxygen collaboration graphs for C++ `Problem` and Python `DTWClustering`
   - This is the clearest current entry point for future API convergence work.

## Skill changes made

Updated `.claude/skills/check-code-quality.md` to make future audits materially stricter and more accurate:

1. **Removed the hardcoded repo path.**
   - Uses `git rev-parse --show-toplevel` with a `pwd -P` fallback.

2. **Made the build/test step multi-config aware.**
   - Detects `CMAKE_CONFIGURATION_TYPES` and uses `--config Release` / `ctest -C Release` when needed.

3. **Added serial reruns of failed tests.**
   - The skill now distinguishes:
     - deterministic failing/crashing tests -> `FAIL`
     - parallel-only failures / flakes -> `WARN`

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

Also corrected internal architecture notes:
- `.claude/design.md` now describes the unified kernel plus policy-based DTW architecture instead of the pre-Phase-3 split.
- `.claude/LESSONS.md` now says DTW variants are distinct recurrence/cost policies, not mere metric swaps, without implying they still require totally separate loops/functions.

## Docs / interface changes made

1. **Added additive `distance` namespaces in all three library surfaces.**
1. **Added `distance` namespaces in all three library surfaces and made them the documented pairwise API.**
   - `dtwc/distance.hpp`
   - `python/dtwcpp/distance.py`
   - `bindings/matlab/+dtwc/+distance/*`
   - Python root exports such as `dtwcpp.dtw_distance(...)` were removed.
   - MATLAB root wrapper files such as `+dtwc/dtw_distance.m` were removed.

2. **Updated the C++ umbrella header to expose the new namespace.**
   - `dtwc/dtwc.hpp` now includes `distance.hpp`.

3. **Updated the docs content to reflect the newer preferred flow.**
   - `docs/content/getting-started/examples.md`
   - `docs/content/getting-started/python.md`
   - `docs/content/getting-started/matlab.md`
   - `docs/content/getting-started/configuration.md`
   - `docs/content/getting-started/supported-data.md`

4. **Expanded Doxygen coverage to checked-in wrapper sources.**
   - `docs/Doxyfile` now includes `python/` and `bindings/matlab/`.
   - Heavy graph settings remained bounded:
     - kept class/collaboration graphs
     - disabled call/caller graphs
     - capped Graphviz worker count

5. **Fixed the broken extra stylesheet reference.**
   - Added `docs/custom.css` so Doxygen no longer points at a missing file.

## Recommended next steps

Priority order:

1. **Resolve the YAML precedence bug before editing docs again.**
   - Either:
     - implement true CLI-wins precedence using `app["--flag"]->count()`, or
     - explicitly document current YAML-wins behaviour and stop claiming otherwise.
   - The code TODO already points at the right fix surface.

2. **Rename or rewrite `test_fast_pam_adversarial.cpp`.**
   - Option A: rename it to legacy/k-medoids/Lloyd terminology.
   - Option B: keep the file name and switch it to actual `fast_pam(...)`.
   - Best outcome: do both categories explicitly, with separate files.

3. **Split pure clustering from reporting side effects.**
   - Legacy Lloyd path should return a `ClusteringResult`-style object and only write CSVs / print via an explicit reporting layer.
   - This simplifies tests, wrappers, CLI integration, and future method unification.

4. **Create one shared enum/string conversion layer.**
   - Suggested target: `dtwc/core/enum_strings.hpp` or similar.
   - Feed it into:
     - CLI transformers / YAML normalisation
     - MATLAB MEX parsers
     - any future docs generation / help text

5. **Unify algorithm naming at the API boundary.**
   - Consider introducing a richer algorithm enum or dispatcher that can represent:
     - FastPAM
     - legacy Lloyd k-medoids
     - FastCLARA
     - CLARANS
     - hierarchical
     - MIP
   - Then keep `Problem.method`, CLI, docs, and bindings aligned to one vocabulary.

6. **Keep the lazy distance path contract explicit.**
   - `distByInd()` is safe as a serial convenience path, not as a parallel fill strategy.
   - If future refactors want fully parallel on-demand assignment, that requires a proper atomic/locked materialisation design rather than assuming packed-matrix writes are benign.

7. **Introduce one shared settings/config object across library bindings.**
   - User expectation is now explicit: a similar flow in C++, Python, and MATLAB:
     - define settings / load YAML
     - load data
     - construct `Problem`
     - run clustering
     - inspect scores / outputs
   - Recommended direction:
     - add a small `Settings` / `ProblemConfig` object in core
     - add YAML loading once in core or a thin shared adapter
     - project that consistently into Python and MATLAB rather than inventing binding-specific loaders

8. **Decide the long-term status of generic vs specific distance entry points.**
   - Current additive approach is good:
     - generic dispatcher: `distance.dtw(..., variant=...)`
     - direct fast/specific entry points: `distance.soft_dtw(...)`, `distance.wdtw(...)`, etc.
   - Still open:
     - whether to expose lower-level fast-specialized names such as `dtw_L`
   - Recommendation: keep the namespace-only public surface for Python/MATLAB and decide separately whether C++ should also hide or formally deprecate low-level free functions.

## Design simplifications that should preserve performance / portability

1. **Do not undo the unified DTW kernel family.**
   - The current policy-based `dtw_kernel_*` setup is the correct simplification: less code, fewer dispatch bugs, no measured perf loss.

2. **Simplify at the orchestration layer, not the hot path.**
   - Good targets:
     - string parsing / enum conversions
     - algorithm naming and dispatch
     - reporting / file I/O
     - config/settings loading
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

> Fix the YAML-vs-CLI precedence mismatch, then rename/split the misleading FastPAM vs Lloyd adversarial tests so the file names, docs, and actual implementation paths line up.

If continuing the docs/API convergence thread instead, the next best follow-on is:

> Introduce one shared settings/config object for C++/Python/MATLAB, then migrate examples so each binding shows the same high-level flow: settings -> data -> Problem -> algorithm -> scores/results.
