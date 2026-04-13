---
name: handoff-2026-04-13-phase3-complete
description: Session handoff — Phase 3 complete (4 parts: resolver + AROW + Soft-DTW + MV AROW). Audit follow-ups B/F/M landed. J blocked (documented).
type: project
---

# Session Handoff — 2026-04-12 → 2026-04-13 — Phase 3 complete + audit follow-ups

Branch: **Claude**. Started on top of `eb7b572`. **15 commits** landed this session.

## Session intent

1. Continue from prior handoff (`handoff-2026-04-12-warping-unification.md`). Uncommitted state was Phase 2 (warping_missing fold) + `check-code-quality` skill.
2. User: "commit before continuing" → then "ultrathink, then Phase 3 refactor."
3. User constraints on Phase 3: no runtime dispatch in inner loop, future-proof, ≤10% perf budget.
4. User: "continue until all phases finish" → Phase 3 parts 1-4 + audit follow-ups.

## Commits (11)

| # | SHA | Title |
|---|-----|-------|
| 1 | `38fa694` | refactor(warping): fold warping_missing into unified DTW kernel (Phase 2) |
| 2 | `94fdd48` | tooling: add check-code-quality skill |
| 3 | `4d92881` | refactor(problem): unify rebind_dtw_fn via templated resolver (Phase 3.1) |
| 4 | `cbdb942` | cleanup: drop dead includes + collapse make_wdtw specialisations |
| 5 | `d595035` | refactor(arow): fold banded AROW into unified kernel (Phase 3.2) |
| 6 | `ee0f798` | refactor(soft-dtw): fold into unified kernel via SoftCell (Phase 3.3) |
| 7 | `08a2b8d` | feat(arow): multivariate AROW via per-channel-skip cost (Phase 3.4) |
| 8 | `4c54089` | test(mip): add Benders decomposition coverage + ignore CTest scratch |
| 9 | `d4c066c` | refactor(io): LoadOptions struct for load_folder/load_batch_file |
| 10 | `85f79d5` | tooling: add .clang-tidy config + amortised-alloc comment |
| 11 | `21c7976` | docs: lessons + TODO notes from Phase 3 + audit session |
| 12 | `7f994ee` | build: add DTWC_REPRODUCIBLE_BUILD option (-ffile-prefix-map) |
| 13 | `375a23e` | refactor(kernel): AROWCell NaN check via std::isnan |
| 14 | `5a0987c` | test(io): end-to-end roundtrip through Problem::{write,read}DistanceMatrix |
| 15 | `90d4488` | cleanup(mmap): drop redundant c_str() on llfio error messages |

## Accomplishments

### Phase 3 — dispatch + kernel family unification

**3.1 — `resolve_dtw_fn<T>` templated resolver** ([dtw_dispatch.hpp](../../dtwc/core/dtw_dispatch.hpp) / [.cpp](../../dtwc/core/dtw_dispatch.cpp)). `Problem::rebind_dtw_fn`'s 130-line nested switch on `{missing_strategy, variant, ndim}` collapsed to 2 lines calling a templated resolver. Explicit instantiations for `T = data_t` (f64) and `T = float` (f32) live in the .cpp. Resolution runs once at rebind; the returned `std::function` body is the existing templated kernel call — zero per-cell dispatch overhead.

**Bug fix (shipped with 3.1):** `Problem::dtw_function_f32()` was hardwired to Standard DTW regardless of `variant_params.variant` / `missing_strategy`. Primary user is `fast_clara`'s chunked-Parquet assignment — callers with variant = WDTW/ADTW/DDTW/SoftDTW or missing-data strategies silently got Standard DTW. Regression tests in `unit_test_mv_variants.cpp` `[f32]` tag.

**3.2 — AROW fold** ([dtw_kernel.hpp AROWCell](../../dtwc/core/dtw_kernel.hpp), [dtw_cost.hpp SpanAROWL1Cost](../../dtwc/core/dtw_cost.hpp)). Extended Cell contract with `seed(cost, i, j)` so AROW can override the (0,0) seed for NaN inputs without breaking StandardCell/ADTWCell (which default to `return cost`). NaN-propagating Cost functor signals missing pairs; AROWCell carries diagonal predecessor. Cross-validated bit-for-bit against legacy `dtwAROW_banded` on {no-NaN, interior-NaN, leading-NaN, trailing-NaN, all-NaN} × bands {1..4} before migration.

**3.3 — Soft-DTW fold** ([dtw_kernel.hpp SoftCell](../../dtwc/core/dtw_kernel.hpp)). Log-sum-exp softmin with max-subtract stabilisation. Sentinel-aware — out-of-bounds predecessors (`maxValue`) excluded from LSE, so first-row/column reduce to `predecessor + cost` (hard accumulation). Cross-validated against legacy `soft_dtw()` on equal/different-length/identical/swap-symmetric inputs × gamma {0.1..10.0}.

**3.4 — MV AROW first-class path** ([SpanMVAROWL1Cost](../../dtwc/core/dtw_cost.hpp)). Previously `make_arow` with ndim > 1 silently flattened to scalar. Added per-channel-skip cost that signals "missing pair" (NaN) only when zero channels are comparable; otherwise sums per-channel L1 over comparable channels. Reduces exactly to scalar AROW when ndim = 1 (verified).

**Perf:** Phase 1 gave 1.54-2.83× speedup on banded paths. Phase 3 added zero perf cost — `BM_dtwBanded/1000/50` 146 μs pre-3.1 → 145 μs post-3.4 (−0.7%, within noise).

### Audit follow-ups (from Phase 3.1 handoff audit WARN items)

**B — MIP Benders coverage** ([unit_test_mip.cpp](../../tests/unit/unit_test_mip.cpp)). Three new tests in `[mip][highs][benders]`: forced Benders produces valid clustering, cost matches direct HiGHS on N=12 k=3, auto-dispatch path exercised. Benders only triggers at N > 200 by default, so every existing MIP test (N ≤ 8) missed `MIP_clustering_byBenders`. Benders warm-starts via `cluster_by_kMedoidsLloyd` → writes medoid CSVs → tests route `output_folder` to `std::filesystem::temp_directory_path()`. Same fix pattern as the Phase 2 stray-CSV leak fix.

**F — LoadOptions struct** ([fileOperations.hpp](../../dtwc/fileOperations.hpp)). `load_folder` / `load_batch_file` had 6 positional parameters each. New `LoadOptions` struct bundles Ndata/verbose/start_row/start_col/delimiter. New overload `load_folder(path, const LoadOptions&)` is the primary API; positional overloads retained as backwards-compat shims. `DataLoader::load()` migrated to the struct form. Tests still use the positional form — all pass unchanged.

**M — `.clang-tidy` config** ([.clang-tidy](../../.clang-tidy)) + amortised-alloc comment ([clarans.cpp:73-83](../../dtwc/algorithms/clarans.cpp)). Conservative starter set: bugprone-*, performance-*, select modernize-*/readability-*. HeaderFilterRegex limits to `dtwc/` + `tests/` (third-party suppressed). Naming enforced via CheckOptions. Note: Homebrew clang-tidy on macOS hits an Apple SDK include-path issue when processing `compile_commands.json` — separate toolchain concern.

### Tests added this session

- `unit_test_mv_missing.cpp` — Phase 2 banded MV regression (prior session)
- `unit_test_mv_variants.cpp [f32]` — 3 cases, 3 assertions (ADTW/WDTW/ZeroCost via `dtw_function_f32()`)
- `unit_test_arow_dtw.cpp [phase3]` — 5 cases, 17 assertions (cross-validation vs `dtwAROW_banded`)
- `unit_test_arow_dtw.cpp [mv][phase3]` — 4 cases, 9 assertions (MV AROW)
- `unit_test_soft_dtw.cpp [phase3]` — 4 cases, 14 assertions (cross-validation vs `soft_dtw()`)
- `unit_test_mip.cpp [benders]` — 3 cases, 16 assertions

All 70/70 ctest binaries pass throughout. Test *count* unchanged (each file = one ctest entry); assertion count grew by ~59.

## Architecture after Phase 3

**One templated core** — `dtw_kernel_{full,linear,banded}<T, Cost, Cell>` in [dtw_kernel.hpp](../../dtwc/core/dtw_kernel.hpp).

**Orthogonal Cost policies** in [dtw_cost.hpp](../../dtwc/core/dtw_cost.hpp): `SpanL1Cost`, `SpanSquaredL2Cost`, `SpanWeightedL1Cost` (WDTW), `SpanNanAwareL1Cost` (ZeroCost missing), `SpanAROWL1Cost` (AROW missing) + MV counterparts.

**Orthogonal Cell policies**: `StandardCell`, `ADTWCell<T>{penalty}`, `SoftCell<T>{gamma}`, `AROWCell`. Contract: `combine(diag, up, left, cost, i, j)` + `seed(cost, i, j)`.

**Dispatch** in [dtw_dispatch.cpp](../../dtwc/core/dtw_dispatch.cpp) — one switch on `missing_strategy` (handled first), one switch on `variant`. Per-T uniform treatment; explicit instantiations for f64 + f32.

**Adding a new variant:** 1 Cost policy + 1 Cell policy + 1 `make_*<T>` helper + 1 switch arm in `resolve_dtw_fn`. No UV/MV duplication.

## Decisions / rationale

1. **Cross-validation before migration.** Every policy fold (3.2 AROW, 3.3 SoftDTW) had a pre-commit test that ran both impls on representative inputs and required bit-for-bit agreement. Caught zero regressions because every migration was gated. Added to LESSONS.md as a durable pattern.

2. **`seed()` extension over Cell-contract bool flag.** AROW's (0,0) with missing needs to return 0, not NaN. Options: (a) add `is_missing_pair` bool to `combine`, (b) encode via NaN cost + Cell handles it. Chose (b) + added `seed()` — one new method, existing cells get a trivial default. Additive, no breakage.

3. **MV AROW per-channel skip, not scalar lift.** Direct scalar→MV interpretation ("any channel missing → diagonal carry") discards usable per-channel info. Chose per-channel-skip semantics (trigger AROW only when ZERO comparable channels) — consistent with `SpanMVNanAwareL1Cost` and reduces exactly to scalar when ndim=1. Documented in code + CHANGELOG because it's a design choice without a specific paper reference.

4. **F32 WDTW routes through `wdtwBanded<float>(..., g)`.** f64 uses Problem-local `wdtw_weights_cache_` (lock-free, serial-populate). f32 uses `detail::cached_wdtw_weights<float>` (thread_local). Slight perf asymmetry, but f32 WDTW is rare and correctness > perf for a rare path. If f32 WDTW ever becomes hot, add an f32 cache (~10 LOC).

5. **LoadOptions struct is additive, positional overloads retained.** New primary API is `load_folder(path, const LoadOptions& = {})`; positional overloads delegate to it. Backwards compatible — existing callers don't change.

6. **Benders tests route output to `temp_directory_path()`.** Same fix pattern as fast_pam / fast_clara CSV leak (Phase 2). The bug would have repeated for anyone writing MIP tests without this.

## Open questions / known blockers

1. **J — Python wheel cross-language parity test** is BLOCKED. `uv pip install -e .` fails at llfio's quickcpplib sub-CMake configure because it can't find `ninja`. scikit-build-core injects ninja path for the top-level build, but the sub-build is sandboxed and doesn't inherit it. Workarounds:
   - **(a) `brew install ninja` system-wide** before running the build — simplest fix for local devs, doesn't help CI.
   - **(b) Make llfio truly optional** via `DTWC_ENABLE_MMAP`. Would need to guard `MmapDistanceMatrix` with `#ifdef DTWC_HAS_MMAP` — touches `Problem::distMat` `std::variant`, `visit_distmat`, CUDA/Metal backends. Non-trivial refactor.
   - **(c) File upstream issue** against quickcpplib for `CMAKE_MAKE_PROGRAM` propagation. Slow path — no guarantee of fix.
   
   Documented in `.claude/TODO.md` under Blocked. User can pick workaround later.

2. **Problem.hpp dtw_fn_t type declaration vs comment.** `using dtw_fn_t = std::function<data_t(...)>` returns `data_t`; the adjacent comment says "Both return double (distance precision is always double)." Because `data_t = double` these coincide today. If `data_t` ever changes, the resolver's `std::function<double(...)>` return type won't match. Latent, not a current bug. Would be a trivial fix: change `dtw_fn_t` to return `double` directly.

3. **Clang-tidy + Apple SDK**: Homebrew clang-tidy can't find `<string>` when processing `compile_commands.json` generated by AppleClang. Config file itself is fine; this is a toolchain include-path issue. Would need `--extra-arg=-isysroot=$(xcrun --show-sdk-path)` or similar in the run command. Not blocking; IDE clangd (Apple-built) works fine with the same `.clang-tidy`.

4. **Hardening item from audit M (reproducible-build flags: `SOURCE_DATE_EPOCH`, `-ffile-prefix-map`)** — not addressed. Polish for Debian-style packaging; skipped this session.

5. **`benchmarks/results/_autorun/`** — from prior session's handoff, still untracked. Undecided whether to commit or gitignore.

## Non-obvious nuggets

- **Phase 3.1 uncovered a second instance of the same silent-dispatch bug class as Phase 1.** `dtw_runtime()` silently ignoring variant was fixed in Phase 1; `dtw_function_f32()` silently ignoring variant AND missing_strategy was fixed in Phase 3.1. Both discovered only when writing unified dispatch infrastructure. Dual-type APIs NEED parity tests. Added as a durable lesson.

- **SoftCell's sentinel handling was the free win.** Existing `soft_dtw()` has an explicit hard-accumulation loop for first row/column AND a separate softmin loop for interior. SoftCell with `maxValue`-skip logic makes boundaries fall out automatically — softmin of a single valid predecessor is that predecessor. Zero special-casing in the kernel. Didn't expect this; the kernel's sentinel convention was already doing 90% of the work.

- **The `cmake.define` / `tool.scikit-build` approach to fixing J got much further than expected** — scikit-build-core DID accept the `CMAKE_MAKE_PROGRAM` hint, passed it to the top-level CMake, and llfio's direct CMake accepted it. It's only quickcpplib's spawn-a-fresh-CMake pattern that sidesteps the whole mechanism. Upstream fix probably belongs in quickcpplib's `QuickCppLibUtils.cmake:79`.

## How to continue next session

1. **J workaround**: try `brew install ninja` + `uv pip install -e .` — lowest-effort path to measure the ≤10% Python/MATLAB parity claim. Independent of any C++ work.

2. **Run `.clang-tidy`** on the codebase with `run-clang-tidy -p build -header-filter='(dtwc|tests)/'` once the Apple SDK issue is worked around (pass `--extra-arg=-isysroot=$(xcrun --show-sdk-path)`). Expect a few dozen bugprone hits on existing code — fix or NOLINT them.

3. **Phase 4 candidates (architectural polish)** — none of these are flagged as defects; all are nice-to-have:
   - Fold `warping_missing_arow.hpp` standalone API to use `dtw_kernel_banded<T, SpanAROWL1Cost<T>, AROWCell>` internally (currently Problem-level uses it; standalone `dtwAROW_banded` still has its own impl). Would remove ~300 LOC.
   - Fold `soft_dtw.hpp` standalone API similarly. Gradient path stays separate — has its own forward/backward matrix ownership.
   - `fast_pam.cpp:46 compute_nearest_and_second` has 6 args — internal helper, not user-facing; refactor value is low.

4. **Reproducible-build flags** (audit hardening item M residual): `-DCMAKE_CXX_FLAGS_INIT="-ffile-prefix-map=$(pwd)=."` + respect `SOURCE_DATE_EPOCH` if set. Ten-line CMake change.

5. **TODO.md cleanup**: the file is from 2026-04-08 and hasn't tracked the Phase 3 work at all. Consider either pruning completed items or reorganising by status. Low priority.
