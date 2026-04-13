---
name: handoff-2026-04-13-phase4-wheel-fix
description: Session handoff — Phase 4 standalone-API fold + Python wheel ninja-propagation blocker resolved. All prior deferred items cleared except upstream nudge to quickcpplib.
type: project
---

# Session Handoff — 2026-04-13 (evening) — Phase 4 + wheel unblock

Branch: **Claude**. Picked up immediately after `handoff-2026-04-13-phase3-complete.md`
(`90d4488` — "cleanup(mmap): drop redundant c_str() on llfio error messages"). **5 new
commits** this session, bringing the branch to 20 commits ahead of `main`.

## Session intent

Prior handoff closed at a "natural finish line" with two items marked deferred:
- Phase 4 standalone API fold (~300 LOC removal, "zero correctness/perf benefit")
- J — Python wheel cross-language parity (blocked on ninja propagation)

User first asked to prune TODO.md, then asked to "continue with the deferred items."
Both deferred items landed this session. The wheel blocker turned out to be harder than
the prior handoff estimated — it took three iterations to converge on the right fix.

## Commits (5)

| # | SHA | Title |
|---|-----|-------|
| 1 | `709ccd5` | docs: prune TODO.md to reflect Phase 3 + audit completion |
| 2 | `ed826c3` | refactor(arow): fold standalone dtwAROW_* public API into unified kernel |
| 3 | `f440bfc` | refactor(soft-dtw): fold standalone soft_dtw() forward pass into unified kernel |
| 4 | `78dd019` | docs(todo): mark Phase 4 standalone API fold complete |
| 5 | `7f79c55` | build(python-wheel): patch quickcpplib to forward make-program across CMake hops |

70/70 ctest green at every step. Working tree clean.

## Accomplishments

### Phase 4 — standalone API folds

**AROW fold** (`ed826c3`). `warping_missing_arow.hpp` (`dtwAROW`, `dtwAROW_L`, `dtwAROW_banded` × span / vector / pointer+size overloads) now delegates to `core::dtw_kernel_{full,linear,banded}<T, SpanAROW{L1,SquaredL2}Cost<T>, AROWCell>`. The hand-rolled `detail::dtwAROW_*_impl` helpers (three full functions × two metric dispatch paths, ~260 LOC) are gone. File shrinks 459 → 214 LOC. Behaviour unchanged — the legacy banded impl used a simpler `ceil/floor` band-bounds calc vs the unified kernel's `round-100` variant, but the legacy's own (now-removed) comment said the difference is harmless because out-of-band cells are sentinel-valued.

**Soft-DTW forward fold** (`f440bfc`). `soft_dtw(x, y, gamma)` forward pass delegates to `core::dtw_kernel_full<T, SpanL1Cost<T>, SoftCell<T>>`. `softmin_gamma()` helper is retained because `soft_dtw_gradient()` still uses it. `soft_dtw_gradient()` itself is unchanged — the Cuturi–Blondel backward pass reads the full forward matrix C *and* writes alignment matrix E, so folding it into the kernel would need C as an out-parameter.

Both folds cross-validated via the existing `[phase3]` kernel-vs-legacy tests from the prior session — those tests become tautological now (both sides call the same kernel) but are left in place as a structural regression guard.

### J — Python wheel ninja-propagation blocker resolved

The blocker: `uv pip install -e .` failed at llfio's nested configure step with `.../ninja --version: no such file or directory`. Prior handoff noted three workarounds (a: `brew install ninja` locally; b: make llfio optional via `DTWC_ENABLE_MMAP`; c: file upstream issue). None were taken — the user suggested a fourth: "download, use cmake patches, run it." That became the fix.

**Three iterations to converge**:

1. **First attempt (rejected)**: env-var `PATH` + `CMAKE_GENERATOR` forwarding before `add_subdirectory(llfio)`. Plausible on paper — `execute_process` inherits env by default. **Failed** because scikit-build-core / uv's ninja lives at an ephemeral path (`/.../uv/builds-v0/.tmpXXX/bin/ninja`) that's rewritten per reinstall. Any stale `CMakeCache.txt` in a sub-build pins a dead path; env `PATH` can't override a hardcoded cache value.

2. **Second attempt (child-spawn patch only)**: pre-clone quickcpplib + patch `download_build_install()` to pass `-G "${CMAKE_GENERATOR}" -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}` to the child. **Got past** the child configure but failed at the Build step because `download_build_install` is actually a superbuild: the child runs `ExternalProject_Add(outcome)`, whose inner cmake invocation gets CMAKE_ARGS from a *template string* built at line 344 of `QuickCppLibUtils.cmake`. That template includes `-G` but not `-DCMAKE_MAKE_PROGRAM` — so the **grandchild** CMake reconfigures from scratch with no make-program hint.

3. **Final fix** (`7f79c55`, verified working): patch **both** sites. Child patch on `download_build_install` at the `COMMAND "${CMAKE_COMMAND}" .` line. Grandchild patch on the `cmakeargs` string in `find_quickcpplib_library`. Sentinel comments `DTWC_NINJA_PROPAGATION_PATCH (child)` / `(grandchild)` per patch so a partially-patched file self-heals on the next configure. Pre-clone quickcpplib into `${CMAKE_BINARY_DIR}/quickcpplib/repo` before `add_subdirectory(llfio)` so llfio's bootstrap finds the pre-existing tree and skips its own git clone.

**Verification**: `env -i HOME=$HOME PATH=/Users/engs2321/.local/bin:/usr/bin:/bin uv pip install --reinstall --no-cache -e .` completes on macOS arm64 in ~90s with no system ninja. `import dtwcpp` works in the resulting `.venv-mac`.

### TODO.md pruning (`709ccd5`, `78dd019`)

Collapsed completed Phase 0/2/4 breakdowns to a one-line reverse-chron log. Promoted J blocker + API-fold deferred to explicit sections with rationale. After the two folds + wheel fix landed, the Blocked section emptied — replaced with "Needs upstream nudge" placeholder (quickcpplib PR for make-program forwarding; once upstream fixes it, our sentinel-guarded patch self-retires).

## Architecture after Phase 4

Unified kernel family now covers **every** Standard / ADTW / WDTW / DDTW / Soft-DTW / AROW / ZeroCost-missing / Interpolate-missing call path. The only remaining per-variant bespoke code lives at the `Problem::rebind_dtw_fn` resolver (Phase 3.1 collapsed it to a templated `resolve_dtw_fn<T>`) and the Soft-DTW gradient (Cuturi–Blondel backward needs its own matrix ownership).

The standalone public-API headers (`warping.hpp`, `warping_adtw.hpp`, `warping_wdtw.hpp`, `warping_missing.hpp`, `warping_missing_arow.hpp`, `soft_dtw.hpp`) are now all thin wrappers over `core::dtw_kernel_*`. DDTW (`warping_ddtw.hpp`) is a thin wrapper that preprocesses the derivative and calls `dtwBanded`.

## Decisions / rationale

1. **Env-var forwarding didn't work, so we patched the source**. Not our first choice — patching upstream-fetched code is unusual — but it's the only approach that survives the stale-cache problem. Partial-patched state is handled via per-patch sentinels, so the fix is idempotent and re-entrant.

2. **Pre-clone rather than post-clone patch**. Rationale: llfio's `QuickCppLibBootstrap.cmake` runs *inside* `add_subdirectory(llfio)`, interleaved with the problematic `find_quickcpplib_library(outcome)` call. We can't intercept between clone and use. Pre-cloning into the exact path llfio's bootstrap expects (`${CMAKE_BINARY_DIR}/quickcpplib/repo`) triggers bootstrap's `if(NOT EXISTS)` shortcut and hands us control.

3. **Patch both child and grandchild**. Patching only the child got past the first configure but left the grandchild ExternalProject_Add failing the same way. Two-line diff across two sites is strictly necessary.

4. **Keep Phase 3 cross-validation tests after fold**. They become tautological (both sides are now the same kernel), but removing them loses structural regression coverage — if someone re-introduces a bespoke AROW impl in the future, the tests would fail on divergence. Low-maintenance, so leave them.

## Open questions / known blockers

1. **quickcpplib upstream fix**. File an issue against `ned14/quickcpplib` for `download_build_install` and `find_quickcpplib_library` lack of `-DCMAKE_MAKE_PROGRAM` forwarding. Once upstream lands a fix + llfio pins a new version, our sentinel-based patch will stop matching and self-retire. Not urgent — our fix works indefinitely.

2. **`tests/integration/test_cross_language.py` content unverified**. The blocker is cleared (wheel now builds in clean env) but I didn't actually run the cross-language test end-to-end. The file exists at `tests/integration/test_cross_language.py` — next session should: activate `.venv-mac`, `uv pip install -e .` (should be a rebuild no-op now), `pytest tests/integration/test_cross_language.py -v`, verify Python ≡ C++ numerical parity within whatever tolerance the test asserts.

3. **`unit_test_clustering_algorithms` flaky SIGABRT**. Still reproduces occasionally under parallel ctest, always passes standalone / on retry. Same behaviour as noted in prior handoff — `init::Kmeanspp` appears to have a race. Not touched this session. Low priority.

4. **Windows + MSVC wheel build**. Our fix should work across generators (Ninja / Unix Makefiles / MSBuild / Xcode) because we forward `CMAKE_MAKE_PROGRAM` generically. But Windows has its own existing blocker (Arrow CPM + ExternalProject flag quoting) that's documented separately in `TODO.md`. The ninja propagation fix doesn't interact with that.

## Non-obvious nuggets

- **The stale-cache problem was the critical insight.** Prior handoff's claim "passing `CMAKE_MAKE_PROGRAM` at the top level via `[tool.scikit-build.cmake.define]` does not propagate" was correct but incomplete — the deeper issue is that a fresh `-D` on the top-level cmake invocation doesn't help the sub-CMake, because the sub-CMake has its own persistent `CMakeCache.txt` from prior reinstall cycles with a dead ninja path. Only a fresh `-D` passed to the *sub-CMake* at configure time can override the stale cache. This changes "propagation" from an env-inheritance problem (easy) to a cache-invalidation problem (requires patching the spawn site).

- **uv's ninja path rewrites per reinstall.** Paths like `/Users/engs2321/.cache/uv/builds-v0/.tmpIUhBMa/bin/ninja` → `.tmpEG0EpC/bin/ninja` → `.tmpwvxE5d/bin/ninja` across three reinstalls. That's why simple PATH-forwarding fails: the directory-prefix you forward today is invalid tomorrow, and the sub-CMake's frozen cache points at yesterday's path.

- **llfio's bootstrap has a quiet "already cloned" fast path** (`if(NOT EXISTS "${CTEST_QUICKCPPLIB_CLONE_DIR}/repo/cmakelib")`). That's the hook we use. If the directory exists it skips cloning entirely, giving us a clean pre-provision mechanism without needing to modify llfio.

- **Sentinel-per-patch, not per-file.** Initial draft used one sentinel (`DTWC_NINJA_PROPAGATION_PATCH`) for both patches, which broke recovery from the partially-patched state I created by iterating the fix. Switching to per-patch sentinels (`(child)` / `(grandchild)`) meant the second patch could still apply when the first already landed from an earlier iteration.

- **Standalone AROW fold's `[phase3]` cross-validation tests still pass after fold, but they test tautology now.** Both legacy and kernel paths now hit the same `core::dtw_kernel_*` under the hood. This isn't a problem — the tests serve as structural regression guards, not semantic ones — but worth calling out so nobody assumes they still catch divergence between two independent implementations.

## How to continue next session

1. **Run `tests/integration/test_cross_language.py`** end-to-end. `source .venv-mac/bin/activate && uv pip install -e . && pytest tests/integration/test_cross_language.py -v`. Fix any numerical-tolerance issues. This validates the actual parity claim that motivated unblocking the wheel.

2. **File quickcpplib upstream issue** for make-program forwarding. Link to this repo's `cmake/Dependencies.cmake` as the workaround.

3. **Sweep any remaining `warping_*.hpp` files** for obvious dead code now that the kernels are unified. Examples: `SquaredL2Dist` in `warping.hpp` might have dead import references; `detail::dispatch_metric` might be superseded by `core::dispatch_metric`. Low priority — lint-tier.

4. **Consider gitignoring `.claude/scheduled_tasks.lock`** — appeared this session as an untracked runtime artifact from the scheduled-wakeup plugin. Not staged in any commit.

5. **Flaky `unit_test_clustering_algorithms` SIGABRT** — if it keeps surfacing, dig into `init::Kmeanspp`. Likely a race on some shared state during parallel ctest.

## State at session end

- Branch: `Claude`, 20 commits ahead of `main`.
- Working tree: clean (modulo untracked `.claude/scheduled_tasks.lock` runtime artifact).
- 70/70 ctest passing.
- Wheel build: **working** in clean env (verified via `env -i PATH=/minimal uv pip install`).
- `import dtwcpp` succeeds in `.venv-mac`, API surface intact.
