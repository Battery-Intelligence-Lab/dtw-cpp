---
name: handoff-2026-04-13-python-wheel-ninja-blocker
description: Standalone problem statement for the `uv pip install -e .` ninja-propagation blocker — approaches to investigate and recommended direction.
type: project
---

# Python Wheel Build — `ninja` Propagation Blocker

Audience: Codex, or future me, coming in cold. Self-contained.

## Symptom

```sh
$ uv pip install -e .
[…]
CMake Error at .../build/quickcpplib/repo/cmakelib/QuickCppLibUtils.cmake:79 (message):
  FATAL: Configure download, build and install of outcome with … stderr was:
  no such file or directory '.../ninja' --version
```

Blocks:
- `tests/integration/test_cross_language.py` — the wheel-parity test that verifies Python bindings ≡ C++ ≡ MATLAB within numerical tolerance. All three frontends share `core::dtw_kernel_*` + Cost/Cell policies (after Phase 3 kernel unification), so agreement should be automatic — but without this test, a binding-layer regression could slip through unnoticed.

Does **not** block: plain `cmake --build build` from the repo root, because the dev venv has `ninja` on its PATH and the top-level CMake propagates it normally.

## Repro

```sh
cd /Users/engs2321/Desktop/git/dtw-cpp
# Ensure no system-wide ninja: `which ninja` returns empty (or remove from PATH)
uv pip install -e .
# → fails at outcome/quickcpplib sub-CMake configure
```

## Root cause (confirmed)

1. Top-level build chain: `uv pip install -e .` → `scikit-build-core` → top-level CMake → FetchContent(llfio).
2. scikit-build-core installs `ninja` (via pyproject.toml `build-system.requires`) into its isolated `pip-build-env` and prepends its bin dir to PATH for its own CMake invocation. Top-level CMake sees ninja fine.
3. llfio's CMake calls `include(QuickCppLibUtils)`, which provides `download_build_install()`. That function spawns **a fresh child CMake process** (`execute_process(COMMAND "${CMAKE_COMMAND}" .)`) to configure outcome (llfio's dependency) in a separate build dir.
4. The child CMake does NOT receive `-G <generator>` or `-DCMAKE_MAKE_PROGRAM=<path>` — QuickCppLibUtils passes neither. The child auto-selects Ninja on macOS and fails because PATH is not reliably inherited (or the original ninja path was sandbox-scoped to scikit-build-core's parent process only).

Exact spawn site — `build/quickcpplib/repo/cmakelib/QuickCppLibUtils.cmake:258-273`:

```cmake
function(download_build_install)
  cmake_parse_arguments(DBI "" "NAME;DESTINATION;INSTALL_PREFIX;GIT_REPOSITORY;GIT_TAG" "CMAKE_ARGS;EXTERNALPROJECT_ARGS" ${ARGN})
  configure_file("${QuickCppLibCMakePath}/DownloadBuildInstall.cmake.in" "${DBI_DESTINATION}/CMakeLists.txt" @ONLY)
  checked_execute_process("Configure download, build and install of ${DBI_NAME} with ${DBI_CMAKE_ARGS}"
    COMMAND "${CMAKE_COMMAND}" .                      # ← no -G, no -DCMAKE_MAKE_PROGRAM
    WORKING_DIRECTORY "${DBI_DESTINATION}"
  )
  checked_execute_process("Build …"
    COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${DBI_DESTINATION}"
  )
  …
endfunction()
```

## Approaches — ranked

### Option 1 (RECOMMENDED): Patch QuickCppLibUtils inline via FetchContent

Two-line change to the `download_build_install` function: forward the parent generator + make-program to the child. Apply via FetchContent's `PATCH_COMMAND`.

Pros:
- Localised — the patch lives in our repo, no upstream round-trip.
- Fixes CI and local dev in one stroke — no `brew install ninja` required.
- Cheap: ~10 lines of CMake + a tiny `.patch` file.

Cons:
- Brittle on llfio/quickcpplib version bumps if upstream rewrites the function.
- Slightly unusual — patching a FetchContent'd dep is uncommon but supported (`FetchContent_Declare(… PATCH_COMMAND …)`).

Patch sketch (apply to `cmakelib/QuickCppLibUtils.cmake`):

```diff
   checked_execute_process("Configure …"
-    COMMAND "${CMAKE_COMMAND}" .
+    COMMAND "${CMAKE_COMMAND}" .
+    -G "${CMAKE_GENERATOR}"
+    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
     WORKING_DIRECTORY "${DBI_DESTINATION}"
   )
```

Where to apply: in `dtwc/cmake/external/llfio.cmake` (or wherever the `FetchContent_Declare(llfio …)` lives). Use `PATCH_COMMAND git apply <patch-file>` or `${CMAKE_COMMAND} -P apply-patch-script.cmake`.

**Open question for investigation**: does llfio's CMake ever `include()` QuickCppLibUtils from multiple locations (submodule, sibling, installed)? If yes, the patch needs to hit each copy. The llfio src tree has at least 4 copies under `build/`:

```
./build/install/share/cmakelib/QuickCppLibUtils.cmake
./build/quickcpplib/repo/cmakelib/QuickCppLibUtils.cmake
./build/quickcpplib/repo/src/quickcpplib/cmakelib/QuickCppLibUtils.cmake
./build/outcome/repo/src/outcome-build/quickcpplib/repo/cmakelib/QuickCppLibUtils.cmake
```

Patching in FetchContent happens post-download pre-configure, so we target whichever copy is authoritative for the quickcpplib FetchContent step. Verify with CMake trace (`--trace-expand`) if the first patch attempt doesn't fix everything.

### Option 2: `[tool.scikit-build.cmake.env]` — untried

pyproject.toml currently sets `cmake.args` but NOT `cmake.env`. `execute_process` inherits environment variables by default, so setting:

```toml
[tool.scikit-build.cmake.env]
CMAKE_MAKE_PROGRAM = "${...dynamic path to ninja from python ninja package...}"
CMAKE_GENERATOR = "Ninja"
```

… would propagate to the child CMake via env inheritance. CMake reads `CMAKE_GENERATOR` and `CMAKE_MAKE_PROGRAM` from env when not set as cache variables.

**Caveat:** resolving the ninja path at pyproject-parse time is awkward because scikit-build-core might install ninja to a path that's not yet known. Workarounds:
- Use `${NINJA_PATH}` where `NINJA_PATH` is set by a pre-configure shell step.
- Use a top-level CMake shim: `find_program(NINJA_EXE ninja)` then `set(ENV{CMAKE_MAKE_PROGRAM} "${NINJA_EXE}")` before llfio FetchContent runs. If the parent CMake's env is inherited by the child, this fixes it without pyproject gymnastics.

This approach is less invasive than patching but needs verification that env really does reach the grandchild CMake process (parent Python → scikit-build-core's CMake → llfio's CMake → quickcpplib's `execute_process` CMake).

### Option 3: Use Python's `ninja` package path explicitly

`pip install ninja` ships `ninja.BIN_DIR` in Python. Before llfio's FetchContent, run:

```cmake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import ninja, os; print(os.path.join(ninja.BIN_DIR, 'ninja'))"
  OUTPUT_VARIABLE NINJA_FROM_PYTHON
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(ENV{CMAKE_MAKE_PROGRAM} "${NINJA_FROM_PYTHON}")
set(ENV{PATH} "${ninja_bin_dir}:$ENV{PATH}")
```

Conceptually a cleaner variant of Option 2 — solves the "where is ninja" problem by asking Python. Still relies on env inheritance reaching the grandchild.

### Option 4: Bundle ninja ourselves

License check: **ninja is Apache 2.0** (https://github.com/ninja-build/ninja/blob/master/COPYING). Compatible with our BSD-3-Clause. quickcpplib license is irrelevant here — we'd bundle *ninja*, not quickcpplib.

But: multi-platform (linux-x64, linux-aarch64, darwin-x64, darwin-arm64, win-x64) × CI matrix. Heavy, and Python's `ninja` package already solves this. Don't.

### Option 5: Force non-Ninja generator for sub-builds

Set `CMAKE_GENERATOR=Unix Makefiles` in the environment. `make` ships with Xcode Command Line Tools on macOS and base-installs on Linux. Sub-CMakes would pick makefiles. No ninja needed.

Risk: slower builds (Ninja is 10-20% faster). Also, on Windows, `make` isn't stock — this fails the Windows CI story. Would need conditional behaviour per-platform.

Acceptable as a CI-only fallback, not a durable fix.

### Option 6: File upstream issue against quickcpplib

Proper long-term fix — the `download_build_install` function should forward the parent generator/make-program by default. Even so, we need a local workaround until upstream releases and llfio updates its quickcpplib pin.

## Recommended direction

**Start with Option 1** (patch via FetchContent PATCH_COMMAND). Fallback to **Option 2+3 combined** (env vars injected from top-level CMake via `ninja` Python package) if patching the fetched source turns out to be fragile.

Also file **Option 6** in parallel — upstream fix benefits everyone and replaces our patch over time.

## Success criterion

Passes:

```sh
# Clean environment, no system ninja
uv pip install -e .
uv run pytest tests/integration/test_cross_language.py -v
```

On both macOS (arm64) and Linux (x64). Windows is bonus, not required for this fix.

## Constraints

- **No system-wide `brew install ninja` as the expected solution** — CI runners won't have it, and it's sidestepping the root cause.
- **Keep llfio / quickcpplib / outcome as FetchContent** — not vendored. We track upstream.
- **No new optional dependency flag** unless it makes llfio fully optional (the `DTWC_ENABLE_MMAP` refactor referenced in TODO, which is a separate larger effort and non-trivial because `MmapDistanceMatrix` lives in `Problem::distMat`'s `std::variant`).

## Files likely to touch

- `dtwc/cmake/external/llfio.cmake` (or wherever `FetchContent_Declare(llfio …)` is) — add `PATCH_COMMAND` or env-var setup
- Possibly a new `dtwc/cmake/patches/quickcpplib-forward-generator.patch` file
- `pyproject.toml` — if going with Option 2 env-var route
- Probably nothing in `tests/integration/test_cross_language.py` itself — it's presumably written and just can't run

## Out of scope

- MATLAB-side wheel parity (`dtwc_mex` is built separately; its parity test would be a different integration file)
- Windows + MSVC parity — own separate blocker per TODO.md
