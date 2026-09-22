# 2026-09-22 — the macOS release archive did not run outside the build machine (X-29)

Branch `design-2.0`, HEAD `ddbc7b6`. Found while implementing X-23 (third-party notices), because
that row named `CMakeLists.txt:290` — the libomp install — as the place the notices obligation
lived. Opening the line showed the block could not execute.

## Machine

Apple M5 Pro, 18 cores, 64 GiB, macOS (Darwin 25.6.0), Apple clang 21.0.0, CMake 4.4.3, Ninja.
`/opt/homebrew/opt/libomp` installed but **not** `brew link`ed (it is keg-only).

## Reproduction — the exact configure line from `release-artifacts.yml:25-31`

```sh
cmake -S . -B build_cirepro -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_HIGHS=ON \
  -DDTWC_ENABLE_LLFIO=OFF -DDTWC_BUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
```

```
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
CMake Error at dtwc/CMakeLists.txt:187 (message):
  OpenMP NOT found — DTWC++ requires OpenMP for parallel execution.
-- Configuring incomplete, errors occurred!
```
`[confirmed]` — configure exit 1. The workflow runs `brew install libomp` but passes no hint, and
libomp is keg-only, so nothing lands in a prefix CMake searches. `macos-unit.yml:20` and
`cuda-mpi-detect.yml:54` avoid this with `brew link --force libomp`; the release and wheel jobs do
neither. Whether GitHub's runner image differs is `[inferred]` — not reproducible here.

## The archive, once configure is given `-DOpenMP_ROOT=/opt/homebrew/opt/libomp`

Same options otherwise; `cmake --build … --target dtwc_cl write-licenses`, then `cpack -C Release`,
then unpack and run:

```
$ otool -L …/dtwc-2.0.0rc1-Darwin-arm64/bin/dtwc_cl
	@rpath/libhighs.1.dylib
	/opt/homebrew/opt/libomp/lib/libomp.dylib
$ otool -l … | grep -c LC_RPATH
0
$ …/bin/dtwc_cl --version
dyld[31745]: Library not loaded: @rpath/libhighs.1.dylib
  Reason: no LC_RPATH's found
real exit=134
```
`[confirmed]`. Two independent defects, one root cause:

1. **No `LC_RPATH`.** HiGHS sets `BUILD_SHARED_LIBS` ON itself
   (`build/_deps/highs-src/CMakeLists.txt:266`), so `dtwc_cl` records
   `@rpath/libhighs.1.dylib`. The only `INSTALL_RPATH` in the project sat inside the dead block
   below, so the installed binary had none and could not start **at all**.
2. **libomp not bundled**, load command absolute.

**Root cause.** `CMakeLists.txt:290` read
`if(APPLE AND OpenMP_CXX_FOUND AND OpenMP_omp_LIBRARY)`. Both conjuncts are false on macOS:

- `OpenMP_omp_LIBRARY` is written **only** at `dtwc/CMakeLists.txt:145`, inside
  `if(… AND WIN32 …)`. AppleClang gives `OpenMP_CXX_LIB_NAMES=libomp`, hence
  `OpenMP_libomp_LIBRARY` (verified in `build/CMakeCache.txt:836,853`).
- `find_package(OpenMP)` runs in `dtwc/CMakeLists.txt:154`, a subdirectory, so `OpenMP_CXX_FOUND`
  does not reach the top-level scope. The configuration summary at `CMakeLists.txt:348` already
  worked around exactly this by reading the target property instead.

## Why no gate caught it

`release-artifacts.yml:41` runs `scripts/smoke_release_archive.py`, which unpacks the archive
outside the checkout and runs the CLI. That is the right shape and still could not catch defect 2:
the build machine has `/opt/homebrew/opt/libomp/lib/libomp.dylib`, so the absolute path resolves
there and nowhere else. It *would* have caught defect 1 — which means the macOS leg of this
workflow has not produced a passing archive, consistent with configure failing first.

## After the fix

`INSTALL_RPATH` unconditional (`@loader_path/../lib`; `$ORIGIN/../lib` on Linux, which had no rule
at all); libomp located by iterating `OpenMP_CXX_LIB_NAMES`; `install_name_tool -change` on the
executable only.

```
$ otool -L …/bin/dtwc_cl
	@rpath/libhighs.1.dylib
	@rpath/libomp.dylib
$ ls …/lib/          → libhighs.1.15.dylib libhighs.1.dylib libhighs.dylib libomp.dylib
$ codesign -vv …/lib/libomp.dylib   → valid on disk
$ …/bin/dtwc_cl --version
2.0.0rc1
exit=0
```
`[confirmed]`.

An intermediate attempt also ran `install_name_tool -id` on the bundled dylib. That left
`libomp.dylib: invalid signature (code or signature have been modified)` and the CLI died with
**exit 137 (SIGKILL) and no dyld message** — Apple Silicon kills an unsigned/invalid image rather
than diagnosing it. The `-id` rewrite is unnecessary: dyld resolves the client's load command, not
the dylib's own identifier.

## Gate discrimination (`[confirmed]`)

`scripts/smoke_release_archive.py` now also checks self-containment and the required notices.
Exercised against both archives:

```
PRE-FIX  escaping deps: ['/opt/homebrew/opt/libomp/lib/libomp.dylib']
POST-FIX escaping deps: []
PRE-FIX raises: archive is missing required notices: [… nanoarrow/LICENSE.txt, … NOTICE.txt]
POST-FIX: passes
```

## Left alone deliberately

`CPACK_COMPONENTS_ALL runtime` (`CMakeLists.txt:330`) is inert because
`CPACK_ARCHIVE_COMPONENT_INSTALL` is OFF, so the archive carries HiGHS's headers, CMake package
files and pkgconfig — 332 entries. **Do not "fix" this by enabling component install**: HiGHS
installs into its own unnamed component, so restricting the archive to `runtime` would drop
`libhighs.*.dylib` and break the CLI again. The current setting is load-bearing.

`build_cirepro/` holds the reproduction and is untracked; removing it is Volkan's call.
