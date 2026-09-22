# X-15 / S-03 — floating-point flag scope, and two defects found alongside

**Date:** 2026-09-22 · **Machine:** Mac (Apple M5 Pro, 18 cores, 64 GiB), AppleClang 21.0.0,
`clang-macos` preset, Release · **Rows:** X-15, S-03 (both W0), plus new X-31 and X-32

## S-03: the leak was real, and larger than the row said

The row named HiGHS and llfio. `compile_commands.json` is the authority on who actually receives a
flag, so it was queried rather than the CMake source read:

```sh
python3 -c "import json; cc=json.load(open('build/compile_commands.json')); \
  print(sum(1 for e in cc if '_deps' in e['file'] and '-fassociative-math' in e['command']))"
```

| Scope | TUs with `-fassociative-math` before | after |
| --- | --- | --- |
| our own code | 161 | **161** |
| fetched dependencies | **146** | **0** |
| — of which Catch2 | 107 | 0 |
| — of which HiGHS (`highs-build` + `highs-src`) | 31 | 0 |
| — of which llfio | 8 | 0 |

All seven relaxations plus `-march=native` were reaching them. The HiGHS number is the one that
matters: reassociating a floating-point sum in simplex or interior-point code can move a pivot or
trip a tolerance. Catch2 at 107 TUs means the framework's own float matchers were built relaxed —
the measuring instrument shared the distortion it was supposed to detect.

**Mechanism.** `add_compile_options()` is directory-scope, and a subdirectory inherits the value in
force when `add_subdirectory()` runs. The flags were set at `CMakeLists.txt:25`;
`dtwc_setup_dependencies()` — where every `CPMAddPackage` lives — is line 240. So the dependencies
were added with the flags already in force. `dtwc_options` is created at line 243, *after* them.

## X-15: the fix

`DTWC_FP_MODEL` (CACHE STRING, `fast` | `strict`, default `fast`) computes `DTWC_FP_FLAGS` in
`cmake/StandardProjectSettings.cmake`; `dtwc_local_options()` applies it to the `dtwc_options`
INTERFACE target (`cmake/ProjectOptions.cmake`), Release and RelWithDebInfo only. Cache, not plain
`set`, so `scripts/machine_facts.py` harvests it into machine records (X-26's requirement).

Coverage was checked rather than assumed: `dtwc++`, `mip-solvers`, `dtwc_cl`/`dtwc_main`, the
nanobind module and **every** test binary already link `project_options` — tests via
`cmake/DtwcTest.cmake:72`, which is the single registration point since `d21ffee`. The one gap was
`benchmarks/`, which linked `dtwc++` but not `project_options`; all 8 targets now do, so a benchmark
compiles inlined kernel code the same way the library does instead of measuring a different build of
it.

`metal_dtw.mm` is the only one of our TUs without the flags, before and after: it compiles as
OBJCXX and never matched `$<COMPILE_LANGUAGE:C,CXX>`.

| Check | Result |
| --- | --- |
| `fast` (default) | `[confirmed]` our TUs 161, dependency TUs 0 |
| `strict` | `[confirmed]` all four sampled relaxations → 0 TUs repo-wide |
| `-DDTWC_FP_MODEL=loose` | `[confirmed]` configure exits 1 naming both valid values |
| full suite, deps now built strict | `[confirmed]` 126/131, the same five V-6 floor failures and no others |

That last row is the one that matters for safety: HiGHS was recompiled without relaxations for the
first time and **no MIP test changed**.

## Conformance is floating-point-model-invariant here `[confirmed]`

X-15's third clause is "conformance pinned under `strict`". Whether the recorded reference — taken
under `fast` — survives `strict` was an open empirical question. It does, and not merely inside the
`1e-12` tolerance. Built `strict`, `cpp_conformance -s` expands to:

```
0.96894972764334841 and 0.96894972764334841   (silhouette)
0.03833333333333334 and 0.03833333333333334   (davies_bouldin)
11.5 and 11.5                                 (dunn)
```

Digit-identical at 17 significant figures, labels and medoids exact. So the clause is a CI leg, not
a redesign: `macos-unit.yml` gained a `conformance-strict` job. Scope of the claim: AppleClang on
arm64 only. `macos-unit.yml` is the **only** Release job that runs ctest in the whole repo — every
other test job is Debug, where these flags do not apply — so the GCC and MSVC branches of
`DTWC_FP_FLAGS` are exercised by nothing. Recorded as V-7.

## X-31: every released CLI archive was `-march=native` `[confirmed]`

`DTWC_ENABLE_NATIVE_ARCH` defaults ON (`StandardProjectSettings.cmake:86`) and applies when
`DTWC_ENABLE_NATIVE_ARCH AND PROJECT_IS_TOP_LEVEL AND NOT DTWC_BUILD_PYTHON` (`:91`).
`release-artifacts.yml` configures `cmake -S . -B build/release ... -DCMAKE_BUILD_TYPE=Release` with
no arch override, on `ubuntu-latest`, `macos-latest` and `windows-latest` — all three conditions
hold. Confirmed locally: the same style of configure prints `-- Architecture tuning: -march=native`,
and the `clang-macos` preset sets nothing about arch, so this is the repo default speaking.

Consequence: each published binary is tuned to whichever ephemeral runner built it, and a user whose
CPU lacks an instruction the runner had gets SIGILL. The wheels were never exposed —
`pyproject.toml:69` sets `DTWC_BUILD_PYTHON=ON`, and the guard excludes them with the comment "to
keep wheel binaries portable". The reasoning was correct and simply never extended to the archives.

Same blind spot as X-29: `smoke_release_archive.py` runs the binary on the machine that built it,
where every instruction is supported by construction. Fixed with
`-DDTWC_ENABLE_NATIVE_ARCH=OFF` in the release configure; `-DDTWC_ARCH_LEVEL=v3` remains the
documented opt-in for a deliberate AVX2+FMA release baseline.

## X-32: the conformance gate certified itself `[confirmed]`

`cpp_conformance.cpp:212-220` regenerated the pinned reference when `DTWC_CONFORMANCE_REGEN=1`
**or when the reference file did not exist**, then read it back and compared it to itself. With the
file absent, every assertion passed trivially; the only signal was a Catch2 `WARN`, which does not
fail a test. Bootstrap and verify shared a trigger.

Regeneration is now explicit-only and a missing reference is a `REQUIRE` failure. Discrimination
check, with the tracked file moved aside and restored:

```
=== EXIT WITH REFERENCE MISSING: 42 ===
cpp_conformance.cpp:229: FAILED:
  REQUIRE( fs::exists(reference_file()) )
```

and `cmp` against the saved copy afterwards reports byte-identical, `git status` clean. The
post-fix failure is executed; the pre-fix pass is read from the diff, not re-run.
