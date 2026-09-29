# DTWC++ — Lessons

Durable rules learned the hard way: a headline, the rule, and at most one pointer. Narrative and anything the
code or a test now enforces were dropped on 2026-09-28; `git log -p -- .claude/LESSONS.md` has the stories.
Append new entries at the end of their section; keep each to a few lines.

## DTW mathematics

- **DTW is not a metric.** It breaks the triangle inequality: no Elkan pruning, no ANN index, no metric-only
  approximation results. Generic MILP bounds on a well-formed cost matrix stay valid.
- **Variants change the recurrence, not the metric.** WDTW, ADTW, DDTW and Soft-DTW are Cost/Cell policies;
  MSM and TWE have their own DPs; DTW-AROW is not zero-cost DTW (it forces diagonal alignment on missing values).
- **Match a reference implementation by reading its source, not only the paper.** aeon's TWE uses |a−b| where
  its name says "squared" and front-pads both series; MSM and TWE are symmetric, so roll the shorter axis.
  *`tests/unit/core/unit_test_msm_twe.cpp`*
- **A rolling-buffer DP writes row 0 into the buffer that becomes `prev` after the first swap.** Otherwise the
  first row reads the sentinel and the distance is the maximum.
- **Soft-DTW self-distance is negative.** Any fixed sentinel collides with some value; NaN is the only safe
  "not computed" mark, and `d(i,i) = 0` is wrong for Soft-DTW.
- **`DTW_I ≤ DTW_D` for an additive local cost and the same band.** A free, independent oracle for the
  multivariate modes. *`tests/unit/core/unit_test_independent_mv.cpp`*
- **Summing channels needs commensurate units.** An additive multivariate cost over raw heterogeneous units is
  arithmetic, not a distance; scale first.
- **TADPole's δ of the densest point is the maximum of the other points' δ** (Begum et al.), not
  Rodriguez–Laio's; ties in ρ need a strict total order (ρ desc, index asc) or labels are nondeterministic.
- **FasterPAM's removal loss is undefined at k = 1** (`inf − inf`); special-case k = 1 to the direct
  arg-min of the row sums.

## Lower bounds and pruning

- **A lower bound cannot cut DTW calls on an exact matrix.** An exact matrix needs every pair; bounds help
  nearest-neighbour search and density pruning (TADPole), and exact-matrix savings come from EAP cell pruning.
  *`baselines/2026-07-08-lb-cascade.md`*
- **Admissibility is a domain contract.** LB_Keogh is admissible for L1 and unrooted squared L2 with an
  envelope whose radius covers the actual window; a negative band means unbanded, never radius 0.
  `compute_envelopes(series, band < 0)` gives the band-0 envelope. *`docs/derivations/02-envelopes-lb-keogh.md`*
- **Carry a bound's provenance.** A bare upper/lower pair does not say which source, window or length made it;
  validate at the public boundary and keep unchecked kernels internal.
- **Metric traits must test units, not family names.** `|a−b|` and `(a−b)²` are different bound units.
- **Admissibility is not tightness, and a decision counter is not an avoided-work counter.** Claim a prune rate
  only from an executed count; reachability needs a case that must prune and one that must not.
- **EAP needs a relaxed prune threshold under reassociation.** `ub·(1 + n·16ε)` is regression-tested, not
  derived; relaxing a threshold only adds computed cells.
- **EAP's exact-matrix speed-up depends on cohesion.** Near-diagonal pairs gain 6–12×, unrelated pairs ~1.5×;
  the paper's 2.88× is a nearest-neighbour regime.
- **Debug a floating-point failure from the exact failing bytes.** A 3-decimal copy of the input lost the
  ULPs that triggered the EAP bug.

## Kernels and performance

- **Generalising can beat specialising.** Template Cost/Cell policies kept one recurrence and ran 1.5–2.8×
  faster than the hand-copied kernels.
- **The row recurrence does not vectorise.** Each cell reads the one written before it, and early abandon
  exits a loop that writes memory. *`baselines/2026-09-22-x04-codegen-report.md`*
- **`-Rpass` is silent under ThinLTO, and a header template reports only where it is instantiated.** Compile
  a probe without LTO to see vectorisation remarks. *`scripts/codegen_report.py`*
- **lld-link's LTO backend runs no SLP vectoriser**, so on Windows a non-LTO listing is not what ships: code
  only SLP packs links as scalar chains. Read the linked binary (`llvm-objdump`). *`baselines/2026-09-29-p1-lanes-fill.md`*
- **A large fill is latency-bound per pair; the PAM swap on a cached matrix is memory-bound.** FastPAM1 gives
  2.95–8.06×, not k×. *`baselines/2026-07-08-faster-pam-bench.md`*
- **Benchmark the kernel you change, on each standard library.** A lookup table for the triangular index regressed 5 %.
- **On the MSVC STL, `std::min({a,b,c})` and `std::min_element` are library calls** (`__std_min_d`,
  `__std_min_element_d`), under cl and under clang on Windows; libc++ inlines both, so a Mac benchmark cannot see it.
  One call per DP cell cost 7.2 ns against 1.36 with a nested, register-carried min. Read the Windows assembly;
  `test_codegen_no_calls` guards the kernels (`baselines/2026-09-29-k1-dp-cell-no-call.md`).
- **clang's Windows driver passes `-relaxed-aliasing`** (no TBAA, as MSVC): a store through a `double *` makes the
  compiler reload every pointer it cannot prove distinct. Copy what a hot loop reads into locals.
- **Float32 is opt-in.** It halves the payload and measured 1.57–1.90× faster; Float64 stays the default.
- **Interleave A and B, and check the power state.** A laptop dropped to low-power mid-run and made unchanged
  code 1.56× slower; one-after-the-other runs would have called it a regression.
- **Before blaming code for a timing gap between two builds, pin where the time goes.** Alignment does not
  cure a fusion cliff. *`baselines/2026-09-23-x27-eigen-gap.md`*
- **Apple Silicon E-cores help** (8 → 12 threads 1.33× more); do not cap to P-cores; `omp_get_max_threads()`
  is the right count.

## Tests and gates

- **Tests must pin the LIVE code path — name the public entry point in a comment.** A fix to a dead duplicate
  with no call sites passed its tests and changed nothing.
- **A structure test does not test optimality.** Label ranges and a converged flag pass a wrong answer; keep an
  independent oracle (brute DP, exhaustive optimum on small N, hand-computed values).
- **A test name must match the path it runs.** A "FastPAM" test that set another method tested that method.
- **A regression test must execute the production arithmetic.** A copy of the formula in the test proves the
  copy.
- **A gate must prove its subject ran.** CTest scores exit 4 (skip) as a pass and exits 0 when `-R` matches
  nothing; `PASS_REGULAR_EXPRESSION` ignores the exit code. Drive binaries from a `cmake -P` script that
  asserts the exit code and the output. *`cmake/DtwcTest.cmake`*
- **Gates are one-sided.** Assert a floor or a named marker, never an upper band on counts.
- **A new gate is shown to bite.** Make a throwaway mutation it must catch, see it fail, revert.
- **A test that regenerates its expected file when it is missing compares the output with itself.** The
  conformance reference is tracked and only rewritten on request.
- **Run full suites serially.** Concurrent ctest runs, and different build matrices, collide on test
  artefacts and working directories.
- **A symmetric matrix cannot prove row- vs column-major copies.** Use a deliberately non-symmetric seam.
- **Dual-type APIs need parity tests.** The Float32 DTW function silently ignored the variant because only
  Float64 was tested.
- **Cross-validate before migrating a dispatch.** Run old and new on representative inputs (NaN positions ×
  bands) and require agreement before switching.
- **`catch (...)` without a rethrow manufactures success.** A swallowed read error printed "loaded" and then
  silently recomputed the matrix.
- **Recount a suite; an old pass total is not a floor.**
- **A zero count proves nothing ran.** Pair a disabled stage with a case that must trigger it.
- **Poison external seams before testing a rejection.** Make the command or file a rejection path would reach
  unusable, so a pass cannot come from the wrong branch.
- **An Arrow gate can pass while running nothing.** `test_io_readers` registers only when Arrow is found: check that
  `ctest -N` lists it (the Arrow shim was deleted once, 2026-09-28).

## C++, OpenMP and correctness

- **An exception must never escape an `omp parallel` region.** Guard in the serial builder; where a throw is
  possible, one `std::exception_ptr` slot per thread, rethrown after the region.
- **Every `omp critical` is named, cold, and in a `.cpp`.** Unnamed criticals serialise against every other
  unnamed critical in the program; a named one in a header breaks GCC LTO with static libraries.
- **Never bound a region with `omp_set_num_threads`.** It changes process-wide state; use a `num_threads`
  clause.
- **A dispatcher that grows an axis enumerates the full cross-product.** Every cell gets an implementation or
  an explicit `InvalidInput` at bind time; a string chain ends in a throwing `else`.
- **A contract enforced at one of three entry points is not enforced.** Three implementations of one concept
  will disagree; keep one.
- **Validate every array you index, not the first one.** A size check on one member of an aggregate is not a
  check on the aggregate.
- **A fingerprint takes every axis the caller controls as a parameter.** A literal L1 in the digest let an
  L2 cache pass for an L1 one.
- **Published caches bind every computation parameter.** A helper taking its own band can publish into a cache
  labelled with another.
- **Lambda captures by value go stale, and a moved `std::function` still points at the moved-from `this`.**
  Capture `this` and rebind after a move.
- **`std::atomic<T>` makes a defaulted move ill-formed.**
- **A maximum finite value is not "no result yet", and a finite no-path sentinel passes `isfinite`.**
  Translate kernel sentinels at the public boundary.
- **`fs::directory_iterator` order is not a contract.** Sort once; NTFS happens to enumerate by name, so the
  bug hides on Windows.
- **A unique name built from a static's address is the same in every process.** Use real entropy.
- **A successful write is checked after `close()`.** A buffered insertion can succeed and the flush fail.
- **`long` is not a width, and `near` / `far` are Windows macros.** Use `std::int64_t` across bindings; name
  variables `nearest`.
- **Public invalid states get typed errors, not `assert`.** Assertions vanish under `NDEBUG`.
- **An unreachable limit needs no check.** A guard on a count nobody can reach is code to maintain and a test
  to keep; widen the type instead.
- **A "be honest, throw" change is a behaviour change on the default path.** Register it like one.
- **An exact method that runs out of iterations throws**, and an error message names a knob the caller has.
- **An incumbent is not a lower bound, and a prune test clears the incumbent by a tolerance, never `>`.**
  Reduced-cost fixing on a certified instance removed an optimal medoid with an exact comparison.

## Numerics and portability

- **NaN means missing or not computed.** `-ffinite-math-only` is never set; `dtwc.hpp` refuses to compile
  under it.
- **`std::mt19937` is bit-exact; the standard distributions are not.** Convert integers to reals yourself for
  cross-library reproducibility. *`dtwc/core/portable_random.hpp`*
- **A last-bit pin is not portable across compilers under the Release reassociation set.** GCC flushed `-0.0`
  to `+0.0` and moved Soft-DTW by 2 ULP; pin to 17 significant figures or a tolerance.
- **`std::stod` honours the C locale; `std::from_chars` does not.** Parse numbers locale-free (fast_float).
- **Field whitespace is ASCII.** `std::isspace` under a UTF-8 locale trims a Latin-1 no-break space.
- **"Free RAM" is three different quantities** across Linux, macOS and Windows; say which one a limit uses.

## I/O and formats

- **A guard for a missing optional dependency lives outside its `#ifdef` (F9).** The Arrow-OFF build is the
  one that must reject a Parquet file; inside the guard the rejection does not exist. *`dtwc/cli/run.cpp`*
- **Never cast an Arrow array to `DoubleArray` without checking its type.** Parquet lists can hold Float32.
- **Arrow field indices and Parquet leaf indices are different namespaces.** A preceding struct owns several
  leaves.
- **A scalar Parquet column is one series; a list cell is one series.** Every reader path uses one selector.
- **Parquet's `total_uncompressed_size` is not decoded RAM.** Budget `num_values × width`; a streaming cap is
  decided before payload I/O and counts the largest row group.
- **Use `LargeList` when offsets may pass `INT32_MAX`,** and validate offsets before access.
- **polars and pandas export `__arrow_c_stream__`, not `__arrow_c_array__`.** Accept both capsules.
- **Prove "works without pyarrow" by blocking the import** (`sys.modules['pyarrow'] = None`), not by
  uninstalling it.
- **`dtwc_cl` names batch rows `1..N` and may write labels out of input order.** Map labels by name.
- **A child's stdout captured on Windows ends lines in `\r\n`.** Strip the CR before comparing.
- **`std::filesystem::path`'s stream operator quotes and escapes** (a Windows separator prints doubled). A test that
  matches a printed path collapses the escape first.

## Build and CMake

- **"Configure exits 0" is not "builds without the dependency".** Compile a translation unit that includes the
  guarded header.
- **Stale caches keep a removed option's value.** Reconfigure fresh, or pass the option, before trusting a gate.
- **`find_package` results are directory-scoped;** only cache entries reach the parent.
- **`add_compile_options()` reaches every fetched dependency.** Put project flags on `dtwc_options`.
- **A PUBLIC compile definition on an OBJECT library does not reach test TUs.** Use a runtime capability query.
- **MSVC flags leak into nvcc.** Wrap them in `$<$<COMPILE_LANGUAGE:C,CXX>:...>`.
- **A Windows DLL linked PUBLIC reaches every test executable.** Attach the runtime directories to every test
  in that directory (and to the `PATH` of spawned CLI children).
- **`string(REPLACE)` with an absent pattern succeeds.** A patch step asserts that it changed something.
- **A pin that looks arbitrary can be load-bearing.** Read why before moving it.
- **Read an upstream guard before trusting it.** Google Benchmark's counters `CHECK` tests the inverse of its
  message; a counter-less run exits 0. *`scripts/run_bench.sh`*
- **A release smoke test that runs the binary on its builder proves nothing about portability.** Check the
  rpath and the `-march` it was built with. *`baselines/2026-09-22-x29-release-archive-not-portable.md`*
- **`install_name_tool` re-signs an executable but not a dylib,** and an invalid signature on Apple Silicon is
  a silent SIGKILL (exit 137). Change only the client's load command (`-change` on the executable).
- **Never include `<windows.h>` in a header reachable from `dtwc.hpp`.**

## Python

- **Print where each module was imported from before a gate.** The pure-Python package and the native core can
  come from different places; an editable finder can supply the package under a detached worktree.
- **Re-route bindings through the C++ Tier-1; do not re-implement it.** Python's copy of the method dispatch
  drifted from C++.
- **Hand a numpy buffer over as an owned object:** build the capsule while a `unique_ptr` still owns it, then
  release. Keep one GIL policy for a class.
- **A reader or binding change is not verified until the Python suite has run** from a fresh `uv` venv. After a
  binding is removed, import the package first: `7ba0b4c` reported Python results that had never run, and
  `import dtwcpp` raised ImportError.
- **Tier-1 routes have no side effects; tests run in a scratch working directory (F45).** A route that wrote
  `./results` collided between concurrent runs. *`tests/matlab/test_tier1_route_parity.m`*

## MATLAB and MEX

- **A MEX inherits MATLAB's already-loaded runtimes.** It shares MATLAB's libomp on macOS (two runtimes abort
  MATLAB) and MATLAB's private MSVC runtime on Windows, where STL ≥ 14.40's constexpr `std::mutex` crashes an
  older MATLAB: build with `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`.
- **Catch, leave the scope that owns native resources, then call `mexErrMsgIdAndTxt`.** Pair `mexLock` with
  `mexAtExit` cleanup.
- **`mxGetScalar` does not check shape, and a raw `static_cast<int>` of a double truncates or is UB.** Use the
  exact-integer helper.
- **A MATLAB batch gate needs a clean exit and executed assertions.** A `verify*` failure does not stop the
  script; put the fresh MEX directory first on the path and check `which`.
- **MATLAB dies at start-up under a long `TMPDIR`** on macOS; run its suite with the default one.
- **CMake 4.2 `FindMatlab` exports `mexFunction` only under MSVC.** A clang MEX on Windows needs
  `/EXPORT:mexFunction` passed to the linker explicitly, or MATLAB reports a missing gateway. *`bindings/matlab/CMakeLists.txt`*

## GPU, Windows toolchain, HPC

- **Guard a pair count at the public entry, and build the smallest failing case at `INT_MAX`.** It is cheap.
- **`CUDA_VISIBLE_DEVICES=-1` forces the no-device branch** on a GPU host.
- **Clamp a logical full window on the host before device integer arithmetic.** `k + w + 1` overflows in a
  kernel before any clip.
- **A per-pair decode is device code.** Its cost is paid once per pair per launch.
- **A profiler can exit 0 having seen no kernel.** Check that it observed one.
- **Git Bash rewrites a leading `/c` argument.** Call `cmd //c`, or run from PowerShell.
- **Sophos may quarantine a Release `dtwc_cl.exe` as 'Generic ML PUA'.** Symptoms: "Permission denied", CLI tests
  "dtwc_cl not found"; the Application event log names the file. Never work around it; a Debug build runs. Volkan
  declined an exclusion (2026-09-28): run the CLI tests from a Debug build of the same tree and say so.
- **ARC:** compute capability is not the CUDA version in the docs (P100 6.0 … H100 9.0); Rome and Broadwell
  nodes lack AVX-512 (`DTWC_ARCH_LEVEL=v3`); Grace Hopper is AArch64.

## Solvers

- **The 2023 x-space LP solvers stay killed.** PDLP / first-order LP arbiter: PDLP cross-checks the LR-core
  bound; matrix-free Kelley wins on either device. *`baselines/2026-07-08-pdlp-bench.md`*
- **The p-median matrix is totally unimodular only for N ≤ 2,** and fractional LP vertices are not
  half-integral (1/3, 1/4 occur): branching is mandatory. *`UNIMODULAR.md`*
- **Two gaps, two tools.** The LR primal repair's gap and the root bound's gap are different quantities.
- **HiGHS PDLP GPU is a build flag, not a runtime switch.**
- **The two MIP adapters flatten differently** (HiGHS row-major, Gurobi column-major, diagonals at
  `i·(N+1)`); equivalent only while D is symmetric.

## Process and records

- **One step per commit.** An omnibus commit makes review, bisection and rollback inseparable.
- **A frozen contract changes only through a dated decision** (`DECISIONS.md` §2).
- **Always verify citations.** Authors, venues and volumes can be invented.
- **Re-run a counterexample after its premise changes.** A killed premise is replaced by a new discriminator,
  not carried by name.
- **Check a recorded number before building a ratchet on it.** The row said 256; the tree held 259.
- **A comment/string stripper under-counts silently.** Compare it with a naive grep.
- **An agent's finding is a hypothesis until the cited line has been opened.**
- **Remote-tracking refs are mutable evidence.** Re-read them at close-out.
- **Before deleting a build dir, grep the surviving `CMakeCache.txt` files for its path.** The 2 KB
  `build/f9-arrow-config` was the Arrow shim of `build/arrow-pyarrow-23`.
- **Agent tooling:** in the Bash tool a `\\` inside a quoted heredoc reaches the program as `\` (write scripts with
  the Write tool); the Workflow tool rejects a `scriptPath` file holding non-ASCII text as "control characters"
  (escape it to `\uXXXX`).
