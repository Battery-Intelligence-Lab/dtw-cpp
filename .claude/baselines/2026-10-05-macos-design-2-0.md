# 2026-10-05 — macOS pass on design-2.0 (the first Mac build since 2026-09-23)

Branch `design-2.0`, base `4dd4dcaf` (the newest code is GC's merge `63415b4f`; W9b not yet merged). Every number
below is `[confirmed]` unless marked otherwise; the commands are the runbook's.

## Machine and toolchain

| Item | Value |
| --- | --- |
| Host | Apple M5 Pro (18 cores: 6 P + 12 E), 64 GiB, Metal 4; macOS 26.6.2 (Darwin 25.6.0), arm64 |
| Compiler | AppleClang 21.0.0.21000334 (`/usr/bin/clang++`), libc++, CMake 4.4.3, Ninja, libomp 23.1.1 (Homebrew) |
| MATLAB | R2026a (`/Applications/MATLAB_R2026a.app`, version 26.1.0.3346908), not on PATH |
| Python | uv-managed CPython 3.12.14 in a venv outside the repo |

## Configure and build (`build/`, the preset `clang-macos`)

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp -DDTWC_ENABLE_GUROBI=OFF
cmake --build --preset clang-macos
```

- `-DDTWC_ENABLE_GUROBI=OFF` once: the tree's cache (2026-09-22) still held the pre-W14b default ON, and since W14b
  an ON that cannot be honoured stops the configure (DECISIONS §3, 10-01). A fresh tree needs no flag.
- Summary: Release, `-march=native` (arm64), OpenMP ON (spec 202011), Metal ON (Apple GPU), HiGHS 1.15.1, llfio,
  fkYAML 0.4.4, Arrow OFF, CUDA OFF, Gurobi OFF; `test_io_readers` unregistered (no Arrow).
- 206 targets, 31 s wall (482 s CPU), **zero warnings**; `metal_dtw.mm` compiled as OBJCXX on the first try: none
  of the 22 Metal commits since 09-28 broke the Objective-C++ build.

## Serial test run (`ctest --test-dir build -C Release -j1 --output-on-failure`)

| Result | Count |
| --- | --- |
| Registered | 95 |
| Passed | 93 |
| Skipped (`MAY_SKIP`) | 1 — `test_cuda_correctness` (no CUDA); `test_cuda_launch_guards` ran its host cases |
| Failed | 1 — `test_codegen_no_calls` (below) |
| Wall | 42 s |

- `test_metal_correctness`: `All tests passed (2168 assertions in 16 test cases)`, no SKIP.
- `test_metal_mmap`: `All tests passed (169 assertions in 5 test cases)`, no SKIP (Metal through `Problem`, dense
  and mapped; the squared-L2 cache; FastCLARA on the GPU device).

### `test_codegen_no_calls` fails on Apple clang: `memset_pattern16` in the lanes kernel

```text
CODEGEN_NO_CALLS tool=clang++ inner_loops=96 calls=2 verdict=FAIL
  _dtwc_probe_lanes_f64: bl	_memset_pattern16
  _dtwc_probe_lanes_f32: bl	_memset_pattern16
```

Apple clang's loop-idiom pass turns a 64-byte constant store loop into a call of the Darwin-only libc function
`memset_pattern16`. In `dtw_kernel_lanes` (`dtwc/core/dtw_kernel.hpp`) it did so twice:

1. the rows above column 0's band, `for i in [hi0, n): for w: s[i].v[w] = maxValue` — one call per row, and the row
   loop is then the innermost loop (this is the one the gate flags);
2. in the column loop, the band's `left[w] = maxValue` for the W lanes — one call per column, with four 32-byte
   q-register spills and reloads around it. The gate does not see it (it inspects innermost loops only); for a
   tight band it is the costlier of the two.

On x86 clang (the Windows box) neither exists: `memset_pattern16` is not in its libc. `-fno-builtin-memset_pattern16`
does nothing (a probe loop kept its ten calls), so the fix is in code, `a332d671`: both fills copy one constant row
(`static constexpr Row kUnreachable`; the rows by assignment, `left` by `std::copy_n`, since a lane-by-lane read of
the row folds back to the constant and the fill with it). After it, on this tree:

```text
CODEGEN_NO_CALLS tool=clang++ inner_loops=96 calls=0 verdict=PASS
```

No `memset_pattern16` remains in either lanes probe. `unit_test_dtw_kernel_lanes` (the lanes against the per-pair
kernels, bit for bit): `All tests passed (175 assertions in 3 test cases)`; `test_dtw`: 30802 in 9. The agent's
scratch driver hashed 38,928 output values of the kernel (f64 and f32, L1 and squared, bands −1…1000, mixed lengths)
identical before and after `[reported by the agent, unverified]`; it also cross-compiled the probe for x86-64 Darwin:
the original header made 6 such calls per precision there too, so an Intel Mac had the same calls. Its advisory
timing (not a claim; the machine was not quiet): narrow-band lanes calls 7–10 % faster, wide and unbanded neutral.

The same idiom sat in the per-pair banded kernel, `dtw_kernel_banded`: two calls per column (the fills of the
cells that left the band, typically one cell each), which the gate cannot see. Fixed in `55911b2a` (+3/−11): the
bounds from `dtw_band_bounds` never decrease with the column, so a cell that left the band is never read again; the
high-side fill ran zero iterations and the low-side fill only fed the seed `left = col[first_row - 1]`, now
`(low == 0) ? col[0] : maxValue`. After it on this tree (verified by me): `CODEGEN_NO_CALLS … inner_loops=97 calls=0
verdict=PASS`; the probe listing has no `memset_pattern16` at all; ctest 95/95; `test_dtw` 30802/9,
`unit_test_dtw_kernel_lanes` 175/3, `unit_test_arow_dtw` 13180/47 `[agent-run]`. The agent's sweep `[reported,
unverified]`: 30,326,400 kernel calls (f64/f32 × L1/squared × four data kinds, n_short 1–300, n_long up to +53,
bands −1…50, plain and early-abandon thresholds) hash identical before and after and match an independent
full-matrix reference; two deliberate seed breaks change the hash; a second sweep over ADTW, AROW, WDTW and SoftCell
(1,272,000 calls) is identical too. `nm -u` over the 89 linked test binaries imports no `memset_pattern*`
(the old header's driver imported it). Noted by the agent: the probe has no f32 banded instantiation, so the gate
never covered `dtw_kernel_banded<float>`; and the `diag` guard and the two bounds vectors are now provably
removable (the bounds are plain arithmetic in j), a candidate simplification, not done.

Timing `[confirmed]`, quiet machine (nothing else running), the agent's harness `bench2` built from the old and the
new header with the library's flags, single thread, f64 L1, `StandardCell`, medians of 3 interleaved rounds:

| band | old ns/cell | new ns/cell | new/old |
| --- | --- | --- | --- |
| 1 | 1.80 | 2.77 | 1.54 |
| 2 | 1.99 | 1.73 | 0.87 |
| 3 | 1.49 | 1.30 | 0.87 |
| 4 | 1.39 | 1.07 | 0.77 |
| 5 | 1.98 | 0.89 | 0.45 |
| 10 | 1.42 | 0.84 | 0.59 |
| 20 | 1.19 | 1.04 | 0.87 |
| 50 | 1.46 | 0.69 | 0.47 |
| 100 | 1.49 | 0.87 | 0.58 |
| 200 | 1.45 | 0.95 | 0.65 |

(n = 1000; n = 3000 and the early-abandon runs give the same ratios within 0.05, except band 16 with abandon at
1.10.) Faster at every band from 2 up; slower only at band 1, three cells per column, where the per-column cost
rose from 5.4 to 8.3 ns.

### Follow-up unit (agent, Opus): the bounds as arithmetic — FALSIFIED, not merged; band 1 explained

Branch `pb/banded-bounds-arith` (`a2d5e4e5`, +8/−25, bit-identical on both sweeps): the two thread_local bounds
vectors, their fill loop and the `diag` guard go; each column calls `dtw_band_bounds` once (the guard was always true:
for `low == 0`, `first_row − 1 = 0 = prev_lo`; for `low > 0`, `first_row − 1 = low − 1 = prev_lo`, and `prev_hi >
low − 1`). Registered band: no band of the per-pair kernel slower than 1.05× the merged `55911b2a`. Quiet machine,
the agent's `bench2` builds, medians of 3 interleaved rounds, ratio `a2d5e4e5 / 55911b2a` (ns per cell):

| band | 1 | 2 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 50 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n=1000, no abandon | 0.97 | 0.94 | 0.97 | **1.20** | **1.10** | **1.14** | 1.00 | **1.09** | **1.12** | 1.04 | 0.97 | 1.00 |
| n=3000, no abandon | 0.97 | 0.93 | 0.94 | **1.19** | **1.10** | **1.13** | **1.08** | **1.08** | 1.02 | 1.04 | 0.97 | 1.00 |
| n=1000, abandon | 0.79 | 0.90 | 0.84 | 0.93 | 0.88 | 0.89 | 0.94 | 0.94 | **1.19** | 0.93 | 0.99 | 1.00 |
| n=3000, abandon | 0.95 | 0.90 | 0.84 | 0.89 | 0.87 | 0.88 | 0.93 | 0.94 | **1.13** | 0.98 | 0.99 | 1.00 |

FALSIFIED at bands 5–12 without early abandon (the unequal-length fill's regime for short series): fewer
instructions per column (66 against 80) but more cycles; the agent's diagnosis is scheduling, not work (a copy whose
column addresses wait on one load, as they waited on the vectors, is fast again). Kept unmerged for the Windows box
to time on x86; if neutral there, the trade-off goes to Volkan. (Absolute ns in this run are about 1.5× the morning's:
the clock had dropped, 4.6 → 2.9 GHz per the agent's `proc_pid_rusage` cycles; interleaving keeps the ratios valid.)

Band 1 `[agent, cycles via proc_pid_rusage]`: the old code's `memset_pattern16` call stored `col[low − 1]` right after
the `diag` read of that cell; with the fills gone each band-1 column re-reads two of the three cells the previous
column just stored, and the column's period is the store→load handoff through `col` (33.5 cycles per column against
21.6 with the call, IPC 7.7). One extra store per column, anywhere before the diag read, restores 20.4–21.4 cycles
(candidate `e13`: a `col[n_long] = maxValue` into a +1 slot; ratio 0.56–0.61 at band 1, 0.93–1.09 elsewhere). A dead
store for one CPU's memory-dependence predictor is not shipped; band 1 (three cells per column) stays as measured.

## Code generation on arm64 (Apple clang, `-O3 -march=native`, the probe with `dtw_dispatch.cpp`'s flags)

Read from the probe listing after the three fixes (`scripts/codegen_report.py`'s compile, `-S`):

- The lanes kernel's DP loop is NEON-packed: f64 as four `.2d` lane pairs, f32 as four `.4s` quads, `x[i]` broadcast
  once per row by `ld1r`, rows loaded and stored as `q` pairs. Per lane pair: one `fabd.2d` (L1) or `fsub`+`fmul`
  (squared), two mins as `fcmgt` + `bit`/`bsl`/`bif` (compare-and-select, the right lowering while NaN must propagate;
  `fminnm` has other NaN rules), one `fadd`. 39 instructions per DP row of 8 f64 lanes, no call. Clang's
  loop-vectorise remark says "not vectorized" for these loops because they are fully unrolled first and packed by
  the SLP vectoriser (`-Rpass=slp-vectorizer`: "Stores SLP vectorized … tree size 26–31"); the report's 1/14
  "vectorized" count is therefore not the SIMD picture here.
- Column 0 of the lanes kernel (`s[i] = combine(max, s[i-1], max, dist)`) stays scalar ("vectorization was impossible
  with available vectorization factors"): O(n) per call against the O(n²) DP, left alone.
- The per-pair kernels: `fabd` + `fminnm` + `fadd` per cell, scalar by nature (one dependency chain).
- The probe's `AbsDiff` was `a < b ? b - a : a - b` (four instructions per lane pair) where every library cost is
  `std::abs(a - b)` (one `fabd`); fixed in the commit after `55911b2a` so the listing matches the shipped code.
  The gate's verdict never depended on it.

## The gate hardened (`98e986fc`): any loop of a probe kernel, and the f32 per-pair kernels

`scripts/codegen_report.py --no-calls` counted a call only in an innermost loop's blocks; the banded kernel's calls
sat in the column loop, which has a child loop. Now any block of any loop counts (`This Loop Header`, `This Inner
Loop Header`, `in Loop: Header=BB…`); the marker and the CMake regex are unchanged (`inner_loops=` still counts
innermost headers). The probe gains `dtwc_probe_kernels_f32`, the twin of the f64 wrapper, since no listing had
`dtw_kernel_banded<float>` before. Verified by me on the main tree:

| header | rule | result |
| --- | --- | --- |
| HEAD | new | `inner_loops=114 calls=0 verdict=PASS` |
| `4dd4dcaf` (neither kernel fix) | new | `inner_loops=112 calls=12 verdict=FAIL` (banded f64 and f32, kernels f64 and f32, lanes f64 and f32) |
| `a332d671` (lanes fixed only) | new | `inner_loops=112 calls=8 verdict=FAIL`, the old rule's exact blind spot (it printed PASS there) |

The agent's census `[reported]`: 274 calls in dtwc probe functions at HEAD, all outside every loop (TLS accessors,
outlined functions, `vector::resize/assign`, `bzero`); 240 loop-code blocks; an x86-64 Darwin cross-compile of the
pre-fix header shows 14 `callq _memset_pattern16`, so the rule reads `call` syntax too. Not run on Linux or Windows
clang: a legitimate call in an outer loop there would now fail the gate (none is known).

## Conformance (`cpp_conformance`, digit for digit)

`DTWC_CONFORMANCE_REGEN=1 build/bin/cpp_conformance` then `git diff` of `tests/conformance/conformance_reference.txt`:
labels and medoids identical; the silhouette differs in its last digit only:

```text
-silhouette,0.96894972764334841   (the tracked reference, recorded on Windows clang, x86-64)
+silhouette,0.9689497276433483    (this Mac)
```

One ulp (relative 1.1e-17), inside the 1e-12 contract; davies_bouldin and dunn identical. The clustering did not
change. Read as the reduction order under `-fassociative-math` differing with the vector width (NEON 2 × f64 vs
AVX2 4 × f64); D-19 and Volkan 10-01 (an epsilon-level difference is fine when the clustering holds). The
reference file was restored; nothing committed.

## Gates

| Gate | Result |
| --- | --- |
| `scripts/check_docs.py --cli build/bin/dtwc_cl` | `DOCS flags checked=392 pages=59 live=60 VERDICT=PASS` |
| `scripts/check_pins.py` | `PINS cmake=18 actions=37 failures=0` |
| `scripts/generate_docs.py --check` | `generated documentation is current` |

## Python (fresh venv outside the repo, runbook commands)

`uv venv --python 3.12`; `CMAKE_ARGS=-DOpenMP_ROOT=… uv pip install ".[test,dev,io]" matplotlib` built the wheel in
73 s. The extension `_dtwcpp_core.cpython-312-darwin.so` is 5.6 MB, HiGHS linked in (static, as `Dependencies.cmake`
forces for a wheel), Metal and Foundation dynamic, libomp by its Homebrew path (a local build; CI's wheel bundles it).
`dtwcpp.gpu_info()` → `Metal: Apple M5 Pro (registryID=0x100000568, max_working_set=51.84 GB)`.

`DTWC_CL_PATH=build/bin/dtwc_cl python -m pytest tests/python -q -p no:cacheprovider`:

```text
1102 passed, 11 skipped, 1 warning in 165.79s
```

Skips: 9 `CUDA not available` (`test_cuda.py`), `test_device.py:120` (GPU present; the unavailable-device path not
exercised), `test_preprocess.py:111` (scipy is installed). So the Metal device path ran through Python. This tree is
without W9b (it rewrote Python's reading, writing and conversion): pytest runs again once W9b lands. Rerun with the
wheel rebuilt at `98e986fc` (every kernel, Metal, probe and gate change of the day): `1102 passed, 11 skipped in
43.52s`, the same skips (the first run's 166 s overlapped two agent builds and the MATLAB suite).

## MATLAB (`build-matlab/`, `-DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2026a.app`)

Same flags as `build/` plus the MATLAB ones; 320 targets, 39 s wall, zero warnings; `dtwc_mex.mexmaca64` built
(1.29 MB; links `@rpath/libomp.dylib` as the CMake repoints it, so MATLAB's own libomp serves, `@rpath/libhighs.1.dylib`
from `build-matlab/lib`, Metal and Foundation).

`ctest --test-dir build-matlab -C Release -R matlab_suite -V`:

```text
matlab_suite: 140 run, 139 passed, 0 failed, 1 incomplete
test_test_api/test_parallelisation_serial_is_honest   Filtered by assumption   (the registered one: an OpenMP MEX filters the serial case)
```

Passed in 48 s (67 s under the verbose rerun while two agent builds ran).

## AddressSanitizer + UBSan (`build-asan/`, RelWithDebInfo, IPO off, Apple clang)

```sh
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DOpenMP_ROOT=/opt/homebrew/opt/libomp -DDTWC_BUILD_TESTING=ON \
  -DDTWC_ENABLE_SANITIZER_ADDRESS=ON -DDTWC_ENABLE_SANITIZER_UNDEFINED=ON -DDTWC_ENABLE_IPO=OFF
cmake --build build-asan
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-asan -j1 -E test_codegen_no_calls --output-on-failure --timeout 1800
```

`-fsanitize=address,undefined` on 116 translation units. **94/94 passed, 1 skipped (`test_cuda_correctness`), 95 s**;
no sanitizer report in the log. `test_codegen_no_calls` is excluded because it replays this tree's flags and the
instrumented probe calls the runtime by design. Build warnings: only clang's `-Wpass-failed` notes that the
`vectorize(enable)` pragma in `z_normalize.hpp` cannot be honoured under instrumentation. LeakSanitizer is not
available on Apple Silicon. This is the first sanitizer run of design-2.0 on macOS (TSan ran on Windows/WSL, R1).

## W4e (Metal cleanup), merged `6cb0c3c1`

`dtwc/metal/metal_dtw.mm` +140/−282: all five pipelines through the one `make_pipeline` helper; one templated
wavefront DP body (`Scope`, `Ptr`) behind the two kernels, their names and `[[buffer(n)]]` indices unchanged; the
dead pair-index plumbing gone (buffers 10 and 11, the dummy buffer, `has_pair_indices`, `effective_pairs`); one RAII
holder owns the four buffers and an `NSAutoreleasePool` (an exception did not drain `@autoreleasepool`), so every
exit releases once. Leaks closed: the lengths and output refusals, the dummy-buffer throw, the in-loop and final
command-buffer errors, a `bad_alloc` from `out.resize`. The agent's leak probe (`compute_distance_matrix_metal`
driven into the output-buffer refusal three times, N chosen so the packed triangle is 1.2 × `maxBufferLength`,
`device.currentAllocatedSize` read afterwards) `[confirmed]`, run by me: 191.8 MB left allocated by the base code,
0.0 MB by the merged code.

Verified by me on this tree after the merge: zero warnings; `test_metal_correctness` 2168/16 and `test_metal_mmap`
169/5 unchanged; ctest 95/95 (1 CUDA skip); the base CLI (built at `4dd4dcaf`) and the merged CLI gave byte-identical
distance matrices, labels, medoids and silhouettes on `--device gpu -k 3` for N=40 L=100 (regtile_w4), N=40 L=600
(wavefront) and the same with `--band 20` (banded_row), N=12 L=3000 (wavefront_global); the agent showed the same for
L=200 (regtile_w8), `--band 100`/`--band 200` and the squared-L2 metric.

The review's one finding, confirmed by reading the loop: the regtile and threadgroup-wavefront routes committed one
command buffer per chunk and checked `last_cmd.error` only, so a chunk failing before the last (the watchdog case
the chunking exists for) left its pairs at 0, which reads as identical series. `bc9469fd` waits for and checks every
buffer; the same evidence set (gate, ctest 95/95, the Metal counts, the four routes byte-identical) passed after it.
Not reproduced: no way to make a Metal command buffer fail on demand was found.
