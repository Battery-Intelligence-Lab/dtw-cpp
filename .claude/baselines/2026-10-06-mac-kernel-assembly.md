# 2026-10-06 — Apple M5 Pro: the CPU assembly DTWC++ ships, against what the core allows

**Question:** on this MacBook (Apple M5 Pro), what assembly do the hot loops compile to in what ships (the CLI,
the Python wheel, the MATLAB MEX), how fast does each run, how close is that to what this core allows, and what
would a change gain?

**Answer.** The DP kernels ship as intended and identically in all three artifacts: the lanes kernel
(`dtw_kernel_lanes`, the equal-length fill) is fully on q-registers with no call, trap or spill, and runs at
0.85–1.02 cycles per cell f64 and 0.44–0.53 f32 (≈ 5 and ≈ 10 Gcell/s on one core), within 1.15× of its
latency bound in every shape and run except the squared cost at L 100 in the second run (1.17–1.21) [confirmed]. That bound is
set by the min, not by the core: `std::min` compiles to `fcmgt`+`bif` (2 ops, 4 cycles), the chain
`fcmgt→bif→fadd` takes 7.0 cycles, and the core has 4 FP pipes. A min that compiles to `fminnm` (exact for every
input the checked paths admit) gives 1.23–1.49× single thread; with 16 doubles / 32 floats per call it reaches
the machine's FP-throughput floor, 1.41–2.00× single thread and 1.48–1.72× in an 18-thread f64 fill, every
output bitwise equal over the sweep [confirmed]. The per-pair kernels (unequal lengths, every other variant) are
scalar 13-instruction loops bound by a 5.1–5.3-cycle `fcmp→fcsel→fadd` chain; computing two columns per pass gives
1.44–1.98× unbanded, bitwise equal [confirmed]. The largest defect is in the wheel: nanobind compiles the binding
TU at `-Os`, LTO keeps that copy of the f64 per-pair kernel, and `-Os` leaves the early-abandon row minimum in the
common loop as a second 6-cycle chain: Python's single-pair `dtw` is 1.15–1.56× and its fill of unequal-length
series 1.52–1.59× slower than the same source at `-O3` (relinked without `-Os`: identical results) [confirmed].
`-march=native` changes no kernel instruction (nor does `-mcpu=apple-m5`) [confirmed].

## Bands (registered before measuring, quoted)

- **B1 packing:** in each LINKED binary, the lanes kernel's per-cell arithmetic runs on q-registers (`.2d` for f64,
  `.4s` for f32). Scalar `d`/`s` arithmetic in the lane loop = FAIL.
- **B2 no call or trap:** no `bl`/`blr`/`b <external>` and no `brk` (libc++ hardening trap) inside any DP inner
  loop of the linked binaries.
- **B3 at bound:** a kernel is "at its bound" when its measured cycles per cell are <= 1.15 x max(latency bound,
  throughput bound). Compute both bounds from the instructions in its linked loop and this machine's MEASURED
  latencies and throughputs (step 3). Otherwise it has headroom of measured/bound.
- **B4 candidate:** a scratch variant counts only if (a) its outputs are bitwise identical to the unmodified kernel
  over a sweep (f64 and f32, L1 and squared cost, lengths 1..1000 including unequal ones for per-pair kernels,
  bands -1, 0, 1, L/10, L, random seeds; hash every output), and (b) it is >= 1.15x single-thread cells/s on at
  least two of the four shapes {L 100, L 1000} x {unbanded, band L/10}, with no shape below 0.97x.

## Machine (`uv run --no-project python scripts/machine_facts.py --build-dir build`)

| Fact | Value |
| --- | --- |
| OS | Darwin 25.6.0 (arm64), macOS 26.6 |
| CPU | Apple M5 Pro — 18 cores / 18 threads |
| Memory | 64.0 GiB |
| GPU | Apple M5 Pro integrated GPU (Metal) |
| Compiler | AppleClang 21.0.0.21000334 (Command Line Tools) |
| Build type / flags | Release, `-O3 -DNDEBUG` (+ `-flto=thin -march=native` and the FP model "fast" per compile_commands) |
| OMP_NUM_THREADS | unset |
| Repository | design-2.0 @ 33877edf (clean), VERSION 2.0.0rc1 |

`sysctl` [confirmed]: perflevel0 "Super" 6 cores, L1D 128 KiB, L2 16 MiB per 6; perflevel1 "Performance" 12 cores,
L1D 64 KiB, L2 8 MiB per 6; cache line 128 B; FEAT_SME2, SME2p1, BF16, I8MM, AFP present. Single-thread clock
4.58–4.60 GHz, measured by a dependent `add x9, x9, #1` chain interleaved with every run (1 add/cycle on Apple
cores; listing in `probes/ubench/ubench.cpp`, `t_int_add_lat`); a lone thread at default QoS ran at that clock, so on
a Super core [inferred: macOS gives no pinning]. Load: every `out/*.txt` starts with `uptime` and the top CPU
consumers; Sophos endpoint processes used 10–170 % of one core during runs (highest, ~170 %, at the start of the
first 18-thread fill), VS Code's renderer 90 % during the first Python pair timing. Single-thread numbers are
medians of 7; fills medians of 5 (3 for the scaling line).

## Method (scratch `S=/private/tmp/claude-504/.../scratchpad/asm-audit`; nothing in the repo written)

1. **Flags.** `clang++ -### {-march=native | (none) | -mcpu=native | -mcpu=apple-m5} -O3 -c probes/empty.cpp`
   (`out/cc1_*.txt`); `clang++ -print-supported-cpus`. Wheel: `uv venv $S/venv --python 3.12`;
   `SKBUILD_BUILD_DIR=$S/wheel-build SKBUILD_BUILD_VERBOSE=true CMAKE_ARGS="-DOpenMP_ROOT=/opt/homebrew/opt/libomp
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" uv pip install -v --python $S/venv/bin/python "/Users/engs2321/git/dtw-cpp[test]"`
   (`out/wheel_build.log`; the build dir and verbosity only make the flags visible). MEX flags from
   `build-matlab/compile_commands.json`. Repo ignored/untracked set checked identical before and after
   (`out/git_ignored_{before,after}.txt`).
2. **Assembly.** `objdump -d --no-show-raw-insn` of `build/bin/dtwc_cl`, the wheel's
   `_dtwcpp_core.cpython-312-darwin.so` (stripped of local symbols: loops found by pattern) and
   `build-matlab/bindings/matlab/dtwc_mex.mexmaca64`; `probes/loops.py`, `probes/lane_loops.py` (innermost
   backward-branch loops), `probes/normregs.py` (diff modulo general-purpose register numbering),
   `probes/b2check.py` (B2 over every innermost loop holding a min feeding an `fadd`), `probes/symsizes.py`.
3. **Core bounds.** Inline-asm microbenchmarks, each timed against the add chain in the same process:
   `probes/ubench/` — `ubench.cpp` (from `gen_ubench.py`: 60 latency/throughput tests), `opord.cpp` (operand order
   and mixed chains), `fcsel_pred.cpp` (does `fcsel` break a chain), `fptp2.cpp` (from `gen_fptp2.py`: FP throughput,
   24 accumulators, rotating sources), `loopasm.cpp`, `lanesasm.cpp`, `w16asm*.cpp`, `wheelloop.cpp` (shipped and
   variant loops replayed verbatim). Build `clang++ -O2 -std=c++20 X.cpp -o X`; their loop bodies checked in
   `objdump`.
4. **Kernel speed.** `probes/kbench/kbench.cpp` includes the real `dtwc/core/dtw_kernel.hpp` and
   `dtwc/core/dtw_cost.hpp`; `probes/kbench/build.sh native|default` replays the compile command of
   `dtwc/core/dtw_lanes.cpp` (`-O3 -DNDEBUG -std=c++20 -flto=thin -arch arm64 -fPIC [-march=native]
   -fno-finite-math-only -Xclang -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math
   -fno-signed-zeros -fno-rounding-math`) and links with ThinLTO. Its shipped lanes and per-pair loops equal the
   linked `dtwc_cl` loops instruction for instruction (modulo GPR numbers) [confirmed]. `kbench_native time 7`
   (run 1, `out/kbench_time_run1.txt`) and `kbench_default time 7` (run 2, wheel flags, `out/kbench_time_run2_default.txt`):
   random walks, 120 M cells (lanes) / 40 M cells (pairs) per measurement, all kernels of a shape interleaved, clock
   calibrated before and after each. `colprobe.cpp`, `dataprobe.cpp`: per-pair speed against column length and data.
5. **Headroom.** Variants are scratch copies of the header (`probes/kbench/make_variants.sh`, `make_skew.py`):
   `v_fmin` (the two `std::min` cells → `std::fmin`), `v_w16`/`v_w32` (`dtw_lanes` = 128 / 256 bytes), `v_fmin_w16`,
   `v_skew2` (kernel 1 computes columns j and j+1 per pass), `v_fmin_skew2`. Bitwise: `kbench_native check`
   (`out/kbench_check_native.txt`). Fills: `kbench_native fill 18 5`, `fill 6 5`, `fill 18 5 f64only` (run 2),
   `KB_SHAPE=s KB_SHIPPED_ONLY=1 kbench_native fill T 3 f64only` (scaling) — the loop of
   `Problem::fillDistanceMatrix_BruteForce` (Problem.cpp:748-813) with `run_openmp`'s `schedule(dynamic, max(1,
   N/(threads*8)))` (parallelisation.hpp:59-63, 104-116). Wheel `-Os`: binding TU recompiled without `-Os` and the
   extension relinked with the wheel's own link line in `$S/nominsize/` (nanobind 3.1.0 headers in `$S/nbvenv`);
   `probes/py_pair_timing.py`, `probes/py_fill_timing.py` run against the installed wheel and against
   `$S/pkg_nominsize` (a copy of the package with the relinked `.so`).

## 1. What ships

| | `-target-cpu` (scheduling model) | features | used by |
| --- | --- | --- | --- |
| `-march=native` | apple-m1 | `+v8.7a` +bf16 +i8mm +dotprod +rdm +lse … (23; drops the default's aes/sha2/sha3/fullfp16) | `build/` (CLI), `build-matlab/` (MEX) |
| no flag | apple-m1 | `+v8.5a` +aes +sha2 +sha3 +fullfp16 +dotprod … (27) | wheels, release archives (level v3 adds nothing on arm64) |
| `-mcpu=native` | **apple-m4** | v8.7a + sme/sme2 … (26) | — |
| `-mcpu=apple-m5` | apple-m5 | v8.7a + sme2p1, cssc, mte … | — |

[confirmed, `out/cc1_*.txt`] AppleClang 21 knows `apple-m5` (`-print-supported-cpus` lists up to apple-m6), but
host detection maps this M5 to `apple-m4`, and on AArch64 `-march` sets features only: no `-tune-cpu` is passed and
the target CPU stays `apple-m1`. Kernel TUs: `build/` and the MEX's core compile with `-O3 -DNDEBUG -std=c++20
-flto=thin -march=native` + the fast FP model (cmake/StandardProjectSettings.cmake:89-105); the MEX's own TU
`dtwc_mex.cpp` with `-O3 -march=native` and no FP-model flags; the wheel's core identical minus `-march`; the wheel's
binding TU `python/src/_dtwcpp_core.cpp` with `-O3 … -fno-stack-protector -Os` (`out/wheel_build.log` line 538:
nanobind's default for `nanobind_add_module`, python/CMakeLists.txt:29,31) [confirmed].

**`-march` does not change the kernels.** All lanes and per-pair inner loops of `kbench_native` (`-march=native`),
`kbench_default` (no flag) and `kbench_m5` (`-mcpu=apple-m5`) are identical (the `-mcpu=apple-m5` scalar loops differ
only in general-purpose register numbers); the wheel's and the MEX's lanes loops equal the CLI's [confirmed:
`asm/lanes_shipped.s`, `out/diff_m5_*.txt`].

## 2. Shipped assembly (trimmed listings in `asm/`)

| loop | where (CLI / wheel / MEX) | insns per cell | min and chain | cost | B1 | B2 |
| --- | --- | --- | --- | --- | --- | --- |
| lanes f64 L1 (dtw_kernel.hpp:460-469) | 1000ecd8c / c6dd8 / 31edfc | 36 per row of 8 = 4.5 | `fcmgt.2d`+`bit/bsl` (min diag, up), `fcmgt.2d`+`bif` (min ·, left), `fadd.2d`; chain left→fcmgt→bif→fadd | `fabd.2d` | PASS | PASS |
| lanes f64 squared | 1000ec020 / c606c / 31e090 | 40 / 8 = 5.0 | same | `fsub.2d`+`fmul.2d`, no FMA into the `fadd` | PASS | PASS |
| lanes f32 L1 / squared | 1000f0310, 1000ee078 / ca35c, c80c4 / 322380, 3200e8 | 36 / 16 = 2.25; 40 / 16 = 2.5 | same on `.4s` | `fabd.4s`; `fsub`+`fmul` | PASS | PASS |
| per-pair kernel 1 f64 L1 (dtw_kernel.hpp:261-268) | 1000ad708 / — / 3fc4 | 13 | `fcmp`+`fcsel` ×2, `fadd`; chain left→fcmp→fcsel→fadd | `fabd` | — | PASS |
| per-pair kernel 2 f64 L1 (:333-340) | 1000ad554 / — / 10128 | 13 | same | `fabd` | — | PASS |
| per-pair f64 squared, f32 L1, f32 squared | 1000ad2a0, 1000bb25c, — / —, 8499c, 84270 / … | 14, 13, 14 | same | `fsub`+`fmul`; `fabd s`; `fsub s`+`fmul s` | — | PASS |
| **wheel per-pair f64 L1, f64 squared** (kernels 1 and 2) | — / 1ec44, 1ee58; 1e770 / — | **16**, **17** | as above **plus** the row minimum `fcmp`+`fccmp`+`fcsel` on every cell (dtw_kernel.hpp:267, :339 not unswitched) | `fabd`; `fsub`+`fmul` | — | PASS |
| ADTW per-pair (separate symbols) | 1000b7968 (linear), 1000b7784 (banded) | 15 | `fadd`(+penalty)→`fcmp`→`fcsel`→`fadd` | `fabd` | — | PASS |
| WDTW per-pair | 1000b58a8 | 20 | as Standard, weight `|i-j|` index per cell | `fabd`+`fmul` | — | PASS |
| lb_keogh (lower_bound_impl.hpp:173-197) | 10009570c / 5db14 … / … | 32 per 8 elements | `x > 0 ? x : 0` → `fmaxnm.2d` (exact) | `fsub.2d` | — | PASS |
| z_normalize (z_normalize.hpp:42-77), Python binding | — / 21744, 217a8, 2182c / — | 14–18 per **2** elements | `-Os` tail-folding: `cmhs`/`xtn`/`tbz`/lane `ld1` per element pair | — | — | PASS |
| FasterPAM `find_best_swap` (fast_pam.cpp:145-158) | 1000880c8–10008812c | ~16 per point, branchy | tri_index per point (`cmp/csel/csel/madd/lsr/add`); `acc += doj - d1` reassociated to `(acc + doj) - d1` | — | — | PASS |
| FasterPAM assignment (fast_pam.cpp:83-99) | 100086c08 | 19 per medoid | tri_index per medoid | — | — | PASS |

[confirmed] Details:

- **Lanes loop:** per row, 4 load instructions (`ldur q`, `ldp q,q`, `ldr q` for `s[i]`, `ld1r.2d` broadcasting
  `x[i]`), 2 `stp q`, 4 `mov.16b` (diag ← up), `subs`/`b.ne`; post-incremented pointers, no other address
  arithmetic, no stack access, no call. The pack (`dtw_kernel.hpp:427-430`) is lane stores (`st1.s {v}[k]`, f32),
  O(nW) per call. Two edge loops are scalar `d`/`s` arithmetic: column 0 (`:436-438`, CLI 1000ec670, 8 unrolled
  scalar cells per row) and the row-0 cell of each column (`:449-453`); they hold 2/L of the cells (2 % at L 100).
  So B1 is PASS for the hot loop and FAIL by the letter for those two edge loops, whose recurrence goes through
  memory (`s[i-1]`) where the main loop carries it in registers. The lanes function calls only the `vector`
  resize and `__tlv_atexit`; no `memset_pattern16` (the 10-05 fix holds).
- **Per-pair loops:** 3 loads per cell, one of them loop-invariant (the fixed column's sample is reloaded every cell
  because the DP store may alias the series), 1 store, 1 rename-eliminated move. First-row/column loops use
  `fminnm` against the `max()` constant, which is exact (one operand is a non-NaN constant).
- **Alignment:** loop entries are 4–128-byte aligned depending on the artifact; forcing 64-byte loop alignment
  (`-Wl,-mllvm,-align-loops=64`) changed no per-pair timing (`out/colprobe_al64_run1.txt`).
- **B2:** `probes/b2check.py` flags 8 / 6 / 150 innermost min+`fadd` loops in CLI / wheel / MEX; in the CLI they are
  Soft-DTW (per-cell `bl _exp`, `bl _log`, `bl _scalbn`: libm by design), MSM (exit branches), the Lagrangian MIP
  root and the fast_float parser — none is a Standard, ADTW, WDTW or lanes loop. Soft-DTW fails B2 by the letter
  (libm calls, not hardening traps); no `brk` in any DP loop.

## 3. Core bounds, measured (single thread, Super core, cycles) [confirmed]

| chain (latency per step) | cycles | | throughput (instructions per cycle) | per cycle |
| --- | --- | --- | --- | --- |
| `fcmgt.2d`→`bif`→`fadd.2d` (lanes cell) | 7.00 | | `fadd`/`fmul`/`fabd`/`fmin`/`fminnm`/`fcmgt`, scalar, `.2d`, `.4s` | 4.0 |
| `fminnm`→`fadd` (scalar or `.2d`/`.4s`) | 4.99–5.00 | | `bif.16b` | 2.8–3.1 |
| `fcmp`→`fcsel`→`fadd` (per-pair cell) | 5.12–5.25 (5.49 random picks) | | `fcmp` / `fcsel` / mixed | 2.0 / 2.0 / 2.46 |
| ADTW `fadd`→`fcmp`→`fcsel`→`fadd` | 7.59 | | `ldr q` / `ld1r` / `ldp q` | 3.0 / 3.0 / 1.42 (2.8 q) |
| row minimum `fcmp`→`fccmp`→`fcsel` | 5.98 | | `str q` / `stp q` | 2.0 / 1.0 (2 q) |
| `fcmgt`+`bif` pair; `fcmp`+`fcsel` pair | 4.00; 2.48 | | integer `add` | 6.95 |
| `fadd`→`fadd`; `fmul`; `fminnm`→`fminnm` | 2.09; 3.00; 1.60 | | `mov.16b` (renamed away) | 9.9 |

Sources: `out/ubench_run1.txt`, `out/opord_run1.txt`, `out/fcsel_pred_run1.txt`, `out/fptp2_run1.txt`,
`out/wheelloop_run1.txt`. The chain operand's position changes nothing (`opord`); `fcsel` keeps a true dependence
whichever operand it picks (5.12–5.49, `fcsel_pred`) — no select prediction. LLVM's scheduling model is not used
(no `llvm-mca` installed). One anomaly: the replayed `v_fmin_w16` loop sustains 32 FP ops in 7.40–7.48 cycles =
4.28–4.31 per cycle, chain broken or not (`out/w16asm2_run1.txt`), above the 4.0 every isolated test shows;
unexplained.

## 4. Shipped kernel speed against its bound (cycles per cell; run 1 `-march=native` / run 2 wheel flags; ratio to bound)

Bounds from §2 instruction counts and §3: lanes f64 = max(7.0 latency, 24 FP ops / 4 = 6.0) per row of 8 =
**0.875**; lanes f64 squared max(7.0, 28/4 = 7.0)/8 = **0.875**; lanes f32 = 7.0/16 = **0.4375** (both costs);
per-pair = max(chain 5.24, 4 `fcmp`/`fcsel` at 2/cycle = 2.0) = **5.24**. Loads, stores and rename are not binding
(≤ 2.0 and 3.6 cycles per row).

| kernel | L 100 | L 100, band 10 | L 1000 | L 1000, band 100 | B3 |
| --- | --- | --- | --- | --- | --- |
| lanes f64 L1 | 0.917 / 0.965 (1.05 / 1.10) | 0.849 / 0.900 (0.97 / 1.03) | 0.931 / 0.983 (1.06 / 1.12) | 0.921 / 0.970 (1.05 / 1.11) | at bound |
| lanes f64 squared | 0.944 / 1.000 (1.08 / 1.14) | 0.974 / 1.023 (1.11 / **1.17**) | 0.942 / 0.993 (1.08 / 1.13) | 0.942 / 0.985 (1.08 / 1.13) | at bound but one shape in run 2 (1.17) |
| lanes f32 L1 | 0.465 / 0.491 (1.06 / 1.12) | 0.441 / 0.464 (1.01 / 1.06) | 0.467 / 0.491 (1.07 / 1.12) | 0.462 / 0.485 (1.06 / 1.11) | at bound |
| lanes f32 squared | 0.485 / 0.513 (1.11 / **1.17**) | 0.503 / 0.530 (1.15 / **1.21**) | 0.471 / 0.497 (1.08 / 1.14) | 0.470 / 0.495 (1.07 / 1.13) | at bound in run 1; 1.17–1.21 in run 2 at L 100 |
| per-pair f64 L1 | 3.03 / 3.09 (0.58) | 4.50 / 4.36 (0.86) | 4.85 / 4.86 (0.93) | 3.89 / 3.89 (0.74) | at (below) bound |
| per-pair f64 squared | 3.08 / 3.16 (0.59) | 2.35 / 2.44 (0.45) | 4.81 / 4.81 (0.92) | 3.90 / 3.94 (0.74) | at (below) bound |
| per-pair f32 L1 | 2.98 / 3.04 (0.57) | 3.59 / 3.64 (0.68) | 4.86 / 4.86 (0.93) | 3.89 / 3.87 (0.74) | at (below) bound |
| wheel per-pair f64 L1 (Python `dtw`) | 4.71 | 4.79 | 5.91 | 5.48 | at its own bound (row-min chain 5.98) |

[confirmed; the wheel row is ns/cell from Python × 4.59 GHz, cycles [inferred]] Gcell/s at run 1: lanes f64 L1
4.9–5.4, f32 9.8–10.4; per-pair 0.94–1.96. The per-pair kernels run **below** their single-chain bound because the
out-of-order window overlaps consecutive columns: at L 1000 the speed approaches the chain (5.10 cycles/cell at
column length ≥ 8192), at short columns it halves (`out/colprobe_run1.txt`: 2.0–3.9 cycles at n 8–128) [confirmed].
The per-pair banded kernel at band 10 is placement-sensitive: `v_skew2` leaves kernel 2's source unchanged, yet its
copy measured 4.65–4.75 cycles/cell against the shipped 4.36–4.50 (L1) and 3.60–3.66 against 2.35–2.44 (squared)
[confirmed]; [inferred] the `col[i]` store-to-load between consecutive 21-cell columns meets memory-dependence
speculation; unexplained.

What this core allows for a Standard L1 cell: 4 FP ops (`fabd`, two mins, `fadd`) at 4 per cycle on 2 lanes →
**0.5 cycles/cell f64, 0.25 f32**, but only with a one-op min; with `fcmgt`+`bif` the floor is 0.75 / 0.375 and the
shipped W=8 sits on its 7-cycle chain above that. Shipped lanes are 1.7–2.0× above the core's floor (L1; squared,
whose floor is 5 ops = 0.625 f64, 1.5–1.6×); the per-pair kernels, one pair at a time, 3–5× above the scalar floor of
1 cycle/cell (4 FP ops at 4 per cycle).

## 5. Headroom (scratch variants; all bitwise identical to the shipped kernel over the sweep)

**NaN first (for `fmin`).** On AArch64 `std::min(a, b)` (libc++ `b < a ? b : a`) has no one-instruction form; `fminnm`
differs from it only when an operand is NaN (and in the sign of a zero). The lanes kernel is reached only through
`resolve_dtw_block_fn` (dtw_lanes.cpp:48-50: Standard, `MissingStrategy::Error`, univariate) from
`Problem::fillDistanceMatrix_BruteForce` (Problem.cpp:776) and one_batch_pam.cpp:101-108, both after
`Problem::validate_fill_request` (Problem.cpp:529, 537, 843), which refuses ±inf always and NaN unless a
missing-data strategy is set (Problem.cpp:697-712, warping.hpp:58-80). On finite input no DP value can be NaN:
`|a-b|` and `(a-b)²` of finite values are in [+0, +inf] (overflow to +inf is possible and allowed), the seed is the
cost, every cell is a min of such values plus a cost, `max()+cost` gives `max()` or +inf, and nothing subtracts DP
values; no `-0` arises (`fabd` and squares give +0, sums of +0 are +0). So `fminnm` is exact on every admitted
input. Missing-data cells (`SpanNanAware*`: cost 0 on NaN; `AROWCell`, dtw_kernel.hpp:181-201: NaN cost carries a
neighbour) never feed NaN to a min either. Only the unchecked per-pair layer, given NaN or ±inf, would answer
differently (`out/nanprobe.txt`: NaN mid-series → `max()` instead of NaN; equal +inf samples → +inf instead of
NaN), which warping.hpp:14-21 already disclaims ("anything else comes back as NaN, as the unreachable max() or as
an ordinary-looking number"). On x86-64-v3 the shipped min is one `vminpd`; `std::fmin` becomes `vminpd` +
`vcmpunordpd` + `vblendvpd` (`asm/x86_fmin_excerpt.s`, cross-compiled), so the change must be AArch64-only:
`#if defined(__aarch64__) || defined(_M_ARM64)`.

**Bitwise sweep** [confirmed, `out/kbench_check_native.txt`]: lanes — every n in 1..1000, bands {-1, 0, 1, n/10, n},
64 partners, 4 generators (uniform, random walk, integers 0–2 with copies of x, ±max/2..max overflowing to +inf),
f64/f32 × L1/squared: 1,280,000 outputs per combination, 314,203 / 314,478 of them +inf; per-pair — every nx in
1..1000 with an equal, a random and a near length, bands {-1, 0, 1, L/10, L, |nx-ny|}, same generators: 72,000
outputs per combination. Every variant's FNV-1a hash equals the shipped kernel's (e.g. lanes f64 L1
`d1b7dc83d39d00be`, per-pair f64 L1 `bb692a1b7a9e864e`); shipped lanes equal shipped per-pair on the sampled lanes.

| variant | change | gains, run 1 / run 2, at L100, L100 b10, L1000, L1000 b100 | B4 | 18-thread fill (6 threads) |
| --- | --- | --- | --- | --- |
| `v_fmin` lanes f64 L1 | `std::fmin` in the cell (dtw_kernel.hpp:74, 91) → `fminnm.2d`, 28 insns/row | 1.30/1.30, 1.42/1.43, 1.26/1.26, 1.29/1.29 | **PASS** | 1.30–1.48 (1.28–1.47) |
| `v_fmin` lanes f64 sq / f32 L1 / f32 sq | same | 1.25–1.49 / 1.26–1.41 / 1.23–1.47 | **PASS** | f32 1.36–1.47 (1.29–1.47) |
| `v_w16` lanes f64 L1 / f32 L1 | 16 doubles / 32 floats per call; one stack reload per row | 1.15/1.15, 1.02/1.03, 1.18/1.19, 1.16/1.16 / f32 1.16–1.19 except b10 1.00 | PASS (marginal) | 1.00–1.07 (1.02–1.11) |
| `v_w16` lanes squared | same | 1.01–1.06 | FAIL | — |
| `v_w32` lanes | 32 doubles / 64 floats; 26 stack loads/stores per row | 0.85–1.12 | FAIL | — |
| **`v_fmin_w16`** lanes f64 L1 | both | 1.90/1.89, 1.64/1.66, 1.99/2.00, 1.93/1.95 | **PASS** | **1.48–1.72 (1.62–1.81)** |
| `v_fmin_w16` f64 sq / f32 L1 / f32 sq | both | 1.52–1.60 / 1.41–1.95 / 1.41–1.63 | **PASS** | f32 1.33–1.66 (1.45–1.74) |
| `v_fmin` per-pair (f64 L1, sq; f32) | `fminnm`, 11 insns/cell | 0.85–1.31; 0.86–0.95; 0.85–1.18 | FAIL | ragged 0.90–0.93 |
| `v_skew2` per-pair (kernel 1 only) | columns j, j+1 per pass: two chains, 21 insns per 2 cells | f64 L1 1.48/1.45, 0.97/0.92\*, 1.97/1.93, 1.00/0.99\*; squared 1.46–1.94 unbanded, 0.65–0.67\* at L100 b10; f32 L1 1.44–1.98 unbanded | FAIL by the letter (\*banded shapes run the unchanged kernel 2) | ragged unbanded **1.54–1.57 (1.44)**, ragged band 10 0.96–1.02 |
| `v_fmin_skew2` per-pair f64 L1 / f32 L1 | both | 1.68/1.63, 1.27/1.20, 1.75/1.71, 1.04/1.03 / 1.00–1.75 | PASS (L1); FAIL squared (0.79–0.88 banded) | ragged 1.49–1.53 (1.57) |

[confirmed: `out/kbench_time_run*.txt`, `out/kbench_fill{18,18_run2,6}.txt`; every fill matrix bitwise equal to
the shipped fill.] Fill shapes: equal N 2000 × L 100 and N 500 × L 1000, unbanded and band L/10 (the lanes path);
ragged N 1000, L 90–110 (the per-pair path). In the fill runs `v_fmin` and `v_fmin_w16` also put `fmin` in the
per-pair kernel, which costs the ragged shapes 0.90–0.94; a lanes-only change leaves the per-pair path as shipped
[inferred]. Against bounds: `v_fmin_w16` runs at 0.47–0.52 cycles/cell f64 L1 against its 0.5 floor (0.94–1.03);
`v_fmin` W=8 at 0.60–0.78 against its 5-cycle chain (0.625): the off-chain `fminnm(diag, up)` costs ~0.8 cycle per
row by competing for the pipes (replay without it: 5.15 cycles/row, `out/lanesasm_run1.txt`); `v_skew2` at 2.46–2.52
against 2.62 (two chains of 5.24) at L 1000.

Fill scaling of the shipped code (f64, 1 → 6 → 18 threads, `out/kbench_fill_scaling.txt`) [confirmed]: equal N 2000
L 100 4.72 → 24.2 → 67.5 Gcell/s (5.1×, 14.3×); equal N 500 L 1000 band 100 4.65 → 23.5 → 66.2 (5.05×, 14.2×);
ragged per-pair 1.50 → 7.69 → 22.3 (5.1×, 14.8×). The 12 "Performance" cores add ≈ 0.77 Super-core each
[inferred]. The same 18-thread shape measured 51–66 Gcell/s in different sessions (all-core thermal or power state
[inferred]); variant ratios come from interleaved repetitions.

## 6. Beyond the bands: the wheel's `-Os` per-pair kernel

The wheel's binding TU is compiled `-Os` (nanobind's default) and its `dtw` (python/src/_dtwcpp_core.cpp:614-623)
instantiates `distance::dtw<double>` → `dtwBanded<double>` → `run_dtw` → `dtw_kernel_banded<double,
SpanL1Cost<double>, StandardCell>` (distance.hpp:139-185 and 43-50, warping.hpp:118-128), the same template the
core's `-O3` `dtw_dispatch.cpp` instantiates for the fill (dtw_dispatch.cpp:114-131). The wheel holds a single
copy (16 instructions, `1ec44`/`1ee58`), the `-Os` one: `-Os` does not unswitch `do_early_abandon`, so the row
minimum (dtw_kernel.hpp:267, :339) runs on every cell as `fcmp`→`fccmp`→`fcsel`, a 5.98-cycle chain longer than the
cell's own [confirmed: listing, `out/wheelloop_run1.txt`: 5.98 vs 5.04 cycles/cell replayed]. Which copy LTO keeps
is the linker's choice (the binding object is first on the link line) [inferred]. f32, which the binding does not
instantiate, keeps the 13-instruction loop. Relinking the same objects with the binding TU at `-O3` restores the
13-instruction loop (+4 % `.so` size: 1,204,320 → 1,253,280 bytes) and its effect, same results:

| Python (`_dtwcpp_core.dtw`, random walks; ns/cell) | wheel | binding without `-Os` | gain |
| --- | --- | --- | --- |
| L 100 / L 100 band 10 | 1.025 / 1.043 | 0.658 / 0.903 | 1.56× / 1.15× |
| L 1000 / L 1000 band 100 | 1.287 / 1.193 | 1.063 / 0.852 | 1.21× / 1.40× |
| `compute_distance_matrix`, ragged N 1000 L 90–110, 18 / 1 threads (ms) | 349 / 5356 | 229 / 3363 | **1.52× / 1.59×** |
| same, equal N 2000 L 100 (lanes path), 18 / 1 threads (ms) | 299 / 4221 | 303 / 4222 | 1.0× |

[confirmed: `out/py_timing_wheel_vs_nominsize.txt` (wheel measured before and after the relinked run, ±0.5 %),
`out/py_fill_timing.txt`; fill checksums equal.] `-Os` also tail-folds `z_normalize`'s three passes (14–18
instructions per 2 elements, `asm/other_loops.s`), but Python's `z_normalize` costs 24 ns/element either way: the
by-value `std::vector<double>` conversion of the binding dominates [confirmed]. The CLI and the MEX compile their
TUs at `-O3` and carry the 13-instruction loop.

## Recommendations, ranked

| # | change | expected gain (measured here) | risk to digit-identical conformance | x86 | NaN semantics | size |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `NOMINSIZE` on `nanobind_add_module` (python/CMakeLists.txt:29, 31); or unswitch the early abandon in source (two loops in kernels 1 and 2) so no build setting can put the row minimum in the common loop | Python ragged fill 1.52–1.59×, single-pair `dtw` 1.15–1.56× | none: the CLI's instructions; outputs equal | x86/Linux wheels likely carry the same `-Os` copy [inferred, not measured] | unchanged | 1 keyword (+4 % `.so`); the source split ~15 lines [not measured] |
| 2 | lanes min → `fminnm` on AArch64, lanes only (a lanes cell, or `std::fmin` behind `#if defined(__aarch64__)`) | 1.23–1.49× single thread; 1.30–1.48× f64 fill at 18 threads, 1.28–1.47× at 6 | none on admitted input (bitwise over the sweep); lanes stay equal to per-pair | must not apply: 3 instructions there | differs only for NaN/±inf through the unchecked layer, already disclaimed | ~5 lines |
| 3 | with 2: `dtw_lanes<T>` = 128 / sizeof(T) (16 doubles, 32 floats) | with fmin 1.41–2.00× single thread; fill 1.48–1.72× (18 threads), 1.62–1.81× (6) | none (bitwise) | without fmin only 1.0–1.19× here; x86 effect unmeasured | as 2 | 1 constant; callers derive W (Problem.cpp:776, one_batch_pam.cpp:108); the last block per row pads up to 15 lanes [inferred: small at N ≥ 500] |
| 4 | per-pair kernel 1: two columns per pass (`v_skew2`), then the same for kernel 2 | unbanded single pair 1.44–1.98×; ragged fill 1.44–1.57×; kernel 2 not attempted | none (bitwise) | likely helps (latency-bound there too) [inferred] | unchanged (same argument roles) | ~40 lines per kernel |
| 5 | minor: lanes column 0 / row 0 in registers like the main loop; FasterPAM's per-point `tri_index` and reassociated accumulator | ≤ 2 % at L 100 [inferred]; FasterPAM not measured | none / reassociation is already allowed by the FP model | — | — | small |

Do not: `fmin` in the per-pair kernels (0.85–0.95× unbanded: the scalar `fminnm` chain loses to the core's overlap
of `fcsel` columns, unexplained), W = 32 (spills, 0.85–1.12×), `-mcpu`/`-march` tuning (no instruction changes).

## Not measured / out of scope

- Metal kernels: the Apple GPU ISA cannot be disassembled with Command Line Tools. Gurobi: a closed solver, not a
  DTWC++ CPU kernel. x86: only the `fmin` lowering, by cross-compilation; no x86 timing.
- SME/SME2 streaming mode (present on this M5): not explored.
- The "Performance" (perflevel1) cores' latencies and throughputs; PMU counters (no kperf access);
  `llvm-mca` (not installed).
- Kernel timing of ADTW, WDTW, Soft-DTW, MSM, TWE, AROW and the multivariate kernels (only the ADTW chain latency);
  the envelope (deque-based, scalar by construction); FasterPAM loop speed.
- Two anomalies left open: 4.3 FP ops/cycle in the `v_fmin_w16` replay against 4.0 everywhere else, and the
  2.35–4.75 cycles/cell spread of identical banded per-pair source at band 10.
- `bench_dtw_baseline` and `bench_openmp_schedule` in `build/bin` date from 09-22 (before the lanes kernel;
  `DTWC_BUILD_BENCHMARK` is OFF in `build/`), so the probes above replace them.

## Files

Kept beside this record in `2026-10-06-mac-kernel-assembly/`: `probes/` — `kbench/` (kbench.cpp, build.sh, and
make_variants.sh with make_skew.py, which regenerate the `v_*.hpp` variants from `dtw_kernel.hpp` at `33877edf`;
colprobe.cpp, dataprobe.cpp, nanprobe.cpp), `ubench/` (the generators gen_ubench.py and gen_fptp2.py and the
hand-written microbenchmarks), `osprobe/zn.cpp`, `lanes_x86.cpp`, `py_pair_timing.py`, `py_fill_timing.py`,
`loops.py`, `lane_loops.py`, `normregs.py`, `b2check.py`, `symsizes.py`; `asm/` — `lanes_shipped.s`,
`perpair_shipped.s`, `variants.s`, `other_loops.s`, `x86_fmin_excerpt.s`. Not kept: the generated sources, the
binaries, the full disassemblies and per-loop listings, the raw outputs (`out/`) and the scratch builds; the tables
above are the record.

## Re-measured by the orchestrator (after the agent finished; load 1.45, nothing building)

Opened: python/CMakeLists.txt:29,31 (no `NOMINSIZE`); nanobind-config.cmake:332,610 (`-Os` unless `NOMINSIZE`);
dtw_lanes.cpp:48-50; Problem.cpp:529, 697-712, 843; warping.hpp:14-21; `asm/lanes_shipped.s` (`fabd.2d`, a
`fcmgt.2d` + `bif`/`bsl` per min, `fadd.2d`, four accumulators).

`KB_SHAPE=0|2 kbench_native time 5`, clock 4.594 GHz [confirmed]:

| shape | shipped cycles/cell | `v_fmin` | `v_fmin_w16` |
| --- | --- | --- | --- |
| lanes f64 L1, L 100 | 0.923 | 1.311× | 1.856× |
| lanes f64 L1, L 100 band 10 | 0.872 | 1.452× | 1.666× |
| lanes f32 L1, L 100 | 0.473 | 1.286× | 1.813× |
| lanes f32 L1, L 100 band 10 | 0.447 | 1.418× | 1.418× |

Per-pair f64 L1, L 100: `v_skew2` 1.488× unbanded; band 10 0.926× (the placement anomaly again) [confirmed].

Python, the agent's wheel against its relink with the binding TU at `-O3` (`py_pair_timing.py`, `py_fill_timing.py`)
[confirmed]: `dtw` L 100 1.0247 → 0.6588 ns/cell (1.56×), L 1000 1.2899 → 1.0618 (1.21×), L 1000 band 100 1.1943 →
0.8545 (1.40×); the ragged fill (N 1000, L 90–110, 18 threads) 350.8 → 230.2 ms (1.52×), checksum 1.721431e+08 in
both; the equal-length fill (N 2000, L 100, the lanes path) 300.3 → 303.0 ms, unchanged.
