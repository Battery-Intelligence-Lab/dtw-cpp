# 2026-09-29 — K1: the DP cell makes no library call

**Question:** on the Windows box (clang + the MSVC STL), do nested two-argument `std::min` (step 1, `8bd6881`)
and carrying the value each DP cell stores into the next cell, with the buffer pointer hoisted (step 2,
`123146b`), make the CPU kernels ≥ 3× faster without changing a digit? Follows
`2026-09-29-windows-kernel-msvc-stl-min.md` (base `ff53782`: `call __std_min_d` in every cell).

**Answer:** yes for the kernels, no for the default fill. Pinned, the linear and banded kernels run at 1.36 ns per
cell (7.0× and 5.7× the interleaved base; 5.3× and 5.4× the quiet baseline's 7.225 ms and 1.395 ms): both bands
hold, even in the worst round. The unbanded all-core fill gains 1.36× only, because it runs the EAPruned kernel,
whose inner loops had no call to remove: that band is FALSIFIED. Every output compared is bit-identical, and MSVC
cl's loops are call-free too.

## Machine

| Fact | Value |
| --- | --- |
| OS | Windows 11 (AMD64) |
| CPU | Intel Core Ultra 9 285: 8 P-cores (logical 0, 1, 10–13, 22, 23) + 16 E-cores, AVX2, no AVX-512 |
| Memory | 127.5 GiB |
| Compiler | Clang 21.1.8 (x86_64-pc-windows-msvc, MSVC STL 14.50), Release, `-O3 -march=native -flto=thin`, `DTWC_FP_MODEL=fast` (`-fassociative-math -fno-signed-zeros …`, no finite-math) |
| Build | `build/` of the K1 worktree: Ninja, IPO, llfio, HiGHS, Gurobi, benchmarks ON (`scripts/machine_facts.py --build-dir build`) |
| Repository | pb/K1 @ `d114677`, clean when measured |

Single-thread runs pinned to P-core 12 (`/affinity 1000`); the fill uses all 24 cores. The machine was shared with
other agents' builds: CPU load was between 23 % and 99 % when rounds started (15 % at the end), so wall-clock is
advisory. The assembly is the proof.

## Band — registered by the orchestrator before the run

| subject | band |
| --- | --- |
| `BM_dtwFull_L/1000`, pinned | ≤ 2.4 ms (≥ 3× faster than 7.225 ms) |
| `BM_dtwBanded/1000/100`, pinned | ≤ 0.46 ms (≥ 3× faster than 1.395 ms) |
| `BM_fillDistanceMatrix/100/1000/-1`, all cores | ≥ 2× faster than 1402 ms (≤ 701 ms) |
| `cpp_conformance` | digit-identical, reference not regenerated |

Predicted before the run: the two kernels hold (about 1.3–1.6 ns/cell, the probe's number); the fill does not.
An unbanded Standard fill binds `dtwFull_eap` (`core/dtw_dispatch.cpp`, `make_standard`), and the EAPruned
kernel's min was already written out by hand, so step 1 cannot reach it; step 2's carry was expected to give it
≤ 1.35×.

## Commands

```bat
rem pinned.bat <exe> <out.json>
start "" /b /wait /affinity 1000 %1 "--benchmark_filter=^BM_(dtwFull_L/(1000|4000)|dtwFull/1000|dtwBanded/(1000|4000)/100)$" --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=%2 --benchmark_out_format=json
rem fill.bat <exe> <out.json>
%1 "--benchmark_filter=^BM_fillDistanceMatrix/(100/1000/-1|50/1000/50)$" --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=%2 --benchmark_out_format=json
```

Three binaries of `bench_dtw_baseline` from the same build directory: base (`ff53782`), step 1 (`8bd6881`) and tip
(`d114677`, the library as at `123146b`). Seven rounds; each round runs all three pinned, then all three on all
cores, and the order rotates each round (base-s1-tip, s1-tip-base, tip-base-s1).

## Numbers [confirmed]

Median over the seven rounds of each run's median of 5 repetitions (real time); spread = lowest–highest round.

All times in ms.

| benchmark | base `ff53782` | step 1 `8bd6881` | tip `d114677` | base / tip | ns/cell, tip |
| --- | --- | --- | --- | --- | --- |
| `BM_dtwBanded/1000/100` | 1.450 (1.404–2.064) | 0.487 (0.470–0.851) | 0.257 (0.245–0.324) | 5.65× | 1.35 |
| `BM_dtwBanded/4000/100` | 6.090 (5.776–7.780) | 1.980 (1.921–2.864) | 1.048 (1.024–1.470) | 5.81× | 1.32 |
| `BM_dtwFull/1000` | 8.131 (7.702–12.573) | 2.093 (2.049–2.949) | 2.133 (2.080–3.448) | 3.81× | 2.13 |
| `BM_dtwFull_L/1000` | 9.525 (7.247–14.887) | 2.726 (2.608–3.095) | 1.362 (1.326–2.079) | 7.00× | 1.36 |
| `BM_dtwFull_L/4000` | 154.933 (116.449–183.985) | 46.704 (42.543–55.978) | 21.636 (21.251–32.511) | 7.16× | 1.35 |
| `BM_fillDistanceMatrix/100/1000/-1` (24 threads) | 1406.837 (1147.454–2021.088) | 1293.353 (1149.278–1974.535) | 1033.088 (867.439–1226.847) | 1.36× | — |
| `BM_fillDistanceMatrix/50/1000/50` (24 threads, band 50) | 53.571 (47.558–91.674) | 20.188 (16.057–28.541) | 10.581 (7.791–10.799) | 5.06× | — |

## Band verdict

| subject | tip | band | verdict |
| --- | --- | --- | --- |
| `BM_dtwFull_L/1000` | 1.362 ms (worst round 2.079) | ≤ 2.4 ms | **held** (5.3× the 7.225 ms baseline) |
| `BM_dtwBanded/1000/100` | 0.257 ms (worst round 0.324) | ≤ 0.46 ms | **held** (5.4× the 1.395 ms baseline) |
| `BM_fillDistanceMatrix/100/1000/-1` | 1033 ms (best round 867) | ≤ 701 ms | **FALSIFIED** (1.36× interleaved) |
| `cpp_conformance` | passes, reference untouched | digit-identical | **held** |

Why the fill does not move [confirmed]: `make_standard` binds `dtwFull_eap` whenever `band < 0`, and the base
EAPruned kernel (the `old_core` copy compiled with the library's flags) has 7 innermost loops and no call in any
of them: its min was written out by hand. Step 1 leaves the fill where it was (best rounds 1147 and 1149 ms; the
medians differ inside the spread), and step 2's carry of `curr[s-1]` gives the 1.36× (best round 867 ms). What is
left is per-cell branching on the pruning window (`prev_lo`/`prev_hi` bounds, `d <= thr`) [inferred]; removing it
means splitting each row into its live segments, a kernel restructure outside K1. The banded fill
(`BM_fillDistanceMatrix/50/1000/50`, the banded Standard kernel) gains 5.1×.

Two further observations. Step 1 alone takes the linear kernel from 9.5 to 2.7 ms; the carry and the hoisted
pointer take it to 1.36 ms, because without TBAA every store forced the buffer pointer and the just-stored value
to be reloaded. The full-matrix kernel gains 3.8× from step 1 and nothing measurable from step 2
(2.09 → 2.13 ms, inside the spread); at 2.1 ns per cell it stays slower than the rolling kernels [inferred: it
writes an 8 MB matrix per pair].

## Inner loops, before and after [confirmed]

`dtw_kernel_linear<double, …, StandardCell>` and `dtw_kernel_banded<…>` for the L1 metric, as `dtwc::dtwFull_L` and
`dtwc::dtwBanded` reach them, compiled with `dtwc/core/dtw.cpp`'s flags from `build/compile_commands.json` plus
`-S -masm=intel`, without `-flto`. Instructions verbatim; the comments are annotations; one or two cells shown
where the loop is unrolled.

Linear, base — per cell: reload the vector's data pointer and the cost lambda's two captured pointers, reload
`dp[i-1, j]` stored one iteration earlier, spill three values, call:

```asm
.LBB24_16:                          ; =>  This Inner Loop Header: Depth=2
    mov     rdi, qword ptr [rbx]              ; short_side.data(), reloaded
    vmovsd  xmm0, qword ptr [rdi + 8*r15 - 8] ; dp[i-1, j], stored one iteration ago
    vmovsd  xmm12, qword ptr [rdi + 8*r15]    ; dp[i, j-1]
    mov     rax, qword ptr [rsi]              ; cost lambda: x, reloaded
    mov     rcx, qword ptr [rsi + 8]          ; cost lambda: y, reloaded
    vmovsd  xmm1, qword ptr [rax + 8*r15]
    vsubsd  xmm1, xmm1, qword ptr [rcx + 8*rbp]
    vandpd  xmm13, xmm8, xmm1                 ; |x - y|
    vmovsd  qword ptr [rsp + 32], xmm11       ; spill diag, up, left
    vmovsd  qword ptr [rsp + 40], xmm12
    vmovsd  qword ptr [rsp + 48], xmm0
    mov     rcx, r13
    mov     rdx, r12
    call    __std_min_d
    vaddsd  xmm0, xmm13, xmm0
    vmovsd  qword ptr [rdi + 8*r15], xmm0
    inc     r15
    vmovapd xmm11, xmm12
    cmp     r14, r15
    jne     .LBB24_16
```

Linear, tip — unrolled ×4, two cells of four: the buffer pointer stays in `rax`; cell 2's `left` is cell 1's
result, still in `xmm2`, and goes straight into `vminsd`; no call:

```asm
.LBB24_46:                          ; =>  This Inner Loop Header: Depth=2
    mov     rbx, qword ptr [r8]               ; cost lambda: x (still reloaded, off the dependency chain)
    mov     r14, qword ptr [r8 + 8]           ; cost lambda: y
    vmovsd  xmm6, qword ptr [rbx + 8*rdi + 8]
    vsubsd  xmm6, xmm6, qword ptr [r14 + 8*rsi]
    vandpd  xmm6, xmm6, xmm3                  ; |x - y|
    vmovsd  xmm7, qword ptr [rax + 8*rdi + 8] ; up = dp[i, j-1]
    vmovsd  xmm8, qword ptr [rax + 8*rdi + 16]; cell 2's up
    vminsd  xmm2, xmm7, xmm2                  ; min(diag, up)
    vminsd  xmm2, xmm5, xmm2                  ; min(., left), left carried from the previous iteration
    vaddsd  xmm2, xmm6, xmm2
    vmovsd  qword ptr [rax + 8*rdi + 8], xmm2 ; dp[i, j], stays in xmm2
    mov     rbx, qword ptr [r8]
    mov     r14, qword ptr [r8 + 8]
    vmovsd  xmm5, qword ptr [rbx + 8*rdi + 16]
    vsubsd  xmm5, xmm5, qword ptr [r14 + 8*rsi]
    vandpd  xmm5, xmm5, xmm3
    vminsd  xmm6, xmm8, xmm7                  ; min(diag = cell 1's up, up)
    vminsd  xmm2, xmm2, xmm6                  ; min(., left = cell 1's result, no reload)
    vaddsd  xmm2, xmm5, xmm2
    vmovsd  qword ptr [rax + 8*rdi + 16], xmm2
    ; ... cells 3 and 4 ...
    add     rdi, 4
    cmp     r9, rdi
    jne     .LBB24_46
```

Banded, base — the same pattern (reload of `col.data()`, reload of `dp[j, i-1]`, three spills, `call __std_min_d`):

```asm
.LBB40_66:                          ; =>  This Inner Loop Header: Depth=2
    vmovapd xmm0, xmm13
    mov     r14, qword ptr [rdi]              ; col.data(), reloaded
    vmovsd  xmm1, qword ptr [r14 + 8*r12 - 8] ; dp[j, i-1], stored one iteration ago
    vmovsd  xmm13, qword ptr [r14 + 8*r12]    ; dp[j-1, i]
    mov     rax, qword ptr [rsi]
    mov     rcx, qword ptr [rsi + 8]
    vmovsd  xmm2, qword ptr [rax + 8*r15]
    vsubsd  xmm14, xmm2, qword ptr [rcx + 8*r12]
    vmovsd  qword ptr [rsp + 64], xmm0
    vmovsd  qword ptr [rsp + 72], xmm13
    vmovsd  qword ptr [rsp + 80], xmm1
    lea     rcx, [rsp + 64]
    lea     rdx, [rsp + 88]
    call    __std_min_d
    vucomisd xmm6, xmm7                       ; do_early_abandon
    vandpd  xmm1, xmm14, xmm10
    vaddsd  xmm0, xmm0, xmm1
    vmovsd  qword ptr [r14 + 8*r12], xmm0
    jb      .LBB40_65
```

Banded, tip — unrolled ×2, one cell of two: `col` in `r15`, the carried value (`xmm9`) feeds the next cell's
`vminsd`, no call (the early-abandon `row_min` stays in the loop: a blend for this cell, a branch for the next):

```asm
.LBB40_79:                          ; =>  This Inner Loop Header: Depth=2
    mov     rcx, qword ptr [rdx]
    mov     r9, qword ptr [rdx + 8]
    vmovsd  xmm9, qword ptr [rcx + 8*r8]
    vsubsd  xmm9, xmm9, qword ptr [r9 + 8*r12]
    vandpd  xmm9, xmm9, xmm7                  ; |x - y|
    vmovsd  xmm12, qword ptr [r15 + 8*r12]    ; dp[j-1, i]
    vminsd  xmm13, xmm12, xmm11               ; min(diag, up)
    vmovsd  xmm11, qword ptr [r15 + 8*r12 + 8]
    vminsd  xmm10, xmm10, xmm13               ; min(., left): left carried
    vaddsd  xmm9, xmm9, xmm10
    vmovsd  qword ptr [r15 + 8*r12], xmm9
    vminsd  xmm10, xmm9, xmm8                 ; row_min, blended in when early abandon is on
    vblendvpd xmm8, xmm8, xmm10, xmm2
    ; ... second cell ...
    jb      .LBB40_78
```

The cost lambda's captured pointers are still reloaded every cell. The cause is the same as for the buffer
pointer: clang's Windows driver passes `-relaxed-aliasing` by default (no TBAA, matching MSVC), so a `double` store
may alias any pointer held in memory, and on Win64 the by-value lambda lives in memory. These loads are off the
recurrence's dependency chain (`vminsd` → `vaddsd` on `left`; 1.36 ns is about 7 cycles per cell [inferred]), so
they were left alone.

## MSVC cl [confirmed]

Second build directory `build-msvc` in the K1 worktree: MSVC 19.50.35723 (vcvars64 of Visual Studio 18), Ninja,
Release (`/O2 /Ob2 /arch:AVX2 /fp:precise /fp:contract /GL`, OpenMP 2.0 via `/openmp:experimental`), CUDA, Arrow and
LLFIO OFF as briefed, HiGHS and Gurobi OFF as well (the kernels use neither; it spared the shared machine the HiGHS
build). `dtwc++` and 18 kernel tests built with 4 warnings, all C4244 inside STL headers from `cli/run.cpp` and
`scores.cpp`. `ctest --test-dir build-msvc -j1 -R <the 18>`: 18/18 passed, `cpp_conformance` included, so cl
reproduces the reference digit for digit as well.

Listings: `dtw.cpp`'s cl command from `build-msvc/compile_commands.json` with `/FA`, without `/GL` (which, like
`-flto`, defers code generation to the link). At base, cl's `dtw_kernel_linear<double, …, StandardCell>` calls
`__std_min_element_d` inside its loops, the main one included: under cl the MSVC STL sends `std::min({…})` through
its vectorised `min_element`, under clang through `__std_min_d`. At the tip, none of the 5 loops of the L1 linear
kernel and none of the 8 of the L1 banded kernel contains a call. The benchmarked linear loop, first cell of four:

```asm
$LL104@dtw_kernel:
    vmovsd  xmm3, QWORD PTR [rdx+rcx*8]        ; up = dp[i, j-1]; the buffer stays in rdx
    vmovsd  xmm0, QWORD PTR [r10+rcx*8]
    vsubsd  xmm1, xmm0, QWORD PTR [r9+r8*8]
    vmovsd  xmm5, QWORD PTR [rdx+r11*8]
    vcomisd xmm6, xmm3
    vandpd  xmm1, xmm1, xmm7                   ; |x - y|
    vmovsd  QWORD PTR up$[rbp-97], xmm3        ; std::min returns a reference: cl picks the
    vmovsd  QWORD PTR diag$[rbp-97], xmm6      ; address of diag or up with cmovbe and loads it
    vmovsd  xmm6, QWORD PTR [rdx+rbx*8]
    lea     r12, QWORD PTR diag$[rbp-97]
    lea     rax, QWORD PTR up$[rbp-97]
    cmovbe  rax, r12
    lea     rbx, QWORD PTR [rbx+4]
    lea     r12, QWORD PTR diag$[rbp-97]
    lea     r11, QWORD PTR [r11+4]
    vminsd  xmm0, xmm4, QWORD PTR [rax]        ; min(., left): left carried in xmm4
    vmovsd  xmm4, QWORD PTR [rdx+rcx*8+8]
    vaddsd  xmm2, xmm1, xmm0
    vmovsd  QWORD PTR [rdx+rcx*8], xmm2        ; dp[i, j]; cell 2 then runs vminsd xmm0, xmm2, [rax]
```

The inner `min(diag, up)` goes through two stack slots (a store-to-load round trip), but off the dependency chain,
which is `vminsd` → `vaddsd` on `left` as with clang. Not timed: the brief asks for the listing and the tests.

## Digit identity [confirmed]

- `cpp_conformance` passes at every step against the untouched reference.
- Old-vs-new probe: the base headers copied into `namespace dtwc::old_core`, compiled with the library's flags in
  one binary with the new ones: full / linear / banded (bands 0, 1, 2, 5, 17, 200) / EAPruned × Standard, ADTW,
  AROW, Soft cells, early abandon on a third of the calls, MSM (c = 0.1, 1), TWE (ν × λ), double and float,
  lengths 1–90 with exact ties, signed zeros and 8 % NaN: 60,000 results, 0 bit differences, at step 1 and at the
  tip (the probe has 80 `call __std_min_d` and 74 `call __std_min_f` sites, in the old kernels and in the triple
  check that follows, so the old side really runs the library helper). `std::min({a, b, c})` against
  `std::min(std::min(a, b), c)` on all 216 triples of {NaN, −0, +0, 1, −1, max}: 0 differences.
- `align_squared` (barycenter DP), base text against new text: 6,000 runs, cost bits, DP matrix and warping
  path identical.
- `dtwc_cl` on 24 configurations (standard at bands −1, 6, 10, 25; DDTW, WDTW, ADTW at −1 and 10; soft-DTW; MSM;
  TWE; float32 standard, ADTW, MSM, TWE; NaN data under zero-cost, AROW and interpolate at −1 and 10), mixed
  lengths 20–260 or 137–143: 96 output files (matrix, labels, medoids, silhouettes) byte-identical to base. The
  base outputs are deterministic (a rerun was identical).

## The check

`test_codegen_no_calls` (`scripts/codegen_report.py --no-calls`, clang only) replays `dtw.cpp`'s compile command on
`scripts/codegen_probe.cpp` with `-S` and fails when an innermost loop of a function whose symbol names `dtwc`
contains `call`/`bl`. Tip: `CODEGEN_NO_CALLS tool=CLANG_~1.EXE inner_loops=72 calls=0 verdict=PASS`. With
`StandardCell` reverted to `std::min({diag, up, left})`: `inner_loops=57 calls=21 verdict=FAIL`, `callq __std_min_d`
/ `__std_min_f` in all five probe functions; with only `twe.hpp` reverted: `calls=1` in
`"??$twe_distance@N@core@dtwc@@YANPEBN_K01NN@Z"`.
