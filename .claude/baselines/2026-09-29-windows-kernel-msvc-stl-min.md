# 2026-09-29 — Windows CPU baseline; the DTW cell calls `__std_min_d`

**Question:** what do the CPU kernels and the fill cost on the Windows box at `4dcc39d`, and does the codegen
explain the gap to the Mac? **Answer:** the scalar DTW recurrence is ~7× slower per cell than on the Mac because
every cell makes an out-of-line library call. Fix scoped as unit K1 (PLAN, phase B).

## Machine

| Fact | Value |
| --- | --- |
| OS | Windows 11 (AMD64) |
| CPU | Intel Core Ultra 9 285 — 8 P-cores (logical 0, 1, 10–13, 22, 23) + 16 E-cores, AVX2, no AVX-512 |
| Memory | 127.5 GiB |
| Compiler | Clang 21.1.8 (x86_64-pc-windows-msvc, MSVC STL), Release, `-O3 -march=native -flto=thin`, FP subset (`-fassociative-math`, no finite-math) |
| Repository | design-2.0 @ `4dcc39d`, `build/` (Ninja, llfio ON, benchmarks ON) |
| Load | quiet: CPU load 1 %, no COMSOL, no builds |

P-core indices from `GetSystemCpuSetInformation` (EfficiencyClass 1). Single-thread runs pinned to CPU 12.

## Commands

```bat
start "" /b /wait /affinity 1000 build\bin\bench_dtw_baseline.exe "--benchmark_filter=^BM_(dtw|wdtw|ddtw|adtw|lb_keogh|z_normalize|compute_envelopes)" --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=kernels_p12.json --benchmark_out_format=json
build\bin\bench_dtw_baseline.exe "--benchmark_filter=^BM_fillDistanceMatrix" --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=fill_all.json --benchmark_out_format=json
```

## Numbers (median of 5, real time) [confirmed]

| benchmark | here | ns / cell | Mac M5 Pro (`2026-09-23-x27-eigen-gap.md`) |
| --- | --- | --- | --- |
| `BM_dtwFull_L/1000` (pinned) | 7.225 ms | 7.2 | 1.06 ms |
| `BM_dtwFull_L/4000` (pinned) | 115.8 ms | 7.2 | 17.6 ms |
| `BM_dtwFull/1000` (pinned) | 8.094 ms | 8.1 | — |
| `BM_dtwBanded/1000/100` (pinned) | 1.395 ms | 6.9 | — |
| `BM_dtwBanded/4000/100` (pinned) | 5.735 ms | 7.1 | — |
| `BM_lb_keogh/1000` (pinned) | 100 ns | 0.10 / element | — |
| `BM_fillDistanceMatrix/100/1000/-1` (24 threads) | 1402 ms | 0.28 aggregate | — |
| `BM_fillDistanceMatrix/50/500/50` (24 threads) | 26.7 ms | — | — |

Raw JSON: session scratchpad `bench/kernels_p12.json`, `bench/fill_all.json` (not kept; the table is the record).

## Cause [confirmed]

The inner loop of `dtw_kernel_linear<double, …, StandardCell>` (compiled with the build's flags, `-S -masm=intel`):

```asm
.LBB0_17:                                   ; one DP cell
    mov     rdi, qword ptr [r13]            ; reload the thread_local vector's data pointer
    vmovsd  xmm0, qword ptr [rdi + 8*rsi - 8] ; reload dp[i-1, j], stored one iteration ago
    ...
    vmovsd  qword ptr [rsp + 48], xmm8      ; spill diag, up, left to the stack
    vmovsd  qword ptr [rsp + 56], xmm9
    vmovsd  qword ptr [rsp + 64], xmm0
    call    __std_min_d                     ; std::min({diag, up, left})
    vaddsd  xmm0, xmm10, xmm0
    vmovsd  qword ptr [rdi + 8*rsi], xmm0
```

`StandardCell::combine` returns `std::min({diag, up, left}) + cost`. The MSVC STL implements
`std::min(initializer_list)` through `__std_min_d`, an out-of-line vectorised range minimum; libc++ inlines it.
`std::min_element` likewise tail-calls `__std_min_element_d`. `std::isnan` and two-argument `std::min` inline
(`vucomisd`, `vminsd`). The same initializer-list form is in `ADTWCell`, `AROWCell`, `msm.hpp`, `twe.hpp` and
`barycenter.cpp` (×2). MSVC builds (the Windows wheels and MEX) use the same STL.

## Probe [confirmed]

`2026-09-29-windows-kernel/min_carry_probe.cpp`: the library's `dtw_kernel_linear` with `StandardCell`, the same
kernel with a nested two-argument min (`std::min(std::min(diag, up), left)`, the comparisons `min_element` makes,
so NaN handling is unchanged), and a copy that also carries `dp[i-1, j]` in a register and hoists the row pointer.
Single thread on CPU 12, 11 interleaved rounds, median:

| L | initializer list | nested min | nested + carried | identical |
| --- | --- | --- | --- | --- |
| 100 | 7.09 ns/cell | 1.10 | 0.88 | yes |
| 1000 | 7.16 | 1.60 | 1.28 | yes |
| 4000 | 7.21 | 1.64 | 1.31 | yes |

Built with the flags above minus `-flto` and the dynamic-CRT trio. Results are bit-identical in all three forms.
