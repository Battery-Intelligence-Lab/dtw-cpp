# 2026-09-29 — Windows GPU fill baseline (RTX 4000 Ada)

**Question:** what does the CUDA pairwise fill cost before phase C touches it? Reference for W4a / W13a bands.

## Machine and build

| Fact | Value |
| --- | --- |
| GPU | NVIDIA RTX 4000 Ada Generation, 20 GB, compute 8.9, driver 596.72 (FP64 rate "Slow", so `Auto` = FP32) |
| CPU | Intel Core Ultra 9 285 (host side only) |
| Build | `build/cuda-verify-0928`: MSVC 14.50 + nvcc 13.0 with `-allow-unsupported-compiler`, Release, sm_89 (recipe in `plans/2026-09-27-audit/phaseA_measurements.md`) |
| Source | built 2026-09-28 03:27 from `f859ab5`; `git diff f859ab5 4dcc39d -- dtwc/cuda` is Z1's deletions of the 1-vs-N / K-vs-N kernels and GPU LB_Keogh plus one error message; the pairwise fill path is unchanged |
| Load | three agents building on the CPU at the time; the kernel runs on the GPU (host overhead advisory) |

## Command

```sh
build/cuda-verify-0928/bin/bench_cuda_dtw.exe "--benchmark_filter=^BM_cuda_(distanceMatrix|scaling_N|scaling_L)" \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

## Numbers (median of 3, real time, FP32 through `CUDAPrecision::Auto`) [confirmed]

| benchmark (N / L) | time | Gcell/s |
| --- | --- | --- |
| `BM_cuda_distanceMatrix/100/100` | 0.248 ms | 200 |
| `BM_cuda_distanceMatrix/100/500` | 13.03 ms | 95 |
| `BM_cuda_distanceMatrix/100/1000` | 40.76 ms | 121 |
| `BM_cuda_distanceMatrix/200/500` | 52.01 ms | 96 |
| `BM_cuda_scaling_N/500` (L = 500) | 323 ms | 97 |
| `BM_cuda_scaling_L/250` (N = 50) | 0.261 ms | 293 |
| `BM_cuda_scaling_L/500` (N = 50) | 3.393 ms | 90 |
| `BM_cuda_scaling_L/2000` (N = 50) | 33.73 ms | 145 |
| `BM_cuda_scaling_L/4000` (N = 50) | 222.2 ms | 88 (3-buffer wavefront path above L = 2048) |

Gcell/s = N(N−1)/2 · L² / time. For scale: the CPU fill `BM_fillDistanceMatrix/100/1000/-1` (FP64, 24 threads) was
1402 ms = 3.5 Gcell/s at `4dcc39d` (`2026-09-29-windows-kernel-msvc-stl-min.md`), before K1 and P1.

Observations for phase C [inferred]: the rate falls from 293 Gcell/s at L = 250 to 90 at L = 500 (13× the time for
4× the cells), a kernel-path change, and L = 4000 runs 39 % below L = 2000's rate on the 3-buffer path; W4a's A/B
should cover both boundaries. The benchmark's largest N is 500; the fills that matter on a GPU are N = 5k–100k.
