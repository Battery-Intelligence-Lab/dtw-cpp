# Apple M2 Max — Metal GPU Benchmarks

All numbers measured on **Apple M2 Max** (38-core GPU, 12 CPU cores = 8P + 4E, 64 GB unified memory), Apple Clang 17, macOS Darwin 25.3.0, 2026-04-12.

Raw Google Benchmark JSON in [results/mac_m2max/](results/mac_m2max/). Reproduce with:
```sh
cmake --preset clang-macos -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_METAL=ON -DDTWC_ENABLE_HIGHS=ON
cmake --build build --config Release --target bench_metal_dtw
./build/bin/bench_metal_dtw --benchmark_min_time=1s
```

Hardware limits:
- `maxThreadgroupMemoryLength` = 32 KB → threadgroup-memory kernel supports max_L ≤ 2730
- `recommendedMaxWorkingSetSize` = 77.76 GB unified (no H2D/D2H)
- Theoretical peak FP32 = 13.6 TFLOPS (38 cores × 128 ALUs × 2 FMA × 1.398 GHz)

---

## 1. CPU thread scaling baseline

Unbanded DTW, `fillDistanceMatrix/100/1000/-1` (4950 pairs × 1000² cells):

| Threads | Time (ms) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 14 618 | 1.00× | 100% |
| 2 | 7 756 | 1.88× | 94% |
| 4 | 4 163 | 3.51× | 88% |
| 6 | 2 913 | 5.02× | 84% |
| 8 (all P-cores) | 2 124 | 6.88× | 86% |
| 10 (8P + 2E) | 1 838 | 7.95× | 80% |
| **12 (8P + 4E)** | **1 599** | **9.14×** | **76%** |

`OMP_PROC_BIND` / `OMP_PLACES` pinning sweep at T=12 (default / close / spread on `cores`): 1606 / 1610 / 1609 ms — **identical within 0.3%**. pinning is a no-op on Darwin + Homebrew libomp. E-cores contribute a real +33% over 8P; don't cap to P-cores. (Lesson saved to [`.claude/LESSONS.md`](../.claude/LESSONS.md).)

---

## 2. Metal vs 12-thread CPU — unbanded (threadgroup kernel, max_L ≤ 2730)

| Workload | CPU (ms) | Metal (ms) | **Speedup** |
|---|---|---|---|
| 50 × 500 | 106 | 6.9 | **15.4×** |
| 100 × 500 | 404 | 26.2 | **15.4×** |
| 100 × 1000 | 1 648 | 139 | **11.9×** |
| 200 × 500 | 1 626 | 103 | **15.8×** |
| 50 × 2 500 | n/a | 151 | — |
| 75 × 500 | 234 | 15.0 | **15.6×** |
| 75 × 1 000 | 988 | 78.5 | **12.6×** |
| 75 × 2 500 | 5 914 | 342 | **17.3×** |

Throughput: **35–52 × 10⁹ DTW cells/sec**, **270–315 GFLOPS** (≈2% of 13.6 TFLOPS peak — expected for memory-bandwidth-bound DP recurrence).

---

## 3. Long series (global-memory kernel kicks in at max_L > 2730)

The global-memory kernel stores the 3 anti-diagonal buffers in device (unified) memory instead of threadgroup memory. Dispatcher picks automatically; pairs are chunked across multiple command buffers to avoid macOS's GPU watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`).

| Workload | CPU (ms) | Metal (ms) | **Speedup** |
|---|---|---|---|
| 10 × 10 000 | 3 058 | **108** | **28.3×** |
| 30 × 10 000 | 16 061 | **929** | **17.3×** |
| **75 × 10 000** | **92 537** (92.5 s) | **5 799** (5.8 s) | **15.9×** |

---

## 4. Banded Metal — before/after the band-range iteration fix

The initial kernel iterated all diagonal cells and wrote INF for out-of-band cells. The optimization clips the inner loop to the band range `[(k - band + 1) / 2, (k + band) / 2]`.

| Workload | Before (full iter + INF) | After (band-range) | Speedup |
|---|---|---|---|
| 30 × 10 000, band = 1 000 | 491 ms | **308 ms** | 1.59× |
| 75 × 2 500, band = 250 | 233 ms | **213 ms** | 1.09× |
| 75 × 10 000, band = 1 000 | 3 205 ms | **2 010 ms** | 1.59× |
| **75 × 10 000, band = 100** (tight) | 2 722 ms | **1 395 ms** | **1.95×** |

---

## 5. User-target workload: 75 × 10 000

| Config | Target | CPU actual | Metal actual |
|---|---|---|---|
| unbanded | CPU < 100 s | **92.5 s** ✓ | **5.80 s** |
| band = 100 | CPU < 4 s | **1.75 s** ✓ | **1.40 s** |
| band = 1 000 (L/10) | — | 17.5 s | 2.01 s |

At tight band (100) Metal is only 1.25× faster than CPU — the fixed per-diagonal barrier + index-setup overhead dominates once actual cell work is small. Further speedup needs a row-major banded kernel that avoids the 2·L-1 diagonal barriers (follow-on work).

---

## 6. Python & MATLAB wrapper overhead

Direct `compute_distance_matrix_metal` vs same path through language wrapper:

| Language | Overhead | Path |
|---|---|---|
| Python (nanobind) | **−0.5% to +0.9%** | `dtwcpp.compute_distance_matrix_metal` |
| MATLAB (MEX, R2024b) | +8.5% to +17% vs direct Metal | `Problem.fill_distance_matrix` |
| MATLAB | **< 1%** vs Python `Problem.fill_distance_matrix` (same path) | same |

The apparent MATLAB overhead (8.5–17%) turned out to be the `Problem::fillDistanceMatrix` wrapper path vs direct `compute_distance_matrix_metal`; that same gap appears in Python when going through `Problem`. Comparing like-for-like, both bindings are within 1% of native C++.

MATLAB 10 k verification:
- 10 × 10 000: 103.81 ms (C++ 108 ms)
- 30 × 10 000: 890.75 ms (C++ 929 ms)

---

## 7. Theoretical vs achieved FLOPs

M2 Max 38-core GPU theoretical peak = **13.6 TFLOPS FP32** (38 × 128 × 2 × 1.398 GHz).

Achieved on Metal DTW (each cell ≈ 6 FP32 ops: sub, abs, 3 min, add):

| Workload | Cells/sec | GFLOPS | % peak |
|---|---|---|---|
| 50 × 500 | 44.4 × 10⁹ | 267 | 1.96% |
| 100 × 500 | 47.3 × 10⁹ | 284 | 2.09% |
| 100 × 1 000 | 35.9 × 10⁹ | 215 | 1.58% |
| 200 × 500 | 48.5 × 10⁹ | 291 | 2.14% |
| 50 × 2 500 | 52.4 × 10⁹ | 315 | 2.31% |

Low FLOP fraction is expected: DTW is a DP recurrence (each cell depends on three neighbors), memory-bandwidth-bound rather than compute-bound. Register-tiling the inner loop (cuDTW++ style) could lift utilization into the 8–12% range (≈ 2–4× current Metal perf on short-medium series); not attempted in this pass.

---

## 8. What's not yet benchmarked / known gaps

- **LB_Keogh pruning on Metal**: the CUDA path computes envelopes + LB_Keogh on-GPU and skips pairs above a threshold. Metal equivalent not yet implemented. Benefit is workload-specific — only helps when user provides a meaningful threshold.
- **Register-tile kernel**: short series (max_L ≤ 128) would see the biggest win. Current wavefront uses ~32 threads per pair and under-utilizes the SIMD group. Implementation is non-trivial MSL + warp-shuffle work.
- **Row-major banded kernel**: would dramatically improve tight-band (e.g. band=100) speedup from the current 1.25× to ≈ 5–10× over CPU, by eliminating anti-diagonal barrier overhead.
- **1-vs-all / k-vs-all Metal variants**: parallel to CUDA's `compute_dtw_one_vs_all` / `compute_dtw_k_vs_all`.
