# 2026-09-30 — W4d: the CUDA cleanup does not slow the fill (RTX 4000 Ada)

**Question:** does W4d (the `KernelOverride` machinery deleted, the device attributes read once, the static
shared-memory fix) change the time of the public CUDA fill? W4d changes host code only; no kernel.

**Answer:** no, by the registered rule. At `e4a1d77` all four cases were within 0.3–1.7 % of the base. At the final
head `d46ac35` three were within 1 %, and `scaling_L/250` came out 8 % *faster* (0.919) against a noisy base (cv
4.8 %). Its registered re-run gave 1.031, inside the band. The re-run's other cases were noisy (base cv up to
11.8 %, CPU load 47 % from other sessions); they are listed but decide nothing. The kernels' SASS is byte-identical to
the base's at every commit.

## Band — registered 2026-09-30 00:53 BST, before the first timed run

- Cases: `BM_cuda_distanceMatrix/100/1000` and `BM_cuda_scaling_L/{250,500,2000}` of `bench_cuda_dtw` (N = 50 for
  scaling_L). The brief named `scaling_L/256`; the registered argument is 250, the same regtile<8> regime.
- Measure: Google Benchmark real time of one `compute_distance_matrix_cuda` call (FP32 through `Auto`, host
  conversion and transfers included); `--benchmark_repetitions=5 --benchmark_report_aggregates_only=true`; the
  median of the 5 repetitions.
- Conditions: base and head run back to back in one session, each under `start "" /b /wait /affinity 0xC03C03`
  (the 8 P-cores, logical CPUs 0, 1, 10–13, 22, 23).
- Base: `bench_cuda_dtw.exe` of `build-cuda` at `830568a` (sha256 `9e349fe9…7c19`, saved before any change). Head:
  the same target built from the last W4d commit.
- Pass: every case within ±5 % (median head / median base in [0.95, 1.05]). A case outside the band is re-run, base
  and head back to back, once; if it is still outside, the band is FALSIFIED for that case and reported.
- Other agents build on the CPU and the GPU also drives the desktop (WDDM), so every number here is `[inferred]`.

## Machine and build

RTX 4000 Ada (compute 8.9, 48 SMs, FP32:FP64 ratio 64, opt-in shared memory 101,376 B per block), driver 596.72;
Intel Core Ultra 9 285. `build-cuda` of the W4d worktree: MSVC 14.50 + nvcc 13.0 (`-allow-unsupported-compiler`),
Release, sm_89 (recipe of `plans/2026-09-27-audit/phaseA_measurements.md`). Heads: `e4a1d77`, then the final
`d46ac35`, whose only change on the CUDA fill path is the non-throwing one-time attribute read.

```bat
start "" /b /wait /affinity 0xC03C03 bench_cuda_dtw.exe "--benchmark_filter=^BM_cuda_(distanceMatrix/100/1000|scaling_L/(250|500|2000))$" --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_out=<json> --benchmark_out_format=json
```

## Results [inferred]

Run 1, head `e4a1d77`, 01:39–01:40 BST, base then head:

| case | base median | head median | head / base | cv base, head |
| --- | --- | --- | --- | --- |
| `BM_cuda_distanceMatrix/100/1000` | 41.007 ms | 40.895 ms | 0.997 | 0.8 %, 1.8 % |
| `BM_cuda_scaling_L/250` | 0.2972 ms | 0.3022 ms | 1.017 | 2.2 %, 2.6 % |
| `BM_cuda_scaling_L/500` | 3.4901 ms | 3.4811 ms | 0.997 | 0.8 %, 1.1 % |
| `BM_cuda_scaling_L/2000` | 34.257 ms | 34.430 ms | 1.005 | 0.6 %, 0.2 % |

Run 2, head `d46ac35`, 02:08 BST, and the registered re-run of the case outside the band, 02:09 BST:

| case | run 2: base, head (ms) | head / base | cv base, head | re-run: base, head (ms) | head / base | cv base, head |
| --- | --- | --- | --- | --- | --- | --- |
| `distanceMatrix/100/1000` | 40.349, 40.429 | 1.002 | 0.2 %, 0.2 % | 40.420, 42.576 | 1.053 | 0.5 %, 1.8 % |
| `scaling_L/250` | 0.3117, 0.2865 | **0.919** | 4.8 %, 0.6 % | 0.2896, 0.2986 | **1.031** | 3.8 %, 2.5 % |
| `scaling_L/500` | 3.4687, 3.4813 | 1.004 | 1.3 %, 1.3 % | 4.1921, 3.5249 | 0.841 | 11.8 %, 1.8 % |
| `scaling_L/2000` | 33.982, 34.337 | 1.010 | 0.3 %, 0.5 % | 36.913, 34.468 | 0.934 | 6.0 %, 0.8 % |

`distanceMatrix/100/1000` reads 0.997, 1.002 and 1.053 over the three pairs; W4d's change on that path is host code
costing well under a microsecond per fill (next section), so the 1.053 is taken as the re-run's noise, not a
slowdown. The base itself ran `scaling_L/250` 14 % slower in run 1 than in `2026-09-29-windows-gpu-fill.md` (0.297
vs 0.261 ms), which is why the band compares against a base measured in the same session.

## Per-launch cost of the new host call [inferred]

The fix calls `cudaFuncGetAttributes` once per wavefront launch: 278–290 ns per call (a standalone nvcc probe
timing 200,000 calls, 3 runs), against a fill of 0.3 ms or more. The three device attributes are read once per
process, 77–103 ns for all three.
