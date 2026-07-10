# Task 5.11 — OpenMP schedule and profiler gate

Date: 2026-07-10 (Europe/London)  
Host: Intel Core Ultra 9 285, 24 physical/logical cores, Windows 11 10.0.26100  
Build: Release, Clang 21.1.8, OpenMP 5.1, `-march=native`  
Status: advisory (the repository plan records that this host is shared)

## Registered schedule sweep

Command:

```powershell
build/bin/bench_openmp_schedule.exe --benchmark_min_time=0.2s `
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true `
  --benchmark_out=build/openmp-schedule.json --benchmark_out_format=json
```

The benchmark uses one identical `schedule(runtime)` triangular pairwise-DTW
loop for every variant (N=64, nominal length=128; ragged by up to 16 samples).

| Schedule | Median real time | Throughput | Verdict |
|---|---:|---:|---|
| `dynamic,1` | 17.674 ms | 114.1k pairs/s | retain |
| `guided,1` | 17.351 ms | 116.2k pairs/s | statistically tied (+1.9%) |
| `dynamic,16` | 124.498 ms | 16.2k pairs/s | reject (7.0× slower) |

`dynamic,16` exposes only four outer-loop chunks at N=64 and therefore leaves
most of the 24-core host idle. `dynamic,1` and guided are too close on this
shared host to justify a production change. The existing adaptive chunking is
retained.

## Hot-kernel timing anchors

Command (two hot paths in the existing benchmark binary):

```powershell
build/bin/bench_dtw_baseline.exe `
  '--benchmark_filter=BM_dtwFull_L/4000$|BM_fillDistanceMatrix/50/500/-1$' `
  --benchmark_min_time=2s
```

Two repeated runs while attempting ETW collection agreed:

| Kernel | Run 1 | Run 2 |
|---|---:|---:|
| `BM_dtwFull_L/4000` | 119.228 ms | 118.168 ms |
| `BM_fillDistanceMatrix/50/500/-1` | 154 ms | 154 ms |

## Cache/PMU profiler gate

Intel VTune and Linux `perf` are not installed. Windows Performance Toolkit is
installed, so both supported local collection paths were attempted:

1. `xperf -on PROC_THREAD+LOADER+PROFILE -stackwalk Profile` failed before
   collection with `NT Kernel Logger: Invalid flags (0x3ec)`.
2. `wpr -start GeneralProfile -filemode` failed before collection with
   `Failed to enable the policy to profile system performance (0xc5585011)`.

Both require an elevated system-performance tracing policy that is unavailable
to this non-administrator session. No cache-miss claim is made.

## SIMD decision

The measure-first gate does **not** authorize a Highway/inter-pair SIMD
prototype: cache/PMU evidence is unavailable, and the schedule sweep found a
load-balancing effect rather than a compute-throughput bottleneck. The deleted
`DTWC_ENABLE_SIMD` surface remains deleted. Owner for a future re-open: release
engineering on a quiet CI/performance host with VTune or `perf stat` access.

