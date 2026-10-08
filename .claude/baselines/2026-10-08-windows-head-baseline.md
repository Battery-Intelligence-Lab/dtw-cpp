# 2026-10-08 — Windows HEAD baseline of phase A: CPU fill, GPU fill, PAM swap (Core Ultra 9 285 + RTX 4000 Ada), design-2.0 at 75d61cfe

**This record answers PLAN phase A's row "HEAD baseline: conformance digits and three benchmarks (CPU fill, GPU fill, PAM
swap)" for Windows.** Question: what do the three phase-A subjects cost at design-2.0 HEAD on this machine, so that a
later step can say "no regression" against a measured, counter-checked base?

**Result [confirmed], one run 2026-10-08 02:08–02:10 at `75d61cfe` (3 rounds; load records 02:08:15 and 02:10:34, machine
quiet; run directory `C:/D/git/wt/tmp/HB/runs/20261008-020758/`).** Every command exited 0, every engagement proof passed,
`cpp_conformance` passes (8 assertions in 1 test case), and all matrix hashes and swap counters are identical across
rounds and thread counts. Medians, tables below: CPU fill, equal lengths (lanes), N 100 L 1000 unbanded 889.153 ms at 1
thread (5.57 Gcell/s) and 54.834 ms at 24 threads (90.27 Gcell/s); ragged lengths (per-pair route) 0.716 ns/cell at 1
thread, 3.99× the lanes' 0.180 ns/cell (computed from the tables); GPU fill 1.14–1.43 s per case, fp32 111.6–354.7 and
fp64 26.7–29.5 Gcell/s, long series (L 4000, `wavefront_global`) 94.4 (fp32) and 29.2 (fp64) Gcell/s; FasterPAM swap
(N 2000, k 10) 24.86 ms at 1 thread.

- **Finding: FasterPAM's swap at N 2000, k 10 takes longer at 24 threads than at 1 (28.16 vs 24.86 ms median).** The
  ranges [25.16-40.29] and [23.15-31.08] overlap; its BUILD is slower too (0.408 vs 0.176 ms). `find_best_swap` is
  sequential and only the refresh after an accepted swap runs on OpenMP, so region overhead against little work is the
  likely cause [inferred, not tested; a test: the same swap at 2, 4, 8 threads on the P-cores].
- **`fast_pam` exposes no evaluated-swap counter**; the bounds stand in (at most 5970 candidates and 11,940,000 lookups;
  2.08 ns per bound lookup at 1 thread). Counters that decide equality: 2 iterations, converged, 9 medoids changed, cost
  18844.735875089202, medoid hash 00426517fe38ab30, at both thread counts.
- Band 250 on the ragged set leaves the matrix hash equal to the unbanded one (`3fb9cf406427ca4f`) yet takes 0.737 s,
  not 0.811 s, at 1 thread: the band covers about half the cells, L(2b+1) - b(b+1) = 338,050 of 640,000 at L 800
  [inferred], so the banded per-pair kernel costs more per cell, as in TP (band 100: 1.09 against 0.71 ns/cell, clang).
- **Not met: criterion 3 (below).** Groups with a range above 10 % of the median were not repeated as a whole run.

## Readiness (`check.txt`: four STALE lines, resolved by the orchestrator; the run used `HB_ALLOW_STALE=1`)

The check flagged the main trees' binaries as older than the merge commit `75d61cfe` (committed 01:55:25):
`build/bin/bench_dtw_baseline.exe` (01:42:53), `build/cuda-verify-0928/bin/bench_cuda_dtw.exe` (01:45:39) and both trees'
`dtwc_core.lib` (01:42:12, 01:44:14). The LRK integrator built them 01:42–01:45 from the merged tree, committed at 01:55.
The orchestrator confirmed they are current: no-op builds of `bench_dtw_baseline` and `dtwc_core` in `build/` and of
`bench_cuda_dtw` and `dtwc_core` in `build/cuda-verify-0928` printed "ninja: no work to do" [confirmed by the
orchestrator; not repeated here]. It rebuilt the harness at HEAD (`rebuild.sh`; `hb_bench_*.exe` 02:07) and ran with
`HB_ALLOW_STALE=1`, so `check.txt` and `run_full.out` print the four STALE lines and "WARNING: running with a failed
readiness check". The other two tests pass: code paths clean in the main checkout; harness built from the tree ids of
HEAD. The sha256 of every executable is in `facts.md`.

## Method

- One command, on a quiet machine: `sh C:/D/git/wt/tmp/HB/run.sh` (predicted 5-6 minutes; observed about 2.7, 02:07:58
  to 02:10:45, harness rebuilt beforehand). `run.sh check` is the readiness test, `run.sh smoke` the one-round tiny
  version. Output in `C:/D/git/wt/tmp/HB/runs/<stamp>/`: `check.txt`, `facts.md`, `load.txt`, `raw/` (every command's
  output verbatim with its command line, affinity mask and process CPU time), `report.md`; console output `run_full.out`.
  Order: readiness check, machine facts, load before, engagement proofs, 3 rounds of every subject, load after, report
  (median [min-max] with the counters). Wall-clock is advisory; counters and hashes decide equality.
- Pinning (Windows has no taskset): the driver sets its own affinity mask just before `CreateProcess`, so the child and
  all its threads start on it. Single-thread rows: `OMP_NUM_THREADS=1`, logical CPU 12 (a P-core, mask `0x1000`).
  24-thread rows: `OMP_NUM_THREADS=24`, all 24 logical processors. GPU rows: the 8 P-cores (`0xC03C03`, the mask of the
  2026-09-29/30 GPU records), `OMP_NUM_THREADS=8`. Process CPU time (kernel + user) is read from the child's handle:
  CPU time / wall time is the number of threads that really ran.
- The two Google Benchmark executables run **in place** from the main checkout's trees (nothing copied out or built in
  them). `hb_bench_cpu.exe` / `hb_bench_gpu.exe` are this kit's own harness (`src/hb_bench.cpp`, built by `rebuild.sh`
  from a `git archive` of HEAD and the trees' static libraries; not part of the repository), because no benchmark
  isolates a ragged-length fill, the PAM swap on a cached matrix, or a Problem fill on the long-series GPU route.

## Machine and builds (`facts.md` of the run confirms these [confirmed])

| Fact | Value |
| --- | --- |
| CPU, memory, OS | Intel Core Ultra 9 285, 24 logical processors (8 P-cores: logical 0,1,10,11,12,13,22,23; 16 E-cores), Windows reports 2500 MHz; 127.5 GiB; Windows 11 Enterprise 10.0.26200, power plan "Ultimate Performance", no battery |
| GPU | NVIDIA RTX 4000 Ada Generation, 20 GB, compute 8.9, driver 596.72, power limit 130 W; FP64 rate "slow" so `Auto` = FP32 |
| `build/` (CPU subjects) | clang++ 21.1.8, Ninja, Release, ThinLTO, `-O3 -DNDEBUG -std=c++20 -march=native -fopenmp -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros -fno-rounding-math` (`DTWC_FP_MODEL=fast`, FMA contraction kept); options ON: BUILD_BENCHMARK, ENABLE_HIGHS, ENABLE_GUROBI, ENABLE_IPO, ENABLE_LLFIO, ENABLE_MMAP, ENABLE_NATIVE_ARCH; CUDA, Arrow, YAML OFF; lanes W = 8 for f64 |
| `build/cuda-verify-0928` (GPU subject) | MSVC 19.50 (14.50) + nvcc 13.0.48 with `-allow-unsupported-compiler`, Release, `/O2 /Ob2 /GL /arch:AVX2 /openmp:experimental /fp:precise /fp:contract`; ENABLE_CUDA ON, `DTWC_CUDA_ARCH_LIST=60;70;75;80;86;89;90`, `CMAKE_CUDA_ARCHITECTURES=89`; BUILD_BENCHMARK ON |
| Repository | design-2.0 at 75d61cfe (clean), VERSION 2.0.0rc1; the harness built from the same commit |

## Subjects

All f64, L1, standard DTW; series from `tests/support/deterministic_series.hpp` (`std::mt19937`, bit-exact on every
platform), so the input of every command is fixed.

| # | Subject | Route (file:line) | Command | Input | Counters printed |
| --- | --- | --- | --- | --- | --- |
| 1a | CPU fill, equal lengths | `Problem::fill_distance_matrix` `dtwc/Problem.cpp:839` -> `fillDistanceMatrix_BruteForce` `:757` -> rows by `run_openmp` `:817` (`dtwc/base/parallelisation.hpp:93`) -> `fill_lanes` `:782` takes the columns 8 at a time through `dtw_block_fn_` (`dtwc/core/dtw_lanes.cpp:43` -> `dtw_kernel_lanes` `dtwc/core/dtw_kernel.hpp:492`) because every length is equal | `bench_dtw_baseline.exe --benchmark_filter=^BM_fillDistanceMatrix/(100/1000/-1\|50/1000/50)$ --benchmark_min_time=5x` at `OMP_NUM_THREADS=1` on CPU 12 and at 24 threads (`BM_fillDistanceMatrix` `benchmarks/bench_dtw_baseline.cpp:295`; the fill is timed, the data set-up is not) | N=100 L=1000 band -1 (4.95 Gcell); N=50 L=1000 band 50 | time per fill, iterations, items/s, process CPU/wall; matrix hash from `hb_bench fill equal 100 1000 -1 1` (same series: base seed 100) at 1 and 24 threads |
| 1b | CPU fill, ragged lengths | same entry; no block of 8 equal-length columns exists, so the per-pair loop `:813` runs `dtw_fn_` (`resolve_dtw_fn` `dtwc/core/dtw_dispatch.cpp:362` -> `make_standard` -> `dtw_kernel_banded` `dtw_kernel.hpp:402`) | `hb_bench_cpu.exe fill ragged 60 800 -1 R` and `... 250 R`, R = 2 at 1 thread, 3 at 24 threads | 60 series of 60 distinct lengths in [700, 900] (length gap 200, so band 250 is feasible), 1770 pairs, 1.13 Gcell unbanded | wall, process CPU/wall, ns/cell (upper bound when banded), `distinct_lengths`, matrix hash |
| 2 | GPU fill, public path | `Problem::fill_distance_matrix` `:864-885` (device gpu) -> `cuda::compute_distance_matrix_cuda` `dtwc/cuda/cuda_dtw.cu:1287` -> `select_route` `:1273` -> `select_kernel` `dtwc/cuda/kernel_selection.hpp:38` | (a) `bench_cuda_dtw.exe --benchmark_filter=^BM_cuda_fill/` (`benchmarks/bench_cuda_dtw.cpp:253`: one warm-up, one timed fill, six cases FP32/FP64 at L 100, 500 (shared-memory wavefront), 2000); (b) `hb_bench_gpu.exe gpu fp32 136 4000 2` and `gpu fp64 68 4000 2` (L 4000 = the long-series global-memory wavefront) | (a) the six sizes of the W13a record: (9000,100,FP32) (1100,500,FP32) (330,2000,FP32) (2700,100,FP64) (520,500,FP64) (140,2000,FP64); (b) N 136 / 68, L 4000, series of base seed 200 | time, Gcell/s, `kernel_used` (the CUDA entry's own string, probed with `gpu <prec> N L 0` for every case), matrix hash, direct GPU time |
| 3 | PAM swap on a cached matrix | `fast_pam` `dtwc/algorithms/fast_pam.cpp:281` -> BUILD (`kmedoids_pp`) -> `swap_phase` `:229` -> `fasterpam_swap_impl` `:179` -> `find_best_swap` `:141` (sequential, one `dist_by_ind` per series); the nearest/second refresh after an accepted swap runs on `run_openmp` | `hb_bench_cpu.exe swap 2000 32 10 15` at 1 thread (CPU 12) and 24 threads | N=2000 L=32 k=10, seed 42, max_iter 100, 15 repetitions on one filled matrix; `fast_pam(prob,k,0)` is the BUILD alone, so swap = (BUILD+SWAP) - BUILD | `iterations`, `converged`, sweeps = iterations + converged, medoids changed vs BUILD, cost, medoid hash, matrix hash; candidates and lookups as the bounds sweeps*(N-k) and sweeps*(N-k)*N |

**Counters `fast_pam` does not expose:** the swaps it accepted and the candidates it evaluated (no counter at
`fast_pam.cpp:179-209`); `iterations`, `converged`, the medoids and the cost decide equality, and the bounds above stand
in. An exact counter needs a code change in `fast_pam.cpp` [inferred from reading the file].

## Engagement proofs (run once before the rounds; each must pass or no round runs; `report.md` lists them verbatim)

| Proof | Mechanism | Result in the run [confirmed] |
| --- | --- | --- |
| threads that ran | `dtwc::test::parallelisation()` in a real OpenMP region of `hb_bench_cpu info`, with the affinity mask in force | 1 thread: max 1, engaged 1, mask 0x1000; 24: max 24, engaged 24, mask 0xffffff; GPU run: max 8, engaged 8, mask 0xc03c03 |
| parallelism of the timed process | process CPU time / wall time in every raw file (whole process, set-up included) | fill: 0.99 at 1 thread, 17.25 [16.84-17.90] at 24 |
| lanes route taken / not taken | `core::resolve_dtw_block_fn<double>` resolves for L1/unbanded/standard (W = 8); 1a has one length, 1b has N distinct lengths (`Problem.cpp:794-797`); ns/cell of 1a vs 1b | block fn resolved 1, W 8; 1b has 60 distinct lengths; 0.180 vs 0.716 ns/cell at 1 thread [the ratio 3.99 is computed from the tables] |
| bit-identical matrices | FNV-1a over the upper triangle; same hash at 1 and 24 threads and across rounds | 100 x 1000 equal-length: IDENTICAL `06c633ef926bbbfa` at 1 and 24 threads; ragged: IDENTICAL `3fb9cf406427ca4f` over all runs, both bands, both thread counts |
| GPU ran | `dtwc::test::gpu()` (a kernel validated against the CPU oracle), the device line, `gpucheck`: GPU matrix against CPU matrix (N 40, L 300; registered tolerance FP64 max relative <= 1e-9, FP32 <= 1e-3) | validated 1, "NVIDIA RTX 4000 Ada Generation (compute 8.9, 20474 MB)"; FP64 max abs 0, max rel 0 (hashes equal `b7e72aa7ce30e8df`); FP32 max rel 9.02904e-07 (GPU hash `7cdd8232d96ed3a5`) |
| GPU route | `kernel_used` of the CUDA entry | L 100: `regtile_w4` (both precisions); L 500 and L 2000: `wavefront`; L 4000: `wavefront_global` (long-series table). Prepare-time probes: FP64 L 2100 already `wavefront_global`, FP32 L 2100 `wavefront`; L 120 `regtile_w4` |
| conformance digits | `build/bin/cpp_conformance.exe`, in place: labels/medoids digit-identical to `tests/conformance/conformance_reference.txt`, scores within 1e-12 | "All tests passed (8 assertions in 1 test case)" |

The existing BM_cuda_fill cases stop at L 2000 (shared-memory wavefront): subject 2b exists to reach the long-series route.

## Criteria (registered 2026-10-08, before any timed run), with the outcome

1. Every command exits 0; every proof passes. A failed proof stops the run. **Met** (`proofs: 0 non-zero exit(s)`).
2. Counters decide equality: matrix hashes identical across rounds and thread counts; swap counters (iterations, sweeps,
   medoids, cost, medoid hash) identical across rounds and thread counts. **Met** (`report.py` prints "NOT IDENTICAL"
   otherwise; no row does).
3. A median is reported with its range; a group whose range exceeds 10 % of the median is run once more as a whole run,
   then recorded as it is (two attempts, then record). **Not carried out**: one full run only. Ranges above 10 % of the
   median: fill 50/1000/50 (1 thread 15 %, 24 threads 36 %), ragged at 24 threads (band -1 28 %, band 250 31 %), PAM
   swap (1 thread 32 %, 24 threads 54 %), PAM BUILD; all other groups are within 6 % (GPU rows within 1.5 %).
4. The numbers are `[confirmed]` timings only if both load records show mean total CPU < 10 % outside the kit, no COMSOL
   process, GPU utilisation < 5 % before the run; otherwise every wall-clock figure is tagged `[inferred]`. **Met**: mean
   4.0 % before, 4.6 % after; COMSOL none; GPU 2 % before.
5. No expected value is registered: this is a base. Earlier Windows records, for reading only and not compared here:
   `2026-09-29-p1-lanes-fill.md`, `2026-09-30-w13a-packed-fill.md`, `2026-09-30-c2-cuda-route.md`,
   `2026-09-29-windows-gpu-fill.md`.

## Load (`runs/20261008-020758/load.txt`)

| when | total CPU (`typeperf`, 10 × 1 s) | busiest processes, % of all 24 logical processors; GPU |
| --- | --- | --- |
| 2026-10-08 02:08:15 (before) | 3.0–4.9 % (mean 4.0) | `system` 1.64 %, `serviceshell` 0.87 %, `spotify` 0.63 %, `wmiprvse` 0.24 %; COMSOL none; GPU 2 %, 6.26 W, 27 °C |
| 2026-10-08 02:10:34 (after) | 3.1–7.7 % (mean 4.6) | `system` 1.60 %, `spotify` 0.65 %, `serviceshell` 0.52 %, `wmiprvse` 0.28 %, `sspservice` 0.28 %, `tpython` 0.17 %; COMSOL none; GPU 2 %, 8.01 W, 48 °C |

## Results (`runs/20261008-020758/report.md`, tables verbatim; per-command output in `raw/`)

### CPU fill, equal lengths (lanes route): `bench_dtw_baseline` `BM_fillDistanceMatrix`

Real time per fill, ms (the benchmark pauses its timer around the data set-up). Gcell/s = N(N-1)/2 * L^2 / time, unbanded only. cpu/wall is the whole benchmark process (set-up included).

| threads | case | time ms | Gcell/s | iterations | process cpu/wall |
| --- | --- | --- | --- | --- | --- |
| 1 thread, CPU 12 | 50/1000/50 | 21.322 [21.002-24.242] (n=3) | (banded) | 5 | 0.99 [0.98-0.99] (n=3) |
| 1 thread, CPU 12 | 100/1000/-1 | 889.153 [888.149-895.052] (n=3) | 5.57 [5.53-5.57] (n=3) | 5 | 0.99 [0.98-0.99] (n=3) |
| 24 threads | 50/1000/50 | 3.241 [3.164-4.339] (n=3) | (banded) | 5 | 17.25 [16.84-17.90] (n=3) |
| 24 threads | 100/1000/-1 | 54.834 [54.204-57.434] (n=3) | 90.27 [86.19-91.32] (n=3) | 5 | 17.25 [16.84-17.90] (n=3) |

### CPU fill, ragged lengths (per-pair route): `hb_bench fill ragged`

Lengths L-100 .. L+100, all distinct (no block of equal-length columns, so the lanes are never used). Wall seconds of `fill_distance_matrix` on a fresh Problem; ns/cell = wall / (sum of Li*Lj over the pairs), an upper bound on the cells computed when banded.

| threads | band | N, L | pairs | distinct lengths | wall s | ns/cell | cpu/wall | matrix hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 thread, CPU 12 | -1 | 60, 800 | 1770 | 60 | 0.811 [0.799-0.816] (n=6) | 0.716 [0.705-0.721] (n=6) | 1.0 [1.0-1.0] (n=6) | 3fb9cf406427ca4f |
| 1 thread, CPU 12 | 250 | 60, 800 | 1770 | 60 | 0.737 [0.728-0.741] (n=6) | 0.650 [0.643-0.655] (n=6) | 1.0 [1.0-1.0] (n=6) | 3fb9cf406427ca4f |
| 24 threads | -1 | 60, 800 | 1770 | 60 | 0.043 [0.035-0.047] (n=9) | 0.038 [0.031-0.041] (n=9) | 16.4 [13.7-16.9] (n=9) | 3fb9cf406427ca4f |
| 24 threads | 250 | 60, 800 | 1770 | 60 | 0.032 [0.030-0.040] (n=9) | 0.028 [0.027-0.036] (n=9) | 19.3 [15.0-23.5] (n=9) | 3fb9cf406427ca4f |

band -1: matrix hash over all runs, both thread counts: IDENTICAL 3fb9cf406427ca4f

band 250: matrix hash over all runs, both thread counts: IDENTICAL 3fb9cf406427ca4f

### PAM swap: `hb_bench swap` (FasterPAM, matrix cached)

`fast_pam(prob, k, 0)` is the BUILD alone (plus the nearest/second state); `fast_pam(prob, k, 100)` is BUILD + SWAP; swap = difference. fast_pam exposes `iterations` (sweeps with a swap before convergence) and `converged`; it does not expose accepted swaps or evaluated candidates, so those are the bounds sweeps*(N-k) and sweeps*(N-k)*N (sweeps = iterations + 1 when converged). The counters decide equality between runs; the time is advisory.

| threads | N, L, k | swap ms | build ms | iterations | converged | sweeps | candidates max | lookups max | medoids changed vs BUILD | cost | medoid hash | ns/lookup (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 thread, CPU 12 | 2000, 32, 10 | 24.86 [23.15-31.08] (n=45) | 0.176 [0.166-0.370] (n=45) | 2 | 1 | 3 | 5970 | 11940000 | 9 | 18844.735875089202 | 00426517fe38ab30 | 2.08 [1.94-2.60] (n=45) |
| 24 threads | 2000, 32, 10 | 28.16 [25.16-40.29] (n=45) | 0.408 [0.232-1.575] (n=45) | 2 | 1 | 3 | 5970 | 11940000 | 9 | 18844.735875089202 | 00426517fe38ab30 | 2.36 [2.11-3.37] (n=45) |

### GPU fill: `bench_cuda_dtw` `BM_cuda_fill` (CUDA tree, RTX 4000 Ada, 8 P-cores)

Public fill (`Problem::fill_distance_matrix`, device gpu), one warm-up, real time of one fill. Route = the CUDA entry's `kernel_used` (proof section).

| case | route | time ms | Gcell/s |
| --- | --- | --- | --- |
| fp32 N 9000 L 100 | regtile_w4 | 1141.6 [1125.7-1142.7] (n=3) | 354.7 [354.4-359.7] (n=3) |
| fp32 N 1100 L 500 | wavefront | 1354.0 [1349.1-1366.4] (n=3) | 111.6 [110.6-112.0] (n=3) |
| fp32 N 330 L 2000 | wavefront | 1433.2 [1422.4-1443.9] (n=3) | 151.5 [150.4-152.7] (n=3) |
| fp64 N 2700 L 100 | regtile_w4 | 1365.6 [1360.1-1375.8] (n=3) | 26.7 [26.5-26.8] (n=3) |
| fp64 N 520 L 500 | wavefront | 1192.1 [1186.7-1194.4] (n=3) | 28.3 [28.2-28.4] (n=3) |
| fp64 N 140 L 2000 | wavefront | 1318.2 [1316.2-1333.8] (n=3) | 29.5 [29.2-29.6] (n=3) |

### GPU fill, long-series route: `hb_bench gpu`

Same call as `long_fill` of the 2026-09-30 C1/C2 records: public fill after one warm-up. `direct_gpu_s` is the GPU time of one direct call of the CUDA entry, which also names the kernel.

| precision | N, L | route | wall s | Gcell/s | direct GPU s | matrix hash |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 136, 4000 | wavefront_global | 1.556 [1.547-1.566] (n=6) | 94.4 [93.8-94.9] (n=6) | 1.547 [1.527-1.548] (n=3) | f6d6b54a596d3997 |
| fp64 | 68, 4000 | wavefront_global | 1.247 [1.238-1.254] (n=6) | 29.2 [29.1-29.5] (n=6) | 1.245 [1.237-1.255] (n=3) | 529a73dcec6b0a66 |

## Commands (copy-pasteable)

```sh
sh C:/D/git/wt/tmp/HB/run.sh check      # readiness (smoke: one tiny round; HB_ALLOW_STALE=1 runs although a test fails)
sh C:/D/git/wt/tmp/HB/run.sh            # the baseline
sh C:/D/git/wt/tmp/HB/rebuild.sh HEAD   # harness only, after the main trees were rebuilt at a new commit
```
