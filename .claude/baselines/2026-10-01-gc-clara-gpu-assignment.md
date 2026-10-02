# 2026-10-01 — GC: CLARA's assignment on the GPU

**Question:** does CLARA's assignment (every series against the k medoids) on CUDA, through the rectangular entry
`compute_medoid_distances_cuda` on the fill's kernels, compute pairs as fast as the fill does at the same length and
precision, and how does it compare with the CPU assignment?

Machine: RTX 4000 Ada (sm_89, 48 SMs), Intel Core Ultra 9 285 (24 logical CPUs), shared with other agents. Worktree
`C:/D/git/wt/GC`, base `558e09a6`. Tree `build-cuda` (MSVC 14.50 + nvcc 13.0, sm_89; C1's recipe, llfio and Arrow off).
Harness: `2026-10-01-gc-clara-gpu-assignment/clara_assign_bench.cpp`, built by its `build.bat` against `build-cuda`'s
library.

## Band — written 2026-10-01 23:35 BST, extended 2026-10-02 01:15 BST (the secondary cases), before any timing

- Case: N = 20,000 series of length 500 (`benchmark_series_set(N, 500, 200)`) against k = 10 medoids (series 0, N/k,
  …), 200,000 pairs; FP32 and FP64. One warm-up, then 5 runs; medians.
- GPU time: the entry's CUDA-event time (each block's upload, launch and download; N = 20,000 is one block). Reference,
  measured in the same session, back to back: the fill `compute_distance_matrix_cuda` at L = 500, CUDA-event time, N =
  1100 (FP32, 604,450 pairs) and N = 520 (FP64, 134,940 pairs), W13a's sizes of at least 1 s a fill.
- Derived expectation from C1's step 3 (the same Preload route; `2026-09-30-c1-cuda-long-series.md`, wall-clock fills
  of L 512): FP32 N 1040 in 1.2798 s = 422.2 kpairs/s = 110.7 Gcell/s, FP64 N 520 in 1.2514 s = 107.8 kpairs/s = 28.3
  Gcell/s; at the same Gcell/s and L = 500: **FP32 442.7 kpairs/s, FP64 113.0 kpairs/s**.
- Pass, per precision: the entry's pairs/s ≥ 0.8 × the same-session fill's; and ≥ 0.8 × the C1-derived expectation (FP32
  ≥ 354 kpairs/s, FP64 ≥ 90.4 kpairs/s). Prediction: 0.95–1.05 of the fill (the same kernels and per-pair work, one
  launch of 200,000 pairs; a pair reads a series once and the 10 medoids stay in cache).
- Secondary cases, the same band against the same-session fill, where the rectangle's kernel compiles to other
  registers than the fill's (`cuobjdump -res-usage`): FP32 L 1024, the Shared wavefront (rectangle 63 registers, fill
  68; C2 saw the fill lose 4–23 % at L 513–2048 when its registers fell and a fourth block fit an SM), fill N = 600,
  entry N = 10,000; FP32 L 3000, the global-memory wavefront (rectangle 38, fill 28), fill N = 184, entry N = 1,000;
  k = 10. Prediction: within 0.9–1.1 of the fill at L 3000; at L 1024 possibly below 1.0, for C2's reason.
- Results: the GPU FP64 assignment's labels and nearest distances hash equal to the CPU's (L1, no FMA in either route).
- CPU reference, reported without a band: the same assignment through the Problem's bound DTW function, `run_openmp`
  over the series (fast_clara's loop), 24 threads, wall-clock median of 5.
- Noise: a case outside the band is re-run once, back to back with its fill; still outside, FALSIFIED (recorded, and the
  route is kept: its results are right, only its speed is in question). The load is recorded with each run.

## Results [inferred, timings; the machine was quiet: CPU load 0–19 %, the GPU idle at 210–405 MHz between cases and at 2310 MHz in them] — pass

Registration committed 01:20:53 BST (`aa114aa9`), before the first timed run (01:21:25–01:23:33 BST, `band_run1.txt`,
harness sha256 `2ab4443ff3f0e86e…`, linked against step 1 `870a3cde`; runner `run_band.sh`). Medians of 5, CUDA-event time:

| case | fill (same session) | entry | entry / fill | entry / C1-derived | registered | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 L 500 | 449.7 kpairs/s (N 1100, 1.3440 s) | 472.2 kpairs/s (0.4235 s) | **1.050** | 1.067 | ≥ 0.8, ≥ 0.8 | pass |
| FP64 L 500 | 113.8 kpairs/s (N 520, 1.1861 s) | 113.9 kpairs/s (1.7557 s) | **1.001** | 1.008 | ≥ 0.8, ≥ 0.8 | pass |
| FP32 L 1024 (Shared) | 120.9 kpairs/s (N 600, 1.4868 s) | 127.9 kpairs/s (N 10,000, 0.7818 s) | 1.058 | — | ≥ 0.8 | pass |
| FP32 L 3000 (global) | 10.5 kpairs/s (N 184, 1.6060 s) | 10.0 kpairs/s (N 1,000, 0.9995 s) | 0.952 | — | ≥ 0.8 | pass |

The entry runs the fill's kernels at the fill's rate: 118.1 Gcell/s against 112.4 (FP32) and 28.5 against 28.4 (FP64)
at L 500, inside the predicted 0.95–1.05. The FP32 Shared kernel's fewer registers (63 against 68) did not cost the
rectangle at L 1024 (1.058). The FP64 assignment's labels and nearest distances hash equal to the CPU's
(`635a683588bf157c` both): the same distances bit for bit.

Against the CPU assignment (24 threads, wall-clock 2.7953 s, 71.5 kpairs/s = 17.9 Gcell/s): the GPU takes 0.4235 s in FP32
(**6.6×**) and 1.7557 s in FP64 (**1.59×**; this GPU runs FP64 at 1/64 of its FP32 rate). With the host's assignment of each
block the wall times are 0.4320 and 1.7640 s.

Memory, whatever N: a block holds at most 2^27 samples and 2^27 distances, so the device holds (k + block) × max_L
samples of T, block × k doubles and the global wavefront's slices (at most the L2's worth), and the host its pinned
staging of the same samples and one block of distances: at L 500 and k 10 a block is 268,435 series, 512 MiB of FP32
or 1 GiB of FP64 samples and 20.5 MiB of distances. The caller keeps O(N): FastCLARA's labels and nearest distances
(16 bytes a series, 1.6 GB at N = 100M).
