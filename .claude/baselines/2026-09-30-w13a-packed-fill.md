# 2026-09-30 — W13a: the GPU fill writes the packed matrix (RTX 4000 Ada)

**Question:** does the packed GPU output (kernels write the packed lower triangle, chunks stream into
`DistanceMatrix::raw()`, no N×N host or device buffer) slow the public CUDA fill?

## Band — registered 2026-09-30 03:50 BST, before the first timed run and before any library change

- Cases: `BM_cuda_fill` of `bench_cuda_dtw` (`Problem::fill_distance_matrix` on device `gpu`, one warm-up fill at the
  same size first), (N, L, precision) = (9000, 100, FP32), (1100, 500, FP32), (330, 2000, FP32), (2700, 100, FP64),
  (520, 500, FP64), (140, 2000, FP64). N is the smallest round size that took at least 1 s per fill at the base in a
  one-repetition sizing run (8000/1000/300/1500/300/80 took 1.15/1.27/1.17/0.43/0.46/0.43 s).
- Measure: Google Benchmark real time of one fill (`Iterations(1)`, `UseRealTime`), `--benchmark_repetitions=5`, the
  median of the 5; Gcell/s = N(N−1)/2 · L² / time.
- Conditions: each binary under `start "" /b /wait /affinity 0xC03C03` (the 8 P-cores); base and head back to back in
  one session.
- Base: `bench_cuda_dtw.exe` of `build-cuda` built from the band commit (library = `e9fffed`). Head: the same target
  built from the last W13a commit.
- Pass: in every case median head time ≤ 1.05 × median base time (a faster head passes; its gain is reported). A case
  outside is re-run, base and head back to back, once; still outside, the band is FALSIFIED for that case, and the fill
  keeps the N×N device write plus one packed copy on the device.
- The GPU also drives the desktop (WDDM) and other agents build on the CPU: every number is `[inferred]` unless the
  machine was quiet (CPU load stated with each run).
