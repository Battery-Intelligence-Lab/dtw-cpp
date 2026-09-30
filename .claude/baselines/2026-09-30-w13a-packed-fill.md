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

**Answer:** no. Every case is inside the band, and the case with the most pairs per second, 9,000 series of length 100
in FP32, is 22 % faster, because the host no longer converts and copies an N×N matrix. The DP loops are the base's
up to register allocation. At N = 20,000 (FP32) one fill's memory above its input fell from 6.0 to 1.6 GiB on the
host and from 1.6 to 1.1 GiB on the GPU. Sizing note: the registered N were chosen with a margin, 1.3–1.5 s per base
fill, not "the smallest round size" of 1 s as the text above says.

Machine and builds: RTX 4000 Ada (sm_89, 20 GB, driver 596.72), Intel Core Ultra 9 285; `build-cuda` of the W13a
worktree, MSVC 14.50 + nvcc 13.0 (`-allow-unsupported-compiler`), Release, llfio ON (so the mapped fill is tested).
Base binary sha256 `0b6353f9…bd0d` (`e9216f4`: library `e9fffed`), head `4a1b8cd8…2cab` (`c67dd3f`). Tools and raw
output: `2026-09-30-w13a-packed-fill/`.

## Band results [confirmed]

Run 2, base then head back to back, 2026-09-30 14:17–14:20 BST, CPU load 3–8 %, GPU idle but for the desktop:

| case (N, L, precision) | base median | head median | head / base | cv base, head | Gcell/s base → head |
| --- | --- | --- | --- | --- | --- |
| 9000, 100, FP32 | 1504.1 ms | 1166.7 ms | **0.776** | 2.9 %, 0.6 % | 269.2 → 347.1 |
| 1100, 500, FP32 | 1570.8 ms | 1562.6 ms | 0.995 | 0.3 %, 0.2 % | 96.2 → 96.7 |
| 330, 2000, FP32 | 1447.2 ms | 1443.5 ms | 0.997 | 1.0 %, 0.1 % | 150.0 → 150.4 |
| 2700, 100, FP64 | 1397.9 ms | 1375.0 ms | 0.984 | 0.6 %, 0.9 % | 26.1 → 26.5 |
| 520, 500, FP64 | 1404.4 ms | 1403.5 ms | 0.999 | 0.5 %, 0.7 % | 24.0 → 24.0 |
| 140, 2000, FP64 | 1324.9 ms | 1339.0 ms | 1.011 | 0.1 %, 0.9 % | 29.4 → 29.1 |

The base alone at 03:50 (CPU load 1 %, run 1) was 1.0–2.6 % faster than the same binary at 14:17 in every case, which
is why the band compares base and head within one session.

## What moved [inferred, advisory]

Kernel time alone (Nsight Systems `cuda_gpu_kern_sum`, one unpinned run each, under 2–93 % CPU load from other agents):

| fill | kernel | base | head | device-to-host copy, base → head |
| --- | --- | --- | --- | --- |
| 9000, 100, FP32 | regtile<float,4> | 1048.1 ms | 1031.6 ms | 13.8 → 34.7 ms |
| 2700, 100, FP64 | regtile<double,4> | 1365.9 ms | 1384.7 ms | 2.2 → 2.1 ms |
| 6000, 200, FP32 | regtile<float,8> | 1485.5 ms | 1475.3 ms | 5.5 → 10.6 ms |
| 1500, 1000, FP32 | wavefront, 3 buffers | 8850.7 ms | 8869.4 ms | — |

The kernels are within ±1.4 % of the base: one contiguous store per pair replaces a row store and a strided column
store. The copy moves the same bytes (N×N floats before, N(N+1)/2 doubles now) but into pageable memory, the matrix
itself, where the base copied into a pinned buffer; it is under 4 % of any fill here. The 22 % at L = 100 is the host
work that went: `convert_result_matrix` (N×N doubles) and the element-wise copy into the packed matrix. The same shows
at L = 200 (regtile<float,8>, N = 6000, three alternating pinned end-to-end runs): base 1.689–1.713 s, head
1.524–1.536 s.

## Memory of one fill at N = 20,000, L = 1000, FP32 [confirmed]

`mem_probe` (a scratch program on the public API: `Problem::fill_distance_matrix` on device gpu; device memory is the
fall of `cudaMemGetInfo`'s free bytes sampled every 2 ms during the fill, host memory is `GetProcessMemoryInfo`):

| | base (`e9fffed`) | head (`c67dd3f`) | accounted for by |
| --- | --- | --- | --- |
| GPU memory during the fill | 1606 MiB | 1106 MiB | base: N×N floats 1526 + series 76; head: 2^27 + N doubles 1024 + series 76 |
| host working set, peak above before | 6183 MiB | 1606 MiB | base: packed 1526 + pinned N×N floats 1526 + N×N doubles 3052 + pinned series 76; head: packed 1526 + pinned series 76 |
| host private bytes, peak above before | 7804 MiB | 2716 MiB | the working set plus about 1.1–1.6 GiB committed but not resident (the driver's, inferred) |
| fill time (advisory, loaded) | 1628.5 s | 1649.3 s | 123 Gcell/s either way; other agents' builds ran alongside the head's run |

Both fills return the same distances (`d(0,1) = 26982.365234375`, `d(N−1,N−2) = 6099.44482421875`).

## SASS: the DP loops [confirmed]

`cuobjdump -sass` of `cuda_dtw.cu.obj`, base against head; `sass_loops.py` compares the innermost loops that hold a
DTW min (FMNMX in FP32, DSETP.MIN in FP64) as instruction shapes (register names by class, `.reuse` hints and branch
displacements dropped). The chunking commit left the kernels byte-identical (`sass_by_function.py`: 8 of 8).

| kernel | DP loops | instructions base → head | difference |
| --- | --- | --- | --- |
| warp<float> | 6 | 459 → 458 | 5 loops identical; one drops a move |
| warp<double> | 2 | 307 → 307 | one identical, one the same instructions reordered |
| regtile<float,4> | 1 | 244 → 242 | drops two integer adds |
| regtile<float,8> | 1 | 488 → 491 | adds three integer index ops (IMAD ×9, IADD3 +6, +7) |
| regtile<double,4> | 1 | 505 → 503 | drops two moves |
| regtile<double,8> | 1 | 963 → 963 | the same instructions reordered |
| wavefront<float> | 2 | 653 → 653 | identical |
| wavefront<double> | 2 | 820 → 819 | adds a move, drops two integer ops |

Total 4439 → 4436. No DP loop gains or loses a floating-point operation, a load, a store, a shuffle, a call or a
spill. Registers per thread (`cuobjdump -res-usage`): unchanged except warp<float> 29 → 32, wavefront<double> 78 → 79
(same allocation granule) and regtile<float,8> 58 → 56; stack and local memory unchanged; the kernels' occupancy is
the base's. Outside the loops the source changed in two places, and the instructions with it (plus the register moves
their allocation brings): the pair index, an int64 `first_pair + pid` (`I2F.F64.S64` in the decode), and the store,
one `STG.E.64` to the packed slot, preceded in FP32 by the widening (`F2F.F64.F32`, a compare with FLT_MAX and two
selects), where the base stored twice, `result[si*N+sj]` and `result[sj*N+si]`.

## Commands

```bat
rem band (band_pair.bat 2 runs bench_fill.bat on the saved base, then the head binary)
start "" /b /wait /affinity 0xC03C03 bench_cuda_dtw.exe "--benchmark_filter=^BM_cuda_fill/" --benchmark_repetitions=5 --benchmark_report_aggregates_only=false --benchmark_out=<json> --benchmark_out_format=json
rem memory
mem_probe.exe 20000 1000 0
rem SASS
cuobjdump -sass build-cuda\bin\CMakeFiles\dtwc++.dir\cuda\cuda_dtw.cu.obj > sass.txt
uv run --no-project python sass_loops.py sass_base.txt sass_head.txt
```

## After the review fixes: FP32, L 100 only [confirmed]

The review (5.1) had each launch copy only slots it defines: the ranges tile the matrix and the device buffer is
zeroed per launch (`cudaMemsetAsync`, 324 MB here), with no diagonal pass after the last. That memset is the new cost,
so the FP32 L = 100 case was re-run at `1affff2`, base then head back to back, 2026-09-30 17:36–17:37 BST, CPU load
3–8 %: base 1529.3 ms, head 1155.9 ms, head/base **0.756** (cv 1.6 %, 0.3 %). Against the recorded run-2 base
(1504.1 ms): 0.769. Against the run-2 head without the memset (1166.7 ms): 0.991. The kernels are byte-identical to
`c67dd3f`'s.
