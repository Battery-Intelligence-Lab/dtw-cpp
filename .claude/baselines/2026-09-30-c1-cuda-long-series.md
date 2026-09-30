# 2026-09-30 — C1: CUDA fills series of any length (RTX 4000 Ada)

**Question:** can the CUDA fill take series longer than a block's shared memory holds (FP32 L > 8446, FP64 L > 4223
on this GPU, a typed refusal since W4d) with the host kernel's distances, and how fast is it against the CPU fill?

**Answer:** yes. Above the shared-memory limit the wavefront keeps its three anti-diagonals in global memory, one slice
per resident block, with the shared kernel's source and arithmetic; every tested distance is the host kernel's bit for
bit, and the eight existing kernels are byte-identical in SASS. FP32 public fill, pinned: 82.1 Gcell/s at L = 10,000
(CPU fill on the 8 P-cores: 38.6) and 38.2 at L = 20,000 (CPU 33.7). The fall above L ≈ 12,000 is the 40 MB L2: capping
the resident blocks so the scratch fits it gives 75.7 Gcell/s at L = 20,000 (probe; step 1b decides it behind a band).

Machine: RTX 4000 Ada (sm_89, 48 SMs, 101,376 B opt-in shared memory per block, 40 MB L2, 360 GB/s, WDDM, driver 596.72),
Intel Core Ultra 9 285. Trees of worktree `C:/D/git/wt/C1`: `build` (clang, Ninja, Release) and `build-cuda` (MSVC 14.50 +
nvcc 13.0 `-allow-unsupported-compiler`, sm_89, llfio ON, HiGHS/Gurobi OFF; recipe of W13a). Base `a7d3a8f`.
Tools and raw output: `2026-09-30-c1-cuda-long-series/`.

## Step 0 — a failed CUDA call's error is consumed where it is reported [confirmed]

Found while replacing the refusal-order test: `CUDA_CHECK` threw on a failed call but left the error as the thread's
last error, which every launch is checked with, so the thread's next fill failed with it ("invalid device ordinal" after
a fill with a device index past the last). New test "A fill the CUDA runtime refuses leaves the thread's next fill
unaffected": fails at `a7d3a8f` (`cuda_dtw.cu:1068: invalid device ordinal`), passes with `cudaGetLastError()` in
`CUDA_CHECK`'s failure branch. Kernels' SASS byte-identical. Commit `13246cb`.

## Step 1 — the global-memory wavefront

- **Design.** `dtw_wavefront_kernel<T, WavefrontBuffers>`: `Shared` is the existing kernel; `Global` compiles the
  preload and double-buffer modes out and points the three buffers at `scratch + blockIdx.x * 3 * max_L`, a new last
  kernel parameter. It always runs persistent (the existing work counter), on min(pairs of the launch, resident
  blocks) blocks; the scratch is allocated with the other device buffers, before the caller's matrix.
  `select_kernel(max_L, sample bytes, dynamic shared bytes)` picks `wavefront_global` where
  `wavefront_buffer_count(L) · L · sizeof(T)` exceeds the opt-in maximum less the kernel's 16 static bytes; the refusal
  (`require_shared_mem_fits`) goes.
- **Tests first** (`test_cuda_correctness`, `[cuda][long]`): FP32 L 8447 and 12,000, FP64 L 4224 and 9406, N = 8, lengths
  {L, L−1, L−700, 3L/4, L/2+1, 2049, 300, 1}; the boundary (first global L and one below, both precisions); 64 series,
  one at the limit and 63 of 1–300 samples (2016 pairs over 288 blocks). Oracle: `dtwFull_L<double>`, and `dtwFull_L<float>`
  on the float-rounded series, compared with `==` on every entry. At base all eight global cases throw "needs 101380 /
  144016 / 101392 / 225760 bytes of shared memory per block" (`step1` red log in the scratchpad); green at head. A
  mutation giving every block slice 0 fails the 64-series case and the four mixed-length cases (6 assertions).
- **SASS** (`step1_sass.txt`, `cuobjdump -sass` of `cuda_dtw.cu.obj`, per kernel, addresses and encodings): the 8
  existing kernels byte-identical to base; `Shared` instantiations 68 / 79 registers as before (CONSTANT[0] 424 → 432,
  the new parameter). New: `Global` 28 (FP32) / 36 (FP64) registers, no local memory; its DP loop has the shared kernel's
  cell (two `__ldg` series loads, three predicated loads, two FMNMX, one add, one store), with generic `LD.E`/`ST.E`
  because the buffer pointers come from the stack array `diag_buf[3]`, as in the shared kernel.
- **CLI, `data/dummy`** (25 series, 5148–9405 samples, `-k 3 --skip-rows 1 --skip-cols 1`): base `--device gpu` exits 1,
  "needs 112876 bytes of shared memory per block". Head `--device gpu` (Auto = FP32, `wavefront_global`, 300 pairs in
  196 ms) and `--device cpu`: `dummy_labels.csv` and `dummy_medoids.csv` identical (also to the clang tree's CPU run);
  distances max relative difference 7.2e-6 (FP32 against FP64; the FP32 tests' tolerance is 1e-4); cost 148361.802 vs
  148361.920.

### Gcell/s against the CPU fill [inferred: advisory, loaded machine]

`long_fill` (`Problem::fill_distance_matrix` of `benchmark_series_set(N, L, 200)`, one GPU warm-up fill, 5 timed fills,
median), each run under `start /affinity 0xC03C03` (8 P-cores). GPU: `build-cuda` (precision Auto = FP32). CPU: the clang
tree (lanes fill, FP64), the faster CPU build. 19:15–19:19 BST, CPU load 18–70 % from other agents.

| N, L | GPU median | GPU Gcell/s | CPU median | CPU Gcell/s | GPU / CPU |
| --- | --- | --- | --- | --- | --- |
| 64, 10,000 | 2.454 s | 82.1 | 5.220 s | 38.6 | 2.1× |
| 48, 20,000 | 11.817 s | 38.2 | 13.395 s | 33.7 | 1.13× |

Spread (max−min)/median ≤ 1 % (GPU), ≤ 5 % (CPU). GPU and CPU `d(0,1)` agree to FP32 precision (3049.4988 / 3049.4976).

Sweep, GPU, N = 48, 3 fills (`step1_sweep.txt`), and the same with the resident blocks capped at 40 MB / (3 · L · 4 B)
(`step1_l2cap_probe.txt`, probe build, reverted):

| L | 8,500 | 10,000 | 12,000 | 14,000 | 16,000 | 20,000 |
| --- | --- | --- | --- | --- | --- | --- |
| scratch of 288 blocks | 29 MB | 35 MB | 41 MB | 48 MB | 55 MB | 69 MB |
| Gcell/s | 83.0 | 82.3 | 78.2 | 64.9 | 51.2 | 37.9 |
| Gcell/s, blocks capped to the L2 | | 82.4 | | 78.5 | 78.3 | 75.7 |

Mechanism [observed]: the throughput falls where the scratch of the resident blocks passes the 40 MB L2 and comes back
when the blocks are capped so that it fits; the distances are unchanged (the grid only schedules pairs).

## Gates (step 1)

CUDA tree: ctest 121 / 0 failed (Metal skips 2); `test_cuda_correctness` 59 cases / 7169 assertions passed (base 56 / 7145; launch guards 4 passed + 1 skip, as at base). Clang tree: ctest 122 = 119
passed + 3 MAY_SKIP (`test_cuda_correctness`, `test_metal_correctness`, `test_metal_mmap`), `cpp_conformance` passed.
`check_docs.py` PASS, `check_pins.py` 0 failures, `generate_docs.py --check` current.
