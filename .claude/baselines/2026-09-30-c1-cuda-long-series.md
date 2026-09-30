# 2026-09-30 — C1: CUDA fills series of any length (RTX 4000 Ada)

**Question:** can the CUDA fill take series longer than a block's shared memory holds (FP32 L > 8446, FP64 L > 4223
on this GPU, a typed refusal since W4d) with the host kernel's distances, and how fast is it against the CPU fill?

**Answer:** yes. Above the shared-memory limit the wavefront keeps its three anti-diagonals in global memory, one slice
per resident block, with the shared kernel's source and arithmetic; every tested distance is the host kernel's bit for
bit, and the eight existing kernels are byte-identical in SASS. FP32 public fill, pinned: 82.1 Gcell/s at L = 10,000
(CPU fill on the 8 P-cores: 38.6) and 38.2 at L = 20,000 (CPU 33.7). The fall above L ≈ 12,000 is the 40 MB L2: step 1b
runs no more blocks than the L2 holds slices for, which passed its band and gives 73.2 Gcell/s at L = 20,000 (2.2× the
CPU fill).

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

## Step 1b — the global route runs at most as many blocks as the L2 holds slices

Change: `DeviceLimits` reads `cudaDevAttrL2CacheSize` once; the global grid is
min(pairs of the launch, resident blocks, max(SMs, L2 bytes / (3 · max_L · sizeof(T)))). Host code only.

### Band — registered 2026-09-30 19:30 BST, before the first timed run of the head

- Measure: `long_fill gpu N L 5` (public fill, precision Auto; FP64 cases through a `long_fill` built with FP64 forced,
  see below), median of 5 fills after one warm-up fill, under `start /affinity 0xC03C03` (8 P-cores). Base: `long_fill`
  linked against step 1 (`9cc754a`, sha256 `aef92d7f…4321`). Head: the same source linked against the step-1b library.
  Base and head back to back in one session, alternating per case.
- Cases, N = 48: FP32 L = 10,000 (cap 349 blocks, not binding: 288 run), 14,000, 16,000, 20,000 (binding: 249, 218,
  174 blocks); FP64 L = 5,000 (not binding) and 10,000 (binding, 174 blocks). No shared-route case: the change is inside
  the global route's grid only (host code; the shared route's launches are untouched).
- Pass, every case: head median ≤ 1.05 × base median. Where the cap binds in FP32 (the gain the step-1 probe showed:
  0.83, 0.65, 0.50): head ≤ 0.90 × base at 14,000 and 16,000, ≤ 0.70 × base at 20,000. No gain is claimed in FP64 (the
  FP64 wavefront is bound by the 1:64 FP64 rate, not memory [assumed]); FP64 10,000 must stay ≤ 1.05.
- Distances: head `d(0,1)` and `d(N−1,N−2)` equal to base's in every case (the grid only schedules pairs).
- Noise: a case outside the band is re-run once, base and head back to back; still outside, the cap is FALSIFIED and does
  not land (no per-precision variant).

### Band results [inferred: CPU load 0–70 % from other agents; the fills are GPU-bound]

Run 1, 19:33–19:39 BST (`step1b_band.txt`); head `long_fill` sha256 `5ade24f2…5111`:

| case (N = 48) | grid base → head | base median | head median | head / base | registered | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 L 10,000 | 288 → 288 | 1.3916 s | 1.3923 s | 1.001 | ≤ 1.05 | pass |
| FP32 L 14,000 | 288 → 249 | 3.5007 s | 2.9055 s | 0.830 | ≤ 0.90 | pass |
| FP32 L 16,000 | 288 → 218 | 5.7578 s | 3.8632 s | 0.671 | ≤ 0.90 | pass |
| FP32 L 20,000 | 288 → 174 | 11.9639 s | 6.1669 s | 0.515 | ≤ 0.70 | pass |
| FP64 L 5,000 | 288 → 288 | 1.0133 s | 0.9766 s | 0.964 | ≤ 1.05 | pass (ranges overlap) |
| FP64 L 10,000 | 288 → 174 | 5.4294 s | 4.1533 s | 0.765 | ≤ 1.05 | pass, and 24 % less time |

Every case's `d(0,1)` and `d(N−1,N−2)` equal base's. The FP64 gain falsifies the band's assumption that the FP64 route is
bound by the FP64 rate alone. The cap lands: the FP32 fill at L = 20,000 is 73.2 Gcell/s, 2.2× the CPU fill of the same
matrix (33.7), where step 1 was 1.13×. The kernels' SASS is step 1's, byte for byte (host change). Tests: the 64-series
case now runs FP64 at twice the limit (L 8448: 206 of 288 blocks, 2016 pairs). CUDA tree ctest 121 / 0 failed,
`test_cuda_correctness` 59 cases / 7169 assertions.

## Step 2 — a 64 KB shared-memory carveout above L = 2048 (W4a: −18 % at FP32 L = 2049)

Change (R1, the brief's rule): the shared-memory wavefront launches through `cudaLaunchKernelEx`, with the per-launch
attribute `cudaLaunchAttributePreferredSharedMemoryCarveout = 64` (% of 100 KB) when max_L > 2048 and none otherwise; the
function attribute is not used, as it is one value per kernel shared by every host thread. Prediction from the block
sizes (3 · L · sizeof(T) + 16 static + 1 KB reserved per block; 100 KB per SM): FP32 2049–2644 runs 2 blocks per SM
instead of 3, with 64 instead of 28 KB of L1 (W4a's regime); FP32 2645–5375 runs 1 block instead of 2–3; FP32 ≥ 5376 and
FP64 ≥ 2688 cannot hold a block in 64 KB, so the driver keeps its own choice; FP64 2049–2090 runs 1 block instead of 2.

### Band — registered 2026-09-30 19:50 BST, before the first timed run of a head

- Measure: `long_fill` public fill, N per case below (about 1.5 s per fill), one warm-up fill, median of 5, pinned
  (8 P-cores). Base: `long_fill` linked against step 1b (`60e2ec7`, sha256 `5ade24f2…5111`). Head: the same source
  against the step-2 library. Base and head back to back per case, one session.
- Cases, FP32 (N): L 1024 (600), 2048 (320), 2049 (272), 2400 (240), 2644 (216), 3000 (184), 4096 (136), 5000 (112),
  6000 (92), 8000 (68), 10,000 (48, global route); FP64 (N): 2048 (136), 2049 (136), 2400 (112), 3000 (92), 4000 (68).
- R1 lands if FP32 L 2049 has head ≤ 0.90 × base (the claimed gain, with margin) and every other case head ≤ 1.05 × base.
- R2, registered now as the alternative if R1 fails: the carveout only where 64 KB holds two blocks (on this GPU FP32
  L 2049–2644; never FP64). It lands if FP32 L 2049, 2400 and 2644 are each ≤ 0.95 × base with 2049 ≤ 0.90, and the cases
  R2 leaves on the base launch parameters (L ≤ 2048, the global route) are ≤ 1.05 in the R1 run (same launch path).
- Noise: a case outside is re-run once, base and head back to back; still outside, it counts. Distances: head `d(0,1)`,
  `d(N−1,N−2)` equal to base's (the carveout does not touch the arithmetic).
- Neither rule lands: FALSIFIED, recorded with the numbers, no code change.

### Band results [inferred: CPU load 0–55 % from other agents; the fills are GPU-bound]

Run 1, 19:47–19:53 BST (`step2_band.txt`); R1 head sha256 `138ee8db…997a`. The re-run of FP32 L 3000 (the only case
outside) at 19:53 (`step2_rerun.txt`):

| case (N) | base median | R1 median | R1 / base | R2 carveout | note |
| --- | --- | --- | --- | --- | --- |
| FP32 L 1024 (600) | 1.5407 s | 1.4857 s | 0.964 | no | launch through `cudaLaunchKernelEx`, no attribute |
| FP32 L 2048 (320) | 1.4621 s | 1.4667 s | 1.003 | no | |
| FP32 L 2049 (272) | 1.4961 s | 1.2327 s | **0.824** | yes | claimed −18 %: reproduced |
| FP32 L 2400 (240) | 1.5899 s | 1.3157 s | **0.828** | yes | |
| FP32 L 2644 (216) | 1.5468 s | 1.2355 s | **0.799** | yes | |
| FP32 L 3000 (184) | 1.6873 s | 1.7988 s | **1.066** | no | re-run 1.7062 → 1.8349 s, **1.075**: outside |
| FP32 L 4096 (136) | 1.7214 s | 1.8063 s | 1.049 | no | |
| FP32 L 5000 (112) | 1.7112 s | 1.7741 s | 1.037 | no | ranges overlap |
| FP32 L 6000 (92) | 2.8026 s | 2.7713 s | 0.989 | no | 64 KB cannot hold a block: the driver's choice |
| FP32 L 8000 (68) | 2.6811 s | 2.6287 s | 0.980 | no | idem |
| FP32 L 10,000 (48) | 1.4179 s | 1.4279 s | 1.007 | no | global route |
| FP64 L 2048 (136) | 1.3330 s | 1.3498 s | 1.013 | no | |
| FP64 L 2049 (136) | 1.6103 s | 1.6819 s | 1.044 | no | 1 block instead of 2; ranges apart |
| FP64 L 2400 (112) | 1.5084 s | 1.5668 s | 1.039 | no | ranges apart |
| FP64 L 3000 (92) | 1.7369 s | 1.6699 s | 0.961 | no | |
| FP64 L 4000 (68) | 1.5767 s | 1.5773 s | 1.000 | no | |

**R1 (the carveout for every L > 2048) is FALSIFIED**: FP32 L 3000, where 64 KB holds one block instead of two, takes 7.5 %
more time on the re-run. **R2 lands**: its three cases gain 17–20 % (2049 ≤ 0.90, all ≤ 0.95) and the cases it leaves on
the base parameters are within 1.05. Distances equal to base's in every case. Code: `DeviceLimits` reads the SM's
shared memory and the per-block reserve once; the shared wavefront launches through `cudaLaunchKernelEx` with the
attribute where 2 · (dynamic + static + reserved bytes) ≤ 64 % of the SM's shared memory and L > 2048.

Confirmation of the R2 build against base, 19:57–19:59 (`step2_confirm_r2.txt`; head sha256 `96705284…5a2d`): FP32 L 2049
0.847, 2644 0.815, 2645 1.002 (not applied), 3000 0.999; FP64 2049 1.002. SASS byte-identical to step 1 (host change).
CUDA tree ctest 121 / 0 failed, `test_cuda_correctness` 59 / 7169 (the regime test runs FP32 L = 2049 with the
attribute, bit for bit). The clang tree compiles none of it (`ninja: no work to do`).

**Not landed after all: step 2 is reverted** (the commit after the compute-capability floor). The review found, and
NVIDIA's runtime API references confirm, that `cudaLaunchAttributePreferredSharedMemoryCarveout` and
`cudaLaunchAttributeValue::sharedMemCarveout` first appear in CUDA 12.5 (absent from the 12.4.0 reference), while the
project builds with CUDA 12.0 and later and the ARC scripts load `CUDA/12.4.0`: the step broke those builds. The other
ways to set a carveout are no better: the function attribute is one value for every host thread (the race W4d removed),
and a second instantiation of the same kernel only to carry the attribute is an abstraction for a knob. The 64 % rule is
also specific to an SM with 100 KB of shared memory in a 128 KB L1 (sm_86/sm_89): on an H100 the same hint would leave
the block count alone and only take L1 [inferred, unmeasured]. The measured gain (17–20 % at FP32 L 2049–2644 on the RTX
4000 Ada) stands for a ruling: raise the CUDA floor to 12.5 and gate the rule on the SM's shared memory, or leave it.

## Step 3 — the preload wavefront compiled apart (W4a: −15–17 % at FP32 L 257–500)

Change: a third instantiation, `dtw_wavefront_kernel<T, Preload>`, compiles only the preload mode (both series and the
three anti-diagonals in shared memory); the host launches it for max_L ≤ 512, where the one kernel ran its preload mode.
W4a's clone without the double-buffer branch needed 44 instead of 68 registers (FP32) and ran 5 instead of 3 blocks per
SM. The `Shared` instantiation, used from L = 513, stays byte-identical (its preload branch is no longer reached);
`kernel_used` stays `wavefront`.

### Band — registered 2026-09-30 20:04 BST, before the first timed run of the head

- Measure, base and head as in step 2: `long_fill`, one warm-up fill, median of 5, pinned, back to back per case. Base:
  step 2 (`ffbede8`, sha256 `96705284…5a2d`).
- Cases, FP32 (N): L 257 (1830), 384 (1280), 512 (1040) — the preload range; 513 (1040), 768 (750), 1024 (600).
  FP64 (N): 257 (1000), 384 (700), 512 (520), 1024 (268).
- Lands if every FP32 preload case has head ≤ 0.90 × base and every other case ≤ 1.05 × base (FP64 preload cases
  included: W4a measured 0.944 at 257 and within ±5 % at 384 and 500).
- Alternative, registered now: if only FP64 preload cases exceed 1.05, the preload kernel is used for FP32 only.
- Noise and distances as in step 2. Neither lands: FALSIFIED, recorded, no code change. SASS: every existing kernel
  byte-identical to step 2's.

### Band results [inferred: CPU load 1–20 %; the fills are GPU-bound]

Run 1, 20:07–20:11 BST (`step3_band.txt`); head sha256 `a5854726…1c54`:

| case (N) | base median | head median | head / base | registered | verdict |
| --- | --- | --- | --- | --- | --- |
| FP32 L 257 (1830) | 1.5949 s | 1.3170 s | **0.826** | ≤ 0.90 | pass |
| FP32 L 384 (1280) | 1.4335 s | 1.1688 s | **0.815** | ≤ 0.90 | pass |
| FP32 L 512 (1040) | 1.4466 s | 1.2798 s | **0.885** | ≤ 0.90 | pass |
| FP32 L 513 (1040) | 1.4660 s | 1.4643 s | 0.999 | ≤ 1.05 | pass |
| FP32 L 768 (750) | 1.4462 s | 1.4457 s | 1.000 | ≤ 1.05 | pass |
| FP32 L 1024 (600) | 1.5052 s | 1.5070 s | 1.001 | ≤ 1.05 | pass |
| FP64 L 257 (1000) | 1.5140 s | 1.2494 s | **0.825** | ≤ 1.05 | pass |
| FP64 L 384 (700) | 1.5484 s | 1.2974 s | **0.838** | ≤ 1.05 | pass |
| FP64 L 512 (520) | 1.4635 s | 1.2514 s | **0.855** | ≤ 1.05 | pass |
| FP64 L 1024 (268) | 1.5167 s | 1.5148 s | 0.999 | ≤ 1.05 | pass |

Lands, in both precisions: 11.5–18.5 % less time at L 257–512; FP64 gains as much as FP32 (W4a's kernel-only clone,
which still compiled the non-preload modes, measured 0.944 at FP64 257). Distances equal to base's. SASS
(`step3_sass.txt`): every existing kernel byte-identical to a7d3a8f's (and step 2's); `Preload` has 40 registers in FP32
and FP64 (the `Shared` kernel 68 / 79), its series loads are `LDS` and its cell is the shared kernel's (two FMNMX, one
add). The regime test (L 257 and 512), the F12 route test (filler 257, banded, both metrics), the two-launch test (257,
persistent) and `test_gpu_matches_cpu_large` (L 500) run it against their oracles. CUDA tree ctest 121 / 0 failed,
`test_cuda_correctness` 59 / 7169; clang tree ctest 122 = 119 + 3 MAY_SKIP, `cpp_conformance` passed.

## After the review

An adversarial review of the branch (read-only) raised, besides the CUDA 12.5 attribute (step 2, reverted):

- **The global route beats the shared one at the top of the shared range** [inferred: 3 fills each, 21:13–21:16 BST,
  CPU load 23–100 % from other agents, GPU otherwise idle]. `route_ab.txt`: the same public fill with the normal
  selection and with a probe build (reverted) that takes the global route for every L > 2048; distances identical.

  | case (N) | shared route | global route | global / shared | shared blocks per SM |
  | --- | --- | --- | --- | --- |
  | FP32 L 2049 (272) | 1.4548 s | 1.5724 s | 1.081 | 3 |
  | FP32 L 3000 (184) | 1.7091 s | 1.6285 s | 0.953 | 2 |
  | FP32 L 4096 (136) | 1.6677 s | 1.6525 s | 0.991 | 2 |
  | FP32 L 6000 (92) | 2.6954 s | 1.7190 s | **0.638** | 1 |
  | FP32 L 8000 (68) | 2.6198 s | 1.7201 s | **0.657** | 1 |
  | FP64 L 2049 (136) | 1.5654 s | 1.3076 s | **0.835** | 2 |
  | FP64 L 3000 (92) | 1.6663 s | 1.2606 s | **0.757** | 1 |
  | FP64 L 4000 (68) | 1.5776 s | 1.2466 s | **0.790** | 1 |

  Every case fits "the shared wavefront above L = 2048 only where three blocks fit an SM": a lead for its own band. The
  brief keeps the selection below the shared-memory limit as it was, so C1 does not change it. The docs page's "each
  kernel is the fastest of those that accept its length range" now names only the warp and register-tile kernels.
- **Tests:** a banded (64) and a squared-L2 FP64 fill at the limit on the global route (banded equal to the host's banded
  kernel bit for bit; squared L2 within 1e-10 relative, as the other squared-L2 tests). The occupancy queries are checked
  (`CUDA_CHECK`), so a failure is reported where it happens, before any allocation.
- **CHANGELOG:** the speed sentences go; the CPU fills were timed under 18–70 % load, and the L2 cap is floored at one
  block per SM (it binds above FP32 L ≈ 72,800, FP64 ≈ 36,400 on this GPU), which the sentence did not say. The numbers
  stay here, advisory.
- **Kept:** `T *__restrict__ scratch` (the buffers are reached through the stack array `diag_buf`; the global kernel's
  scratch loads are `LD.E`, never the read-only path); the extra persistent blocks the occupancy query counts but a
  per-launch carveout would not hold (moot after the revert).
