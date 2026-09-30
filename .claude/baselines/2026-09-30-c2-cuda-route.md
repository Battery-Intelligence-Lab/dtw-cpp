# 2026-09-30 — C2: the CUDA wavefront route above L = 2048, and the Shared kernel without its preload branch

**Question 1:** above L = 2048, is the global-memory wavefront faster than the shared-memory one wherever fewer than
three shared blocks fit an SM (C1's probe: FP32 L 6000/8000 in 0.64/0.66 of the time, FP64 L 2049–4000 in 0.76–0.84)?

**Question 2:** does the Shared kernel lose anything when its preload branch goes (no launch reaches it since C1's
separately compiled `Preload` kernel)?

Machine: RTX 4000 Ada (sm_89, 48 SMs, 100 KB shared memory per SM, 1 KB reserved per block, 99 KB opt-in per block,
40 MB L2), Intel Core Ultra 9 285, shared with other agents. Trees of worktree `C:/D/git/wt/C2`: `build` (clang) and
`build-cuda` (MSVC 14.50 + nvcc 13.0, sm_89; C1's recipe). Base `6d8c1f2` (design-2.0 with C1 merged). Measuring tool:
`2026-09-30-c1-cuda-long-series/long_fill.cpp` (public fill, `Problem::fill_distance_matrix`), built by its `build.bat`
against each tree; raw output in `2026-09-30-c2-cuda-route/`.

## Step 1 — the route rule

Rule: up to L = 2048 the wavefront keeps its anti-diagonals in shared memory (at most 32 KB a block, which every
supported GPU holds); above L = 2048 it does so only where three blocks — each its three anti-diagonals, the kernel's
static bytes and the runtime's reserved bytes — fit the shared memory of an SM, and takes the global-memory wavefront
otherwise. "Blocks per SM" comes from `cudaDevAttrMaxSharedMemoryPerMultiprocessor` and
`cudaDevAttrReservedSharedMemoryPerBlock`, read once with the other device limits. On this GPU the shared route above
2048 is then FP32 L ≤ 2757 and no FP64 length (base: FP32 L ≤ 8446, FP64 L ≤ 4223).

### Band — registered 2026-09-30 21:34 BST, before any code change

- Measure: `long_fill` (`gpu` = precision Auto = FP32 on this GPU, `gpu64` = FP64), one warm-up fill, median of 5
  fills, under `start /affinity 0xC03C03` (8 P-cores). Base: `long_fill` linked against `6d8c1f2`. Head: the same source
  linked against the step-1 library. Base and head back to back per case, one session.
- Cases (N): FP32 L 2049 (272), 2400 (240), 3000 (184), 4096 (136), 6000 (92), 8000 (68), 8446 (64); FP64 L 2049 (136),
  2400 (112), 3000 (92), 4000 (68), 4096 (68), 4223 (64). FP32 2049 and 2400 keep the shared route (a control); every
  other case moves to the global route. FP64 L > 4223 is global at base and head (same launch): not measured.
- Pass: every case head ≤ 1.02 × base; and the gains the probe saw: FP32 L 6000 and 8000 ≤ 0.75 × base, FP64 L 2049,
  3000 and 4000 ≤ 0.90 × base. Distances: head `d(0,1)` and `d(N−1,N−2)` equal to base's in every case (both routes
  compute the same cells).
- Noise: a case outside the band is re-run once, base and head back to back; still outside, the rule is adjusted once
  (the adjustment recorded before its run) or FALSIFIED.

### Band results [inferred: CPU load 81–100 % from other agents; the fills are GPU-bound, base and head back to back]

Run 1, 21:48–21:53 BST (`2026-09-30-c2-cuda-route/step1_band.txt`); base `long_fill` sha256 `2e626cee…`, head `782b0fc9…`:

| case (N) | route base → head | base median | head median | head / base | registered | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 L 2049 (272) | shared → shared | 1.4408 s | 1.4492 s | 1.006 | ≤ 1.02 | pass |
| FP32 L 2400 (240) | shared → shared | 1.5014 s | 1.5109 s | 1.006 | ≤ 1.02 | pass |
| FP32 L 3000 (184) | shared → global | 1.6870 s | 1.6135 s | 0.956 | ≤ 1.02 | pass |
| FP32 L 4096 (136) | shared → global | 1.6530 s | 1.6533 s | 1.000 | ≤ 1.02 | pass |
| FP32 L 6000 (92) | shared → global | 2.6790 s | 1.7136 s | **0.640** | ≤ 0.75 | pass |
| FP32 L 8000 (68) | shared → global | 2.6075 s | 1.7156 s | **0.658** | ≤ 0.75 | pass |
| FP32 L 8446 (64) | shared → global | 2.6324 s | 1.7892 s | **0.680** | ≤ 1.02 | pass |
| FP64 L 2049 (136) | shared → global | 1.5536 s | 1.3053 s | **0.840** | ≤ 0.90 | pass |
| FP64 L 2400 (112) | shared → global | 1.5088 s | 1.2213 s | **0.809** | ≤ 1.02 | pass |
| FP64 L 3000 (92) | shared → global | 1.6975 s | 1.2971 s | **0.764** | ≤ 0.90 | pass |
| FP64 L 4000 (68) | shared → global | 1.6106 s | 1.2643 s | **0.785** | ≤ 0.90 | pass |
| FP64 L 4096 (68) | shared → global | 1.6623 s | 1.3250 s | **0.797** | ≤ 1.02 | pass |
| FP64 L 4223 (64) | shared → global | 1.5654 s | 1.2346 s | **0.789** | ≤ 1.02 | pass |

The rule lands unchanged. Every case's `d(0,1)` and `d(N−1,N−2)` equal base's. The ratios reproduce C1's probe from
another session (FP32 L 6000/8000: 0.638/0.657 then, 0.640/0.658 now). Code: `select_kernel(L, sample bytes, SM shared
bytes, per-block overhead)`; `DeviceLimits` reads the SM's shared memory and the per-block reserve once; the per-block
opt-in check goes (up to L = 2048 a block needs at most 32 KB; above it three blocks fitting an SM implies one fits a
block). Host code only: the SASS of every kernel is byte-identical to base. Tests: the route boundary, the regime test
at L = 2049, the 4095/4096 test and the many-pairs test derive their expected route from the device's SM shared memory
and reserve (first global length FP32 2758, FP64 2049 here); `test_cuda_launch_guards` pins the rule for this GPU and for
an H100's 228 KB SM (FP32 6398/6399, FP64 3199/3200) without a GPU. CUDA tree ctest 122 / 0 failed (Metal skips 2),
`test_cuda_correctness` 60 cases / 7232 assertions; clang tree ctest 123 = 120 passed + 3 MAY_SKIP, `cpp_conformance`
passed; `check_docs`, `check_pins`, `generate_docs --check` green.
