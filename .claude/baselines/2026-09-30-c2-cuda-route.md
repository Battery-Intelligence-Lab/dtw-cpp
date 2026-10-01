# 2026-09-30 — C2: the CUDA wavefront route above L = 2048, and the Shared kernel without its preload branch

**Question 1:** above L = 2048, is the global-memory wavefront faster than the shared-memory one wherever fewer than
three shared blocks fit an SM (C1's probe: FP32 L 6000/8000 in 0.64/0.66 of the time, FP64 L 2049–4000 in 0.76–0.84)?

**Question 2:** does the Shared kernel lose anything when its preload branch goes (no launch reaches it since C1's
separately compiled `Preload` kernel)?

**Answers:** (1) yes: the rule landed inside its band, FP32 L 6000–8446 in 0.64–0.68 and FP64 L 2049–4223 in 0.76–0.84
of the time, no case above 1.006. (2) yes, in FP32: FALSIFIED. The branch is dead but holds the FP32 kernel at 68
registers and three blocks per SM; without it FP32 L 513–1024 and 2048 took 4–23 % more time (FP64 gained 10–14 % at
513–1024). The branch stays, with a comment. All timings are advisory (CPU load 75–100 % from other agents; the fills are
GPU-bound and base and head ran back to back).

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

## Step 2 — the Shared kernel without its preload branch

Since C1 the host launches the separately compiled `Preload` kernel up to L = 512, so the `Shared` instantiation's
preload branch is never reached. The change: `preload` is `Mode == Wavefront::Preload`, a compile-time constant, so the
`Shared` kernel drops the staging code and the per-cell choice between shared and `__ldg` series loads. Its SASS and
registers change; every other kernel's must not.

### Band — registered 2026-09-30 22:01 BST, before the change

- Measure as in step 1. Base: `long_fill` of step 1 (`63d5d0a`, sha256 `782b0fc9…`). Head: the same source against the
  step-2 library.
- Cases (N), the `Shared` route after step 1: FP32 L 513 (1040), 768 (750), 1024 (600), 1025 (600), 1500 (424), 2048
  (320), 2049 (272), 2400 (240), 2757 (208); FP64 L 513 (520), 768 (350), 1024 (268), 1025 (268), 1500 (200), 2048 (136).
- Pass: every case head ≤ 1.02 × base; distances equal to base's; the SASS of the warp, regtile, `Preload` and `Global`
  kernels byte-identical to step 1's. A case outside is re-run once; still outside, the change is adjusted once or
  FALSIFIED.

### Band results [inferred: CPU load 75–100 %; GPU-bound fills, back to back] — FALSIFIED

Run 1, 22:05–22:11 BST (`step2_band.txt`; head `long_fill` sha256 `cb68423d…`), and the registered re-run of the cases
outside, 22:11–22:14 (`step2_rerun.txt`). Without the branch the `Shared` kernel needs 50 registers in FP32 (was 68) and 62
in FP64 (was 79) (`step2_registers.txt`): five blocks per SM instead of three in FP32, four in FP64. Every other kernel's
SASS was byte-identical to step 1's.

| case (N) | run 1 head / base | re-run | verdict |
| --- | --- | --- | --- |
| FP32 L 513 (1040) | **1.107** | **1.101** | outside |
| FP32 L 768 (750) | **1.062** | **1.065** | outside |
| FP32 L 1024 (600) | **1.044** | **1.055** | outside |
| FP32 L 1025 (600) | 0.867 | | pass |
| FP32 L 1500 (424) | 0.904 | | pass |
| FP32 L 2048 (320) | **1.229** | **1.223** | outside |
| FP32 L 2049 (272) | **1.029** | **1.040** | outside |
| FP32 L 2400 (240) | 1.018 | | pass |
| FP32 L 2757 (208) | **1.032** | **1.031** | outside |
| FP64 L 513 (520) | 0.856 | | pass |
| FP64 L 768 (350) | 0.873 | | pass |
| FP64 L 1024 (268) | 0.899 | | pass |
| FP64 L 1025 (268) | 0.998 | | pass |
| FP64 L 1500 (200) | 0.982 | | pass |
| FP64 L 2048 (136) | 1.004 | | pass |

**FALSIFIED.** The branch is dead code, but it holds the FP32 kernel at 68 registers and three blocks per SM, which the
FP32 fills that read their series through L1 need (the same effect as W4a's L = 1024 clone, 1.275). No single adjustment
keeps the deletion clean: `__launch_bounds__` cannot lower occupancy, and capping the grid or padding shared memory is a
per-precision knob. The branch stays, with a comment in the kernel giving these numbers so nobody deletes it again.
Distances equal to base's in every case. Lead for its own band: the FP64 `Shared` kernel at four blocks per SM took
0.86–0.90 of the time at L 513–1024.

## Gates at the end

CUDA tree: ctest 122 / 0 failed (Skipped: `test_metal_correctness`, `test_metal_mmap`); `test_cuda_correctness` 60 cases /
7232 assertions; `test_cuda_launch_guards` 5 passed + 1 device skip, 54 assertions; every kernel's SASS byte-identical to
base (`6d8c1f2`). Clang tree: ctest 123 = 120 passed + 3 MAY_SKIP, `cpp_conformance` passed (no regeneration); the step-2
comment compiles nothing there. `data/dummy` (L up to 9405, the global route at base and head): `--device gpu` (FP32,
300 pairs in 194 ms) gives `--device cpu`'s labels and medoids, distances within 7.2e-6 relative. `check_docs`,
`check_pins`, `generate_docs --check` green.

## 2026-10-01 — C3: the FP64 Shared kernel without the preload branch

**Question:** compiled for FP32 only, does the Shared kernel's preload branch give the FP64 kernel step 2's gain (62
registers instead of 79; 0.86–0.90 of the time at L 513–1024) and leave FP32 as it is?

The change: `preload`'s Shared-mode term also requires `std::is_same_v<T, float>`, a compile-time constant, so the FP64
`Shared` instantiation drops the staging code and the per-cell choice between shared and `__ldg` series loads (as step 2
did for both), and FP32's condition folds to what it is now. Worktree `C:/D/git/wt/C3`, base `9f4fcca` (design-2.0;
its `dtwc/cuda` equals step 2's commit `22f6d3e`, and its SASS dump is byte-identical to step 2's final one). Trees as in
step 1, of this worktree.

### Band — registered 2026-10-01 01:53 BST, before the change and before any timing

- Predicted: FP64 `Shared` 79 → 62 registers; FP32 `Shared` stays at 68; every kernel's SASS but the FP64 `Shared`
  one's byte-identical to base. Blocks per SM at 256 threads with the fill's dynamic shared memory, from the driver's
  occupancy API on the extracted cubin: base 3 at every band length (base measured); head 4 at L 513, 768, 1025 and
  1500, but 3 at L 1024 and 2048, where shared memory (blocks of 24,592 and 32,784 bytes plus 1,024 reserved, allocated in
  128-byte units, of 102,400) allows no fourth block. A gain at L 1024 would not come from occupancy.
- Measure: `long_fill` (C1's, plus an FNV-1a 64 hash of the whole packed matrix printed after the timed fills,
  `c3_long_fill.cpp`), FP64 (`gpu64`), one warm-up fill, median of 5 fills, under `start /affinity 0xC03C03` (8
  P-cores). Base: `long_fill` linked against `9f4fcca`; head: the same source against the changed library; base and
  head back to back per case, one session (`c3_band.sh`); each case waits until no other CUDA process from `C:\D\git`
  is on the GPU, and a case with one there after it is re-run.
- Cases (N), step 2's FP64 ones: L 513 (520), 768 (350), 1024 (268), 1025 (268), 1500 (200), 2048 (136).
- Pass: L 513, 768 and 1024 head ≤ 0.93 × base; L 1025, 1500 and 2048 head ≤ 1.03 × base; in every case the matrix
  hash, `d(0,1)` and `d(N−1,N−2)` equal to base's; the FP32 kernels' SASS byte-identical to base's (FP32 is then not
  timed) and only the FP64 `Shared` kernel's SASS differs.
- Noise: a case outside is re-run once, base and head back to back; still outside, FALSIFIED (the change has no knob
  to adjust): the kernel stays as it is and only this record lands.

### Band results [inferred: CPU load 100 % from other agents; the fills are GPU-bound, base and head back to back, and no other CUDA process from `C:\D\git` was on the GPU before or after any case] — pass

Run 1, 02:07–02:10 BST (`c3_band.txt`; base `long_fill` sha256 `54f3f231…`, head `94816a68…`). Blocks per SM from the
driver's occupancy API (`c3_kernels.txt`), as predicted:

| case (N) | blocks per SM base → head | base median | head median | head / base | registered | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| FP64 L 513 (520) | 3 → 4 | 1.4865 s | 1.2535 s | **0.843** | ≤ 0.93 | pass |
| FP64 L 768 (350) | 3 → 4 | 1.4350 s | 1.2696 s | **0.885** | ≤ 0.93 | pass |
| FP64 L 1024 (268) | 3 → 3 | 1.5007 s | 1.3218 s | **0.881** | ≤ 0.93 | pass |
| FP64 L 1025 (268) | 3 → 4 | 1.3004 s | 1.2918 s | 0.993 | ≤ 1.03 | pass |
| FP64 L 1500 (200) | 3 → 4 | 1.5757 s | 1.5120 s | 0.960 | ≤ 1.03 | pass |
| FP64 L 2048 (136) | 3 → 3 | 1.2994 s | 1.3068 s | 1.006 | ≤ 1.03 | pass |

The change lands. In the three gaining cases head's slowest fill beat base's fastest; base against itself, run before
the change, gave 0.997–1.006 in the same six cases (`c3_aa.txt`); the ratios reproduce step 2's (0.856, 0.873, 0.899;
0.998, 0.982, 1.004). Every case's whole matrix, `d(0,1)` and `d(N−1,N−2)` equal base's, in L1 (the band) and in
squared L2 (`c3_sql2.txt`, one fill a case). Registers: FP64 `Shared` 79 → 62; every other kernel, the six FP32 ones
included, keeps its registers and byte-identical SASS. Step 2's comment gave the FP32 kernel without the branch five
blocks per SM: that code compiled alone gets four from the occupancy API (50 registers are allocated as 56 a thread), and
the comment now says four.

Mechanism [inferred, not profiled]: not occupancy. L 1024 gains as much at three blocks per SM in both builds, and the
two-buffer cases that go from three to four gain 1–4 %. In the three-buffer cell loop base issues the series loads and
the subtraction twice, under complementary predicates (the run-time `preload` choice), and head once (`c3_kernels.txt`).
FP64 runs at 1/64 of the FP32 rate here, 2 lanes per SM a clock: 222 G lane-operations/s at the 2310 MHz sampled during
the L 1024 fills. Counting predicated-off instructions (the L1-or-squared choice is a predicated pair in every build), a
cell issues 7 FP64 instructions at base (24.9 Gcell/s), 6 at head (28.5) and 6 in the two-buffer path (29.1–29.6):
171–178 G/s each, 77–80 % of that peak, so the FP64 pipe sets the rate. One instruction of seven fewer predicts 0.86 of
the time; measured 0.84–0.89. In FP32 the duplicate is a full-rate `FADD`, and the branch's registers hold the kernel at
three blocks per SM, which its fills need (step 2).

Code: `preload`'s Shared term requires `std::is_same_v<T, float>`; the kernel comment gives both reasons. The launch
comment no longer gives the FP64 Shared kernel 79 registers (C1's measurement it cites stands).

Gates: CUDA tree ctest 113 / 0 failed (Skipped: `test_metal_correctness`, `test_metal_mmap`); `test_cuda_correctness`
60 cases / 7232 assertions at base and head; `test_cuda_launch_guards` 5 passed + 1 device skip, 54 assertions. Clang
tree: ctest 114 = 111 passed + 3 MAY_SKIP, `cpp_conformance` passed; the change compiles nothing there (`ninja: no work
to do` after it), so its run at base is head's too. `check_docs`, `check_pins`, `generate_docs --check` green. No
CHANGELOG line: CUDA is 2.0-born (v1.0.0 shipped no `dtwc/cuda`), and a faster FP64 fill changes no result.

Open, not changed here: step 1's rule counts three blocks' bytes without shared memory's 128-byte allocation unit, so
FP32 L 2751–2757 take the shared route at two blocks per SM (L 2750: three; `c3_kernels.txt`). Step 2's FP32 L 2757 base
ran at 88.9 Gcell/s against 107.5 at L 2400. Rounding each block up to 128 bytes moves the boundary to 2750: its own band.
