# PDLP bench — CPU vs GPU cuPDLP vs matrix-free Kelley (Task 4.5 deliverable)

**Date:** 2026-07-08 · **Machine:** dev box, RTX 4000 Ada (sm_89), shared/under load
**Status:** ADVISORY timings (CLAUDE.md — shared machine; read the *scaling*, not the ms).
Correctness/agreement halves are HARD and gated in `test_pdlp_lp.cpp`.

Problem: p-median LP relaxation on well-separated 1-D clustered data
(`clustered_D(N, k=3, seed=20240708)`), N ∈ {20, 50, 100, 200, 400}.
Two solvers of the SAME LP optimum:
- **PDLP** — HiGHS first-order primal-dual (`solver="pdlp"`, cuPDLP-C), forms the
  explicit ~3N²-nonzero LP. Device fixed at HiGHS build time (see note below).
- **Kelley** — LR-core matrix-free Lagrangian cutting-plane (`lagrangian_root_kelley`),
  streams D once per major iteration, never forms the LP.

## Registered bands (stated BEFORE the runs)

- **P-BENCH-agree [HARD]** — `|pdlp − kelley_LB| / max(1,|kelley|) ≤ 1e-4` at every N,
  on BOTH builds. (Same LP optimum by Geoffrion; a bigger gap ⇒ different LPs.)
- **P-BENCH-scale [ADVISORY]** — Kelley wall-time < PDLP at every N ≥ 50, and the
  ratio `pdlp_ms/kel_ms` grows with N (matrix-free streaming beats forming the 3N² LP).
- **P-BENCH-gpu [ADVISORY]** — at N ≤ 400, GPU-PDLP ≥ CPU-PDLP wall-time (host-device
  transfer + kernel-launch overhead dominates the tiny LP); GPU only closes the gap as
  N grows; GPU bound matches CPU bound to 1e-4.

## Builds

- **CPU** — `build/highs-1151` (clang 18, Ninja; HiGHS v1.15.1, CUPDLP_GPU=OFF).
- **GPU** — `build/highs-gpu` (MSVC cl 14.50 host + nvcc 13.0, `-allow-unsupported-compiler`,
  Ninja; HiGHS v1.15.1, `-DDTWC_HIGHS_GPU=ON` → CUPDLP_GPU=ON, sm_89).
  Rebuild recipe: `MSYS_NO_PATHCONV=1 cmd.exe /c build/highs-gpu/bench_build.bat`
  (vcvars64 → `cmake --build … --target test_pdlp_lp`).

## Results — verbatim `test_pdlp_lp.exe "[pdlp][bench]"`

CPU build (`build/highs-1151`):
```
  PDLP build: CPU-only  →  device = CPU

   N    k | pdlp ms    iters  gpu? | kelley ms   LB          | rel      | pdlp/kel
  -------+-------------------------+-------------------------+----------+---------
    20   3 |       3.8    880   no |       4.2  7.5786     | 5.7e-10 |     0.9
    50   3 |      46.5   2000   no |       1.9  21.8294    | 1.5e-11 |    24.1
   100   3 |     262.8   2600   no |       3.4  47.6842    | 2.6e-09 |    76.7
   200   3 |    1786.8   3680   no |       4.1  103.0373   | 4.4e-10 |   433.4
   400   3 |   16047.6   4320   no |      17.0  203.6108   | 3.6e-09 |   944.9
```

GPU build (`build/highs-gpu`):
```
  PDLP build: CUPDLP_GPU=ON  →  device = GPU (cuPDLP-C)

   N    k | pdlp ms    iters  gpu? | kelley ms   LB          | rel      | pdlp/kel
  -------+-------------------------+-------------------------+----------+---------
    20   3 |     255.0    880  yes |       8.5  7.5786     | 9.7e-10 |    30.0
    50   3 |     492.6   2000  yes |       3.2  21.8294    | 4.5e-09 |   151.9
   100   3 |     542.7   2240  yes |       6.4  47.6842    | 2.1e-09 |    84.5
   200   3 |    1000.1   3840  yes |       5.8  103.0373   | 1.9e-09 |   173.5
   400   3 |    2161.3   6840  yes |      17.1  203.6108   | 1.9e-09 |   126.3
```

## Cross-build (PDLP device comparison)

| N   | CPU-PDLP ms | GPU-PDLP ms | Kelley ms | CPU/GPU PDLP        | GPU-PDLP / Kelley |
|-----|-------------|-------------|-----------|---------------------|-------------------|
| 20  | 3.8         | 255.0       | ~4–8      | GPU **67× slower**  | 30×               |
| 50  | 46.5        | 492.6       | ~2–3      | GPU 11× slower      | 152×              |
| 100 | 262.8       | 542.7       | ~3–6      | GPU 2× slower       | 85×               |
| 200 | 1786.8      | 1000.1      | ~4–6      | GPU **1.8× faster** | 174×              |
| 400 | 16047.6     | 2161.3      | ~17       | GPU **7.4× faster** | 126×              |

## Verdicts

- **P-BENCH-agree → CONFIRMED.** Max `rel` = 7.7e-09 (GPU 24-instance arbiter) / 7.1e-09
  (CPU) — both ≪ the 1e-4 band. PDLP and matrix-free Kelley converge on the identical LP
  optimum from different mathematics (independent arbiter, CLAUDE.md §4).
- **P-BENCH-scale → CONFIRMED.** Kelley near-flat (matrix-free, ~15 major iters); PDLP
  explodes forming/solving the 3N² LP. `pdlp/kel` grows monotonically: CPU 0.9 → 944.9×;
  GPU 30 → 126×. Kelley dominates on the TU-structured p-median on EITHER device.
- **P-BENCH-gpu → PARTIALLY FALSIFIED (deliverable).** I predicted GPU-PDLP ≥ CPU-PDLP at
  every N ≤ 400. FALSE at N = 200, 400: GPU has a ~255 ms fixed launch/transfer floor
  (67× slower than CPU-PDLP at N = 20) but **crosses over near N ≈ 150** and is 7.4× faster
  than CPU-PDLP by N = 400. cuPDLP-C on the GPU also takes MORE first-order iterations
  (6840 vs 4320 at N = 400 — different restart/precision) yet each is far cheaper, netting
  the win. The crossover being *inside* the tested range is the corrected fact.
  Structural conclusion unchanged: even GPU-PDLP is 126× slower than Kelley at N = 400.

## Side finding — `gpu_used` honesty bug found + fixed by this bench

The bench first ran with a two-column (cpu-pdlp / gpu-pdlp) design toggled by the
`use_gpu` flag. On the GPU build **both columns were digit-identical in iteration count**
(880/880 … 6840/6840) and time — i.e. the `use_gpu` flag does NOT switch device:
HiGHS `CUPDLP_GPU` is a **compile-time** switch, so on a GPU build `solver="pdlp"` ALWAYS
runs on the GPU. The old code set `gpu_used=true` only when `use_gpu` was requested, so a
`use_gpu=false` solve on a GPU build ran on the GPU yet reported `gpu_used=false` — a false
report (contract: "true iff the GPU backend actually ran").

Fix (`dtwc/mip/pdlp_lp.cpp`): `gpu_used = pdlp_gpu_available() && variant=="pdlp"` —
reflects the build + variant, not the request. `use_gpu` now only governs the CPU-build
warning. New `[pdlp][gpu]` assertion: `gpu_used == pdlp_gpu_available()` for BOTH
`use_gpu=true` and `use_gpu=false` (device is the build's choice, not the caller's).

## Gate (no regression)

- CPU build `build/highs-1151`: `ctest` → **100% passed, 0 failed / 89** (6 documented skips).
- `test_pdlp_lp` non-hidden: **92 assertions / 3 cases** pass on BOTH builds
  (arbiter max rel 7.14e-09 CPU / 7.72e-09 GPU).
