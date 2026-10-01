# W7ef — Soft-DTW on the linear kernel, Interpolate buffers, the dead variant surface (2026-10-01)

Branch `pb/W7ef` from `da6304de` (design-2.0 with W7d merged). The base binaries for the measurements are
built in a detached worktree of the same commit (`C:/D/git/wt/W7ef-base`, `dtwc_cl` only, preset
`clang-win`, benchmarks ON). Data generator, launcher, timing script, raw CSVs and probes:
`2026-10-01-w7ef-variant-surface/`. Everything is `[confirmed]` (observed on this tree) unless marked.

## Bands, registered before any measurement

Machine: Intel Core Ultra 9 285, P-cores = logical CPUs 0,1,10,11,12,13,22,23 (affinity mask `C03C03`),
`OMP_NUM_THREADS=8`; shared with other agents' builds, so every time is `[inferred]` under load.

(a) Memory. `dtwc_cl -i softdtw_8x8000 --skip-rows 1 --skip-cols 1 -k 3 -m pam --variant softdtw`: 8 random
walks of 8,000 samples (28 pairs). Base keeps a thread_local 8,000 × 8,000 double matrix (512 MB) in every
thread that ran a pair, so its peak working set should be near 8 × 512 MB = 4.1 GB. Head keeps one rolling
column of 8,000 doubles per thread. Measured by `peakmem.exe` (GetProcessMemoryInfo on the exited child:
PeakWorkingSetSize, PeakPagefileUsage). PASS iff head's peak working set < 200 MB and < base / 10. The 28
distances must agree within 1e-12 relative (the pre-registered Soft-DTW tolerance).

(b) Time. Fill time = the `-v` clock at "FastPAM converged" minus the clock at "Data loaded" (fill plus a
k = 3 FastPAM), W7d's method. Five repetitions per binary, base and head alternating back to back (base
first on odd repetitions), both pinned to `C03C03`.

- Soft-DTW: `softdtw_32x1000` (32 random walks of 1,000 samples, 496 pairs), `--variant softdtw`.
- Interpolate: `interp_1000x64.csv` (1,000 series of 64 samples, one per row; every second series has NaN
  runs at 0–2, 20–26 and 60–63), `--missing-strategy interpolate` (499,500 pairs).

Band for each: median(head) <= median(base) + spread(base), spread = max − min of the five base repetitions.

(c) Registered after step 6, before any WDTW timing: the bound WDTW closure changed (one `make_wdtw<T>`; the
float32 fill reads the bound table instead of the thread_local cache), so its fill is timed the same way on
`plain_1000x64.csv` (1,000 random walks of 64 samples, no NaN, 499,500 pairs, where a per-pair cost would
show), `--variant wdtw` in float64 and with `--dtype float32`, under the band of (b).

## Results

(a) Memory, `peakmem.exe C03C03 dtwc_cl ... --variant softdtw`, `OMP_NUM_THREADS=8`:

| binary | peak working set | peak private | fill + FastPAM (-v clocks, under load) |
| --- | --- | --- | --- |
| base `da6304de` | 3,430.9 MB | 3,430.9 MB | 12.16 s |
| step 1 | 13.9 MB | 6.2 MB | 12.63 s |

PASS: 247× less, no O(n·m) per thread. The two `dtwc_distance_matrix.csv` files are byte-identical (all 28
Soft-DTW distances digit-identical).

Soft-DTW values. Moving the value to `dtw_kernel_linear` as is moved float64 values by at most 9.7e-16
relative (inside the band) but float32 values by 1.6e-7 (1.3 float ulps; 1e-12 relative is below float32's
resolution, so no reordered float32 sum can meet it): `SoftCell` summed its three exponentials as diag, up,
left, and the linear kernel passes dp[i, j-1] as `up` where the full kernel passed dp[i-1, j]. Summing diag,
left, up instead restores the full kernel's order: `parity.cpp` (dtwFull pointer/vector L1/sq, soft_dtw both
argument orders and the gradient at gamma 1e-3, 0.1, 0.7, 1, 10, denorm_min and min, the Interpolate facade
and WDTW, 8 shapes from 1×1 to 513×700, float64 and float32; 304 lines, hex floats) is digit-identical to
base: `parity_diff.py` → "changed: 0", at every step of the unit.

(b), (c) Fill times, head `441e64ad` (ms, medians of five; raw rows in the CSVs):

| run | case | base median (spread) | head median | head / base | band |
| --- | --- | --- | --- | --- | --- |
| 1 | Soft-DTW 32 × 1,000 | 1,942.0 (94.8) | 1,860.9 | 0.958 | PASS |
| 1 | Interpolate | 255.1 (19.6) | 294.6 | 1.155 | **FAIL** |
| 1 | WDTW float64 | 344.6 (308.1) | 300.5 | 0.872 | PASS |
| 1 | WDTW float32 | 315.2 (109.6) | 265.1 | 0.841 | PASS |
| 2 | Interpolate | 259.6 (27.7) | 277.9 | 1.070 | PASS |
| 3 | Interpolate | 268.4 (66.9) | 287.0 | 1.069 | PASS |
| 3 | Interpolate, no NaN in the data | 251.3 (84.2) | 267.1 | 1.063 | PASS |
| 3 | ZeroCost (code unchanged) | 264.3 (26.9) | 296.0 | 1.120 | FAIL |
| A/A | ZeroCost, head binary in both arms | 287.0 (123.7) | 263.5 | 0.918 | PASS |
| 4 | Standard (lanes, unchanged) | 54.6 (1.8) | 53.8 | 0.985 | PASS |
| 4 | ADTW (unchanged) | 360.4 (113.9) | 342.9 | 0.951 | PASS |
| 4 | ZeroCost (unchanged) | 272.9 (82.9) | 279.3 | 1.023 | PASS |

The Interpolate band failed on the first run and passed on the next two. The fill timing cannot settle a
few per cent here: the same binary in both arms differs by 8 %, and ZeroCost, whose code this unit does not
touch, failed the band once (1.12) and passed once (1.02). Counters and a pinned single-thread timing of the
bound closures settle it (`closure_probe.cpp`: `Problem::dtw_function()` / `dtw_function_f32()` on 63–64-
sample pairs, one P-core, median of 9 blocks of 20,000 calls, three runs each, `closure_probe.txt`):

| closure | heap allocations per call, base → head | ns per call, base → head (3 runs) |
| --- | --- | --- |
| Interpolate, no NaN | 2 → 0 | 3,161–3,184 → 3,120–3,137 |
| Interpolate, one gappy series | 2 → 0 | 3,250–3,260 → 3,194–3,217 |
| Interpolate, two gappy series | 2 → 0 | 3,264–3,308 → 3,223–3,263 |
| WDTW float64, a length the series have | 0 → 0 | 3,298–3,323 → 3,302–3,313 |
| WDTW float64, another length (a DBA centroid) | 1 → 0 | 4,403–4,436 → 4,175–4,208 |
| WDTW float32, a length the series have | 0 → 0 | 3,288–3,359 → 3,282–3,293 |
| WDTW float32, another length | 0 → 0 | 4,172–4,186 → 4,168–4,188 |

Same checksum of every distance in both builds (12469744.942036184). Verdict: Interpolate is 1 % faster per
pair and allocates nothing; WDTW is unchanged where the table holds the length and 5.5 % faster where it does
not; Soft-DTW's fill is 4 % faster (one run, `[inferred]` under load). No closure is slower.

Assembly (no LTO, `probe.cpp` compiled with the library's flags, `inner.py`): the Soft-DTW inner loop is the
same shape before and after, three `exp` and one `log` call per cell plus the SoftGammaScale branch to
`scalbn`; base 72 instructions over the loop's blocks (`dtw_kernel_full`, two loads per cell), head 80
(`dtw_kernel_linear`, one load per cell, the up-value carried in a register, plus the early-abandon select,
a run-time test in this out-of-line instantiation; `[inferred]` that inlining with T(-1) folds it away). The listings stay in the session scratchpad (`w7ef/asm/`).

## Gates at head `1f7ce199`

Measured at `441e64ad` above; `1f7ce199` adds the fixes of an adversarial review (separate size checks and
exact sizing for the gradient's two buffers, one WDTW table per distinct length, `core::validate` in
`distance::dtw(x, y, band, metric)`, `ddtw`, `missing` and `arow`, which returned the L1 distance for
metric 7 and now raise `InvalidInput`) and stale text. Every gate below ran at `1f7ce199`.

- clang tree: serial ctest 96 = 93 passed + 3 MAY_SKIP skipped (`test_cuda_correctness`,
  `test_metal_correctness`, `test_metal_mmap`); base 98 = 95 + the same 3. Gone with their subjects:
  `unit_test_scratch_matrix`, `unit_test_time_series`.
- CUDA tree (MSVC 19.50 + nvcc 13.0, `build-cuda`, under vcvars64): 95 = 93 passed + 2 MAY_SKIP skipped
  (`test_metal_correctness`, `test_metal_mmap`); `test_cuda_correctness` passed (84 s). No C4244 from
  `wdtw_weights`; `cl /W4` on a float32 WDTW TU: 3 C4244 at base, 0 at head.
- `cpp_conformance` passed (no Soft-DTW row exists; nothing regenerated).
- 31 `dtwc_cl` runs (`w7ef_cli_runs.sh`): every output file, stdout, stderr and exit code byte-identical to
  base, at every step and at `1f7ce199`.
- Python (fresh venv, wheel built from each worktree): base 1124 passed / 19 skipped, head 1124 / 19.
- clang MEX, R2024b `matlab_suite`: base and head 146 run, 145 passed, 0 failed, 1 incomplete
  (`test_parallelisation_serial_is_honest`, filtered by assumption).
- `check_docs.py` PASS (386 flags), `check_pins.py` 0 failures, `generate_docs.py --check` current.
