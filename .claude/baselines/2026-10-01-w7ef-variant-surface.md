# W7ef — Soft-DTW on the linear kernel, Interpolate buffers, the dead variant surface (2026-10-01)

Branch `pb/W7ef` from `da6304de` (design-2.0 with W7d merged). The base binaries for the measurements are
built in a detached worktree of the same commit (`C:/D/git/wt/W7ef-base`, `dtwc_cl` only, preset
`clang-win`, benchmarks ON). Data generator, launcher and timing script: `2026-10-01-w7ef-variant-surface/`.
Everything is `[confirmed]` (observed on this tree) unless marked.

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

## Results

(a) Memory, `peakmem.exe C03C03 dtwc_cl ... --variant softdtw`, `OMP_NUM_THREADS=8`:

| binary | peak working set | peak private | fill + FastPAM (-v clocks, under load) |
|---|---|---|---|
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
base: `parity_diff.py` → "changed: 0".
