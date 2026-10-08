# RL report from raw.txt

# run 2026-10-08 20:03:22 rounds=3 compilers=clang,msvc,gcc cpu=12 vcpu=2
# wsl: systemd running, waited 0 s for its jobs; jobs now: 0; up 2.79 s
# info clang_base: info clang 21.1.8  W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info clang_B: info clang 21.1.8  W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info clang_A: info clang 21.1.8  W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info msvc_base: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info msvc_B: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info msvc_A: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info gcc_base: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info gcc_B: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info gcc_A: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# run done in 767 s

## clang: B against base, median over rounds; ratio = base ms / B ms per round

| fill | W | base ms | B ms | base Gcell/s | B Gcell/s | ratio (median) | range | band | matrix hash base / B | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 3075.78 | 456.63 | 1.615 | 10.881 | 6.765 | 6.735-6.839 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | 5.272 / 5.308 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 142.99 | 38.52 | 34.746 | 128.971 | 3.536 | 3.417-3.712 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 2653.18 | 387.38 | 1.425 | 9.763 | 6.920 | 6.811-6.937 | >= 1.5: pass | `d0c4ea00ede2475c` = B | 1.1780 (0.1120, 0.0391) | 5.324 / 5.298 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1096.51 | 322.50 | 37.842 | 128.663 | 3.387 | 3.343-3.422 | >= 1.5: pass | `cd125a5433bb9769` = B | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 146.92 | 21.60 | 33.817 | 229.985 | 6.727 | 6.185-6.837 | >= 1.5: pass | `d0429e2d299c55ed` = B | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 136.62 | 38.91 | 36.367 | 127.694 | 3.555 | 3.346-3.569 | >= 1.5: pass | `76e1c6ac3e6ce69d` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 160.02 | 159.97 | 124.921 | 124.958 | 0.997 | 0.967-1.096 | 0.97-1.03: pass | `fa485cbe1fa07d20` = B | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 180.79 | 180.03 | 131.725 | 132.284 | 0.998 | 0.997-1.016 | 0.97-1.03: pass | `79637ca9e991a308` = B | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 83.09 | 84.13 | 58.431 | 57.706 | 0.977 | 0.974-1.019 | 0.97-1.03: pass | `bed68cfaa32b7445` = B | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 2597.72 | 705.17 | 30.695 | 113.074 | 3.684 | 3.449-3.897 | >= 1.5: pass | `64d2a0d49855c763` = B | 1.0023 (0.0007, 0.0015) | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B Gcell/s | B ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 30.695 | 112.291 | 3.503 | 3.491-3.839 | 113.074 | 3.684 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B: yes; lanes vs per-pair bitwise mismatches base + B: 0; outside the route bound: 0 -> pass
- ragged sweep (B): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- CLI (w8b_cli_runs.sh + w8b_cli_extra.sh, data/dummy): byte-identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 1
  - round 0 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 4.7-5.264 GHz
- band: 10/10 shapes in band; matrix hashes base == B (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict clang (B): PASS**

## msvc: B against base, median over rounds; ratio = base ms / B ms per round

| fill | W | base ms | B ms | base Gcell/s | B Gcell/s | ratio (median) | range | band | matrix hash base / B | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 8 | 3355.55 | 3691.16 | 1.481 | 1.346 | 0.906 | 0.905-0.968 | >= 0.97: FAIL | `b28673ce59c47ce8` = B | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 8 | 165.16 | 159.49 | 30.083 | 31.151 | 1.041 | 1.015-1.060 | >= 0.97: pass | `b28673ce59c47ce8` = B | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 8 | 2657.55 | 3052.34 | 1.423 | 1.239 | 0.870 | 0.869-0.957 | >= 0.97: FAIL | `d0c4ea00ede2475c` = B | 1.0809 (0.0608, 0.0140) | n/a |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 8 | 1141.98 | 1380.68 | 36.335 | 30.053 | 0.829 | 0.825-0.833 | >= 0.97: FAIL | `cd125a5433bb9769` = B | 1.0246 (0.0207, 0.0033) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 178.10 | 151.93 | 27.896 | 32.701 | 1.149 | 1.145-1.172 | >= 0.97: pass | `d0429e2d299c55ed` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 8 | 164.29 | 159.31 | 30.241 | 31.187 | 1.031 | 1.029-1.115 | >= 0.97: pass | `76e1c6ac3e6ce69d` = B | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 8 | 677.51 | 672.30 | 29.505 | 29.734 | 1.005 | 1.002-1.038 | >= 0.97: pass | `fa485cbe1fa07d20` = B | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 8 | 775.24 | 782.03 | 30.719 | 30.453 | 0.996 | 0.989-1.001 | >= 0.97: pass | `79637ca9e991a308` = B | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 8 | 93.78 | 94.29 | 51.768 | 51.485 | 0.990 | 0.985-1.021 | >= 0.97: pass | `bed68cfaa32b7445` = B | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 8 | 2658.94 | 2753.50 | 29.988 | 28.958 | 0.966 | 0.923-1.039 | >= 0.97: FAIL | `64d2a0d49855c763` = B | 1.0011 (0.0003, 0.0007) | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B Gcell/s | B ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 29.988 | 29.672 | 0.969 | 0.953-1.028 | 28.958 | 0.966 | = base | 1.0009 (0.0002, 0.0007) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B: yes; lanes vs per-pair bitwise mismatches base + B: 0; outside the route bound: 0 -> pass
- ragged sweep (B): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- pinned probes whose clock moved by more than 3% over their fills: 0
- band: 6/10 shapes in band; matrix hashes base == B (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict msvc (B): FALSIFIED**

## gcc: B against base, median over rounds; ratio = base ms / B ms per round

| fill | W | base ms | B ms | base Gcell/s | B Gcell/s | ratio (median) | range | band | matrix hash base / B | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 3932.29 | 480.59 | 1.263 | 10.338 | 8.216 | 8.088-8.239 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | 5.424 / 5.356 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 200.25 | 39.48 | 24.811 | 125.850 | 5.049 | 4.957-5.420 | >= 2.0: pass | `b28673ce59c47ce8` = B | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 3345.42 | 384.63 | 1.130 | 9.833 | 8.743 | 8.627-8.822 | >= 1.5: pass | `d0c4ea00ede2475c` = B | 1.1780 (0.1120, 0.0391) | 5.322 / 5.283 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1802.25 | 322.79 | 23.023 | 128.549 | 5.576 | 5.543-5.661 | >= 1.5: pass | `cd125a5433bb9769` = B | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 123.60 | 24.20 | 40.198 | 205.288 | 5.107 | 4.808-5.265 | >= 1.5: pass | `d0429e2d299c55ed` = B | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 205.74 | 45.29 | 24.148 | 109.711 | 4.481 | 4.320-4.547 | >= 1.5: pass | `c640e49a859012ae` / `1618537f9c633a7b` (FMA caveat) | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 149.32 | 153.68 | 133.870 | 130.076 | 0.969 | 0.955-0.977 | 0.97-1.03: FAIL | `fa485cbe1fa07d20` = B | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 180.31 | 176.38 | 132.078 | 135.019 | 1.022 | 1.007-1.044 | 0.97-1.03: pass | `79637ca9e991a308` = B | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 84.06 | 82.81 | 57.755 | 58.623 | 1.015 | 1.006-1.030 | 0.97-1.03: pass | `bed68cfaa32b7445` = B | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 2972.98 | 763.00 | 26.820 | 104.503 | 3.873 | 3.850-3.930 | >= 1.5: pass | `64d2a0d49855c763` = B | 1.0023 (0.0007, 0.0015) | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B Gcell/s | B ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 26.820 | 106.466 | 3.958 | 3.921-3.970 | 104.503 | 3.873 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq no; per-pair hashes base == B: yes; lanes vs per-pair bitwise mismatches base + B: 2,692,955; outside the route bound: 0 -> pass
- ragged sweep (B): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 264,419; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0.4933, f64 L1 0, f64 Sq 0.4943 -> pass
- clustering (clust/, data/dummy uncut): clustering identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 5
  - round 2 B f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.877-5.224 GHz
  - round 0 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.172-5.393 GHz
  - round 1 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.239-5.522 GHz
  - round 2 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.094-5.544 GHz
  - round 1 B f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.086-5.527 GHz
- band: 9/10 shapes in band; matrix hashes base == B (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict gcc (B): FALSIFIED**

