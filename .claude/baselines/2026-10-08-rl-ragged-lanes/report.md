# RL report from raw.txt

# run 2026-10-08 23:14:11 rounds=3 compilers=clang,msvc,gcc cpu=12 vcpu=2 cand=B3
# wsl: systemd running, waited 0 s for its jobs; jobs now: 0; up 3.05 s
# info clang_base: info clang 21.1.8  W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info clang_B3: info clang 21.1.8  W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info clang_A: info clang 21.1.8  W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info msvc_base: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info msvc_B3: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info msvc_A: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info gcc_base: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info gcc_B3: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# info gcc_A: info gcc 13.3.0 W f64 16 f32 32 x86_64 1 AVX2 1 FMA 1 omp threads 24
# run done in 898 s

## clang: B3 against base, median over rounds; ratio = base ms / B3 ms per round

| fill | W | base ms | B3 ms | base Gcell/s | B3 Gcell/s | ratio (median) | range | band | matrix hash base / B3 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 3103.37 | 458.87 | 1.601 | 10.827 | 6.785 | 6.711-6.833 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | 5.266 / 5.251 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 146.77 | 40.20 | 33.852 | 123.597 | 3.630 | 3.458-3.696 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 2673.88 | 388.91 | 1.414 | 9.725 | 6.875 | 6.798-6.966 | >= 1.5: pass | `d0c4ea00ede2475c` = B3 | 1.1780 (0.1120, 0.0391) | 5.287 / 5.277 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1097.92 | 336.38 | 37.793 | 123.355 | 3.264 | 3.258-3.354 | >= 1.5: pass | `cd125a5433bb9769` = B3 | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 159.59 | 21.98 | 31.133 | 226.004 | 7.095 | 6.794-7.261 | >= 1.5: pass | `d0429e2d299c55ed` = B3 | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 139.03 | 41.65 | 35.736 | 119.297 | 3.490 | 3.255-3.518 | >= 1.5: pass | `76e1c6ac3e6ce69d` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 172.95 | 172.97 | 115.585 | 115.571 | 1.000 | 0.967-1.051 | 0.97-1.03: pass | `fa485cbe1fa07d20` = B3 | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 182.05 | 184.14 | 130.812 | 129.332 | 0.982 | 0.969-0.994 | 0.97-1.03: pass | `79637ca9e991a308` = B3 | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 84.65 | 85.26 | 57.348 | 56.940 | 0.996 | 0.992-1.029 | 0.97-1.03: pass | `bed68cfaa32b7445` = B3 | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 2673.44 | 736.54 | 29.825 | 108.258 | 3.630 | 3.628-3.769 | >= 1.5: pass | `64d2a0d49855c763` = B3 | 1.0023 (0.0007, 0.0015) | n/a |
| f64 L1 N 8 L 8000 unbanded, 1 thread pinned | 16 | 649.71 | 653.47 | 2.758 | 2.742 | 0.994 | 0.992-0.994 | 0.97-1.03: pass | `88dd1bba61d35f0b` = B3 | - | 5.267 / 5.261 |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B3 Gcell/s | B3 ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 29.825 | 105.465 | 3.544 | 3.440-3.884 | 108.258 | 3.630 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B3: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B3: yes; lanes vs per-pair bitwise mismatches base + B3: 0; outside the route bound: 0 -> pass
- ragged sweep (B3): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- CLI (w8b_cli_runs.sh + w8b_cli_extra.sh, data/dummy): byte-identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 5
  - round 0 base f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.749-5.27 GHz
  - round 0 B3 f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.755-5.263 GHz
  - round 1 B3 f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.805-5.327 GHz
  - round 2 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 4.978-5.279 GHz
  - round 0 base f64 L1 N 8 L 8000 unbanded, 1 thread pinned: clock 4.803-5.342 GHz
- band: 11/11 shapes in band; matrix hashes base == B3 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict clang (B3): PASS**

## msvc: B3 against base, median over rounds; ratio = base ms / B3 ms per round

| fill | W | base ms | B3 ms | base Gcell/s | B3 Gcell/s | ratio (median) | range | band | matrix hash base / B3 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 8 | 3385.00 | 3372.65 | 1.468 | 1.473 | 1.000 | 0.999-1.004 | >= 0.97: pass | `b28673ce59c47ce8` = B3 | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 8 | 176.16 | 171.66 | 28.203 | 28.943 | 1.026 | 0.987-1.042 | >= 0.97: pass | `b28673ce59c47ce8` = B3 | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 8 | 2696.97 | 2686.70 | 1.402 | 1.408 | 0.997 | 0.984-1.006 | >= 0.97: pass | `d0c4ea00ede2475c` = B3 | 1.0809 (0.0608, 0.0140) | n/a |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 8 | 1288.72 | 1360.59 | 32.198 | 30.497 | 1.008 | 0.947-1.009 | >= 0.97: pass | `cd125a5433bb9769` = B3 | 1.0246 (0.0207, 0.0033) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 185.79 | 187.49 | 26.742 | 26.500 | 0.940 | 0.939-1.078 | >= 0.97: FAIL | `d0429e2d299c55ed` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 8 | 181.74 | 185.15 | 27.337 | 26.834 | 1.004 | 0.982-1.005 | >= 0.97: pass | `76e1c6ac3e6ce69d` = B3 | 1.0077 (0.0014, 0.0063) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 8 | 726.26 | 707.22 | 27.525 | 28.266 | 0.983 | 0.924-1.027 | >= 0.97: pass | `fa485cbe1fa07d20` = B3 | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 8 | 820.59 | 811.55 | 29.022 | 29.345 | 1.009 | 0.987-1.011 | >= 0.97: pass | `79637ca9e991a308` = B3 | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 8 | 100.26 | 102.43 | 48.420 | 47.398 | 0.979 | 0.979-0.997 | >= 0.97: pass | `bed68cfaa32b7445` = B3 | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 8 | 2819.16 | 2833.11 | 28.284 | 28.144 | 1.015 | 0.995-1.030 | >= 0.97: pass | `64d2a0d49855c763` = B3 | 1.0011 (0.0003, 0.0007) | n/a |
| f64 L1 N 8 L 8000 unbanded, 1 thread pinned | 8 | 2755.88 | 2760.28 | 0.650 | 0.649 | 0.995 | 0.970-1.005 | >= 0.97: pass | `88dd1bba61d35f0b` = B3 | - | n/a |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B3 Gcell/s | B3 ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 28.284 | 26.297 | 1.023 | 0.930-1.074 | 28.144 | 1.015 | = base | 1.0009 (0.0002, 0.0007) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B3: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B3: yes; lanes vs per-pair bitwise mismatches base + B3: 0; outside the route bound: 0 -> pass
- ragged sweep (B3): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- pinned probes whose clock moved by more than 3% over their fills: 0
- band: 10/11 shapes in band; matrix hashes base == B3 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict msvc (B3): FALSIFIED**

## gcc: B3 against base, median over rounds; ratio = base ms / B3 ms per round

| fill | W | base ms | B3 ms | base Gcell/s | B3 Gcell/s | ratio (median) | range | band | matrix hash base / B3 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 16 | 4063.40 | 502.27 | 1.223 | 9.892 | 8.129 | 8.058-8.232 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | 5.209 / 5.207 |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 16 | 214.29 | 42.59 | 23.185 | 116.668 | 4.868 | 4.659-5.504 | >= 2.0: pass | `b28673ce59c47ce8` = B3 | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 16 | 3446.94 | 396.38 | 1.097 | 9.541 | 8.710 | 8.527-8.785 | >= 1.5: pass | `d0c4ea00ede2475c` = B3 | 1.1780 (0.1120, 0.0391) | 5.187 / 5.204 |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 16 | 1873.72 | 344.06 | 22.145 | 120.601 | 5.479 | 5.425-5.534 | >= 1.5: pass | `cd125a5433bb9769` = B3 | 1.0535 (0.0423, 0.0084) | n/a |
| f32 L1 N 1000 L 90-110 unbanded, 24 threads | 32 | 122.64 | 22.08 | 40.512 | 225.005 | 5.554 | 4.499-6.039 | >= 1.5: pass | `d0429e2d299c55ed` = B3 | 1.0345 (0.0058, 0.0275) | n/a |
| f64 Sq N 1000 L 90-110 unbanded, 24 threads | 16 | 209.00 | 45.26 | 23.772 | 109.768 | 4.601 | 4.573-4.884 | >= 1.5: pass | `c640e49a859012ae` / `1618537f9c633a7b` (FMA caveat) | 1.0166 (0.0029, 0.0134) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 16 | 160.79 | 154.34 | 124.326 | 129.519 | 1.011 | 0.944-1.049 | 0.97-1.03: pass | `fa485cbe1fa07d20` = B3 | - | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 16 | 217.02 | 198.76 | 109.736 | 119.818 | 0.973 | 0.820-1.092 | 0.97-1.03: pass | `79637ca9e991a308` = B3 | - | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 16 | 92.67 | 96.97 | 52.388 | 50.065 | 0.973 | 0.956-1.012 | 0.97-1.03: pass | `bed68cfaa32b7445` = B3 | - | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 16 | 3187.04 | 803.26 | 25.019 | 99.266 | 3.890 | 3.726-4.091 | >= 1.5: pass | `64d2a0d49855c763` = B3 | 1.0023 (0.0007, 0.0015) | n/a |
| f64 L1 N 8 L 8000 unbanded, 1 thread pinned | 16 | 689.13 | 687.35 | 2.600 | 2.607 | 1.008 | 0.998-1.009 | 0.97-1.03: pass | `88dd1bba61d35f0b` = B3 | - | 5.249 / 5.239 |

A (rows and columns longest first; for the record, no band) on the same shape:

| fill | base Gcell/s | A Gcell/s | A ratio (median) | A range | B3 Gcell/s | B3 ratio (median) | A's matrix | A's lanes: computed/useful (padding, duplicates) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 25.019 | 100.282 | 4.012 | 4.008-4.031 | 99.266 | 3.890 | = base | 1.0019 (0.0004, 0.0015) |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B3: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq no; per-pair hashes base == B3: yes; lanes vs per-pair bitwise mismatches base + B3: 2,692,955; outside the route bound: 0 -> pass
- ragged sweep (B3): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 264,419; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0.4933, f64 L1 0, f64 Sq 0.4943 -> pass
- clustering (clust/, data/dummy uncut): clustering identical: yes
- pinned probes whose clock moved by more than 3% over their fills: 3
  - round 0 B3 f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned: clock 4.864-5.24 GHz
  - round 0 base f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned: clock 5.008-5.191 GHz
  - round 2 B3 f64 L1 N 8 L 8000 unbanded, 1 thread pinned: clock 4.801-5.256 GHz
- band: 11/11 shapes in band; matrix hashes base == B3 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict gcc (B3): PASS**

