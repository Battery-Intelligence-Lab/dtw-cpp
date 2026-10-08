# RL report from raw_cl.txt

# run 2026-10-08 21:13:06 rounds=3 compilers=msvc cpu=12 vcpu=2 cand=B2
# info msvc_base: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# info msvc_B2: info cl 195035723 W f64 8 f32 16 x86_64 1 AVX2 1 FMA 0 omp threads 24
# run done in 316 s

## msvc: B2 against base, median over rounds; ratio = base ms / B2 ms per round

| fill | W | base ms | B2 ms | base Gcell/s | B2 Gcell/s | ratio (median) | range | band | matrix hash base / B2 | B's lanes: computed/useful (padding, duplicates) | clock GHz base / B2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 L1 N 1000 L 90-110 unbanded, 1 thread pinned | 8 | 3343.52 | 3345.29 | 1.486 | 1.485 | 0.999 | 0.994-1.000 | >= 0.97: pass | `b28673ce59c47ce8` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 1000 L 90-110 unbanded, 24 threads | 8 | 159.49 | 178.83 | 31.152 | 27.782 | 0.951 | 0.869-1.030 | >= 0.97: FAIL | `b28673ce59c47ce8` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 150 L 100-1000 unbanded, 1 thread pinned | 8 | 2665.71 | 2662.88 | 1.419 | 1.420 | 1.003 | 0.998-1.010 | >= 0.97: pass | `d0c4ea00ede2475c` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 500 L 100-1000 unbanded, 24 threads | 8 | 1160.28 | 1159.37 | 35.762 | 35.790 | 0.985 | 0.951-1.002 | >= 0.97: pass | `cd125a5433bb9769` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 2000 L 100 unbanded, 24 threads | 8 | 679.09 | 695.55 | 29.436 | 28.740 | 0.976 | 0.963-1.027 | >= 0.97: pass | `fa485cbe1fa07d20` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 500 L 1000 band 100, 24 threads | 8 | 780.43 | 779.04 | 30.515 | 30.569 | 1.002 | 0.992-1.002 | >= 0.97: pass | `79637ca9e991a308` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 1000 L 90-110 band 30, 24 threads | 8 | 94.53 | 94.82 | 51.357 | 51.200 | 0.997 | 0.986-1.003 | >= 0.97: pass | `bed68cfaa32b7445` = B2 | - (no ragged block in the lanes under cl) | n/a |
| f64 L1 N 8000 L 40-60 unbanded, 24 threads | 8 | 2657.20 | 2715.39 | 30.008 | 29.365 | 0.985 | 0.956-1.049 | >= 0.97: pass | `64d2a0d49855c763` = B2 | - (no ragged block in the lanes under cl) | n/a |

- equal-length sweep: 16,560,000 lane outputs per build; lane hashes base == B2: f32 L1 yes, f32 Sq yes, f64 L1 yes, f64 Sq yes; per-pair hashes base == B2: yes; lanes vs per-pair bitwise mismatches base + B2: 0; outside the route bound: 0 -> pass
- ragged sweep (B2): 3,312,000 lane outputs; vs per-pair bitwise mismatches L1 0, squared 0; outside the route bound 0; worst dev f32 L1 0, f32 Sq 0, f64 L1 0, f64 Sq 0 -> pass
- pinned probes whose clock moved by more than 3% over their fills: 0
- band: 7/8 shapes in band; matrix hashes base == B2 (L1 everywhere, squared under clang): yes; outputs as registered: yes
- **verdict msvc (B2): FALSIFIED**

