# MV report from raw.txt

clock of CPU 12 (bin/clk.exe, a dependent register-add chain, median of five per round): round 0 5.271 GHz, round 1 5.276 GHz, round 2 5.260 GHz

## msvc: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single thread) or cand Gcell/s / base Gcell/s (fill), per round

| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand |
| --- | --- | --- | --- | --- | --- | --- |
| f32 L1 L 100 unbanded | 16/32 | 0.65157 | 0.11305 | 5.559 | 1.620-5.764 | 54.84 / 19.03 |
| f32 L1 L 100 band 10 | 16/32 | 0.65683 | 0.11562 | 5.681 | 5.538-5.762 | 55.28 / 19.46 |
| f32 L1 L 1000 unbanded | 16/32 | 0.65193 | 0.09918 | 6.586 | 5.919-6.610 | 54.98 / 16.74 |
| f32 L1 L 1000 band 100 | 16/32 | 0.64010 | 0.10330 | 6.323 | 5.909-6.360 | 54.03 / 17.39 |
| f32 Sq L 100 unbanded | 16/32 | 0.63831 | 0.11676 | 5.526 | 5.439-5.577 | 53.72 / 19.71 |
| f32 Sq L 100 band 10 | 16/32 | 0.66975 | 0.11751 | 5.699 | 1.747-5.700 | 56.37 / 19.78 |
| f32 Sq L 1000 unbanded | 16/32 | 0.64418 | 0.10800 | 5.920 | 1.794-6.015 | 54.33 / 18.23 |
| f32 Sq L 1000 band 100 | 16/32 | 0.63937 | 0.11542 | 5.540 | 5.467-5.633 | 53.97 / 19.43 |
| f64 L1 L 100 unbanded | 8/16 | 0.70871 | 0.21381 | 3.402 | 1.030-3.500 | 29.88 / 18.05 |
| f64 L1 L 100 band 10 | 8/16 | 0.76284 | 0.24817 | 2.976 | 2.868-3.226 | 32.17 / 20.95 |
| f64 L1 L 1000 unbanded | 8/16 | 0.73469 | 0.21778 | 3.381 | 3.371-3.502 | 30.92 / 18.34 |
| f64 L1 L 1000 band 100 | 8/16 | 0.72587 | 0.21200 | 3.424 | 3.035-3.531 | 30.64 / 17.90 |
| f64 Sq L 100 unbanded | 8/16 | 0.68342 | 0.24144 | 2.838 | 2.788-2.844 | 28.82 / 20.38 |
| f64 Sq L 100 band 10 | 8/16 | 0.68471 | 0.27531 | 2.570 | 2.315-2.687 | 28.81 / 23.24 |
| f64 Sq L 1000 unbanded | 8/16 | 0.71018 | 0.24487 | 2.889 | 2.721-2.900 | 29.95 / 20.65 |
| f64 Sq L 1000 band 100 | 8/16 | 0.68968 | 0.80188 | 0.867 | 0.692-2.914 | 29.11 / 67.49 |

| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) | range | matrices |
| --- | --- | --- | --- | --- | --- | --- |
| N 2000 L 100 unbanded, 24 threads | 8/16 | 29.86 | 78.51 | 2.629 | 2.615-2.643 | identical |
| N 500 L 1000 band 100, 24 threads | 8/16 | 30.46 | 76.60 | 2.514 | 2.484-2.546 | identical |
| N 1000 L 100 +-10 ragged unbanded, 24 threads | 8/16 | 30.40 | 30.08 | 0.947 | 0.866-1.133 | identical |

- sweep: 16,560,000 lane outputs per build, hashes base == cand: NO, lanes vs per-pair mismatches (base + cand): 1343025; fill matrices equal: yes
-   cand f32 L1 W 32: hash = base, mismatches against per-pair 0 of 4,140,000, outside the route bound 0, max dev 0.000 (bound 2)
-   cand f32 Sq W 32: hash differs from base, mismatches against per-pair 670,401 of 4,140,000, outside the route bound 0, max dev 0.328 (bound 2)
-   cand f64 L1 W 16: hash = base, mismatches against per-pair 0 of 4,140,000, outside the route bound 0, max dev 0.000 (bound 2)
-   cand f64 Sq W 16: hash differs from base, mismatches against per-pair 672,624 of 4,140,000, outside the route bound 0, max dev 0.309 (bound 2)
- band: every single-thread shape >= 2.0x: min 0.867 (16 shapes) -> fail; both equal-length fills >= 1.5x: 2.629, 2.514 -> pass; ragged fill in 0.97-1.03x: 0.947 -> fail; bitwise -> fail
- not the band, for the record: speed clauses -> fail; L1 lanes bitwise (= per-pair, hashes = base) -> yes; squared lanes outside the route bound -> 0

- clang: dtw_lanes.cpp assembly identical to base: yes (asm/clang_check.txt; identical assembly needs no timing under the band)
- **verdict: FALSIFIED**
