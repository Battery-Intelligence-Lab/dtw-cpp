# XL report from raw.txt

clock of CPU 12 (the clang probes' dependent register-add chain, median per round): round 0 5.278 GHz, round 1 5.298 GHz, round 2 5.292 GHz

## clang: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single thread) or cand Gcell/s / base Gcell/s (fill), per round

| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand |
| --- | --- | --- | --- | --- | --- | --- |
| f32 L1 L 100 unbanded | 16/32 | 0.08062 | 0.04541 | 1.786 | 1.445-1.818 | 6.88 / 7.62 |
| f32 L1 L 100 band 10 | 16/32 | 0.06576 | 0.05499 | 1.209 | 1.139-1.219 | 5.58 / 9.22 |
| f32 L1 L 1000 unbanded | 16/32 | 0.08475 | 0.04304 | 1.964 | 1.941-1.980 | 7.17 / 7.29 |
| f32 L1 L 1000 band 100 | 16/32 | 0.08152 | 0.04842 | 1.685 | 1.658-1.685 | 6.86 / 8.18 |
| f32 Sq L 100 unbanded | 16/32 | 0.08091 | 0.04480 | 1.806 | 1.796-1.836 | 6.87 / 7.55 |
| f32 Sq L 100 band 10 | 16/32 | 0.06371 | 0.05514 | 1.148 | 0.758-1.243 | 5.39 / 9.36 |
| f32 Sq L 1000 unbanded | 16/32 | 0.08459 | 0.04327 | 1.942 | 1.941-1.974 | 7.14 / 7.30 |
| f32 Sq L 1000 band 100 | 16/32 | 0.08030 | 0.04807 | 1.670 | 1.647-1.676 | 6.80 / 8.16 |
| f64 L1 L 100 unbanded | 8/16 | 0.15173 | 0.08848 | 1.715 | 1.714-1.729 | 6.41 / 7.50 |
| f64 L1 L 100 band 10 | 8/16 | 0.11685 | 0.09055 | 1.283 | 1.277-1.316 | 4.93 / 7.71 |
| f64 L1 L 1000 unbanded | 8/16 | 0.16721 | 0.08488 | 1.970 | 1.940-1.971 | 7.08 / 7.22 |
| f64 L1 L 1000 band 100 | 8/16 | 0.15737 | 0.09043 | 1.738 | 1.736-1.763 | 6.62 / 7.71 |
| f64 Sq L 100 unbanded | 8/16 | 0.15115 | 0.08923 | 1.696 | 1.684-1.705 | 6.40 / 7.58 |
| f64 Sq L 100 band 10 | 8/16 | 0.11373 | 0.09028 | 1.264 | 1.260-1.270 | 4.85 / 7.63 |
| f64 Sq L 1000 unbanded | 8/16 | 0.16772 | 0.08666 | 1.937 | 1.928-1.948 | 7.11 / 7.29 |
| f64 Sq L 1000 band 100 | 8/16 | 0.15776 | 0.08988 | 1.754 | 1.738-1.761 | 6.67 / 7.61 |

| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) | range | matrices |
| --- | --- | --- | --- | --- | --- | --- |
| N 2000 L 100 unbanded, 24 threads | 8/16 | 101.61 | 124.40 | 1.194 | 1.118-1.226 | identical |
| N 500 L 1000 band 100, 24 threads | 8/16 | 112.93 | 131.95 | 1.170 | 1.166-1.186 | identical |
| N 1000 L 100 +-10 ragged unbanded, 24 threads | 8/16 | 36.28 | 35.84 | 0.988 | 0.967-1.006 | identical |

- sweep: 16,560,000 lane outputs per build, hashes base == cand: yes, lanes vs per-pair mismatches (base + cand): 0; fill matrices equal: yes
- f64 L1 unbanded (V3's prediction about 2x; no band): 1.715, 1.970
- band: every single-thread shape >= 1.1x: min 1.148 (16 shapes) -> pass; both equal-length fills >= 1.1x: 1.194, 1.170 -> pass; ragged fill in 0.97-1.03x: 0.988 -> pass; bitwise -> pass
- **verdict clang: PASS**

## msvc: candidate (cand) against base, median over rounds; ratio = base ns / cand ns (single thread) or cand Gcell/s / base Gcell/s (fill), per round

| shape | W base/cand | base ns/cell | cand ns/cell | ratio (median) | range | cycles/row step base/cand |
| --- | --- | --- | --- | --- | --- | --- |
| f32 L1 L 100 unbanded | 16/32 | 0.64037 | 0.82598 | 0.770 | 0.762-0.775 | 54.22 / 139.86 (clang's clock) |
| f32 L1 L 100 band 10 | 16/32 | 0.66588 | 0.84615 | 0.786 | 0.757-0.796 | 56.32 / 143.38 (clang's clock) |
| f32 L1 L 1000 unbanded | 16/32 | 0.65780 | 0.85295 | 0.772 | 0.743-0.777 | 55.69 / 144.61 (clang's clock) |
| f32 L1 L 1000 band 100 | 16/32 | 0.64686 | 0.84902 | 0.762 | 0.741-0.766 | 54.63 / 143.40 (clang's clock) |
| f32 Sq L 100 unbanded | 16/32 | 0.65076 | 0.78144 | 0.829 | 0.819-0.847 | 55.16 / 132.29 (clang's clock) |
| f32 Sq L 100 band 10 | 16/32 | 0.67487 | 0.80408 | 0.842 | 0.826-0.845 | 56.99 / 136.32 (clang's clock) |
| f32 Sq L 1000 unbanded | 16/32 | 0.66225 | 0.79662 | 0.828 | 0.818-0.832 | 55.93 / 134.98 (clang's clock) |
| f32 Sq L 1000 band 100 | 16/32 | 0.64827 | 0.78998 | 0.823 | 0.803-0.831 | 54.89 / 133.93 (clang's clock) |
| f64 L1 L 100 unbanded | 8/16 | 0.71674 | 0.71721 | 1.000 | 0.983-1.028 | 30.37 / 60.72 (clang's clock) |
| f64 L1 L 100 band 10 | 8/16 | 0.74424 | 0.75805 | 0.982 | 0.692-1.024 | 31.54 / 64.26 (clang's clock) |
| f64 L1 L 1000 unbanded | 8/16 | 0.73929 | 0.71916 | 1.003 | 0.987-1.029 | 31.33 / 60.90 (clang's clock) |
| f64 L1 L 1000 band 100 | 8/16 | 0.73993 | 0.70276 | 0.995 | 0.995-1.053 | 31.24 / 59.35 (clang's clock) |
| f64 Sq L 100 unbanded | 8/16 | 0.66650 | 0.68955 | 0.892 | 0.696-0.967 | 28.14 / 58.45 (clang's clock) |
| f64 Sq L 100 band 10 | 8/16 | 0.70985 | 0.71522 | 0.992 | 0.914-0.995 | 30.09 / 60.63 (clang's clock) |
| f64 Sq L 1000 unbanded | 8/16 | 0.69671 | 0.69998 | 0.964 | 0.928-1.018 | 29.53 / 59.11 (clang's clock) |
| f64 Sq L 1000 band 100 | 8/16 | 0.69686 | 0.67784 | 1.028 | 1.002-1.031 | 29.50 / 57.39 (clang's clock) |

| fill (f64 L1, OpenMP default threads) | W base/cand | base Gcell/s | cand Gcell/s | ratio (median) | range | matrices |
| --- | --- | --- | --- | --- | --- | --- |
| N 2000 L 100 unbanded, 24 threads | 8/16 | 29.52 | 28.20 | 0.964 | 0.955-0.990 | identical |
| N 500 L 1000 band 100, 24 threads | 8/16 | 30.49 | 28.82 | 0.945 | 0.933-0.948 | identical |
| N 1000 L 100 +-10 ragged unbanded, 24 threads | 8/16 | 29.55 | 29.68 | 1.009 | 0.993-1.071 | identical |

- sweep: 16,560,000 lane outputs per build, hashes base == cand: yes, lanes vs per-pair mismatches (base + cand): 0; fill matrices equal: yes
- f64 L1 unbanded (V3's prediction about 2x; no band): 1.000, 1.003
- band: every single-thread shape >= 1.1x: min 0.762 (16 shapes) -> fail; both equal-length fills >= 1.1x: 0.964, 0.945 -> fail; ragged fill in 0.97-1.03x: 1.009 -> pass; bitwise -> pass
- **verdict msvc: FALSIFIED** -- but msvc was disqualified before timing by the assembly: cl's f32 lanes keep every lane's left on the stack at W 32 (5 of 16 at W 16); these rounds are information

