Fill seconds as the CLI stamps them (from 'Data loaded' to the FastPAM k=1 result): median (range) over 3 round(s) x 5 runs at 24 threads, x 1 pinned to CPU 12 at 1 thread; then Gcell/s of the median. Ratio = dtwc_cl_msvc.exe / dtwc_cl.exe seconds of the medians (above 1: dtwc_cl.exe, clang-cl, faster), and the range of the per-round ratios.

| shape | threads | dtwc_cl_msvc.exe s (range), Gcell/s | dtwc_cl.exe s (range), Gcell/s | speed-up (per-round range) | cost |
| --- | --- | --- | --- | --- | --- |
| (a) N 2000, L 100, unbanded | 24 | 0.680 (0.665-0.704), 29.40 | 0.169 (0.160-0.181), 118.61 | **4.04** (3.87-4.06) | same |
| (a) N 2000, L 100, unbanded | 1 | 14.884 (13.467-14.912), 1.34 | 1.829 (1.812-1.845), 10.93 | **8.14** (7.30-8.22) | same |
| (b) N 500, L 1000, band 100 | 24 | 0.780 (0.769-0.814), 30.52 | 0.187 (0.180-0.198), 127.59 | **4.18** (4.15-4.24) | same |
| (b) N 500, L 1000, band 100 | 1 | 18.026 (17.989-18.036), 1.32 | 2.240 (2.235-2.267), 10.63 | **8.05** (7.95-8.05) | same |
| (c) N 1000, L 90-110, unbanded (ragged) | 24 | 0.171 (0.160-0.178), 29.22 | 0.139 (0.136-0.145), 35.82 | **1.23** (1.18-1.24) | same |
| (c) N 1000, L 90-110, unbanded (ragged) | 1 | 3.398 (3.379-3.415), 1.47 | 3.041 (3.041-3.049), 1.64 | **1.12** (1.11-1.12) | same |

band: PASS: the archive ships as built (dtwc_cl.exe fills each equal-length shape >= 1.2x as fast as dtwc_cl_msvc.exe at 24 threads, no shape below 0.95x)
problems: none
