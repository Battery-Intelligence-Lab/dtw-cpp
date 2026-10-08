# 2026-10-08 — CA: the Windows archive's dtwc_cl.exe (clang-cl) against dtwc_cl_msvc.exe (cl) — PASS

**Question.** Volkan, 2026-10-08, on the release archive: "yes archive too, you can also keep msvc compiled versions
with "_msvc" suffix like dtwc_cl_msvc.exe, so we can write in documentation in our tests we had seen this much speedup
so we suggest use of this but it is up to you or in case you have any issues." Unit CA builds the Windows archive's
`dtwc_cl.exe` with the pinned clang-cl and packs cl's build beside it as `dtwc_cl_msvc.exe`. How much faster is the
clang-cl build, the number the docs quote?

**Registered band (the orchestrator's, fixed before any run; DECISIONS 2026-10-08):** the archive ships as built if
`dtwc_cl.exe` fills each equal-length shape ≥ 1.2× as fast as `dtwc_cl_msvc.exe` at 24 threads and no shape is below
0.95×; otherwise Volkan decides before the merge.

**Setup.** Windows 11, Intel Core Ultra 9 285 (8 P + 16 E cores, 24 threads), 128 GB. The archive
(`dtwc-2.0.0rc1-Windows-AMD64.zip`, 312 files) built locally from `pb/CA` (`8b6bf4ef`) as `release-artifacts.yml`
will build it (LLVM 23.1.1's clang-cl; cl for the fallback), unpacked outside the repo; both executables run from it.
Data: fixed-seed random walks written once (`kit/gen_data.py`, SHA-256 manifest), one series per CSV row. Shapes:
(a) N 2000, L 100 unbanded; (b) N 500, L 1000, band 100; (c) N 1000, L 90–110 unbanded (ragged: the one-pair route
in both builds). Time: the CLI's own stamps from "Data loaded" to the FastPAM k = 1 result, so the span includes a
small clustering step and the ratio errs low. 24 threads (5 runs a round; no OMP variable) and one thread
(`OMP_NUM_THREADS=1`, pinned to CPU 12, affinity read back from the CLI's process); 3 rounds alternating which exe
goes first. Kit and results beside this record in `2026-10-08-ca-clang-cl-archive/`. Prepared by unit CA (Opus 5.5);
timed by the orchestrator, 2026-10-08 19:59–20:03.

**Load** (`results/load.txt`): total CPU 2.3–9.2 % before, 1.6–5.2 % after (10 one-second samples each); the busiest
process 1.6 % of all cores; no COMSOL.

## Result [confirmed]

Median (range) of the fill seconds over 3 rounds; Gcell/s of the median. Speed-up = dtwc_cl_msvc.exe seconds /
dtwc_cl.exe seconds (above 1: dtwc_cl.exe, clang-cl, faster), with the range of the per-round ratios.

| shape | threads | dtwc_cl_msvc.exe s (range), Gcell/s | dtwc_cl.exe s (range), Gcell/s | speed-up (per-round range) | cost |
| --- | --- | --- | --- | --- | --- |
| (a) N 2000, L 100, unbanded | 24 | 0.680 (0.665-0.704), 29.40 | 0.169 (0.160-0.181), 118.61 | **4.04** (3.87-4.06) | same |
| (a) N 2000, L 100, unbanded | 1 | 14.884 (13.467-14.912), 1.34 | 1.829 (1.812-1.845), 10.93 | **8.14** (7.30-8.22) | same |
| (b) N 500, L 1000, band 100 | 24 | 0.780 (0.769-0.814), 30.52 | 0.187 (0.180-0.198), 127.59 | **4.18** (4.15-4.24) | same |
| (b) N 500, L 1000, band 100 | 1 | 18.026 (17.989-18.036), 1.32 | 2.240 (2.235-2.267), 10.63 | **8.05** (7.95-8.05) | same |
| (c) N 1000, L 90-110, unbanded (ragged) | 24 | 0.171 (0.160-0.178), 29.22 | 0.139 (0.136-0.145), 35.82 | **1.23** (1.18-1.24) | same |
| (c) N 1000, L 90-110, unbanded (ragged) | 1 | 3.398 (3.379-3.415), 1.47 | 3.041 (3.041-3.049), 1.64 | **1.12** (1.11-1.12) | same |

band: **PASS**: the archive ships as built. "same": equal clustering costs. Unit CA's CLI comparison on data/dummy:
`dtwc_cl_msvc.exe` 156 of 156 files byte-identical to the clang++ build; `dtwc_cl.exe` 155 of 156, the one difference
a hierarchical run's printed total cost one ulp apart (a regrouped sum, which `-fassociative-math` allows); labels,
medoids, matrices and silhouettes identical.

**For the docs:** on equal-length series `dtwc_cl.exe` filled the matrix about 4× faster than `dtwc_cl_msvc.exe` with
24 threads (4.04–4.18×) and about 8× on one thread (8.05–8.14×). These are the numbers installation.md and the
CHANGELOG quote.

**Not covered:** the workflow on GitHub's runner, a start on a machine without the Visual C++ Redistributable (the
smoke's PE import check stands in for it), cache misses.
