# 2026-10-08 — CW: the Windows wheel built by clang-cl against the MSVC wheel — PASS

**Question.** Volkan, 2026-10-08, after MV: "Okay move the wheel to cl if you cannot make it happen, unless there is a
problem with the license. Because speed is the most important right now." Unit CW builds the Windows wheels with a
pinned clang-cl (LLVM 23.1.1, `scripts/fetch_llvm_windows.sh`), bundles its `libomp.dll`, and makes clang-cl link
LLVM's OpenMP runtime and take the project's FP flags. Does the wheel CI will build keep WM's speed-up?

**Registered band (the orchestrator's, fixed before any run; DECISIONS 2026-10-08):** the clang-cl wheel ships as
built if at 24 threads both lanes fills ((a) N 2000 L 100 unbanded, (b) N 500 L 1000 band 100) are ≥ 1.2× the MSVC
wheel's and no shape at any thread count timed is below 0.95×; otherwise Volkan decides before the merge.

**Setup.** Windows 11, Intel Core Ultra 9 285 (8 P + 16 E cores, 24 threads), 128 GB. Two fresh venvs (CPython 3.12):
`C:/D/git/wt/venv/cw-msvc`, the MSVC wheel from design-2.0 `f5ad16a2` as CI builds it today, and
`C:/D/git/wt/venv/cw-clang`, the clang-cl wheel from `pb/CW` (`3619c828`) built by cibuildwheel 3.4.1 locally with
the workflow's CIBW settings and repaired by the pinned delvewheel (`libomp` and `msvcp140` bundled). WM's shapes and
driver (`2026-10-07-wm-wheel-msvc-vs-clang-cl.md`): (a), (b) lanes fills, (c) N 1000 L 90–110 unbanded (the per-pair
route), (d) one pair of L 1000 through `dtwcpp.distance.dtw`, unbanded and band 100; one thread pinned to CPU 12 and
24 threads; 3 rounds alternating which wheel goes first. Kit and results beside this record in
`2026-10-08-cw-clang-cl-wheel/`. Prepared by unit CW (Opus 5.5); timed by the orchestrator, 2026-10-08 19:54–19:59.

**Load** (`results/load.txt`): total CPU 2.9–8.6 % before, 2.9–4.2 % after (10 one-second samples each); the
busiest process 1.6 % of all cores; no COMSOL. A Tanium scan holding one core ended before the run (19:54).

## Result [confirmed]

Medians (range) over 3 rounds, Gcell/s; ratio = clang-cl / MSVC of the medians (above 1: clang-cl faster).

| shape | threads | MSVC Gcell/s | clang-cl Gcell/s | clang-cl/MSVC | shortest timed s | result |
| --- | --- | --- | --- | --- | --- | --- |
| (a) N 2000, L 100, unbanded: lanes fill | 1 | 1.341 (1.332-1.345) | 11.019 (10.883-11.074) | **8.22** | 1.81 | same |
| (a) N 2000, L 100, unbanded: lanes fill | 24 | 31.088 (30.761-31.441) | 123.256 (122.845-127.712) | **3.96** | 0.63 | same |
| (b) N 500, L 1000, band 100: lanes fill | 1 | 1.324 (1.312-1.453) | 10.780 (10.747-10.787) | **8.14** | 2.21 | same |
| (b) N 500, L 1000, band 100: lanes fill | 24 | 31.133 (30.898-31.190) | 134.231 (132.265-134.608) | **4.31** | 0.53 | same |
| (c) N 1000, L 90-110, unbanded: per-pair fill | 1 | 1.470 (1.469-1.474) | 1.628 (1.625-1.633) | **1.11** | 3.07 | same |
| (c) N 1000, L 90-110, unbanded: per-pair fill | 24 | 31.478 (31.363-33.056) | 38.614 (36.306-38.843) | **1.23** | 0.52 | same |
| (d) dtw, one pair, L 1000, unbanded | 1 | 1.387 (1.378-1.399) | 1.407 (1.406-1.415) | **1.01** | 0.50 | same |
| (d) dtw, one pair, L 1000, band 100 | 1 | 0.859 (0.857-0.859) | 0.881 (0.862-0.883) | **1.03** | 0.50 | same |

band: **PASS**: the clang-cl wheel ships as built (both lanes fills ≥ 1.2× at 24 threads, no shape below 0.95×).
"same": the two wheels' results are equal (the kit's check); unit CW also found labels and medoids identical in 48 of
48 clusterings, with no difference in any cost or matrix entry.

**Reading [inferred].** WM measured 4.40–4.68× on one thread and 3.14–3.33× at 24 on the lanes fills (8a63cda3).
Now 8.1–8.2× and 4.0–4.3×: since WM, clang's x86-64 lanes went to 128 bytes (XL), and CW made clang-cl take the
project's FP flags, which WM's build ignored. MSVC still scales further with threads (its scalar lanes are
compute-bound per core), so the ratio falls from one thread to 24.

**Not covered:** CI itself (the runner's Visual Studio detection, cp39–311 and cp313–314); "OMP: Error #15" beside
Intel's OpenMP, documented in installation.md, not exercised; cache misses (no hardware counters here yet).
