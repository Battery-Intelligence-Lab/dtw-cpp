# 2026-10-01 — W13b: one finite scan at each matrix intake

Question: with a matrix value from outside the fill checked once where it enters, do the loops that only read a filled
matrix drop their per-lookup checks with identical results, and is the PAM swap no slower? Criterion set before the
runs: conformance digit-identical; dtwc_cl CSVs byte-identical on data/dummy; bench counters identical, time advisory.

Branch pb/W13b, base e4a1bca. Commits: 5605f4e (intake scan, +158/−51), 3599ebc (FasterPAM loops unchecked, +6/−30),
a7f5af1 (OrderedMedoidObjective non-throwing, one objective check, +27/−54), 8895c95 (tests, +6/−233). Total +197/−368.

## Gates [confirmed]

| gate | base | after each step | head |
|---|---|---|---|
| serial ctest, `build/` (clang, llfio, HiGHS, Gurobi) | 114 = 111 passed + 3 MAY_SKIP | same at steps 1–4 | 114 = 111 + 3 (cuda/metal correctness, metal_mmap) |
| cpp_conformance | passed | passed at steps 1–4 | passed |
| dtwc_cl, 20 configurations on data/dummy (E1's runner) | reference | files, stdout, stderr, exit codes byte-identical at steps 1–4 | identical |
| pytest, fresh venv | 1176 passed, 19 skipped | step 1: 1176 / 19 | 1176 / 19 (see note) |
| matlab_suite, clang MEX, R2024b | 149 run, 148 passed, 0 failed, 1 incomplete | step 1: 150 / 149 / 0 / 1 | 150 / 149 / 0 / 1 |
| serial ctest, `build-arrow` (Arrow + Parquet, llfio off) | not run | step 3: 117 = 114 + 3 | 117 = 114 + 3; test_io_readers and test_fast_clara_assignment_contract ran |

Note: the first head pytest run had 7 failures, all CLI tests that pick the newest dtwc_cl under the repo
(`_hpc.find_dtwc_binary`); that was `build-arrow`'s, which exits 0xC0000135 without pyarrow's DLLs on PATH. With them on
PATH the 7 pass and the full run is 1176 passed, 19 skipped.

## Intake rows bite [confirmed]

Each C++ row of "Every matrix intake refuses a distance that is not finite, naming the pair" fails alone when its scan is
replaced by the old NaN-only test (commit point, CSV, checkpoint, mmap). The Python row (test_api.py, ±inf) and the MATLAB
row fail with their binding's scan replaced the same way.

## Assembly, fast_pam.cpp [confirmed]

compile_commands flags, no `-flto`, `-S -masm=intel`. Instructions on the common path (no update), per lookup:

| loop | base | head | removed |
|---|---|---|---|
| find_best_swap (in swap_phase) | 27 | 21.5 (43 per 2, unrolled ×2) | `vmovq; bzhi; movabs 0x7FF0000000000000; cmp; jge` to the throw |
| compute_nearest_and_second (OpenMP body) | 24 | 18.5 (37 per 2, unrolled ×2) | `vmovq; bzhi; cmp; jge` to the throw |

OpenMP bodies with a catch funclet in fast_pam.cpp: 2 at base (nearest/second, k = 1), 0 at head. ordered_medoid_objective
at head: per element `vmovsd xmm0,[x]; vaddsd xmm0,xmm0,[rbp+24]; vmovsd [rbp+24],xmm0`, unrolled ×8 in point order,
then one finite check. At base the accumulator's add was an out-of-line call per element. The zero canonicalization in
`value()` is folded away under `-fno-signed-zeros` at base and head alike.

## PAM swap bench (E1's bench_pam_swap.cpp) [counters confirmed; time inferred: machine under load]

Counters identical in all 12 runs, base and head: (2000, 10) sweeps 4, cost 3565.852605417439, hash ddb78d417a641718;
(2000, 50) 2, 2160.9379984901775, 8db78e5116aac6d6; (4000, 10) 1, 7121.1746602151261, d69dfe2050dfa8bb; (4000, 50) 2,
4349.0299483885392, fbb6a4bf2ba323f7.

Median ms, base | head, runs alternating:

| N, k | unpinned, 24 threads, 5 reps | P-cores (mask C03C03), 8 threads, 11 reps |
|---|---|---|
| 2000, 10 | 124.3, 112.9, 104.6 \| 84.6, 76.7, 81.3 | 94.8, 122.7, 89.7 \| 79.9, 102.0, 81.8 |
| 2000, 50 | 52.6, 48.6, 51.9 \| 96.3, 77.2, 67.2 | 58.2, 63.1, 56.2 \| 44.0, 57.2, 46.7 |
| 4000, 10 | 171.3, 162.7, 179.7 \| 177.1, 149.5, 166.0 | 175.1, 205.6, 192.8 \| 168.1, 193.1, 208.2 |
| 4000, 50 | 302.0, 271.0, 258.0 \| 223.0, 246.1, 215.8 | 251.5, 343.1, 375.4 \| 251.2, 265.9, 346.6 |

The unpinned (2000, 50) head medians came with maxima of 193–277 ms; pinned, head is below base in all three pairs.

## Cost of one intake scan [inferred: machine under load]

`all_computed(where)` on a filled heap matrix, min of 5, against the fill it would precede (sine series):
N 4000, L 16: 4.6 ms vs fill 149.9 ms (3.0 %); N 4000, L 100: 6.7 vs 1306.8 ms (0.5 %); N 20000, L 16: 251.6 vs
2578.8 ms (9.8 %). The fill's commit-point scan therefore runs only after a write through the mutable
`distance_matrix()` (commit after 8895c95): fill of a freshly mapped matrix, N 10000, L 1, 6 runs each, accessor
untouched 66.1–86.9 ms, touched 95.8–143.8 ms.

Conclusion: results identical; the swap is no slower and mostly a little faster. Open: a fill of finite series can
overflow (±DBL_MAX series give +inf; soft-DTW at gamma = DBL_MAX gives −inf) and is not scanned, so Lloyd's
assign_clusters keeps its check.
