# 2026-10-01 — W8c: Python and MATLAB `compute_distance_matrix` through `Problem`

Question: does the matrix the Problem fill returns equal the matrix of the Python binding's own loop that W8c
deletes, and is the Problem route slower?

Branch pb/W8c, base 171b724c. Commits 3c412c42 (Python), b79bbea9 (MATLAB). Logs are `C:/D/git/wt/W8c-*.log` and
`C:/D/git/wt/W8c-matrices/` (disposable; scripts were scratch).

## Method [confirmed]

Base = `_dtwcpp_core.compute_distance_matrix(series, band, metric)` (the deleted loop, `dtwBanded` / `dtwFull_L` per
pair) from the base wheel, copied to `C:/D/git/wt/W8c-base-pkg` before the edit; head = `dtwcpp.compute_distance_matrix`
from the head wheel (MSVC wheel, `CMAKE_GENERATOR` unset). Same interpreter, same inputs, `numpy.savez` of each result.

## Matrices: 23 configurations, base vs head [confirmed]

All 23 are byte-identical, max |difference| 0.

- data/dummy (25 ragged series of 5148-9405 samples): full DTW, L1 and squared Euclidean.
- data/dummy cut to 700 samples (equal length): band 0, 10, 100, -1, each in L1 and squared Euclidean.
- data/dummy subsampled by 10 (ragged, 515-941 samples): band 426 (the smallest feasible), 476, -1, L1.
- 40 random ragged series of 30-59 samples: band 30 and -1, L1 and squared Euclidean.
- N = 0, N = 1, three series of one sample.
- Control, a code path the unit does not touch: a Problem with `missing_strategy` zero_cost, arow and interpolate on
  three series holding NaN.

MATLAB head (`dtwc.compute_distance_matrix`, clang MEX, R2024b) on data/dummy cut to 700 samples (25 x 700), Band 0,
10, -1, against the base Python matrix (CSV, `%.17g`): max |difference| 0 for each.

## Behaviour that is not a matrix, base vs head [confirmed]

| Input | Base | Head |
|---|---|---|
| NaN in a series | `InvalidInput` "compute_distance_matrix: series[0][2] is NaN ..." | `InvalidInput` "Problem::fill_distance_matrix: series '0' (index 0)[2] is NaN ..." |
| +inf in a series | `InvalidInput` "series[0][3] is +inf" | `InvalidInput` "series '0' (index 0)[3] is +inf" |
| band 2 on lengths 4, 10, 6 | the matrix, with 1.7976931348623157e+308 where no path fits | `InvalidInput`, "band = 2 is narrower than the length difference between series '0' (index 0, length 4) and series '1' (index 1, length 10) ... smallest feasible band is 6" |
| metric 'bogus' | binding: `InvalidInput` "unknown metric 'bogus'. Valid: l1, squared_euclidean."; the Python wrapper in front of it: `ValueError` "Unknown metric" | `InvalidInput`, the binding's message |
| metric 'l2sq' | binding accepts; the Python wrapper refused | accepted |

The infeasible-band rule is the fill's own (`validate_fill_request`, D-12): `InvalidInput` on every device.

## Timing: 200 series x 1000 samples (random walks), L1 [inferred]

Machine shared (other agents; CPU load 5-25% during the second round, 100% at the start of the first), so wall-clock
is advisory. Per block 5 repetitions after a 20-series warm-up; blocks alternate base, head, base, head, base, head
(same process model: one interpreter per block). Second round, medians per block (min-max):

| | base | head |
|---|---|---|
| full DTW | 3.133 (3.010-3.329), 2.945 (2.845-3.087), 2.818 (2.774-3.208) s | 1.001 (0.631-1.012), 0.804 (0.602-1.038), 0.926 (0.587-0.968) s |
| band 50 | 0.354, 0.307, 0.307 s | 0.094, 0.099, 0.097 s |

Pooled medians of the 15 repetitions: full DTW 3.01 s base, 0.94 s head; band 50 0.312 s base, 0.099 s head. First
round (loaded): base 4.43 / 6.20 / 8.01 s, head 0.98 / 2.71 / 1.06 s. One quiet-machine base block before the edit
(CPU load 6%): full 3.824 (3.734-3.879) s, band 50 0.360 (0.350-0.382) s. The matrix checksum (`D.sum()`) agrees to
all printed digits in every block: 642706933.412037 (full), 854376699.359075 (band 50).

Reading: the Problem fill is faster, not slower, by about 3x, with the rank gap far outside the spread. [inferred]
cause: the old loop called the scalar per-pair kernel; the fill takes equal-length columns W at a time through the lane
kernel (`dtw_block_fn_`, `fillDistanceMatrix_BruteForce`), which the matrices above show to be bit-identical to the
per-pair one on this data.

## Gates at head [confirmed]

- serial `ctest -j1` (main build, no Arrow): 95 tests, 92 pass, 3 MAY_SKIP skips (test_cuda_correctness,
  test_metal_correctness, test_metal_mmap), 0 failed; base the same. cpp_conformance passes. No C++ under dtwc/ changed.
- pytest: base 1095 passed / 19 skipped (1114 collected), head 1094 passed / 19 skipped (1113 collected): -2
  `test_api.py::TestCpuMatrixRoute::test_matrix_is_digit_identical_to_compute_distance_matrix[-1|5]`, one renamed
  (`test_raw_matrix_binding_reads_the_metric_from_the_cpp_table` ->
  `test_compute_distance_matrix_reads_the_metric_from_the_cpp_table`), +1
  `test_a_band_narrower_than_the_length_difference_is_refused`.
- matlab_suite (clang MEX, R2024b): base 142 run / 141 passed / 0 failed / 1 incomplete, head 141 / 140 / 0 / 1: -1
  `test_mex_input_validation/test_valid_double_matrix_still_works`; four rejection cases renamed
  `test_distance_matrix_*_rejected` -> `test_set_data_*_rejected`. The one incomplete is
  `test_test_api/test_parallelisation_serial_is_honest`.
