# 2026-09-30 — P5: OneBatchPAM's batch table fill runs on the SIMD lanes kernel

Question: `one_batch_pam` fills its N x m table (m = 220 at N = 2000) with m(N-1) = 439,780 DTW calls, one per pair
through `Problem::dtw_function()`. At the base (`ab5ec7c`) that is about 95 % of a call (P4:
`baselines/2026-09-30-p4-onebatch-assign.md`). Do the W = 64 / sizeof(T) columns of a row, through the compiled lanes entry
`core::resolve_dtw_block_fn` (`core/dtw_lanes.cpp`, as `Problem::fillDistanceMatrix_BruteForce` does), make the table
fill at least 2x faster without changing a table entry, label, medoid, cost or DTW call count?

## Band and model, registered BEFORE any code change [registered]

Workload (P4's): N = 2000, k = 10, L = 200 equal-length random walks (fixed seed), full band, L1, float64, automatic batch
m = 220 (27 blocks of 8 columns and a tail of 4), 24 threads, base and head binaries alternating, medians of >= 5 rounds.
Wall-clock on this shared machine is advisory; the load is noted at each run.

Model. The P1 lanes kernel ran 26x faster single-thread than the per-pair path of the fill (0.26 against 6.4 ns per cell,
loaded) and 5-15x in the 24-thread fill. The base's per-pair fill here costs 20.0 s at 1 thread for 1.76e10 cells
(P4: 1.14 ns per cell), so the lanes at the quiet-machine 0.17-0.26 ns per cell per pair give 3.0-4.6 s at 1 thread:
4-7x. The tail block (4 of 8 lanes used) wastes 4 x 27 + 4 lanes of 224, 1.8 %. The swap phase and the assignment are not
changed (0.004 s and 0.041 s at base).

| id | quantity | band | predicted |
|---|---|---|---|
| B1 | table-fill phase, base / head, 24 threads, median | >= 2x | PASS, 4-7x (FALSIFIED if < 2x) |
| B2 | whole `one_batch_pam` call, base / head, 24 threads, median (reported, no band) | none | (0.96 s) / (0.04 + 0.04 + fill head) = 3-5x |
| B3 | N x m table, labels, medoids, cost, `distance_evaluations` base vs head | bit-identical | identical: each lane is bitwise the per-pair kernel (P1) |
| B4 | lane loop of the object the new call reaches | packed (`vminpd`/`vminps` on ymm) | packed: `core/dtw_lanes.cpp` compiles `-fno-lto` under clang/MSVC ABI (P1) |

## Change [confirmed]

`dtwc/algorithms/one_batch_pam.cpp` (commit `1b69614`, base `ab5ec7c`, +44 −11 lines). The batch fill takes row i's columns
W at a time (`core::dtw_lanes<T>` = 8 for float64, 16 for float32). A block goes through `core::resolve_dtw_block_fn<T>`
(`core/dtw_lanes.cpp`, compiled `-fno-lto`: the one entry Problem's fill uses, resolved once before the parallel region) when
the function exists (Standard DTW, `MissingStrategy::Error`, univariate) and every series of the block is as long as series i;
otherwise each column goes through `dtw()` as before. The last block repeats its final column in the lanes past the batch and
drops them (P1's ruling; a per-pair tail would cost 4 columns at about 25 times a lane's price); a block that holds series i
computes that self-pair and drops it, so `distance_evaluations` counts what it counted. The loop shape is unchanged: one
writer per row, `run_openmp(fill_row, n, n > 64)`, the block function and the series read-only. No mutex, atomic or critical
(`git grep -nE "omp critical|omp atomic|std::mutex|std::atomic" dtwc/algorithms`: empty). `one_batch_pam.cpp` instantiates no
kernel: `llvm-nm` of its object lists only `U resolve_dtw_block_fn<float>` and `<double>`.

Not shared with Problem's fill: the two loops differ (Problem's asks the matrix which pairs are known and never meets a
diagonal; this one gathers columns through the batch order and meets the diagonal), and E1 works in Problem's fill region.

## Commands

Sources and raw output in `2026-09-30-p5-onebatch-lanes/` (scratch paths inside point at `C:/D/git/wt/tmp/P5-scratch/`): build
the base (`ab5ec7c`) and the head (`1b69614`) library, apply `scratch_instr.py` to `one_batch_pam.cpp` in each, rebuild the
`dtwc++` target, `build_driver.sh bench_base|bench_head`, revert the instrumentation; then `run_ident.sh bench_base|bench_head`
(diff the two outputs; `identity_base.txt` is the base's), `run_band.sh 3 6 24 2000 10 200 0 0 0 -1 -1` and
`run_band.sh 1 3 1 2000 10 200 0 0 0 -1 -1` (`band_24_threads.txt`, `band_1_thread.txt`), `uv run --no-project python
analyse.py <log> <rounds per process>`.

## Numbers [confirmed]

Machine: Intel Core Ultra 9 285 (24 logical CPUs, 8 P + 16 E), Windows 11, clang 21.1.8, Release, ThinLTO, `-O3 -march=native`,
the project FP subset. Workload: `p5_bench 2000 10 200 0 0 0 -1 -1` (N = 2000, k = 10, L = 200, float64, L1, full band,
automatic batch m = 220; random walks with normal steps, `mt19937_64(20260930)`). The base and the head binary are the same
driver linked against the base and the head library, each instrumented by `scratch_instr.py` (stage timer and FNV-1a hash of the
table; never committed). 6 alternating process pairs, 3 calls per process (18 calls each), `OMP_NUM_THREADS=24`. Load: total CPU
30-45 % from other work before and after the run, 75-92 % during it, which includes the benchmark's own 24 threads (`typeperf`).
Medians [lowest-highest process-pair ratio]:

| quantity | base | head | base / head |
|---|---|---|---|
| table fill, 24 threads | 0.929 s | 0.2005 s | **4.61** [4.44-4.81] |
| whole call, 24 threads | 0.976 s | 0.248 s | **3.87** [3.81-4.09] |
| table fill, 1 thread (3 pairs, 1 call each) | 19.51 s | 3.01 s | 6.46 [6.00-6.60] |
| whole call, 1 thread | 20.40 s | 3.93 s | 5.19 [4.95-5.27] |

An earlier 24-thread run of the same comparison, with the first form of the gather loop (the `lanes &&` test ran all W
columns before stopping; it changes the loop over W spans, not a kernel), gave 4.49 [4.43-4.90] for the fill and 3.79
[3.71-4.22] for the call; its last process pair met a load burst (base 1.99 s) and is in that range.

Table cost per pair-cell at one thread: head 3.014 s / (439,780 pairs x 40,000 cells) = 0.171 ns; base 1.11 ns. P1 measured the
packed W = 8 loop at 0.17-0.18 ns per pair-cell and the unpacked one at 0.36. So the loop that ran here is the packed one
[inferred from the match; the assembly below is the direct evidence]. The 24-thread speed-up over one thread is 21.0x at the
base and 15.0x at the head [confirmed]; why the lanes scale less was not investigated (E-cores with 256-bit operations and
memory traffic are candidates, unexamined). The table is no longer 95 % of the call: at 24 threads the call is fill 0.20 s,
the rest 0.05 s (assignment and swap).

| id | result | verdict |
|---|---|---|
| B1 table fill, base / head, 24 threads | 4.61x (band >= 2x; predicted 4-7x) | PASS |
| B2 whole call | 3.87x (predicted 3-5x) | as predicted, no band |
| B3 identity | below | PASS |
| B4 packed loop | below | PASS |

## Identity (B3) [confirmed]

`run_ident.sh` runs 24 configurations through the base and the head driver and compares, line for line, the hash of the N x m
table (`raw`, every bit), the call count, and the hash of labels + medoids + total cost + estimated objective: float64 and
float32; L1 and squared L2; all series one length, one eighth, two eighths, four eighths and every series 3 samples longer;
band full, 6 and 0; batch automatic, 13, 17 and 100 (not multiples of W); N = 37 and 65 to 70 (below and above the 64-row
parallel threshold, rows that are a tail alone); L = 1 and 2; and the benchmark's N = 2000, k = 10, L = 200 for both
precisions. The diff against `identity_base.txt` is empty at `OMP_NUM_THREADS` = 8 and 24, and at 1 thread once the
single-thread warning lines are dropped. The benchmark's hashes: table `a00eefb248cb87ec`, call `db63c95b8a23b53b`, cost
`560855.36730926076`, evaluations 459770, in every timed call of both binaries.

Committed test `unit_test_one_batch_pam`, "the batch table through the lanes is bitwise the per-pair table": the batch is the
whole data set (70 series), so every medoid is a table column; labels, total cost (bits) and the call count N(N-1) are compared
with a serial scan over `Problem::dtw_function()` / `dtw_function_f32()`, for 7 cases (float64 and float32, equal lengths full
and banded, L1 and squared L2, two mixed-length sets) x 4 seeds (the seed is the batch order). It reads N x k table entries, not
the whole table; the whole table is B3's hash. Bite checks (source mutated, test rebuilt, source restored and `cmp`-ed with the
commit): lane results to swapped slots fail 48 of 84 assertions; dropping the equal-length test fails 17 of 73 (the case ends
early); counting the self-pair fails 28 of 84.

## Assembly (B4) [confirmed]

- `llvm-objdump -d -M intel build/bin/CMakeFiles/dtwc++.dir/core/dtw_lanes.cpp.obj` (native COFF; every other object of the
  library is ThinLTO bitcode): `vminpd` 24 (12 in each of the two float64 kernels), `vminps` 16, scalar `vminsd` 32 and
  `vminss` 64 (remainders and the n = 1 path). The float64 L1 loop, unrolled x2, as P1 recorded it: `vbroadcastsd ymm9, [x + 8*i]`,
  `vsubpd ymm10, ymm9, ymm3`, `vandpd` with the abs mask, `vminpd ymm14, ymm12, ymm5`, `vminpd ymm7, ymm7, ymm14`,
  `vaddpd ymm7, ymm10, ymm7`, `vmovupd [dp], ymm7`, then the same on ymm6 for lanes 4-7, and again for i + 1.
- The linked driver (`/map`): `dtw_kernel_lanes<double, L1Dist, StandardCell>` is one function at `0x140003a00`, from
  `dtwc++:dtw_lanes.cpp.obj`; `one_batch_pam` is at `0x140568c70`, from the LTO output. The driver calls nothing but
  `one_batch_pam`, so every entry into the lane code is this path. (Counting the entries with a small Win32 debugger crashed on
  its first run and was dropped; the evidence is the object, the map and the 0.171 ns.)
- `one_batch_pam.cpp` alone (`-S -masm=intel`, the `compile_commands.json` flags without `-flto`): no `vminpd`/`vminps`, no
  call to `dtw_kernel_lanes`, two references to `resolve_dtw_block_fn`, 0 `lock`/`xchg`/`cmpxchg`, as at the base. The row
  body is now `FixedBatchDistances::...::operator()` called from `run_openmp` (at the base it was inlined into it): its calls
  are four `call qword ptr [rax+16]` (the float32 and float64 block call and per-pair call) and the cold throws
  (`throw_wrong_precision` x2, `_Xbad_function_call`, `InvalidInput`); the base had two `[rax+16]` calls and the same throws.

## Gates [confirmed]

- Serial `ctest -j1 -C Release`: base `ab5ec7c` 123 tests, 100 % passed, 3 skipped (`test_cuda_correctness`,
  `test_metal_correctness`, `test_metal_mmap`, all MAY_SKIP); head the same, 123 and the same 3. `cpp_conformance` passes against
  the unchanged `conformance_reference.txt` (it has no OneBatchPAM row, so B3 and the unit test are the identity evidence).
  `unit_test_one_batch_pam`: 12 cases, 10638 assertions (base 11 cases, 10554).
- `check_docs.py --cli build/bin/dtwc_cl.exe` VERDICT=PASS (386 flags), `check_pins.py` 0 failures, `generate_docs.py --check` current.
- Python gate: fresh venv (`C:/D/git/wt/venv/P5`, wheel built from the worktree with `--reinstall`, `CMAKE_GENERATOR` unset),
  `DTWC_CL_PATH` the worktree's `dtwc_cl.exe`: 1191 passed, 19 skipped, 0 failed (6 min 28 s). `tests/python` is untouched by this
  unit, so its collection is the base's; no base wheel was built to recount it. `test_api.py` and `test_sklearn_estimator.py`
  run `method="onebatch"` through the binding.
- TSan, R1's harness (WSL2, clang 18.1.3, LLVM libomp + Archer, `OMP_NUM_THREADS=8`), a clone of `pb/P5` at `1b69614` in
  `~/dtwc-tsan-p5`, `unit_test_one_batch_pam` (12 cases, 10638 assertions): exit 0, 0 `WARNING: ThreadSanitizer`. The loop shape
  did not change (still one writer per row), so this is a confirmation, not a gate. Bite: with `lane` made `static` (one array
  shared by every worker), exit 66 and 152 reports, all in the lane lambda's write to `out` (`core/dtw_lanes.cpp:43`); the clone
  was restored afterwards (`git checkout`, `git diff --stat` empty). The harness's own positive control is R1's
  (`2026-09-30-r1-tsan.md`). Logs in WSL: `~/tsan/logs-p5/unit_test_one_batch_pam-head.log`, `-bite.log`.

## Conclusion

The table fill took the lanes entry that Problem's fill already had; nothing else moved. At N = 2000, k = 10, L = 200 the fill
is 4.6x faster at 24 threads (6.5x at one) and the call 3.9x (5.2x), with every table entry, label, medoid, cost and call count
unchanged over 24 configurations. The lane loop the call reaches is the packed one (`vminpd ymm` in `dtw_lanes.cpp.obj`, 0.171 ns
per pair-cell at one thread). What the call still spends at 24 threads is the fill (0.20 of 0.25 s); the lanes scale worse with
threads (15x over one thread) than the scalar path did (21x), which was not examined. Not done: an MSVC `cl` build of this path
(P1 found `cl` does not pack the lane loop, so there the gain would be smaller; not built or measured), a timing of float32,
squared L2 or a band (only float64 L1, full band, was timed; the others are covered for identity), and a timing on real data of
mixed lengths (a block whose series differ in length falls back to the one-pair path, by construction).
