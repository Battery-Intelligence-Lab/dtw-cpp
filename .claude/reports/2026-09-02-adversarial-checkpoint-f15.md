# Adversarial review — mid-fill checkpoint + F15 portable generator (2026-09-02)

Gate: `build/highs-1151`. Read-only on source. `ctest -R "checkpoint|problem|distance_matrix"` → **15/15 passed, 0 failed** (matches the implementer's count). `unit_test_checkpoint` 210 assertions / 14 cases; `unit_test_deterministic_series` 177 / 6 — both exactly as reported.

## Change 1 — mid-fill interval checkpoint

**C1-1 [MEDIUM, PLAUSIBLE] Autosave hard-codes `MetricType::L1`; every other save/load site takes the caller's metric.**
`Problem.cpp:846` and `Problem.cpp:1106` call `save_checkpoint(*this, checkpoint.directory, core::MetricType::L1)`, but `dtwc_cl.cpp:1687` and `:1924` use `cache_metric`. `Problem.cpp:583-585` states the metric is in the identity precisely so "a later L1 run [does not] accept the wrong matrix". `CheckpointOptions` has no metric field, so a SquaredL2 user cannot make them agree. Reachable only with `--device cuda --metric squared_euclidean` (CPU rejects non-L1 at `dtwc_cl.cpp:382-384`). Two consequences: (a) mid-fill generations are rejected on resume → silent full recompute; (b) if the process dies between the post-switch autosave and the CLI end save, `CURRENT` names an L1-tagged generation holding SquaredL2 values, which a later `--metric l1` run accepts. Not reproduced — no CUDA in this build. Fix: plumb the metric into `CheckpointOptions`, or assert `L1` when enabling.

**C1-2 [MEDIUM, CONFIRMED] O(N²) per save, unbounded generation growth, undocumented.**
Each `save_checkpoint` writes the whole N×N CSV plus three full packed scans (`checkpoint.cpp:469,475-481,527-545`); nothing prunes (`checkpoint.cpp:411` removes only on failure). Total ≈ O(N³/interval) time *and bytes*. Reproduced: `dtwc_cl -i data/dummy -k 2 --checkpoint <d> --checkpoint-interval 5` → 6 generations; a second run → **7** and growing; 56 KB at N=25. At the default `save_interval=100`, N=10 000 is ~100 generations × ~2 GB. Neither `checkpoint.hpp`, `docs/api-contract-2.0.md` §2.7 nor `docs/content/getting-started/checkpointing.md` mentions cost or retention. Judgement: document both; the default interval is otherwise sane.

**C1-3 [LOW, CONFIRMED] Disabled-path change beyond the stated fix.** `io::read_csv` treats an empty CSV field as uncomputed (`core/matrix_io.hpp:141-142`) and `read_distance_matrix` performs **no** identity check. Previously a following fill wiped and recomputed; now a partial, unfingerprinted CSV is trusted. Correct for checkpoints, weaker for `--dist-matrix`. No test pins either semantics.

**C1-4 [LOW, CONFIRMED] CLI validation is late.** `--checkpoint-interval` without `--checkpoint` exits 1 with a clear message, but only *after* the data load. `--checkpoint-interval 0` or negative is rejected at fill time (`InvalidInput`), not at parse.

**C1-5 [LOW] Test gaps.** No coverage for the empty-`directory` `InvalidInput` branch; none for the new header claim that a throwing save leaves the previous generation valid; none for the GPU post-switch single save; none for the CLI flag; none pinning C1-3.

**Refuted.** (a) No stale-matrix path: every semantic setter (`Problem.hpp:433,497,514`, `Problem.cpp:330,339`) and every drift detector routes through `refresh_distance_matrix()` → `resize(0)` (`Problem.cpp:255-258`); `Problem.cpp:704` and `:872-877` were already size-guarded, so the wiped `resize(N)` in BruteForce was pure waste. Fix verified end-to-end: same CLI run 6.94 s → **0.19 s** on resume. (b) Blocks disjoint and complete — measured `pairs_computed` = 135, 220, 280, 315, 325 (exact prefix sums of 25×26/2), ceil(25/5)=5 + 1 CLI end save = 6 generations; per-*row* cost is one integer add, no mutex/atomic/alloc. (e) No double-save: the Pruned→BruteForce downgrade (`Problem.cpp:964-972`) precedes the switch, so `effective != BruteForce` cannot fire after a BruteForce fill; all of it is gated on `enabled`. (h) `run_openmp` uses `#pragma omp parallel for` with no `nowait` (`parallelisation.hpp:124-145`) — implicit join before the save; OpenMP-OFF is a serial loop. (d) A throwing save leaves a partially filled matrix and `is_distance_matrix_filled()` = `size>0 && all_computed()` (`Problem.hpp:578-585`) → false.

**Verdict: fix-first** — C1-1 (metric) then merge; C1-2 is a docs edit.

## Change 2 — portable F15 generator

Independently recomputed the schedule with no project headers (`clang++ -std=c++20 -O0 -ffp-contract=off`): **all 29 registered bit patterns match exactly** (`kScalarBits`, `kRowsBits`, `kAcceleratorBits`). Endpoints exact: min = −1.0 / −10.0 exactly, max = 0.99999999999999978 / 9.9999999999999982, both strictly inside. `|k| ≤ 2^52` ⇒ `int64→double` exact; `×0x1p-52` exact; `×0x1.4p-49` (= 10·2⁻⁵², exactly representable) is one correctly-rounded multiply. Uniform over 2^53 points and injective (10·2⁻⁵² = 2.22e-15 > ulp(10) = 1.78e-15).

**C2-1 [LOW, PLAUSIBLE]** The argument omits `FLT_EVAL_METHOD==2` (x87). It is safe — `10k` needs ≤56 significant bits ≤ 64, so long-double intermediate rounding is exact and the double rounding is harmless — but that holds by luck, not by the stated reasoning; add it to the header comment. Likewise the claim "regardless of FP flags" should read "under round-to-nearest": `fesetround` would move the `kDecaScale` results (the `kUnitScale` path is immune).

**C2-2 [LOW, CONFIRMED]** The report's reproduction command points at `<scratchpad>/f15_probe.cpp`, absent from the repo, so a reader cannot rerun it. (Catch2 prints the actual bits on mismatch, so this is a doc nit.)

**C2-3 [LOW, CONFIRMED]** The source guard matches `uniform_real_distribution<`; a CTAD reintroduction (`std::uniform_real_distribution dist(-1.0,1.0)`) or `generate_canonical` evades it. The 29 bit `CHECK`s still catch both, so not blocking.

**Refuted.** Consumers verified: `unit_test_mpi.cpp:45-180` asserts only symmetry, zero diagonal, positivity and agreement with a serial recompute (1e-10/1e-12); `test_cuda_correctness.cpp:45`, `test_cuda_lb_keogh.cpp:48`, `test_metal_correctness.cpp:46` bind `&accelerator_series_set` and compare against a runtime CPU reference; the five `benchmarks/bench_*.cpp` only time. No hard-coded number derived from the old data. (CMake floor ≥177/≥6 equals the measured value exactly — zero headroom.)

**Verdict: merge.**
