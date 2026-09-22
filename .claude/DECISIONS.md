# DTWC++ — Decisions

The standing decisions of the project and its append-only decision log. This file is what
`docs/api-contract-2.0.md` means by "a dated PLAN.md decision entry". The digest of everything
decided before 2026-09-21 is in `PLAN-archive-2026-09-21-research-release-campaign.md` (killed ideas
and binding decisions, lines 1113–1215 of that file); the verbatim 2026-07/08 decision archives it
points to were removed on 2026-09-21 and live in git history at `9c08074`.

## 1. Killed ideas — do not reopen without overturning the recorded evidence

- **FastDTW:** verified trap ("much slower than exact DTW" — Wu & Keogh, TKDE 2022).
- **BanditPAM/++:** dominated by FasterPAM on precomputed matrices.
- **Elkan / triangle pruning on DTW:** invalid — DTW is not a metric.
- **ONNX export; R / Julia bindings:** no sensible story / deferred (native competitors saturate).
- **The 2023 custom OSLP solver:** retired (`f7064b3`); the "third solver" is PDLP-as-arbiter, not a revival.
- **SIMD before correctness + measurement:** the old dispatched route ignored bands and variants and
  was removed; 1.29× was an operation-count estimate, while route-specific SIMD timings disagreed. No
  PMU artefact proves a universal memory-bound ceiling. R2-D17 is the only door — see `PLAN.md` R5.
  *2026-09-21 addition:* the compiler itself reports the row recurrence as non-vectorisable
  (loop-carried dependency); see `baselines/2026-09-21-macos-first-baseline.md`.
- **"≥ 25 % fewer full DTW calls on an exact matrix" via lower bounds:** unachievable — an exact matrix
  needs every DTW; LB early-abandon recomputes abandoned pairs (`Pruned` is a pessimisation for exact
  matrices). Exact-matrix work reduction = EAP cell pruning / TADPole pair-skip only.
- **≥ 10× swap speed-up from the FastPAM decomposition on a cached matrix:** falsified — the advisory
  N = 1000 table reports 2.95×–8.06× and is non-monotone. Both variants retain N² lookups per
  iteration; attributing the result to memory bandwidth is **[inferred]** until a counter gate proves it.
- **LB pruning inside PAM / MIP / LR-core:** not admissible — those consumers read the whole matrix.
- **PDLP as the production p-median solver:** falsified — matrix-free Kelley dominates on either device
  (945× CPU / 126× GPU at N = 400); PDLP remains a cross-validation arbiter only.
- **Extending the hand-written CMake URL-pin lexical deny-list:** two registered pivots reached 63/63
  yet audits still bypassed it; F36 must replace the design, not add a third special case.
- **Parallel `find_best_swap`:** measured ~10× slower than the sequential scan.

**Not planned (distinct from killed — reopen with clustering evidence, proposed 2026-09-22, D-15):** ERP,
LCSS, EDR, ShapeDTW, Itakura parallelogram. Holder, Middlehurst & Bagnall (KAIS 2024) find MSM/TWE with
k-medoids best and DTW ≈ Euclidean for clustering; ShapeDTW (Zhao & Itti 2018) is descriptors + DTW_D and
reproducible by preprocessing; fixed parallelograms lose to learned bands (Ratanamahatana & Keogh 2004).
The library already has the two distances the review ranks first; lead with them.

**Libraries considered and not adopted (dependency review 2026-09-22, standing rule 14 — reopen only with a
measurement or a portability failure).** In every case the repository already solves the problem, usually
deliberately:

| Candidate | Licence | Why not |
| --- | --- | --- |
| fast_float | Apache-2.0 / MIT / BSL-1.0 | every loader already uses `std::from_chars` (`fileOperations.hpp:197`, `core/matrix_io.hpp:139`); `std::stod` was removed after an `LC_NUMERIC=de_DE` bug (regression test `unit_test_distance_matrix_csv.cpp:558`) |
| xxHash / any hash library | BSD-2 | `core/sha256.hpp` is dependency-free, NIST-vector tested, chosen *because* `std::hash` is implementation-defined and salted (`sha256.hpp:6`); CRC32 and the avalanche64 row digest are in-tree and on disk |
| PCG / xoshiro | Apache-2.0 / CC0 | `core/portable_random.hpp` already has Lemire bounded, Fisher-Yates, weighted index and selection sampling over `std::mt19937_64`. The residue is legacy `init::` on `std::shuffle` / `uniform_int_distribution` (`initialisation.cpp:140,170,182`) — X-12 / A-21 finish it, no library needed |
| mio (mmap) | MIT | does not cover `barrier` granularity or `try_lock_file`; only relevant as part of D-18, not as an addition |
| magic_enum | MIT | the string↔enum tables *are* the cross-language contract (aliases, kebab keys, stability); compiler-hack reflection cannot express them |
| toml++ / nlohmann-json / glaze | MIT | CLI11 `from_config` + fkYAML already read the config; `Config` needs an alias table and `schema`, not a second parser |
| {fmt} | MIT | C++20 + `std::format`; error text is not a bottleneck |
| kokkos-mdspan | Apache-2.0 WITH LLVM-exception | `tri_index` exists; a packed triangle is not an mdspan layout |
| Highway / xsimd | Apache-2.0 / BSD-3 | hand-written SIMD is killed; parked behind R2-D17 and the X-04 codegen report |
| rapidcheck | BSD-2 | Catch2's `GENERATE` covers invariant tests (symmetry, zero diagonal, band ⊆ full, LB ≤ exact) |
| Taskflow / TBB | MIT / Apache-2.0 | OpenMP + `PairRange` is the scheduling seam; `dtwc/` has no `std::thread` fan-out to unify |
| ankerl::unordered_dense | MIT | only if S-13 proves the MSVC `unordered_map` move-assign hazard is real |
| faiss / hnswlib | MIT / Apache-2.0 | ANN indexes assume a metric; DTW is not one (same reason Elkan is killed) |
| scikit-learn `check_estimator` | BSD-3 | already a dev extra and already run (`tests/python/test_sklearn_estimator.py:76`) |

**Adopted:** libpfm4 (MIT) through Google Benchmark's `BENCHMARK_ENABLE_LIBPFM`, benchmark-only, Linux-only
(X-24, D-17). **Conditional, tied to a row that does not exist yet:** PocketFFT (BSD-3, header-only) *only* if
TS01 k-Shape outgrows the naive O(n²) cross-correlation — it is the licence-clean alternative to FFTW (GPL);
CORE-MATH (MIT) *only* if D-19 chooses to pin exp/log.

**Conventions that stay (v1.0.0 / JOSS surface):** local cost L1, band in integer cells, no final square
root, Σd (not Σd²) medoid objectives with k-medoids++ seeding. Interoperability is handled by additive
tokens and a documented conversion table, never by flipping a default.

## 2. Standing rules (binding)

1. **Lock-free, high-performance by design** (Volkan, 2026-09-02). No lock in a hot path; every
   critical is named, cold, and justified.
2. **No silent fallback** — device, threading, format, solver. A request that cannot be honoured is a
   typed error naming the fix. Guards that must fire without a dependency live outside its `#ifdef`.
3. **Optional dependencies stay optional**, including Gurobi, Metal, MPI, llfio, Arrow, YAML.
4. **Gates assert the subject ran**; CTest scores a skip as a pass unless told otherwise. CLI
   behaviour is proven by driving the real binary.
5. **Register the band before the run.** FALSIFIED is a deliverable. Two attempts, then record.
6. **Numbers live in `.claude/baselines/`**, verbatim, tagged `[confirmed]` or `[inferred]`.
7. **Disagreement → a third computation.** "No-op" needs digit-identical output. Stash and re-run
   before calling anything "pre-existing". Oracles are validated on non-degenerate cases.
8. **Wall-clock is advisory; counters decide.**
9. **One commit per task on green**; bookkeeping rides in separate `docs:` commits.
10. **Evidence effort proportional to consequence.**
11. **FP model:** `-fassociative-math` without `-ffinite-math-only`; NaN means missing / uncomputed;
    no `infinity()` sentinels.
12. **Python `Problem` is single-thread-per-instance**, with the GIL released consistently.
13. **Tier-1 routes are side-effect-free** (no CWD-relative writes). Names are UTF-8 end to end.
14. **Dependencies:** rapidcsv removed; YAML config through CLI11 `from_config` + fkYAML (flags beat
    the file, unknown keys are errors); no further ancillary library without ledger evidence.
15. **Lower bounds:** the public `Webb` name is kept for compatibility but the implementation is
    all-index `LB_Webb_NoLR` plus a separately proved tail cap; only `production ≤ exact-predicate NoLR`
    has a loosening direction; Enhanced dominates matching-direction Keogh at effective `V = 1`, and no
    ordering is claimed at `V ≥ 2` (D3, F54, F55, F57 — closed 2026-08-09).
16. **`.claude/MISSING.md` and `READ.md` were retired by `0449f7c`; do not recreate them.**
17. **Agents never push, tag, publish, SSH out, submit to ARC, rewrite history, or delete an untracked
    `build*/`.** Those are Volkan's actions.
18. **Avoid copyleft** (Volkan, 2026-09-22). We ship BSD-3-Clause; prefer MIT / BSD / Apache-2.0 /
    BSL-1.0 dependencies. Weak copyleft (MPL-2.0, and Eigen is the only one we have) needs a recorded
    reason and its §3.2 notice in every binary artefact; strong copyleft (GPL/LGPL) is never linked —
    which is also why FFTW is not an option if k-Shape ever wants an FFT (PocketFFT, BSD-3, is).
    Prefer a **mature portable library over an in-tree rewrite** (Volkan, 2026-09-22): this is not a
    monolith, and platform corners — mapping growth, durable flush, file locks over network
    filesystems — are what such libraries exist to have already got right (D-18).

## 3. Design 2.0 decisions of 2026-09-07 (Volkan: "Go" — every III.10 recommendation adopted)

C-05 `Auto` = BruteForce for exact matrices, `Pruned` explicit only · C-09 / C-19 delete the unwired MV
lower bounds, wire `MetricType::L2` to a token · C-11 move the foundation headers into `dtwc/base/` ·
C-13 refuse mmap caching for a custom distance function · O-09 drop the 1.x artefact filenames with a
contract addendum · B-14 one `DTWC_` option prefix, old names warned for a release · B-19 keep the 33
shims, collapse the F22 apparatus to one compile probe per compiler family · A-05 keep CLARANS · G-07
defer MPI wiring · T-03 / T-08 / T-09 / T-17 test merges and deletions approved as a class, veto by
name · T-04 / T-12 shrink the slow tests.

The 2026-09-21 multi-agent-sweep handoff calls these "still open". It is wrong; the spec header and
III.10 record them as adopted. Two of them — **O-09** (drop the 1.x artefact filenames) and **C-09**
(delete the unwired surface) — are *adopted but re-opened* by the proposed amendment A6 (`design.md`
§2 compatibility policy): they stand until Volkan answers `PLAN.md` D-2 and D-3.

## 4. Where the old plan and the spec disagree

| # | Old binding text | Spec / today | Resolution |
| --- | --- | --- | --- |
| 1 | simplify only after R3, with zero behaviour change | refactor now, with deliberate behaviour changes (C-05, O-06, O-09, B-14, G-04, A-07) | superseded by the 2026-09-07 approval |
| 2 | "never ask" (Codex rule) | DECIDE items go to Volkan | superseded |
| 3 | **"1.x shims STAY"** | O-09 (adopted 09-07) drops the 1.x artefact filenames | **re-opened by A6 — see `PLAN.md` §7, D-2** |
| 4 | F13: medoid-scan consolidation is R4-owned | A-18: the seven scans stay separate | A-18 (measured rationale) |
| 5 | F22: "do not rerun" | B-19 collapses its apparatus | superseded by B-19 |
| 6 | contract changes need a decision entry | "break other public API freely" | `design.md` §2 (proposed) |
| 7 | floors are recorded in `AGENTS.md` | `AGENTS.md` is gone | `tests/floors.cmake` is the source; the run-log records the measurement |

## 5. Decision log (append-only; newest last)

- **2026-09-21 — PROPOSED, awaiting Volkan.** `design.md` §11 amendments A1–A10: `io` below
  `session` and `Data` in `core`; `CliConfig` promoted to a library `Config`; execution target as a
  `Problem` setting with one resolver; `PairRange` as the unit of work; a codegen report beside
  counters and oracles; the compatibility policy and break register; waves W9 (interface) and W10
  (scale-out); the twelve verified sweep findings as rows S-01…S-12 plus S-13; floors measured on
  every supported platform. None of this is authorised until approved.
- **2026-09-22 — PROPOSED, awaiting Volkan.** Four-lens review (Python API / C++ language / C++ software
  design / time-series science) → `design.md` amendments A11–A18, rows S-14…S-20 and X-13…X-22, decisions
  D-12…D-16. Every finding that entered the plan was re-opened at its cited line by the orchestrating
  session; rejected or corrected on the way: "Python is entirely 2.0-only" (v1.0.0 shipped a pybind
  `Problem` wrapper with 1.x names — those keep shims), "MSM/TWE not exported" (reachable through Tier-2
  `variant_params`; missing only from the pairwise surface), "`dtwc.test.parallelisation()` missing"
  (it exists).
- **2026-09-22 — PROPOSED, awaiting Volkan.** Dependency review (Volkan's question: is a lightweight,
  well-licensed library missing?). Answer: essentially no — §1's candidate table records thirteen rejections
  with the in-tree facility that already covers each. Outcome: rows X-23 (third-party notices) and X-24
  (libpfm4 counters), decisions D-17 / D-18 / D-19. Two agent claims were re-opened and **corrected**: Eigen
  5.0.1 has no LGPL modules and upstream removed `EIGEN_MPL2_ONLY` (the ILUT note at
  `IncompleteLUT.h:88-96` records Saad's relicensing to MPL2), so there is nothing to fence — the real Eigen
  obligation is MPL-2.0 §3.2 in the wheel; and `third_party.txt` *is* built for CLI release archives
  (`release-artifacts.yml:34`), it is only the wheel job that omits it.
- **2026-09-22 — done.** X-23 (third-party notices) implemented, closing ledger row B-16, and it
  surfaced a release blocker recorded as X-29: the macOS CLI archive could not start on any machine
  but its builder (no `LC_RPATH`, absolute libomp path), because the block meant to fix that guarded
  on a variable only the Windows branch ever sets. Evidence, reproduction and the post-fix run are in
  `.claude/baselines/2026-09-22-x29-release-archive-not-portable.md`. Three notes worth keeping:
  nanoarrow's upstream `LICENSE.txt` has an appended flatcc section, so a generic Apache-2.0 text
  would have been the wrong file; `CPACK_COMPONENTS_ALL runtime` is **load-bearing while inert** —
  enabling `CPACK_ARCHIVE_COMPONENT_INSTALL` would drop `libhighs.*` from the archive, because HiGHS
  installs into its own unnamed component; and `scripts/check_docs_contract.py` is **red on HEAD**
  (`D2 CTest drift: expected one test_lb_keogh_derivation policy block`), reproduced in a clean
  worktree at `ddbc7b6` — `d21ffee` moved test registration to `dtwc_add_test(...)` and the checker
  still greps for the old `if(TARGET …)` block. Unowned; needs a row.
- **2026-09-22 — done.** X-24's W0 half: `DTWC_BENCHMARK_PMU` → `BENCHMARK_ENABLE_LIBPFM` (libpfm4,
  MIT, benchmark-only, never redistributed — D-17). The row was filed as plumbing; the finding is
  that google/benchmark **has no working runtime guard** for a counters request it cannot serve. A
  build without libpfm4 accepts `--benchmark_perf_counters`, prints one stderr line, and writes a
  complete JSON with no counter fields at exit 0 — confirmed by running it here. Its own
  `BM_CHECK` (`benchmark_runner.cc:323`, v1.9.5) tests the inverse of its message and lives in a
  branch ordinary benchmarks never enter. `scripts/run_bench.sh` now verifies the counters are
  actually in the JSON and renames a counter-less file to `*.no-counters.json` at exit 65, which is
  the part that keeps rule 3 (no silent fallback) true for the O-22 artefact. Evidence, including
  the guard-discrimination pair, in `.claude/baselines/2026-09-22-x24-benchmark-pmu-counters.md`.
  The success path is unproven anywhere here and is queued as V-5, together with a probe that would
  settle the upstream inversion by execution rather than by reading.
- **2026-09-21 — done (housekeeping, no product change).** Root `PLAN.md` archived into `.claude/`;
  new `.claude/PLAN.md`, `MAP.md`, `CHARTER.md`, this file; 105 superseded records, four unloadable
  skill files, twelve one-off evidence scripts, generated plots and unused figures removed (all
  recoverable at `9c08074`); two gate scripts repointed, assertions unchanged.
