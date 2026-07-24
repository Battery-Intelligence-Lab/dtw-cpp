# DTWC++ Development TODO

## Volkan's instructions

Now we are doing a huge-refactor and upgrade our DTWC++ library.

I want a very detailed analysis and a plan for Opus 4.8 -effort=xhigh to implement. Please you do yourself do not implement the plan but call it as the subagent to implement. Your output tokens are very valuable. So you need to focus on high-level thinking and guiding other agents. Not implementing things yourself nor bloating your context. I want following things

1) Top-to-down library and interface redesign.

2) Interface is consistent in all languages (C++, MATLAB and Python, like how Casadi is doing)

3) I want device selection like Pytorch so you create the DTWC environment then you set the device somehow. So it is like device=cpu, device=gpu, device=hpc. The cpu and gpu are local, and hpc is the SLURM interface we have. So it should use the credientials to connect HPC in the ".env" file. If they are not there then it should give an error that the connection is not established for the reason (no password -> then tell user how to put their things to .env, or wrong password etc. then tell user, so informative message then close). Maybe we could have some lazy loading so that if it is hpc then it doesn't load the data, or if the data is too big then it uses some mmap or something else. Meticulously decide these important design questions.

4) Automated compilation for executables and mex files, python wheels for all platforms. I think we can consider uploading the pypi when we are hundred percent sure our software is working. So we will release as DTWC++ 2.0.

5) The code is cross-platform, works on all supported platforms (windows, macos (both intel and amd), Linux (ubuntu)).

6) All parallelisation etc. things work out of the box. I don't want it to cannot activate parallelisation due to missing oneTBB etc. then fall back to sequential. Otherwise I would be happier probably for using std algorithms but this was the issue. Maybe we could make user-facing test interface like. dtwc.test.parallelisation() so this tries how many cores and how we can use it. Same for GPU testing. Once these functions are called it can just report back how many gpu what it is using etc.

7) See literature and other abilities we can add.

8) Zero overhead abstraction. So if we have the ability to choose L1 and L2 norms or inject another cost function. These should have nearly zero cost, you could in C++ especially inject things with compile time. And in other languages you could compile multiple options then the main function can select so the selection is not on the hotpath. Or anything that doesn't sacrifice speed. I want this library to be the fastest available DTWC++ library for large data etc. Or maybe multi-dim DTWC++ etc.

9) See my other attempts of writing my own solver, it doesn't work nicely but we could I think can improve. See also my work in UNIMODULAR.md where I believe this problem is almost unimodular, so in MIP programming I believe we could solve this very easily with a much more clever branching rather than leaving the solver to take the branches. And having a large MILP is impossible when you have lots of time series. LP would be more feasible and probably reducing this to a network problem and then solving somehow to global optimality would be amazing. Maybe you could throw another Claude Fable with max effort to investigate the math in the UNIMODULAR.md and maybe come up with a better solver also using my previous attempts. It would be nice to have something nice.

10) Once everything is there, we should update the documentation website.

11) Please think deeply and also remind me if I forgot anything. Like maybe you could write huge CUDA kernels other things or improve some algorithms to make this library EVEN FASTER. You could use some profiler some other thing see cache hit etc. You are free to change data types, how to hold data, how to do things. As long as this library is very fast, accurate, and portable.

## Reconciled campaign record

**Last reconciled:** 2026-07-23. Evidence ledger commit: `512bbc4`.

This file is the live index, not an evidence store. Exact source anchors,
commands, verbatim outputs, stale-subclaim corrections, and the registered
verdict rules are in
`.claude/baselines/2026-07-23-r1-todo-reconciliation.md`.
`PLAN.md` owns ordering and exit gates.

- **CLOSED-BY** names the delivering commit/task and current proof.
- **STILL-OPEN** names its PLAN finding/phase or operator/community owner.
- **NOT-REPRODUCIBLE** retires a historical claim contradicted by a direct
  current-tree probe.
- **MIXED** is used only when every subclaim has an explicit sub-verdict.

### Known bugs and cleanup — 21/21 adjudicated

| ID | State | Reconciled record |
|---|---|---|
| K01 | DONE | **CLOSED-BY `a992183`:** CUDA >2048 wavefronts select the correct three-buffer route; the old direct-8K wording was unsupported. |
| K02 | DONE | **CLOSED-BY `a992183`, `6a53be7`:** Metal pair decode uses shared 64-bit arithmetic and exact correction loops. |
| K03 | DONE | **CLOSED-BY `a992183`, `6a53be7`:** Metal pair counts/work indices were widened; compact active-pair widths remain in R3's general width lens. |
| K04 | DONE | **CLOSED-BY `0c173be`:** untrusted mmap distance-matrix layout arithmetic is overflow-checked. |
| K05 | DONE | **CLOSED-BY `0c173be`:** mmap data-store interior offsets are bounded, monotone, and aligned. |
| K06 | DONE | **CLOSED-BY `bb72bb0`:** MATLAB inputs are validated before `mxGetDoubles`; fresh-MEX gate 61/61. |
| K07 | DONE | **CLOSED-BY `f156474`, `1f4d985`:** HiGHS/Gurobi non-optimal/error states produce typed failures, not assertion-dependent extraction. |
| H01 | DONE | **CLOSED-BY `ae0796b`:** live CPU Soft-DTW dispatch validates gamma and reaches Soft-DTW. |
| H02 | DONE | **CLOSED-BY `ffb7a8d`:** live multivariate L2 dispatch uses `MVL2Dist`; the earlier scalar overclaim is retired. |
| H03 | DONE | **CLOSED-BY `442676a`:** public default data type and CLI default are Float64; stale docs are owned by R1 docs truth. |
| H04 | DONE | **CLOSED-BY `a992183`:** CUDA pair/result indexing is 64-bit at the overflow boundary. |
| H05 | DONE | **CLOSED-BY `a07b63f`:** Arrow IPC/Parquet readers validate type, dimensions, and list bounds; fresh Arrow gate 390/390. |
| H06 | DONE | **CLOSED-BY `8ca7354`:** production FastPAM uses the O(N²)-per-iteration FastPAM1 decomposition. |
| H07 | DONE | **CLOSED-BY `eeca641`, `24ef4e5`, `30411a7`:** resident/chunked FastCLARA share seeded sampling and parallel assignment. |
| H08 | DONE | **CLOSED-BY `abb0fb1`, `f621016`, `5fd3770`:** the real CLI rejects malformed device/metric selections without silent fallback. |
| H09 | DONE | **CLOSED-BY `2b70f34`:** `TimeSeries::view()` preserves `ndim`. |
| H10 | OPEN | **MIXED:** main dependency pins/optionality and Codecov integrity closed by `fbab32a`, `b77a75c`, `3f827c2`; mutable example-project archive → **R3/F11**; quickcpplib upstreaming → **UP01/community**. |
| C01 | DONE | **CLOSED-BY `998c4c7`:** dead/misclassifying integer/zero/one helpers removed. |
| C02 | DONE | **CLOSED-BY `ae0796b`, `ffb7a8d`:** duplicate dead metric dispatch removed; stale prose is R1 docs truth work. |
| C03 | DONE | **CLOSED-BY `998c4c7`:** dead SIMD surface removed; killed idea remains killed and stale prose is R1 docs truth work. |
| C04 | DONE | **CLOSED-BY `a992183`, `4583443`, `1eb8609`, `e5bfd20`, `3061a31`:** decode SSOT, band semantics, medoid scans, CSV emitters, and exact generator/dense-reference duplicates are resolved; purpose-specific variants and independent arbiters remain deliberately separate. |

### Backlog, deferred, operator, and questions — 32/32 adjudicated

| ID | State | Reconciled record |
|---|---|---|
| P01 | DONE | **CLOSED-BY Task 5.11 `8debf1d`:** OpenMP schedule sweep retained the measured adaptive policy. |
| S01 | DONE | **CLOSED-BY `3550bb9`, `84693d4`:** Parquet row requests are grouped so each row group is read once. |
| S02 | DONE | **CLOSED-BY-REJECTED `f74e346`:** the registered boundary study retained the canonical sample-size rule; undocumented sqrt(N) scaling stays killed. |
| S03 | OPEN | Streaming CLARA assignment-state resume → **R5/2.1**. |
| S04 | OPEN | Permanent synthetic Parquet resident-versus-stream parity fixture → **R3/F8**. |
| G01 | OPEN | **MIXED:** capability dispatch/sm_90 build/local Ada run closed by `08f1d6e`, `ecc522c`; real H100 validation → **R5/operator**. |
| G02 | OPEN | Wire the existing CUDA K-vs-all kernel into production streaming CLARA → **R5**. |
| G03 | OPEN | **MIXED:** Float32 device path closed by `08f1d6e`, `54615fb`; 80-GB H100-scale validation → **R5/operator**. |
| G04 | DONE | **NOT-REPRODUCIBLE:** wavefront preload is live for lengths 257–512 or when forced. |
| G05 | OPEN | Multi-stream CUDA pipeline; current production path is one serialized stream → **R5**. |
| B01 | OPEN | First PyPI release/trusted publisher → **R6/operator**; workflow/dry-run preparation is complete. |
| B02 | DONE | **CLOSED-BY `c4ba175`, Task 6.2 `8debf1d`:** MATLAB Phase 2 is delivered; fresh-MEX gate 61/61. |
| B03 | OPEN | **MIXED:** tracked MEX removed by `8debf1d`; explicit `*.mexw64` ignore protection → **R1 tracked-junk audit**. |
| M01 | DONE | **CLOSED-BY-REJECTED `af67486`:** odd-cycle band falsified; the recorded LR/B&B chain remains binding. |
| A01 | OPEN | Two-phase within-/cross-group clustering → **R5/2.1**, only after a quality/complexity band is registered. |
| A02 | OPEN | **MIXED:** device-aware selection closed by `4e59d05`; measured resource/cost model → **R4 then R5**. |
| PL01 | DONE | **CLOSED-BY `0d56fb9`, `5363ac1`, `7f79c55`:** CPU/wheel path executed on Apple M2 Max. |
| PL02 | OPEN | Arrow CPM on Windows+Clang → **2.1/community, BLOCKED-UPSTREAM**; F9's system-Arrow route is distinct. |
| PL03 | OPEN | Arrow CPM on Windows+MSVC → **2.1/community**; currently untested, with no “should work” claim. |
| PL04 | OPEN | Portable/truthful CMake presets (no developer-absolute LLVM path; floor 3.26) → **R3/F16**. |
| D01 | DONE | **CLOSED-BY Task 7.3 `8debf1d`:** conversion guide delivered at `docs/content/guides/data-formats.md`. |
| D02 | DONE | **CLOSED-BY Task 7.3 `8debf1d`:** Mermaid website architecture diagram delivered. |
| D03 | OPEN | Byte-identical, separately referenced docs logos → **R1 tracked-junk audit**; preserve both consumers if deduplicated. |
| DEF01 | OPEN | DDTW recurrence fusion remains an explicit **2.1/R5 non-goal** unless profiling reopens it. |
| DEF02 | DONE | **CLOSED-BY-SUPERSEDED `9511efd`, `9becd53`, `ef978e2`:** full semantic/content cache identity supersedes filename-plus-size. |
| DEF03 | OPEN | **MIXED:** nanoarrow C Data ingestion closed by `3f827c2`; replacing Arrow C++ file readers remains an explicit **2.1 non-goal**. |
| DEF04 | OPEN | HIP backend → **post-2.0/community-owned**. |
| BLK01 | OPEN | Real Oxford ARC submit/poll/download validation → **R6/operator-owned**; local chain is covered, and agents must not submit. |
| Q01 | DONE | **CLOSED-BY `442676a`:** Float64 is the default; explicit Parquet Float32 remains. |
| Q02 | DONE | **CLOSED-BY `ffb7a8d`, `d4a9c6e`:** multivariate L2 is a real norm, not an L1 alias. |
| Q03 | DONE | **CLOSED-BY `fbab32a`, `b77a75c`:** llfio pinned to `b17613fb2149a93b0cc7022c8e649dbf5a015b90`. |
| UP01 | OPEN | quickcpplib generator-forwarding issue → **post-2.0/community-owned**; no upstream issue is claimed yet. |

## Historical delivery summary

These lines describe delivery events, not current-open state. PLAN and the
reconciled tables above are authoritative.

- **2026-06-30** — **PyTorch-style device API + HPC auto-dispatch** (`f933003`, `cceae71`): unified `device()` → `load()` → `cluster()` flow and local SLURM envelope. A real ARC run remains BLK01.
- **2026-06-01** — **Full-repo adversarial audit:** recorded 95 confirmed findings in `.claude/summaries/handoff-2026-06-01-adversarial-audit.md`; subsequent closure state is tracked above, not by that historical snapshot.
- **2026-04-13** — **Python wheel build unblocked** (`7f79c55`): verified macOS arm64 without a system Ninja. This does not close the Windows llfio-ON wheel item.
- **2026-04-13** — **Standalone API fold** (`ed826c3`, `f440bfc`): missing-data and Soft-DTW forward paths delegate to unified kernels; Soft-DTW gradient remains separate.
- **2026-04-13** — Audit hardening (`7f994ee`, `375a23e`, `5a0987c`, `90d4488`): reproducible-build option, NaN cleanup, matrix roundtrip, and mmap lint repair.
- **2026-04-13** — Audit follow-ups (`4c54089`, `d4c066c`, `85f79d5`): MIP Benders tests, `LoadOptions`, and a non-gating `.clang-tidy` configuration.
- **2026-04-13** — **Kernel unification** (`4d92881`, `d595035`, `ee0f798`, `08a2b8d`): templated resolver, AROW/Soft-DTW cell policies, and f32 dispatch repair.
- **2026-04-12** — **Warping-kernel unification** (`eb7b572`, `38fa694`): selected Standard/WDTW banded benchmarks measured 1.54–2.83×; the missing-data fold itself was within measurement noise.
- **2026-04-09 onward** — **RAM-aware chunked CLARA** (`3550bb9`; corrected by `f74e346`, `30411a7`, `84693d4`): row-group streaming, medoid pinning, Float32, and OpenMP assignment; permanent parity remains R3/F8.
- **2026-04-09** — **UCR benchmark suite** (`c370236`): all 128 datasets ran across four recorded architectures; artifacts are in `benchmarks/ucr_benchmark_results.*` and `docs/content/benchmarks/ucr.md`.
- **Earlier** — **Data access/I/O/f32** (`420f764`): spans, zero-copy CLARA sampling, storage policy, Arrow/Parquet readers, conversion CLI, and explicit Float32/Float64; `442676a` later made Float64 the default.
- **Earlier** — **Pruned/banded DTW** (`7551018`): rolling band column, early abandon, ADTW pruned strategy, native tuning, safe math flags, and O(n) envelope. Committed JSON reports `BM_dtwFull_L` +45.5–103.5%; banded cases were −2.2% to +2.3%.
