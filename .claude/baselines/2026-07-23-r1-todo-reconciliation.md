# R1 TODO reconciliation — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `83a2048` (`docs: close R0 adjudication`)
- Environment: Windows, PowerShell; current tree clean at registration.
- Source record: `.claude/TODO.md`, last reconciled commit before R0:
  `874edd5` (2026-07-06).

The inventory parser found exactly 53 live records:

```text
unchecked=49
checked=1
open_questions=3
```

Section totals:

```text
7  Critical
10 High
4  Dead code / cleanup
1  Performance
4  Streaming CLARA
5  CUDA
3  Bindings
1  MIP Solver
2  Algorithms & Scale
4  Platform
3  Documentation
4  Deferred
1  Blocked
3  Open questions
1  Needs a PR / upstream nudge
TOTAL=53
```

The first 21 records are the known-bug/cleanup audit (including the already
closed FastPAM record). The remaining 32 are backlog, deferred, blocked,
question, and upstream-work records. R1 will reconcile all 53, not only the
approximately 30 anticipated by PLAN.

## Preregistered verdict rules

Every parent record receives exactly one durable verdict:

- **CLOSED-BY** — a named commit/task plus current-tree or executable evidence
  confirms the claim was fixed or delivered. A checked box without that pair
  fails the gate.
- **STILL-OPEN** — current semantic source anchors plus an executable probe when
  feasible confirm the work remains. A defect/cleanup record must map to a
  unique numbered R3 finding; a product/performance/backlog record must name its
  owning campaign phase or explicit operator/community owner.
- **NOT-REPRODUCIBLE** — a direct current-tree probe contradicts the historical
  claim. The output and tested boundary are quoted; absence by code inspection
  alone is insufficient when the real binary or installed runtime can decide.

Composite historical bullets may split into subclaims with different verdicts,
but the parent row must name every sub-verdict. Operator-only actions are
recorded as **STILL-OPEN / OPERATOR-OWNED**, never treated as local failures.
Environment-impossible runtime checks use `[BLOCKED-ENV]` with verbatim probe
output and retain the truthful open status.

## Acceptance band

- All 53 inventory records appear in the final ledger and rewritten TODO.
- The 21 known-bug/cleanup records have one of the three registered verdicts;
  every STILL-OPEN defect has a unique R3 finding and current source/probe
  evidence.
- The 32 remaining records have a verdict plus a campaign/operator/community
  owner when open.
- Stale file:line anchors and stale counts are removed or replaced by semantic
  anchors verified at this base.
- No program-behavior file changes in R1.
- A final parser reports 53/53 adjudicated, zero unclassified historical
  records, and zero duplicate R3 IDs.

## Evidence ledger

Append one row per inventory record before rewriting `.claude/TODO.md`.
Exploratory checks do not count as closure evidence until their command/output
or named source artifact is recorded here.

## Focused current-tree closure gate

Registered before execution:

- Directly run `test_decode_pair`, `unit_test_mmap_data_store`,
  `unit_test_mmap_distance_matrix`, `unit_test_dtw_api`,
  `unit_test_time_series`, `unit_test_fast_clara`, `unit_test_cli_args`,
  `unit_test_mip`, and `test_supply_chain_pinning` from the rebuilt canonical
  directory.
- Every binary must exit zero, report at least one executed assertion and one
  executed case, and contain no skip marker. CTest green alone is insufficient.
- The Arrow reader closure reuses the fresher F9 enabled-build evidence:
  390/390 assertions in all 11 cases with no skip. The MATLAB input-validation
  closure reuses the latest fresh-MEX 61/61 artifact because R1 changes no
  program files.
- Any current focused failure downgrades the corresponding historical item to
  STILL-OPEN regardless of its closing commit.

### Focused gate result

Build:

```text
[0/2] Re-checking globbed directories...
ninja: no work to do.
```

Direct Catch2 summaries (the binaries emitted no skip marker):

```text
[test_decode_pair]
All tests passed (444 assertions in 7 test cases)

[unit_test_mmap_data_store]
All tests passed (2752 assertions in 6 test cases)

[unit_test_mmap_distance_matrix]
All tests passed (1343 assertions in 37 test cases)

[unit_test_dtw_api]
All tests passed (26 assertions in 14 test cases)

[unit_test_time_series]
All tests passed (39 assertions in 12 test cases)

[unit_test_fast_clara]
All tests passed (842 assertions in 21 test cases)

[unit_test_cli_args]
All tests passed (149 assertions in 25 test cases)

[unit_test_mip]
All tests passed (194 assertions in 17 test cases)

[test_supply_chain_pinning]
All tests passed (20 assertions in 3 test cases)
```

**Verdict: PASS.** All nine binaries executed assertions and cases, exited
zero, and emitted no skip marker.

## Adjudicated evidence ledger

The identifiers below are permanent reconciliation keys for the 53 source
records. `K`, `H`, and `C` cover the 21 known-bug/cleanup records; the remaining
32 retain their source-section grouping. A composite row is `MIXED` only when
every subclaim has an explicit sub-verdict.

### Known bugs and cleanup (21/21)

| ID | Verdict | Evidence and current owner |
|---|---|---|
| K01 | **CLOSED-BY** `a992183` | [confirmed] CUDA selects the two-buffer route only for `max_L <= 2048` and the three-buffer route above it (`dtwc/cuda/cuda_dtw.cu:145-147,1306-1317`). The committed RTX run is `.claude/baselines/2026-07-07-cuda-first-runtime-verification.md`; its 4096-length case logged the three-buffer route. The historical “8K run” wording was false: 8K is covered by the route predicate, not by a direct 8K allocation. |
| K02 | **CLOSED-BY** `a992183`, hardened by `6a53be7` | [confirmed] Metal consumes the shared 64-bit decode source from `dtwc/detail/decode_pair.hpp:88-116`; exact correction loops make the FP32 square-root seed non-decisive. Current `test_decode_pair`: 444 assertions in 7 cases. Metal device runtime remains unexecuted on this Windows host. |
| K03 | **CLOSED-BY** `a992183`, completed by `6a53be7` | [confirmed] Metal pair counts, work indices, and decoded indices use MSL `long` (`dtwc/metal/metal_dtw.mm:93-103,212-219,334-340,760-773`); the current N=70,000 host regression exceeds the int32 pair-count boundary. Separate compact active-pair buffers remain under R3's integer-width lens, not this historical defect. |
| K04 | **CLOSED-BY** `0c173be` | [confirmed] `MmapDistanceMatrix::checked_layout` checks every multiplication and addition before accepting an untrusted header (`dtwc/core/mmap_distance_matrix.hpp:131-172`). Current gate: 1,343 assertions in 37 cases, including crafted `n=2^62`. |
| K05 | **CLOSED-BY** `0c173be` | [confirmed] `MmapDataStore::open` validates table-size arithmetic, the first offset, monotonicity, alignment, and every interior offset (`dtwc/core/mmap_data_store.hpp:236-266`). Current gate: 2,752 assertions in 6 cases, including a corrupt interior offset. |
| K06 | **CLOSED-BY** `bb72bb0` | [confirmed] `require_real_double` validates MATLAB class, complexity, sparsity, emptiness, and dimensionality before `mxGetDoubles` (`bindings/matlab/dtwc_mex.cpp:105-118`). Fresh-MEX evidence is 25 validation cases within the 61/61 floor in `.claude/baselines/2026-07-10-phase8-remediation.md` and `2026-07-10-phase67-floors.md`. |
| K07 | **CLOSED-BY** `f156474`, typed-error migration `1f4d985` | [confirmed] HiGHS checks model handoff, run result, and model status; Gurobi checks solver status and rethrows (`dtwc/mip/mip_Highs.cpp:160-198`, `dtwc/mip/mip_Gurobi.cpp:114-148`). Current HiGHS gate: 194 assertions in 17 cases, including 2 assertions in the focused infeasible case. Gurobi is source-verified, not runtime-verified in this audit. |
| H01 | **CLOSED-BY** `ae0796b` | [confirmed] the live CPU route validates parameters and directly dispatches Soft-DTW (`dtwc/core/dtw.cpp:29-33,75-86`). Current focused probe: 4 assertions in 2 cases; full API gate: 26 assertions in 14 cases. |
| H02 | **CLOSED-BY** `ffb7a8d` | [confirmed] the sole live multivariate dispatcher maps L2 to `MVL2Dist` (`dtwc/warping.hpp:246-293`); the public-route regression returns 20 rather than L1's 28 (6 assertions). The earlier `ae0796b` change touched a dead duplicate and did not close this defect. Scalar L1/L2 equality was a stale overstatement because `sqrt((a-b)^2) == abs(a-b)` for one channel. |
| H03 | **CLOSED-BY** `442676a` | [confirmed] `default_data_t` is `double` (`dtwc/settings.hpp:30`) and the real CLI reports `float64 (default, full precision)`. The pre/post f64 fingerprints and 240-assertion gates are in the Phase-1 archive. Stale “float today” prose is assigned to the R1 docs-truth audit. |
| H04 | **CLOSED-BY** `a992183` | [confirmed] CUDA decoded indices are `std::int64_t`, the matrix-index expression has a production `static_assert` (`dtwc/cuda/cuda_dtw.cu:192-197`), and shared row arithmetic is 64-bit (`dtwc/detail/decode_pair.hpp:50-70`). The 444-assertion suite includes N=46,342. |
| H05 | **CLOSED-BY** `a07b63f` | [confirmed] Arrow IPC validates value type, offsets, and nonzero dimensions; Parquet handles Float64/Float32 explicitly and validates list ranges (`dtwc/io/arrow_ipc_reader.hpp:97-162`, `dtwc/io/parquet_reader.hpp:87-155`). Fresh Arrow-ON gate: 390 assertions in all 11 cases with no skip (`.claude/baselines/2026-07-23-f9-arrow-gate.md`). |
| H06 | **CLOSED-BY** `8ca7354` | [confirmed] production `fast_pam` and `fast_pam_seeded` select FastPAM1's O(N²)-per-iteration decomposition; the O(N²k) direct-sum path is an explicit reference/benchmark variant. Registered objective agreement is 1e-9 (`.claude/baselines/2026-07-08-faster-pam-bench.md`). |
| H07 | **CLOSED-BY** `eeca641`, hardened by `24ef4e5` and `30411a7` | [confirmed] resident and chunked FastCLARA share `mt19937_64` plus `portable_sample_indices`; resident assignment is OpenMP static and chunked assignment is OpenMP dynamic (`dtwc/algorithms/fast_clara.cpp:160-189,248-339,362-371,515-522`). Current focused gate: 419 assertions in 2 cases. |
| H08 | **CLOSED-BY** `abb0fb1`, hardened by `f621016` and `5fd3770` | [confirmed] the real CLI rejects malformed/unknown devices, unsupported CPU metrics, and unavailable CUDA without silent fallback. The four probes exited 1 with actionable diagnostics; parser/validation anchors are `dtwc/dtwc_cl.cpp:273-376,1001-1014,1425-1430,1497-1510`. |
| H09 | **CLOSED-BY** `2b70f34` | [confirmed] both lvalue conversion and `view()` preserve `{data, timesteps, ndim}` (`dtwc/core/time_series.hpp:68-70`). Current gate: 39 assertions in 12 cases. |
| H10 | **MIXED**: historical main-tree pin defects **CLOSED-BY** `fbab32a`/`b77a75c`/`3f827c2`; repo-wide example integrity **STILL-OPEN → R3/F11**; quickcpplib upstreaming **STILL-OPEN / COMMUNITY-OWNED → UP01** | [confirmed] llfio and quickcpplib use exact SHAs, llfio is optional, main CPM archives have hashes, and Codecov is action-SHA pinned (`cmake/Dependencies.cmake:25,46,67,103,131,159-171,211-233,367`; `.github/workflows/documentation.yml:111`). `examples/cpp/example_project/CMakeLists.txt:12` still downloads mutable `refs/heads/documentation_update.zip` without `URL_HASH`; the current checker does not scan it. The exact-SHA quickcpplib configure workaround remains until upstream resolves UP01; configure-time execution alone is not adjudicated as an unfixed security defect. |
| C01 | **CLOSED-BY** `998c4c7` | [confirmed] the dead/misclassifying `is_integer`, `is_zero`, and `is_one` helpers have no definitions or C++ callers. |
| C02 | **CLOSED-BY** `ae0796b`, completed by `ffb7a8d` | [confirmed] only the live `dtwc::detail` dispatchers remain in `dtwc/warping.hpp:265-293`; `dtwc/core/dtw_cost.hpp:73-80` records the single-home rule. Stale CHANGELOG/warping comments are R1 docs-truth corrections, not an open code defect. |
| C03 | **CLOSED-BY** `998c4c7` | [confirmed] the unbuildable SIMD option/reference surface is gone and PLAN keeps the idea killed. Stale comments in `cmake/StandardProjectSettings.cmake` and CHANGELOG are R1 docs-truth corrections; this row must not reopen SIMD. |
| C04 | **MIXED**: decode SSOT **CLOSED-BY** `a992183`; band bounds **STILL-OPEN → R3/F12**; nearest-medoid scans **STILL-OPEN → R3/F13**; CSV emitters **STILL-OPEN → R3/F14**; benchmark/test generators and CPU oracles **STILL-OPEN → R3/F15** | [confirmed] decode has one generated source in `dtwc/detail/decode_pair.hpp`. CPU, CUDA, and Metal retain distinct band expressions; production retains separate nearest-medoid scans in FastPAM/CLARANS/FastCLARA; four CSV emitters retain separate precision loops; five benchmark-local and three test-local random/oracle implementations remain. The historical “four forms”, “4×”, and “byte-identical across eight files” counts are stale; the eight generators use different ranges/shapes. R3 must pin behavior before any R4 consolidation. |

### Backlog, deferred, operator, and question records (32/32)

| ID | Verdict | Evidence and current owner |
|---|---|---|
| P01 | **CLOSED-BY** Task 5.11 `8debf1d` | [confirmed] `.claude/baselines/2026-07-10-openmp-profile.md` records `dynamic,16` about 7× worse and `dynamic,1`/guided tied; the adaptive policy was retained. Artifact relocation/repair: `a3ebf2c`. |
| S01 | **CLOSED-BY** `3550bb9`, hardened by `84693d4` | [confirmed] `read_rows_impl` sorts requested rows, groups them by Parquet row group, and reads each group once. |
| S02 | **CLOSED-BY-REJECTED** `f74e346` | [confirmed] the boundary study in `.claude/baselines/2026-07-12-fast-clara-boundaries.md` retained `max(40+2k, min(N, 10k+100))`; undocumented `sqrt(N)` scaling is a killed divergence and must not be silently reintroduced. |
| S03 | **STILL-OPEN → R5/2.1** | [confirmed] dense/final checkpoints exist, while streaming assignment state is rejected rather than resumable. This is a product feature, not a current correctness defect. |
| S04 | **STILL-OPEN → R3/F8** | [confirmed] a one-off 8-series/four-row-group parity run exists, but no permanent resident-versus-stream synthetic Parquet fixture pins the route. |
| G01 | **MIXED**: capability dispatch **CLOSED-BY** `08f1d6e`; local Ada runtime **CLOSED-BY** `ecc522c`; H100 validation **STILL-OPEN / OPERATOR-OWNED → R5** | [confirmed] sm_90 fat-binary support exists and local RTX Ada execution is recorded. No real H100 run is claimed. |
| G02 | **STILL-OPEN → R5** | [confirmed] `compute_dtw_k_vs_all` is public and benchmark/test exercised (`2aedaac`) but has no production algorithm caller; streaming CLARA does not use it. A checked statement in the immutable plan archive is stale historical prose. |
| G03 | **MIXED**: Float32 device path **CLOSED-BY** `08f1d6e`, hardened by `54615fb`; 80-GB H100-scale validation **STILL-OPEN / OPERATOR-OWNED → R5** | [confirmed] the f32 path exists. No 80-GB HBM3-scale device run is claimed. |
| G04 | **NOT-REPRODUCIBLE** | [confirmed] the purported dead preload branch is reachable for wavefront lengths 257–512 or when forced; it is not dead code. |
| G05 | **STILL-OPEN → R5** | [confirmed] `3ad88d4` provides one stream plus pinned memory; production calls remain serialized and do not pipeline multiple streams. |
| B01 | **STILL-OPEN / OPERATOR-OWNED → R6** | [confirmed] workflow and dry-run work closed in Tasks 6.1/6.6 (`8debf1d`), but trusted-publisher configuration and the first PyPI release are operator actions. |
| B02 | **CLOSED-BY** `c4ba175` and Task 6.2 `8debf1d` | [confirmed] MATLAB Phase 2 API coverage is delivered; the current fresh-MEX floor is 61/61. |
| B03 | **MIXED**: tracked MEX removal **CLOSED-BY** `8debf1d`; ignore hygiene **STILL-OPEN → R1 tracked-junk audit** | [confirmed] no `*.mexw64` binary is tracked or present, but no explicit ignore rule prevents its reintroduction. |
| M01 | **CLOSED-BY-REJECTED** `af67486` | [confirmed] the registered odd-cycle band was falsified; follow-on LR/B&B evidence is `fd49ff4`, `8375aef`, and `789cc52`. PLAN kills reopening without overturning that evidence. |
| A01 | **STILL-OPEN → R5/2.1** | [confirmed] no production within-group/cross-group clustering route exists. It requires a preregistered quality/complexity band before implementation. |
| A02 | **MIXED**: device-aware selection **CLOSED-BY** `4e59d05`; resource/cost model **STILL-OPEN → R4 then R5** | [confirmed] current auto-selection remains a fixed threshold rather than a measured resource/cost model. R4 first establishes one selection-policy home; R5 may tune only against registered counters/quality. |
| PL01 | **CLOSED-BY** `0d56fb9`, hardened by `5363ac1` and `7f79c55` | [confirmed] the CPU path and wheel ran on a real Apple M2 Max. The artifact proves M2 Max, not an otherwise unverified “Mac Studio” chassis claim. |
| PL02 | **STILL-OPEN / BLOCKED-UPSTREAM → 2.1/community** | [confirmed] the Windows+Clang Arrow CPM path remains blocked by upstream ExternalProject flag quoting. F9's system-Arrow build does not validate this separate route. |
| PL03 | **STILL-OPEN → 2.1/community** | [confirmed] Windows+MSVC Arrow CPM remains untested; “should work” was unsupported prediction and is removed. |
| PL04 | **STILL-OPEN → R3/F16** | [confirmed] `CMakePresets.json` still hardcodes a Windows LLVM path and declares CMake 3.21 while root `CMakeLists.txt` requires 3.26. The historical pyproject mismatch subclaim is stale: `pyproject.toml` already requires 3.26. |
| D01 | **CLOSED-BY** Task 7.3 `8debf1d` | [confirmed] the conversion guide exists as `docs/content/guides/data-formats.md`; the historical requested filename is stale. |
| D02 | **CLOSED-BY** Task 7.3 `8debf1d` | [confirmed] the website architecture page contains the Mermaid diagram. |
| D03 | **STILL-OPEN → R1 tracked-junk audit** | [confirmed] `docs/docs_logo.png` and `docs/static/docs_logo.png` share SHA-256 `4CE78596704E8A5F3EE4F46C8FAF9C5B754488158A29002CEAFCB2FD210BA051`; both paths are referenced, so any deduplication must preserve consumers. |
| DEF01 | **STILL-OPEN / EXPLICIT NON-GOAL → 2.1/R5** | [confirmed] DDTW derivative fusion is absent. Reopen only with profile evidence that the derivative pass is material. |
| DEF02 | **CLOSED-BY-SUPERSEDED** `9511efd`, with mmap-v3 integrity `9becd53`/`ef978e2` | [confirmed] cache identity now hashes full semantic/content identity rather than the weaker proposed filename-plus-size scheme. |
| DEF03 | **MIXED**: nanoarrow C Data ingestion **CLOSED-BY** `3f827c2`; replacing Arrow C++ file readers **STILL-OPEN / EXPLICIT NON-GOAL → 2.1** | [confirmed] the C Data interface exists, but Parquet/IPC file readers still use Arrow C++. |
| DEF04 | **STILL-OPEN / COMMUNITY-OWNED → post-2.0** | [confirmed] no HIP backend exists; the recorded policy is community contributions only. |
| BLK01 | **STILL-OPEN / OPERATOR-OWNED → R6** | [confirmed] local submit/poll/download/name-mapping work is covered by `f933003`, `cceae71`, `d3bdae6`, `bfee0eb`, and `daf2e36`; no real Oxford ARC run exists. AGENTS forbids submitting one from this session. |
| Q01 | **CLOSED-BY** binding decision and `442676a` | [confirmed] `default_data_t = double`; explicit Parquet Float32 remains available. |
| Q02 | **CLOSED-BY** binding decision and `ffb7a8d` | [confirmed] multivariate L2 is a real norm and the live dispatcher uses `MVL2Dist`; missing-data L2 is covered by `d4a9c6e`. |
| Q03 | **CLOSED-BY** `fbab32a`, strengthened by `b77a75c` | [confirmed] llfio is pinned to exact SHA `b17613fb2149a93b0cc7022c8e649dbf5a015b90`. |
| UP01 | **STILL-OPEN / COMMUNITY-OWNED → post-2.0** | [confirmed] no upstream issue URL/artifact exists. The local exact-SHA, sentinel-guarded quickcpplib workaround remains; filing/upstream coordination is an operator/community action. |

## Newly routed R3 findings

These identifiers are unique within the campaign and are the only new R3
routes created by this reconciliation:

| Finding | Registered subject and first decisive gate |
|---|---|
| F11 | Example-project dependency integrity: fail a repo-wide checker on every mutable/unhashed remote archive, then pin or hash `examples/cpp/example_project/CMakeLists.txt` without weakening the current main-dependency checks. |
| F12 | Cross-backend band semantics: use unequal lengths and a non-degenerate forcing case to compare CPU, CUDA, and real Metal before declaring the differing formulas equivalent. Windows source inspection alone cannot close it. |
| F13 | Nearest-medoid assignment copies: pin FastPAM, CLARANS, and both FastCLARA storage/precision routes digit-for-digit before deciding whether consolidation is behavior-neutral. |
| F14 | CSV emitter drift: pin byte-identical output, locale, non-finite handling, and precision across all four dense/mmap stream/visitor routes before consolidation. |
| F15 | Benchmark/test generator and CPU-oracle drift: inventory semantic variants, retain intentional ranges/shapes, and prove any shared utility preserves seeded bytes and oracle values. The old “byte-identical” premise is falsified. |
| F16 | Preset portability/truth: a clean configure probe must reject stale minimum-version metadata and must not require one developer's absolute LLVM installation path. |

**Ledger verdict: PASS.** All 53 source records have a verdict: 21/21
known-bug/cleanup and 32/32 backlog/deferred/operator/question records. No
historical parent record is unclassified. The six new open defect/cleanup
routes use unique finding IDs F11–F16; the pre-existing permanent Parquet
fixture remains F8.
