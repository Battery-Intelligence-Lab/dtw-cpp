# R1 record hygiene — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `80b4485` (`docs: register multivariate flat-buffer finding`)
- Scope: `.claude/UNIMODULAR.md`, `.claude/LESSONS.md`,
  `.claude/CITATIONS.md`, the contradictory FasterPAM baseline narrative,
  the permanent checker, and the owning PLAN/handoff records.
- Constraint: record corrections only; no runtime behavior changes.

## Preregistered acceptance band

1. Add one permanent deterministic checker before editing the three records and
   demonstrate that it fails on the inherited stale claims.
2. `.claude/UNIMODULAR.md` has exactly one freshness header containing
   `85eabcd`, `fd49ff4`, `8375aef`, and `789cc52`. Historical proposals remain
   identifiable as historical; present-tense formulation, computational
   verdicts, and LR-core architecture agree with §8 and the implemented chain.
3. `.claude/LESSONS.md` retains every lesson concept while correcting the
   dependency workflow, benchmark labels, removed SIMD claims, Float64 default,
   typed-validation rule, MATLAB/HiGHS scope, ARC architecture advice, closed
   F7/F9 state, immutable fix anchor, and exact maximum-gap value.
4. `.claude/CITATIONS.md` has one canonical entry each for FasterPAM, TC-DTW,
   and Shokoohi-Yekta; the Duran-Mateluna, DDTW, MathWorks, CUDA 13.0, Chao,
   Weber/Freifeld, HiGHS v1.15.1, Arrow, and nanoarrow records use verified
   primary identities and immutable/versioned links. Loog is retained.
5. `.claude/MISSING.md` and `.claude/READ.md` remain absent. Git history must
   show their deliberate deletion in `0449f7c`; they are not recreated merely
   to satisfy stale PLAN wording.
6. External URL availability is verified manually against primary/official
   sources and recorded below, but is not a deterministic hard gate. The
   permanent script pins the verified citation identities and URL strings.
7. Final gates:

   ```text
   .venv/Scripts/python.exe scripts/check_record_hygiene.py
   git diff --check
   git log --all --diff-filter=D --summary -- .claude/MISSING.md .claude/READ.md
   ```

   The checker exits zero with exact final line
   `record hygiene checks passed`; `git diff --check` has no diagnostics; the
   deletion log names commit `0449f7c`.

## Inherited-state probes

```text
missing_exists=False
read_exists=False
```

The deliberate-red checker output is recorded before any record correction.

```text
Traceback (most recent call last):
  File "C:\D\git\dtw-cpp\scripts\check_record_hygiene.py", line 220, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_record_hygiene.py", line 211, in main
    check_unimodular()
    ~~~~~~~~~~~~~~~~^^
  File "C:\D\git\dtw-cpp\scripts\check_record_hygiene.py", line 37, in check_unimodular
    raise AssertionError("UNIMODULAR must contain exactly one current freshness header")
AssertionError: UNIMODULAR must contain exactly one current freshness header
```

## Primary-source verification

Accessed 2026-07-23:

- **[confirmed]** Schubert and Rousseeuw's official arXiv record
  (`https://arxiv.org/abs/2008.05171`) gives the full FasterPAM title, authors,
  journal reference *Information Systems* 101 (2021), article 101804, and DOI
  `10.1016/j.is.2021.101804`.
- **[confirmed]** Elsevier's article page
  (`https://www.sciencedirect.com/science/article/pii/S0377221722008864`) gives
  Cristian Duran-Mateluna, Zacharie Ales, and Sourour Elloumi; *European Journal
  of Operational Research* 308(1), 84--96; 1 July 2023; DOI
  `10.1016/j.ejor.2022.11.033`; and the reported 238,025-client/site scale.
- **[confirmed]** SIAM's DDTW proceedings page
  (`https://epubs.siam.org/doi/10.1137/1.9781611972719.1`) gives Eamonn J. Keogh
  and Michael J. Pazzani, pages 1--11, and the pinned DOI.
- **[confirmed]** MathWorks' `mexErrMsgIdAndTxt` page
  (`https://www.mathworks.com/help/matlab/apiref/mexerrmsgidandtxt.html`) says
  that the call terminates the MEX file, returns control to MATLAB, does not run
  `mexAtExit`, and automatically frees MATLAB-created allocations. Its C++ MEX
  page
  (`https://www.mathworks.com/help/matlab/matlab_external/creating-c-mex-files.html`)
  separately says that destructors run for objects that leave scope on error.
  Neither page supports the inherited categorical `longjmp` claim.
- **[confirmed]** NVIDIA's versioned CUDA 13.0 programming-guide and
  `cudaDeviceProp` archive pages opened at
  `https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html`
  and
  `https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/structcudaDeviceProp.html`.
- **[confirmed]** The official arXiv records identify the full Chao et al. author
  list (`https://arxiv.org/abs/2603.14899`), Daniel Shen and Min Chi plus the
  exact TC-DTW title (`https://arxiv.org/abs/2101.07731`), and Ron Shapira Weber
  and Oren Freifeld plus the exact SoftDTW-CUDA-Torch title
  (`https://arxiv.org/abs/2602.17206`).
- **[confirmed]** HiGHS' versioned/tagged v1.15.1 release, PDLP source tree,
  and root CMake
  file opened at `https://github.com/ERGO-Code/HiGHS/releases/tag/v1.15.1`,
  `https://github.com/ERGO-Code/HiGHS/tree/v1.15.1/highs/pdlp`, and
  `https://github.com/ERGO-Code/HiGHS/blob/v1.15.1/CMakeLists.txt`.
- **[confirmed]** The official Arrow C Data and PyCapsule specifications opened at
  `https://arrow.apache.org/docs/format/CDataInterface.html` and
  `https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html`.
  The tagged nanoarrow 0.8.0 release opened at
  `https://github.com/apache/arrow-nanoarrow/releases/tag/apache-arrow-nanoarrow-0.8.0`.
- **[confirmed]** Springer's canonical Shokoohi-Yekta et al. article page
  (`https://link.springer.com/article/10.1007/s10618-016-0455-0`) gives the
  exact title, five authors, *Data Mining and Knowledge Discovery* 31, 1--31,
  and DOI `10.1007/s10618-016-0455-0`.
- **[confirmed]** Loog's official arXiv record
  (`https://arxiv.org/abs/2102.02291`) gives the author and title. The IEEE
  endpoint was unavailable from this environment; the 2012 venue, pages 1--6,
  and DOI `10.1109/MLSP.2012.6349714` were cross-checked against DBLP and the TU
  Delft research portal rather than claimed as a successfully opened IEEE
  primary page.

External availability is advisory evidence only. The deterministic checker
pins the identities and versioned/tagged URL text without making a network
request.

## Environment and live source probes

The environment probe at the registered base printed:

```text
5.1.26100.8457
Python 3.13.7
git version 2.51.0.windows.1
Claude
80b4485
```

The mixed Python import probe that corrected the inherited non-editable-wheel
lesson printed verbatim:

```text
package=C:\D\git\dtw-cpp\python\dtwcpp\__init__.py
api=C:\D\git\dtw-cpp\python\dtwcpp\_api.py
core=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
```

The current file:line spot-check printed:

```text
dtwc/mip/benders.cpp:177:  // Objective: minimize sum theta_j
dtwc/mip/benders.cpp:317:    // --- Generate disaggregated Benders cuts ---
dtwc/CMakeLists.txt:211:  message(STATUS "Arrow + Parquet linked — IPC and Parquet reading enabled")
dtwc/Problem.hpp:70:  bool warm_start = true;          ///< Run FastPAM first and feed as MIP start.
dtwc/mip/lagrangian_root.cpp:480:LagrangianResult lagrangian_root_exact(const double *D, int N, int k,
dtwc/mip/lagrangian_root.cpp:633:                 "lagrangian_root_exact: node cap %ld reached at N=%d k=%d before the tree "
cmake/Coverage.cmake:10:    set_tests_properties(${TARGET_NAME} PROPERTIES SKIP_RETURN_CODE 4)
tests/unit/algorithms/unit_test_fast_clara.cpp:712:TEST_CASE("FastCLARA in-RAM uses the portable seeded sample contract",
```

## Independent residual review

The first post-edit checker green was not accepted as final evidence. Two
read-only reviews found claims outside its initial marker set; the records and
checker were corrected before the final gate:

- **[confirmed]** `.claude/baselines/2026-07-06-phase0.md:175-194` records
  Float32 speedups of 1.57×–1.90×. Its narrow ten-pair accuracy probe does not
  establish clustering invariance.
- **[confirmed]** The verbatim table at
  `.claude/baselines/2026-07-08-faster-pam-bench.md:48-53` records
  non-monotone 2.95×–8.06× naive/FastPAM1 ratios. A dated correction now
  supersedes that baseline's contradictory 2.3× prose; no PMU artifact proves
  the memory-bound explanation.
- **[confirmed]** `.claude/baselines/2026-07-08-pdlp-bench.md:52-85`
  separates the measured CPU/GPU/Kelley ratios and brackets, but does not
  measure an exact PDLP crossover or prove the full p-median matrix TU.
- **[confirmed]** `benchmarks/bench_parquet_access.py` is the surviving I/O
  experiment source: it generates 30 fixture files but uses only ten random
  length-500 pairs for its pure-Python DTW timing, and no output transcript
  survives.
- **[inferred]** Commit `420f764` historically records 199.6 MB to 9.7 MB
  (20.58×) for a generated battery fixture. With no raw transcript, those
  numbers are preserved explicitly as inherited prose rather than asserted as
  a confirmed measurement; the old broad timing rules were removed.
- **[confirmed]** `.claude/baselines/2026-07-23-f9-arrow-gate.md:168-181`
  records the fresh PyArrow-23 CMake route executing 390 assertions in all 11
  Arrow/Parquet cases with no skip. The canonical Arrow-OFF skip and the
  complementary Arrow-ON execution are now stated separately.
- **[confirmed]** The live Benders master has N `theta_j` variables and
  disaggregated all-facility cuts (`dtwc/mip/benders.cpp:177-204,317-361`);
  LR-core uses tolerance-guarded fixing
  (`dtwc/mip/reduced_cost_fixing.cpp:61-85`) and a capped y-only tree that
  returns loudly uncertified on exhaustion
  (`dtwc/mip/lagrangian_root.cpp:480-641`).

The permanent checker now pins these residual classes, scopes all four commit
IDs to the single freshness header, and rejects the malformed
Ghouila-Houri/odd-cycle, Benders, SIMD, FastPAM, mmap, I/O, MATLAB, CMake, and
optional-dependency claims that escaped the first pass.

## Final gates

Python syntax compilation exited zero with no output:

```text
.venv/Scripts/python.exe -m py_compile scripts/check_record_hygiene.py
```

The deterministic checker printed:

```text
record hygiene checks passed
```

The preregistered unstaged `git diff --check` command exited zero but did emit
line-ending conversion warnings:

```text
warning: in the working copy of '.claude/CITATIONS.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.claude/LESSONS.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.claude/UNIMODULAR.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.claude/baselines/2026-07-08-faster-pam-bench.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.claude/summaries/handoff-2026-07-23-r1-reconciliation.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'PLAN.md', LF will be replaced by CRLF the next time Git touches it
```

Verdict: **FALSIFIED as worded** for the preregistered "no diagnostics"
expectation. The command found no whitespace errors and exited zero, but
`core.autocrlf` produced the six diagnostics above. The decisive staged-diff
gate below checks the actual commit payload without conflating those
worktree-conversion warnings with whitespace errors.

The retirement-history command printed:

```text
commit 0449f7cddccdd91faaf025d4aa67e2eabad82331
Author: Volkan Kumtepeli <volkan.kumtepeli@gmail.com>
Date:   Thu Apr 2 23:26:41 2026 +0100

    added yaml

 delete mode 100644 .claude/MISSING.md
 delete mode 100644 .claude/READ.md
```

After staging the exact eight-file task payload, both whitespace gates exited
zero with no output:

```text
git diff --cached --check
```

```text
git diff --check
```

The syntax compile again exited zero with no output, and the staged-state
checker again printed:

```text
record hygiene checks passed
```

## Verdict

**PASS.** The permanent checker, exact staged-diff whitespace gate, empty
worktree-diff gate, and retirement-history gate meet the corrected registered
band. The six pre-stage CRLF conversion warnings are retained above rather
than mislabeled as an empty diagnostic stream. Both independent final reviews
were green after the residual corrections.
