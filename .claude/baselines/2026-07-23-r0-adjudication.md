# R0 adjudication — 2026-07-23

## Scope and baseline

- Branch: `Claude`
- Code base entering R0: `c36ba27` (`docs: hand off the F7 streaming session`)
- Campaign-record commit made during R0: `6a92e70`
- Environment: Windows, PowerShell, shared host; wall-clock observations are
  advisory only.
- Last committed-tree baseline [confirmed:
  `.claude/summaries/handoff-2026-07-13-f7-streaming.md`]: 113/113 CTest
  targets passed, zero failed, six capability skips.
- Current generated test inventory before the decisive run [confirmed:
  `ctest -N -C Release` in `build/highs-1151`]: 114 targets after adding the
  direct F10 suite. The R0 gate therefore requires all 114 current targets,
  not merely the historical floor of 113.

## Diff classification before execution

| Paths | Class | Initial verdict |
|---|---|---|
| `PLAN.md`, `AGENTS.md`, `.claude/PLAN-archive-2026-07-20-phases0-9.md` | docs / campaign governance | KEEP; committed separately as `6a92e70` |
| `.github/workflows/ubuntu-unit.yml` | F9 | REPAIR; the proposed gate checked 12 test cases but did not assert the registered 348 executed assertions |
| `tests/unit/unit_test_cli_args.cpp` | F8/F7-adjacent | KEEP as Arrow-OFF routing coverage; insufficient by itself to close F8 |
| `tests/unit/core/unit_test_distance_sampling_weights.cpp`, `tests/unit/algorithms/unit_test_fast_pam.cpp`, `tests/unit/unit_test_clustering_algorithms.cpp`, F10 `CHANGELOG.md` paragraph | F10 | REPAIR; validate non-finite selected entries and remove the false proportional-sampling claim before execution |
| `dtwc/algorithms/fast_pam.cpp` | other: R3 integer-width audit | REPAIR; the widened swap helpers leave `fast_pam_seeded` with an unchecked `size_t`→`int` narrowing and let `fast_pam` fill the matrix before the result-width rejection |
| `.claude/CITATIONS.md`, `.claude/UNIMODULAR.md` | docs / provenance | REPAIR; Vinod bibliography is verified, but the unread full text cannot support an unqualified “same program” claim |
| `.claude/LESSONS.md` | docs (floating-point correction) + F9 lesson | KEEP after source-reference verification; split into the corresponding docs/F9 commits |
| `.claude/TODO.md` | docs / R1 preparation | KEEP as an explicit staleness warning; full reconciliation remains R1 |

## Registered bands (written before decisive execution)

### F10 focused gate

- The newly registered selected-NaN case must fail before the production
  repair and pass after it.
- Final direct suite: zero failed assertions; nonnegative weights exact,
  selected outputs exactly `0.0`, signed translation exact on the binary64
  fixtures, degenerate total exactly `0.0`, and every non-finite input rejected
  with a typed exception.
- Seeded `Kmeanspp_seeded`, unseeded/seeded k-means++, and
  `fast_pam_seeded` signed/degenerate behavioral cases all execute and pass.

### F9 Arrow gate

- Workflow YAML passes a local parser and `actionlint` when available.
- Test binary output contains no skip marker.
- Catch2 output reports at least 12 executed cases and at least 348 executed
  assertions; missing or unparsable counts fail closed.
- Primary closure requires a local Arrow-ON build executing
  `test_io_readers`; hosted CI remains operator verification.

### Canonical R0 gate

- `cmake --build build/highs-1151` exits zero.
- Every currently registered target runs to a CTest pass: 114/114 after the
  F10 suite addition, zero failed.
- Exactly the six documented capability skips remain: CUDA ×2, Metal ×3,
  `test_io_readers` ×1. Any additional skip is a failure.
- Changed suites are also run directly so Catch2 assertion/case output proves
  they executed.

## Evidence

Decisive command output and verdicts are appended below immediately after each
run. Exploration is labelled explicitly and is not used as closure evidence.

## F10 adjudication

**KEEP after REPAIR**, committed as `20b894d`
(`fix: reject non-finite D-sampling inputs`). The preregistered red and final
51-assertion direct seam suite, behavioral callers, and 114-target canonical
gate are recorded verbatim in
`.claude/baselines/2026-07-23-f10-sampling.md`.

## Remaining adjudications

The preliminary F9 band above required 12 cases. Source inventory falsified
that number before the decisive run: the enabled suite contains 11 cases. The
dedicated preregistration in
`.claude/baselines/2026-07-23-f9-arrow-gate.md` replaced it with the reachable
floor of 11 cases while retaining the 348-assertion floor. That artifact records
the deliberate reds, fresh Arrow-ON build, final 390/390 assertions in 11
cases, CTest execution, imported DLLs, parser mutants, and final canonical gate.

- **FastPAM width edit — KEEP after REPAIR.** Commit `f8ff7d3` moves the
  `size_t`-to-`int` check to every public entry before effects and removes
  repeated narrowing in the supported kernels. Evidence:
  `.claude/baselines/2026-07-23-fast-pam-index-width.md`.
- **Interrupted CLI/FastCLARA tests — RECLASSIFY, then KEEP after REPAIR.**
  They are F7 planner/guard coverage, not F8 resident/streaming equivalence.
  Commit `7c71602` kills three routing mutants and drives the real CLI.
  Evidence: `.claude/baselines/2026-07-23-f7-routing-coverage.md`.
- **F9 Arrow workflow — KEEP after REPAIR.** Commits `833f570` and `0c91c9b`
  repair the two failures exposed by the fresh enabled build; `e323197` adds
  the execution-count gate; `b79e271` records closure. Evidence:
  `.claude/baselines/2026-07-23-f9-arrow-gate.md`.
- **Floating-point LESSONS hunk — KEEP after REPAIR.** Commit `c627826`
  replaces the false `-ffast-math`/single-flag causal claim with the emitted
  Clang/GCC Release flag set, labels the historical cause inferred, and leaves
  the factor-16 derivation open in R2-D4. Focused retained gate:

  ```text
  Filters: [eap] [exact]
  Randomness seeded to: 725833122
  ===============================================================================
  All tests passed (1011 assertions in 3 test cases)
  ```

- **TODO staleness hunk — KEEP after REPAIR.** Commit `8775156` removes stale
  source-line anchors and the false digit-identical-medoid claim, marks the one
  adjudicated FastPAM entry CLOSED-BY `8ca7354`, and leaves every other entry
  explicitly unverified for R1. `git log -- .claude/TODO.md` confirms
  `874edd5` (2026-07-06) was the last prior file commit.
- **MIP provenance hunks — KEEP after REPAIR.** Commit `85eabcd` removes the
  contradicted “same 0/1 program” and unsupported independent-lineage/audience
  claims. Publisher/Crossref metadata and the Vinod author-page abstract are
  recorded in `.claude/CITATIONS.md`; current solver rows are confirmed in
  `dtwc/mip/mip_Highs.cpp`, `mip_Gurobi.cpp`, and `pdlp_lp.cpp`.

## Final verdict

**PASS — R0 CLOSED.** Every inherited path has a KEEP-after-gate,
KEEP-after-REPAIR, RECLASSIFY-after-REPAIR, or campaign-governance verdict and
is committed. The final command

```text
git status --porcelain=v1
```

printed no output at `85eabcd`; no inherited change remains uncommitted. The
canonical behavioral closure remains the post-F9 114/114 CTest run with zero
failed and exactly six registered capability skips, recorded verbatim in the
F9 artifact.

The claim most likely to need later correction is the fine-grained historical
attribution of the classical p-median rows between Balinski and
ReVelle-Swain; the current wording follows the named primary records plus the
Marín-Pelegrín history and is not used as mathematical evidence for the TU
analysis.
