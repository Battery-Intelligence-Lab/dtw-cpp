# Handoff — 2026-09-21 — multi-agent sweep (W0 continuation)

Base: branch `design-2.0`, HEAD `22c4c5f`, tree **clean** (verified mid-session; all
agents this session were read-only by construction — no edits made to the repo).

## Session state

Nothing was committed or edited this session. Three agent waves were launched against the
tree; their results live in the workflow transcripts (below), not yet consolidated into a report.

## Project state as read (confirmed)

- Spec `.claude/specs/2026-09-07-design-2.0-campaign.md` + 129-row ledger
  `.claude/specs/2026-09-07-diff-ledger.md`. W0 plan `.claude/plans/2026-09-07-w0-baseline-tooling.md`
  = 14 tasks (Task 0, 1a, 1b, 1c, 2, 3, 3b, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13).
- **Done**: Task 0 (branch/baseline/run-log), 1a/1b/1c — commits `d21ffee`, `22c4c5f`.
  Every test now registers through `dtwc_add_test`; floors in `tests/floors.cmake`.
- **Next unstarted**: Task 2 — conformance reference read-only + hash-pinned (ledger T-15).
- ~10 III.10 DECIDE rows still open (C-05, C-09/C-19, C-11, C-13, O-09, B-14, B-19, A-05,
  G-07, T-04/T-12).

## Confirmed myself this session (not agent-reported)

Task 2's premise, corrected. The handoff of 2026-09-07 says
`tests/cpp_conformance.cpp:212` "regenerates the reference then compares against it".
Two corrections:

1. The file is at **`tests/conformance/cpp_conformance.cpp`**, not `tests/cpp_conformance.cpp`.
2. The gate is **not universally fake**. `conformance_dir()` (lines 67–69) resolves from the
   compile-time `DTWC_TEST_DATA_DIR` into the **source tree**, and
   `tests/conformance/conformance_reference.txt` **is git-tracked** (`git ls-files`).
   So in a normal checkout the auto-regen branch does not fire and the comparison is real.

The residual holes at `tests/conformance/cpp_conformance.cpp:211–219` are narrower but real:

- `DTWC_CONFORMANCE_REGEN=1` rewrites the **tracked** ground truth **in the source tree**,
  then compares `live` against what it just wrote. No hash pin, so a silent regen is invisible
  unless someone reads `git status`. A test mutating tracked data is itself a defect.
- `regen` is also true when the reference file is simply **absent** (sparse checkout, source
  export, accidental deletion) — the test then passes vacuously instead of failing loudly.

Fix shape for Task 2 (unchanged in spirit, sharper in detail): make the test read-only —
absent reference = hard `FAIL`, never regenerate; pin the reference by SHA-256 checked in the
test; move regeneration into a separate opt-in tool/target, not the test binary.

## Agent waves launched (results NOT yet consolidated)

All three ran read-only. To recover results, read `journal.jsonl` in the transcript dir, or
resume with `Workflow({scriptPath, resumeFromRunId})` — completed agents return cached results.

| Wave | Run ID | Shape | Transcript dir |
|---|---|---|---|
| 1 | `wf_724fa8b7-1f9` | 6 finders → adversarial verify per finding | `…/subagents/workflows/wf_724fa8b7-1f9` |
| 2 | `wf_51ebd254-acb` | 9 finders → verify, + 3-lens design panel on the DECIDE rows | `…/subagents/workflows/wf_51ebd254-acb` |
| 3 | `wf_da2a1086-cc9` | 20 narrow 90-second targets, no verify stage | `…/subagents/workflows/wf_da2a1086-cc9` |

Transcript root: `C:\Users\engs2321\.claude\projects\c--D-git-dtw-cpp\fdb43eba-c2dc-49ce-aec5-d24cd7625c7c\subagents\workflows\`
Scripts root: `…\fdb43eba-c2dc-49ce-aec5-d24cd7625c7c\workflows\scripts\`

Wave 1 dimensions: conformance-gate, test-floors, core-correctness, spec-vs-reality,
build-optionality, api-surface.
Wave 2: python-bindings, matlab-mex, algorithms, io-formats, openmp-concurrency, ci-workflows,
error-handling, docs-changelog, perf-hotpath + design panel (maintainer / user / risk lenses).
Wave 3: gitignore, floors-zero, may-skip, catch2-tags, dtws-format, dataloader-csv,
rng-determinism, band-edges, nan-contract, naked-new, include-leak, cli-options, exit-codes,
cmake-required, source-path-defines, test-writes-tree, changelog, python-gil, scores-edge,
todo-open.

## Next steps

1. Consolidate the three waves' `journal.jsonl` into one report
   `.claude/reports/2026-09-21-multiagent-sweep.md`, separating survived from refuted findings.
2. **Re-verify every surviving finding yourself** before acting — agents over-report; a finding
   is a hypothesis until the cited line is opened. (This session already found the standing
   handoff overstated the Task 2 gate.)
3. Then execute W0 Task 2 with the corrected premise above.
4. The design panel's output is input to the III.10 DECIDE rows — it does **not** resolve them.
   Volkan decides; nothing may be silently pre-decided.

## Open questions (unchanged, plus one new)

- OPEN: all ~10 III.10 DECIDE rows.
- OPEN (new): does any W0 work already committed silently pre-decide a DECIDE row? Wave 1's
  `spec-vs-reality` dimension was asked exactly this; answer is in its transcript, unread.
- OPEN: latency- vs memory-bound regime (PMU task, W8).
- OPEN: whether IPO inlines `dist_by_ind` into algorithm TUs (W0 Task 12 measures).

## Status honesty

Nothing in this session was built, run or tested. No test suite was executed, so there is **no
new baseline** and no regression claim is possible. The only claims I confirmed by opening files
are the ones in "Confirmed myself" above; everything in the agent waves is unverified.

---

## HARVEST (appended after the waves were stopped)

All three waves stopped on the user's timebox. Returned before stop:
wave1 **19/29** agents, wave2 **0/9** (killed too early, nothing recoverable), wave3 **8/20**.
Raw consolidated output (39 findings + skeptic verdicts, full text):
`.claude/reports/2026-09-21-multiagent-sweep.md` (119 KB).

Coverage is PARTIAL by construction. Wave 2's nine dimensions (python-bindings, matlab-mex,
algorithms, io-formats, openmp-concurrency, ci-workflows, error-handling, docs-changelog,
perf-hotpath) and its 3-lens design panel produced NOTHING and must be re-run.

### Verified by me, not by an agent (opened the cited lines myself)

1. **Explicit floors silently shadow the measured table.** `cmake/DtwcTest.cmake:121-130`
   consults `DTWC_TEST_FLOOR_<target>` only under `if(NOT ARG_ASSERT_FLOOR)` /
   `if(NOT ARG_CASE_FLOOR)`, with no comparison. An explicit `ASSERT_FLOOR 1` in
   `tests/CMakeLists.txt` therefore wins over a measured floor of, e.g., 13, silently.
   CAUTION (from the skeptic agent, plausible and unverified): some entries sit below the
   table deliberately, on a build-flavour branch, so a naive `max()` would break them.
   Do not "fix" this without reading each site.
2. **`22c4c5f` ships no CHANGELOG.md entry.** `git show --stat 22c4c5f` touches exactly
   `cmake/DtwcTest.cmake` and `tests/CMakeLists.txt`. The repo PR checklist requires a
   CHANGELOG update on every PR. Same check not yet run for `d21ffee`.
3. **`unit_test_mpi`'s pass regex is the skip announcement** — `tests/CMakeLists.txt:238-243`:
   with `DTWC_ENABLE_MPI=OFF` (every W0 matrix) the MARKER is
   `"MPI not enabled \(DTWC_ENABLE_MPI=OFF\)[.] Skipping MPI tests[.]"`, registered
   `MARKER_ONLY`. NUANCE the agent missed: the comment at 232-237 states this deliberately
   (the binary owns `main()`, prints no Catch2 summary, so it cannot be floored). This is a
   knowingly-accepted hole, not an accident - but it is still a test that cannot fail, and
   the ON-branch marker is explicitly "transcribed ... and unverified".

### Highest-value UNVERIFIED agent claims (re-open the line before acting)

- `dtwc/Problem.cpp:685` - `dist_by_ind` hard-codes `d(i,i)=0`, said to be wrong and
  sign-breaking for SoftDTW (self-distance is not 0 there). If true this is a correctness bug,
  not a cleanup.
- `CMakeLists.txt` / `dtwc/CMakeLists.txt` - the `dtwc++` target and its public headers are
  never installed or exported; `cmake --install` is claimed to ship only the CLI binary.
  Two independent agents raised this.
- The no-OpenMP build is claimed to be `FATAL_ERROR` by default and exercised in NO CI job -
  that would directly violate repo non-negotiable #3 (optional deps stay optional).
- Release-only aggressive FP relaxations are claimed to be directory-scope, reaching vendored
  HiGHS, while every unit-test CI job builds Debug - so the NaN contract is never tested where
  it is at risk.
- `--distance-matrix` load failure and checkpoint save failure are each claimed to be
  swallowed, with exit code 0.
- `fast_pam()` is claimed to consume the process-global mutable `randGenerator` via an
  unseeded `init::Kmeanspp` - a reproducibility hazard the conformance test depends on.

### Correction to carry forward

Wave 1 labelled the conformance gate "fake-green" outright. That is OVERSTATED: the reference
IS git-tracked, so in a normal checkout the comparison is real. The precise holes are the two
in "Confirmed myself" above. Do not repeat the stronger claim.
