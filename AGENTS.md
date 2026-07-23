# AGENTS.md — Working rules for autonomous agents in DTWC++

You are working the campaign in `PLAN.md`. This file is HOW you work; PLAN.md is
WHAT you work on. Read both before touching anything. Supporting record:
`.claude/LESSONS.md` (gotchas — read it, it will save you hours),
`.claude/CITATIONS.md`, `.claude/design.md`, `.claude/TODO.md`, the plan archive
`.claude/PLAN-archive-2026-07-20-phases0-9.md`, and session handoffs in
`.claude/summaries/`.

## Prime directives

1. **Run continuously. Never wait for a human.** No questions, no check-ins, no
   "shall I proceed". If something is environment-impossible, take the named
   fallback or record `[BLOCKED-ENV]` with verbatim probe output and move on.
2. **Never fabricate.** State as fact only what you verified. Every number in
   any report exists in a nameable artifact (file, run-log, cited source).
   Quote outputs, errors, and data VERBATIM — paraphrased evidence is corrupted
   evidence. Tag load-bearing claims **[confirmed]** (name the evidence) or
   **[inferred]** (name what would confirm it). "I must check first" is a
   complete answer; filling a gap from pattern-matching is not.
3. **Commit relentlessly.** The last two runs died with hours of work
   uncommitted. One conventional commit per task/finding the moment its gate is
   green; never accumulate more than one task uncommitted; keep PLAN.md
   checkboxes and the current handoff file truthful in real time so any death
   costs one task, not a session.
4. **Register bands before decisive runs.** The quantitative pass/fail band goes
   into the script BEFORE the run; the verdict prints against it. Unregistered
   runs are exploration, never evidence. A falsified band is a deliverable —
   record FALSIFIED with the numbers; no rescue-tuning past 2 attempts.
5. **Killed ideas stay killed.** Before starting any branch, search PLAN.md's
   Killed-ideas section, the archive, and LESSONS.md. Re-opening requires
   overturning the recorded kill evidence explicitly.

## Safety rails (absolute)

- Never write outside the git project root. Data files are read-only.
- Never `git push`, never tag, never publish (PyPI/TestPyPI/conda), never SSH or
  submit to SLURM/HPC — all operator actions. Local commits only.
- Never delete untracked local `build*/` directories (they encode verified
  recipes); inventory, don't destroy.
- Python dependencies via `uv add/remove` only — never pip.
- Optional deps (OpenMP, HiGHS, Gurobi, CUDA, Metal, MPI, llfio, Arrow) stay
  optional; the core must always build with all of them OFF.

## Verification discipline

- **BUILD → RUN → ANALYSE → FIX.** A clean compile proves nothing; read the
  artifact or execute it. Record the baseline (numbers + failing-test names +
  base commit) BEFORE changing anything; report deltas against it by name.
- **Drive the real binary** for anything CLI-facing. A green unit test on a
  helper proves the helper works, never that it is reachable (the F7/D1 bug
  passed its unit test AND the full gate while the real CLI ignored the flag).
- **Gates must assert the subject RAN.** ctest counts a skip as a pass
  (`SKIP_RETURN_CODE 4`). When a suite can skip, assert on the test binary's own
  output: skip message absent, executed-assertion count ≥ floor.
- **Arbiters.** Two computations disagree → neither judges; build a third from
  different mathematics. "No-op / refactor-only / identical behaviour" claims
  require digit-identical outputs on a recorded case, not code inspection.
  A test failing after your change: stash, rerun, compare digit-for-digit
  before you may say "pre-existing".
- **Oracles validated on NON-degenerate cases.** Symmetric/uniform/zero-forcing
  fixtures let sign and factor errors cancel; always include a case that would
  catch them.
- **Numbers ledger for open problems:** a small table of independently computed
  values whose disagreement localises the fault; attack the largest unexplained
  gap, not the best story.
- Every derivation: units checked, assumptions stated where they enter, each
  approximation named with regime + leading-order error term.

## Repository conventions

- **Commits:** Conventional Commits, one per task/finding, precise subject
  (never "Added more algorithms and tests"). Do not stage PLAN.md edits inside
  a code commit — plan/bookkeeping updates ride in their own `docs:` commits.
- **CHANGELOG.md** (Unreleased): one line per user-visible change, always.
- **Run-logs:** `.claude/baselines/YYYY-MM-DD-<topic>.md` — decisive outputs
  verbatim, bands + verdicts, environment noted. Never `benchmarks/baselines/`.
- **LESSONS.md:** add an entry for every new bug class or expensive surprise.
  **CITATIONS.md:** every source used, verified; paywalled reads flagged
  [inferred].
- **Decision log:** any departure from the plan, any resolved ambiguity, any
  `[BLOCKED-ENV]` → dated entry in PLAN.md §Binding decisions (digest style).
- **Handoff:** append-as-you-go file `.claude/summaries/handoff-YYYY-MM-DD-<topic>.md`
  — accomplishments, decisions, next steps, open questions, exact resume point.
- C++20 minimum; no naked `new`/`delete` in core; buffer > thread_local >> heap
  in hot paths; match existing style; every changed line traces to a task.
- No silent fallbacks: undeliverable capability ⇒ typed error or loud warning.
- Guards that must fire on a build WITHOUT an optional dep live OUTSIDE that
  dep's `#ifdef` (canonical gate builds `DTWC_ENABLE_ARROW=OFF`).

## Build & gate recipes (proven; details in archive §Proven recipes)

- **Canonical gate:** `build/highs-1151` (clang + Ninja + Release, HiGHS ON,
  llfio ON, Arrow OFF). Floor: `ctest` → **114/114, 0 failed**, 6 capability
  skips (cuda×2, metal×3, io_readers×1 — the io_readers skip is expected in
  this Arrow-OFF build; F9 is closed by its separate Arrow-ON executable gate).
  Rebuild first: `cmake --build build/highs-1151` (expect "no work to do" on a
  clean tree).
- **llfio-OFF build:** `build/nollfio` — must configure, build, and pass with
  capability skips.
- **CUDA:** `build/cuda-verify` (MSVC host + nvcc 13.0, RTX 4000 Ada sm_89 is
  LOCAL — run the CUDA tests for real). GPU-HiGHS rebuild:
  `MSYS_NO_PATHCONV=1 cmd.exe /c build/highs-gpu/bench_build.bat` (MSYS mangles
  `/c` → `C:\` without the env var). nvcc rejects clang-21 as host; CUDA 13
  needs `-allow-unsupported-compiler` for new MSVC.
- **Python:** rebuild the extension via a configure dir with
  `-DDTWC_BUILD_PYTHON=ON`, copy the fresh `.pyd` + `libomp.dll` into the venv,
  verify import + one NEW symbol before pytest (stale-`.pyd` false-greens are
  real). Floor: 407 passed / 11 skipped (418 collected) on the fresh wheel.
  Windows llfio-ON wheel is a known OPEN item (see PLAN R6).
- **MATLAB:** R2024b + R2025b installed; run via `matlab -batch`. addpath ORDER
  matters — add the fresh `build/mex-verify/bin` LAST so it prepends ahead of
  any stale MEX (wrong order → 0xc0000005 from an old binary; check
  `which('dtwc_mex','-all')` first). Floor: 61/61 on the expanded local gate.
- **Conformance (permanent parity gate):** labels/medoids digit-identical across
  all 4 routes; silhouette 0.96894972764334841, DB 0.038333333333333337,
  dunn 11.5 vs `tests/conformance/conformance_reference.txt` (≤1e-12 rel).
- **Floating-point model:** the build is NOT `-ffast-math` — an explicit subset
  incl. `-fassociative-math`, deliberately WITHOUT `-ffinite-math-only`
  (`cmake/StandardProjectSettings.cmake:59-70`); kernels avoid `infinity()`
  sentinels for this reason. Debug FP issues from exact seeded bytes, never
  from rounded printouts.
- **Shell:** Git Bash paths are `/c/D/git/dtw-cpp`; working directory persists
  between commands — prefix `cd /c/D/git/dtw-cpp &&` when in doubt.
- Wall-clock on this machine is ADVISORY (shared host). Hard perf gates =
  machine-independent counters (allocations, cell counts, prune rates) or
  quiet CI.

## Reporting standard

Lead with the outcome and the verdict against the registered band. Separate
confirmed from inferred so a reader can tell them apart from prose alone. Name
the rollback for anything irreversible or runtime-unconfirmed, and name the one
claim you most expect to be wrong. A report that hides a failure is worse than
the failure.
