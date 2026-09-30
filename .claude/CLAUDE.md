# DTWC++ — Claude Runbook

## Read in this order, and no further

0. The newest `.claude/summaries/handoff-*.md` — where the last session stopped.
1. `.claude/CHARTER.md` — what Volkan asked for, verbatim.
2. `.claude/MAP.md` — what is where today, the target interface, the invariants. **Read this instead of the tree.**
3. `.claude/PLAN.md` — the phases A–G; read your step, then its row in
   `.claude/plans/2026-09-27-audit/advisor_exec.md` §2 (or the wave in `chair.md` §4).

On demand: `DECISIONS.md` (killed ideas, standing rules, rulings) and `LESSONS.md` (rules learned the hard
way). Context is a budget: never bulk-read `baselines/`, `LESSONS.md`, `CHANGELOG.md` or the audit folder;
grep them for the symbol or finding id you need.

## Non-negotiables

1. No runtime dependence on repo-relative paths. Never put the repo root on an include path (on a
   case-insensitive filesystem the root `VERSION` file shadows `<version>`).
2. A user-visible change against v1.0.0 gets a CHANGELOG line. Every change adds or adjusts tests; keep lint
   clean.
3. Optional dependencies only (HiGHS, Gurobi, CUDA, Metal, llfio, Arrow, YAML; OpenMP unless
   `DTWC_ALLOW_SEQUENTIAL`). The core builds without them. No silent fallback, ever: a request that
   cannot be honoured is a typed error.
4. **Use subagents for isolated work** — research, verification, analysis, implementation. Each gets
   its own context. When two or more return, consolidate: agreements, conflicts, key numbers
   (500–1500 tokens). Run them in parallel where possible; use a separate adversarial agent to check
   quality. An agent's finding is a hypothesis until the cited line has been opened.
5. Lessons go to `.claude/LESSONS.md` (≤ 300 lines: headline, rule, one pointer), citations to
   `.claude/CITATIONS.md`, rulings to `.claude/DECISIONS.md` §3 (one line each).
6. **Always `uv`** for Python — never pip. Stdlib-only scripts: `uv run --no-project python <script>`.
7. C++20 minimum. No naked `new` / `delete` in core.
8. **Write state to disk before the session ends**: run the `session-handoff` skill
   (`.claude/skills/session-handoff/`). Keep the newest two handoffs; delete older ones.
9. **No over-engineering.** Minimal edits. Simple > clever. A helper must delete at least two copies.
   If a workflow will repeat, make it a skill.
10. **A break of what v1.0.0 shipped needs a solid reason** (`DECISIONS.md` §2): silently wrong, unsound,
    blocks the cross-language contract, or provably unreachable. 2.0-born surface is pre-tag. People use
    this library.
11. Never `git push`, tag, publish, SSH out, submit to ARC, rewrite history, or delete an untracked
    `build*/`. Those are Volkan's actions. Each proven step is one local commit on `design-2.0`
    (approved 2026-09-28).

## Verification discipline

- Register the band before the run; FALSIFIED is a deliverable; two attempts, then record.
- "No-op" means digit-identical conformance output. Wall-clock is advisory; counters decide.
- A gate must prove its subject **ran** — CTest scores a skip as a pass unless told otherwise. CLI
  behaviour is proven by driving the real binary.
- Stash and re-run before calling a failure "pre-existing". Disagreement → a third computation.
- Numbers go to `.claude/baselines/` verbatim, tagged `[confirmed]` or `[inferred]`.
- Full test runs are serial evidence (`ctest -j1`). Every C++ test writes to its own scratch directory
  (`tests/support/scratch_directory.hpp`), so runs of different build trees may overlap; two runs of the same build
  tree may not (the `FIXTURE_ROOT` and `cmake -P` CLI tests keep one directory inside it).

## Build and gates (macOS; other platforms in `MAP.md` §3)

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp
cmake --build --preset clang-macos
ctest --test-dir build -C Release -j1 --output-on-failure
uv run --no-project python scripts/check_docs.py --cli build/bin/dtwc_cl
uv run --no-project python scripts/check_pins.py        # gitleaks runs in CI
uv run --no-project python scripts/generate_docs.py --check
```

ctest does not run `tests/python`. Any change that reaches a binding, a reader or a user-visible default also runs
the Python suite from a fresh venv outside the repo (`$V` a scratch directory):

```sh
uv venv $V/venv --python 3.12
CMAKE_ARGS="-DOpenMP_ROOT=/opt/homebrew/opt/libomp" uv pip install --python $V/venv/bin/python ".[test,dev,io]" matplotlib
cd $V && DTWC_CL_PATH=<repo>/build/bin/dtwc_cl $V/venv/bin/python -m pytest <repo>/tests/python -q -p no:cacheprovider
```

## Key files

- `.claude/cpp-style.md`, `.claude/python-style.md` — coding conventions.
- `docs/api-contract-2.0.md` — the 2.0 surface as written; it retires in phase G. What is frozen is
  `DECISIONS.md` §2 rule 1.
- `cmake/DtwcTest.cmake` — `dtwc_add_test`: a test passes on Catch2's summary with ≥ 1 assertion in ≥ 1 case,
  no failure, and no skip unless registered `MAY_SKIP`.
- `.claude/commands/` — the user-facing slash commands shipped with the repo.

## PR checklist

- [ ] Tests added or updated — each pins a named contract against an independent oracle
- [ ] CHANGELOG line if the change is user-visible against v1.0.0
- [ ] Docs updated (if user-facing)
- [ ] Optional deps remain optional
- [ ] `check_docs.py` and `check_pins.py` green; a new gate shown to bite

## Guidelines

- Buffer > thread_local >> heap allocation: already enforced everywhere.
- Contiguous arrays in hot paths; `Data::p_vec` as `vector<vector<data_t>>` is correct for
  variable-length series.
- Algorithms work on indices and a distance oracle; series stay where they are.
