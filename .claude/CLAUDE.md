# DTWC++ — Claude Runbook

## Read in this order, and no further

0. The newest `.claude/summaries/handoff-*.md` — where the last session stopped.
1. `.claude/CHARTER.md` — what Volkan asked for, verbatim.
2. `.claude/MAP.md` — what is where, today. **Read this instead of the tree.**
3. `.claude/design.md` — the target architecture, the compatibility policy, the invariants.
4. `.claude/PLAN.md` — your wave's card only; then the ledger rows it names
   (`.claude/specs/2026-09-07-diff-ledger.md`) and the deep-dive section a row cites.

Context is a budget. A wave card lists its *context pack*: the only source files to open before
starting. Never bulk-read `reports/`, `baselines/`, `LESSONS.md`, `CHANGELOG.md` or a `PLAN-archive-*`;
grep them for the symbol or row you need. `DECISIONS.md` holds the killed ideas and standing rules.

## Non-negotiables

1. No runtime dependence on repo-relative paths. Never put the repo root on an include path (on a
   case-insensitive filesystem the root `VERSION` file shadows `<version>`).
2. Every PR: update CHANGELOG.md (Unreleased), add or adjust tests, keep lint clean.
3. Optional dependencies only (OpenMP, HiGHS, Gurobi, CUDA, Metal, MPI, llfio, Arrow, YAML). The core
   builds without them. No silent fallback, ever: a request that cannot be honoured is a typed error.
4. **Use subagents for isolated work** — research, verification, analysis, implementation. Each gets
   its own context. When two or more return, consolidate: agreements, conflicts, key numbers
   (500–1500 tokens). Run them in parallel where possible; use a separate adversarial agent to check
   quality. An agent's finding is a hypothesis until the cited line has been opened.
5. Lessons go to `.claude/LESSONS.md`, citations to `.claude/CITATIONS.md`. Both are pinned by gate
   scripts: append, do not reword existing entries.
6. **Always `uv`** for Python — never pip. Stdlib-only scripts: `uv run --no-project python <script>`.
7. C++20 minimum. No naked `new` / `delete` in core.
8. **Write state to disk before the session ends**: run the `session-handoff` skill
   (`.claude/skills/session-handoff/`). Keep the last few handoffs; delete older ones.
9. **No over-engineering.** Minimal edits. Simple > clever. A helper must delete at least two copies.
   If a workflow will repeat, make it a skill.
10. **A break needs a solid reason** (`design.md` §2): silently wrong, unsound, blocks the
    cross-language contract, or provably unreachable. Additive first. People use this library.
11. Never `git push`, tag, publish, SSH out, submit to ARC, rewrite history, or delete an untracked
    `build*/`. Commit only when asked. Those are Volkan's actions.

## Verification discipline

- Register the band before the run; FALSIFIED is a deliverable; two attempts, then record.
- "No-op" means digit-identical conformance output. Wall-clock is advisory; counters decide.
- A gate must prove its subject **ran** — CTest scores a skip as a pass unless told otherwise. CLI
  behaviour is proven by driving the real binary.
- Stash and re-run before calling a failure "pre-existing". Disagreement → a third computation.
- Numbers go to `.claude/baselines/` verbatim, tagged `[confirmed]` or `[inferred]`.
- Full test runs are serial evidence (`ctest -j1`): concurrent runs collide on test artefacts.

## Build and gates (macOS; other platforms in `PLAN.md` §2.1)

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp
cmake --build --preset clang-macos
ctest --test-dir build -C Release -j1 --output-on-failure
uv run --no-project python scripts/repo_map.py layers     # upward-edge ratchet
python3 scripts/check_record_hygiene.py; python3 scripts/check_repo_hygiene.py; python3 scripts/check_docs_contract.py
python3 scripts/check_supply_chain_pins.py   # the fourth gate — omitting it here left it red for weeks (X-33)
```

## Key files

- `.claude/cpp-style.md`, `.claude/python-style.md` — coding conventions.
- `docs/api-contract-2.0.md` — the frozen 2.0 surface; a change needs a dated entry in `DECISIONS.md`.
- `tests/floors.cmake` — generated pass floors (`scripts/measure_test_floors.py`).
- `.claude/commands/` — the user-facing slash commands shipped with the repo.

## PR checklist

- [ ] Tests added or updated — each pins a named contract against an independent oracle
- [ ] CHANGELOG.md updated (Unreleased)
- [ ] Docs updated (if user-facing)
- [ ] Optional deps remain optional
- [ ] Upward-edge count not higher; gate scripts green

## Guidelines

- Buffer > thread_local >> heap allocation: already enforced everywhere.
- Contiguous arrays in hot paths; `Data::p_vec` as `vector<vector<data_t>>` is correct for
  variable-length series.
- Algorithms work on indices and a distance oracle; series stay where they are.
