# DTWC++ — Claude Runbook

## Non-negotiables

1. No runtime dependence on repo-relative paths.
2. Every PR: update CHANGELOG.md, add/adjust tests, keep lint clean.
3. Optional dependencies only (OpenMP, HiGHS, CUDA). Core must build without them.
4. Use parallel agents where possible. Separate adversarial agents to check quality.
5. Add lessons to `.claude/LESSONS.md`, citations to `.claude/CITATIONS.md`.
6. **Always use `uv`** for Python — never pip.
7. C++17 minimum. No naked new/delete in core.

## Key files

- `.claude/design.md` — architecture, layered design, DTW variant rules, binding strategy
- `.claude/LESSONS.md` — critical gotchas to avoid repeating
- `.claude/TODO.md` — remaining work
- `.claude/cpp-style.md` / `.claude/python-style.md` — coding conventions

## PR checklist

- [ ] Tests added/updated
- [ ] CHANGELOG.md updated (Unreleased)
- [ ] Docs updated (if user-facing)
- [ ] Optional deps remain optional
