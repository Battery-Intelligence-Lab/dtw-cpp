# DTWC++ — Claude Runbook

## Non-negotiables

1. No runtime dependence on repo-relative paths.
2. Every PR: update CHANGELOG.md, add/adjust tests, keep lint clean.
3. Optional dependencies only (OpenMP, HiGHS, CUDA). Core must build without them.
4. **Use subagents for isolated work.** Spawn subagents for research, verification, and analysis. Each gets its own context window. When 2+ subagents return results, consolidate: identify agreements, flag conflicts, extract key data (500-1500 tokens max). Use parallel agents where possible. Separate adversarial agents to check quality.
5. Add lessons to `.claude/LESSONS.md`, citations to `.claude/CITATIONS.md`.
6. **Always use `uv`** for Python — never pip.
7. C++20 minimum. No naked new/delete in core.
8. **Write state to disk before session end.** Use `/session-handoff` to write accomplishments, decisions, next steps, and open questions to `.claude/summaries/`. This is critical for continuity across sessions.
9. **No over-engineering.** Minimal edits. Simple > clever. Reuse existing skills. If a workflow will repeat, create a skill.
10. Don't randomly read files, read only relevant ones. 

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

## Guidelines
- Buffer > thread_local >> heap allocation: already enforced everywhere
- No naked `new`/`delete` in core: already enforced
- Contiguous arrays in hot paths: `Data::p_vec` as `vector<vector<data_t>>` is correct for variable-length series