---
name: session-handoff
description: Write the end-of-session handoff for DTWC++ — what was done, what was verified, decisions, next steps, open questions — to .claude/summaries/, and update PLAN.md status. Use before ending any working session on this repository, or when the user says "handoff", "wrap up" or "save state".
---

# Session handoff

The next session starts with no memory of this one. The handoff is the only bridge, so it states
facts with their evidence and never inflates them.

## Steps

1. **Collect the facts** (do not reconstruct them from memory):
   - `git status --short` and `git log --oneline <base>..HEAD` — what changed, what is committed,
     what is staged or untracked.
   - Gate results you actually ran this session, with the command and the numbers. If a suite was
     not run, say so.
2. **Write** `.claude/summaries/handoff-YYYY-MM-DD-<topic>.md` (absolute date; one file per session;
   80 lines at most) with exactly these sections:
   - `Base` — branch, HEAD, tree state.
   - `Done` — one line per outcome, each with its commit or path.
   - `Verified by me` — claims you confirmed by opening the line or running the command.
   - `Reported by agents, unverified` — keep these apart; an agent finding is a hypothesis.
   - `Decisions` — taken by Volkan this session (quote him) versus proposed and awaiting him.
   - `Next steps` — ordered, each naming its `PLAN.md` phase and step.
   - `Open questions`.
   - `Status honesty` — what was built, run and tested, and what was not.
3. **Update `PLAN.md`**: the step marks (☐ ◐ ☑) and commit hashes, and nothing else. New rulings go to
   `DECISIONS.md` §3 as one line; measurements go to `.claude/baselines/`.
4. **Prune**: keep the newest two handoffs and delete the rest with `git rm`; history keeps them.
   Code, tests and docs never cite a handoff: a comment states its reason itself.
5. **Lessons**: if the session taught something durable, append one entry to `.claude/LESSONS.md`
   (headline, rule, one pointer). Keep it short; no gate pins the text.

## Rules

- Never write "tests pass" without the command and the count. Never call a failure "pre-existing"
  without having stashed and re-run.
- Correct earlier handoffs when they were wrong, by name, in `Verified by me`.
- Do not commit or push unless Volkan asked.
