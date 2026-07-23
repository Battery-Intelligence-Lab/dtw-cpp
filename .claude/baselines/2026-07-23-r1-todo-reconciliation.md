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

