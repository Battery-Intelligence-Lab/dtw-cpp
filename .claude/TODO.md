# DTWC++ — open items outside the campaign plan

Contributors: this is the list `CONTRIBUTING.md` points to. The 2.0 campaign itself (waves, gates,
decisions) is in `PLAN.md`; the project brief is in `CHARTER.md`. Rows closed before 2026-07-23 were
removed on 2026-09-21; their reconciliation record is
`.claude/baselines/2026-07-23-r1-todo-reconciliation.md` and the full table is in git history
(`9c08074`).

## Features wanted (2.1 unless pulled forward — see `PLAN.md` W10)

| ID | Item |
| --- | --- |
| S03 | Streaming CLARA: resume of the assignment state. |
| G02 | Wire the existing CUDA K-vs-all kernel into production streaming CLARA. |
| G05 | Multi-stream CUDA pipeline; the production path is one serialised stream. |
| A01 | Two-phase within-group / cross-group clustering — only after a quality / complexity band is registered. |
| A02 | A measured resource / cost model for device-aware method selection (device-aware selection itself is done, `4e59d05`). |
| TS01 | SBD distance and k-Shape (Paparrizos & Gravano, SIGMOD 2015): the widely used shift-invariant, matrix-free competitor; naive O(n²) cross-correlation, no FFT dependency. |
| TS02 | NN-chain hierarchical clustering (Murtagh 1983; Müllner 2011): O(N²) single / complete / average linkage, retiring the `max_points = 2000` guard. |
| TS03 | Adjusted mutual information (Vinh, Epps & Bailey, JMLR 2010) beside ARI / NMI. |

## Operator-owned (Volkan)

| ID | Item |
| --- | --- |
| B01 | First PyPI release through the trusted publisher; the workflow and dry-run are ready. |
| BLK01 | A real Oxford ARC submit / poll / download run. The local chain is covered; agents must not submit. |
| G01, G03 | H100 validation: kernel-level sm_90 bands and the 80 GB Float32 path are advisory until a real run exists. |

## Upstream, community, or explicit non-goals

| ID | Item |
| --- | --- |
| PL02 | Arrow through CPM on Windows + Clang — blocked upstream; the system-Arrow route is separate and works. |
| PL03 | Arrow through CPM on Windows + MSVC — untested; no "should work" claim. |
| DEF01 | DDTW recurrence fusion — non-goal unless profiling reopens it. |
| DEF03 | Replacing the Arrow C++ file readers with nanoarrow — non-goal for 2.x (C-Data ingest is done, `3f827c2`). |
| DEF04 | HIP backend — post-2.0, community-owned. |
| UP01 | quickcpplib generator-forwarding issue — post-2.0; no upstream issue filed yet. |

Tracked in `PLAN.md` instead: H10 → F11 / F36 (supply-chain pins), PL04 → F38 (preset metadata guard).

## Closed rows that other documents cite

| ID | State | Record |
| --- | --- | --- |
| M01 | DONE | **CLOSED-BY-REJECTED `af67486`:** the odd-cycle band was falsified; the recorded LR / B&B chain remains binding. |
