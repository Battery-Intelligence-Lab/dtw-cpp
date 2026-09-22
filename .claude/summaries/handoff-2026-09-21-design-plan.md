# Handoff — 2026-09-21 — Design 2.0: map, design, plan, cleanup (new Mac)

## Base

Branch `design-2.0`, HEAD `9c08074`. **Nothing committed**: 136 deletions and one rename are staged
(`git rm` / `git mv`), 6 files modified, 7 paths untracked. `AGENTS.md` was already deleted in the working
tree when the session began (not by this session). No library code (`dtwc/`, `tests/`, `bindings/`,
`python/`) was touched.

## Done

- Toolchain on the new Mac: `brew install cmake ninja doxygen graphviz libomp` (Apple clang 21 was present).
- `.claude/CHARTER.md` — Volkan's instructions verbatim (today's two messages + the brief moved from `TODO.md`).
- `.claude/MAP.md` — the as-is map (317 lines): layer cards, flows with anchors, mechanical facts, build, CI,
  tests, deep-dive caveats. `scripts/repo_map.py` regenerates the mechanical half (`layers`, `symbols`).
- `.claude/design.md` — rewritten as the target architecture; §11 lists ten amendments to the 09-07 spec.
- `.claude/PLAN.md` — new plan: machine dependencies, wave graph, row-level ordering, wave cards with context
  packs, carried-over R-tracks, new rows S-01…S-13 / X-01…X-10 / H-1…H-6, break register, 11 decisions.
- `.claude/DECISIONS.md` — killed ideas, standing rules, the 09-07 outcomes, old-plan-vs-spec conflicts, log.
- `.claude/CLAUDE.md` rewritten (reading order, context budget, rules absorbed from `AGENTS.md`);
  `.claude/skills/session-handoff/SKILL.md` created (the runbook referenced a skill that did not exist).
- Root `PLAN.md` → `.claude/PLAN-archive-2026-09-21-research-release-campaign.md` (verbatim + archive note);
  the three path reads in `check_docs_contract.py` / `check_record_hygiene.py` repointed, assertions unchanged.
- Removed: 105 `.claude` records (5 plan archives, 40 baselines, 25 reports, 35 handoffs), 4 unloadable flat
  skill files (+ their two entries in the docs gate's scan list), 12 one-off evidence scripts + fixture,
  5 generated plots, 3 one-off LaTeX artefacts, 5 unused figures, `.cmake-format.yaml`. `.claude/` went from
  180 tracked files / 3.6 MB to 78 files / 2.3 MB; the session-start reading set is ~83 KB.
- `.claude/TODO.md` trimmed to open items (row M01 kept: a docs source cites it). CHANGELOG line added.
- `.claude/baselines/2026-09-21-macos-first-baseline.md` — first macOS build/test baseline + codegen probe.

## Verified by me (command run or line opened)

- Build: `clang-macos` preset + `-DOpenMP_ROOT=/opt/homebrew/opt/libomp`, 434 targets, 2 min 31 s, OpenMP,
  HiGHS, llfio, Metal ON. Serial `ctest`: **126/131 pass, 2 CUDA skips, 5 fail — all five print "All tests
  passed" and fail only the W0 floor regex** (floors measured on Windows; case counts match, assertion counts
  do not).
- `check_docs_contract.py` **fails at HEAD** (`D2/D3 CTest drift`): it pins the registration blocks `d21ffee`
  replaced. Per-assertion run before and after this session's cleanup: identical (14/16).
  `check_record_hygiene.py` and `check_repo_hygiene.py` pass before and after.
- Include graph against the target layers: 18 upward edges, 17 into `Problem.hpp`, 1
  `system_memory.cpp → DataLoader.hpp`; no cycles. All four `io → session` includes are `→ Data.hpp`.
- Doxygen XML: `Problem` has 175 members (105 public functions); three "where it runs" vocabularies and three
  precision vocabularies exist.
- Codegen probe under the real Release flags: `lb_keogh` and `z_normalize` vectorise; `compute_envelopes` and
  the DTW row recurrence do not (compiler reasons recorded).
- Last release tag is `v1.0.0`; `VERSION` is `2.0.0rc1`. Pushes to `design-2.0` trigger no workflow.
- Anchors for S-01 (`Problem.cpp:685`), S-04 (`dtwc_cl.cpp:1584, :1845`), S-13 (`Problem.cpp:151`),
  `configure_device` (`api.cpp:94`).
- **Correction to `handoff-2026-09-21-multiagent-sweep.md`:** the III.10 DECIDE rows are *not* open; the spec
  header and III.10 record all of them as adopted on 2026-09-07.

## Adversarial review of the four documents (done, fixes applied)

A separate agent checked coverage (129/129 ledger rows now named or covered), 40 anchors, the numbers, the
mermaid graph and spec conflicts, and returned 14 defects. All fixed: the W3 layer gate was unreachable
(mip's six edges need O-16 and a new X-11 in W4 — gate is now 15 → 6, W4 takes it to 0); D-4 must edit
`check_docs_contract.py:2070-2077` in the same commit (the CI gate reads the "private" report folder);
`check_ipo_inlining.py` does not exist yet (design.md said it did); `DtwcError` is `dtwc::Error`; D7 is a
hard dependency of W2 (graph + ordering updated); `settings.hpp` cannot drop `<random>` until `randGenerator`
moves (new X-12); D-8 reworded (merge target stays `Claude`; a PR to `main` is CI only); O-09 / C-09 are
"adopted, re-opened by A6", not "open"; `repo_map.py` now resolves bare includes and knows `dtwc/base/`.

## Four-lens orthogonal review (2026-09-22, Volkan's request; applied)

Four parallel read-only reviewers — Python API (van Rossum lens), C++ language (Stroustrup lens), C++
software design (Iglberger lens), time-series science — returned 49 findings. Consolidated into
`design.md` A11–A18, rows S-14…S-20 / X-13…X-22, decisions D-12…D-16, three TODO rows. Every finding that
entered the plan was re-opened at its cited line by me; the ones that changed the plan most:

- **Python Tier-1 re-implements C++** (`_api.py:461-500`; `dtwc::cluster` is not bound) while MATLAB makes one
  call (`dtwc_mex.cpp:1488`) → W9's first row: `cluster()` = `run(Config)`.
- **Release history**: v1.0.0 has no MATLAB file and only a pybind `Problem` wrapper with 1.x names → a
  PRE-TAG class in design.md §2; 2.0-only binding names are fixable at no user cost until the tag (D-11).
- **Execution leaks into the cache identity** (`Problem.cpp:367-369, 397-399`) → `DistanceConfig` semantics
  only (A11).
- **Infeasible band returns a finite `max()`** that passes `isfinite` guards (`warping.hpp:403`,
  `medoid_assignment_policy.hpp:43`) → R1 break S-15 in X-02, decision D-12.
- **FP model**: `-fassociative-math` at directory scope makes every reduction width-dependent → X-15
  `DTWC_FP_MODEL=strict|fast`, conformance pinned under `strict`.
- **Oracle shape** reconciled between two lenses: one concrete `PackedOracle` (dense and mmap both read
  `data_[tri_index]`) for filled-matrix algorithms; the concept only for lazy consumers and fakes; `row(i)`
  as a gather (no contiguous row in a packed triangle).
- **Scores allocate N²** (`scores.cpp:122`) → X-20 with D18; **`dtw_path` promoted to 2.0** (X-17; the backtrack
  exists in `barycenter.cpp:184-224`); interop tokens `euclidean` / `window_fraction`; D19 (missing data).
- Rejected or corrected: "Python entirely 2.0-only" (1.x pybind names keep shims), "MSM/TWE not exported"
  (reachable via Tier-2 `variant_params`; missing only from the pairwise surface → S-14),
  "`dtwc.test.parallelisation()` missing" (exists), "hide `Pruned` pre-tag" (C-05 stands → D-16).

## Reported by agents, not re-opened by me

The layer cards in `MAP.md` (agents spot-checked 6–18 claims each against source and corrected several report
errors); the remaining S-rows (S-02, S-03, S-05…S-12); the R-track status table (R2 3/18, 33 open findings);
the README / docs-deploy defects listed under W8; the claim that the Pages upload step has no `path:`.

## Decisions

Taken by Volkan this session: none beyond the charter text. **Proposed and awaiting him:** `PLAN.md` §7
D-1…D-11 — most importantly D-1 (the design amendments), D-2 (keep the 1.x artefact filenames), D-4 (a report
headed "PRIVATE — do not push to GitHub" is tracked and present on `origin/Claude` and `origin/design-2.0`).

## Next steps

1. Volkan reviews `design.md` §11 and `PLAN.md` §7; then commit this session's changes (one `docs:` commit for
   records, one for the script edits is reasonable).
2. W0 repairs, in order: W0-R1 (portable floors), W0-R2 (docs gate pins), W0-R5 (CI on this branch), W0-R3, W0-R4.
3. W0 Task 2 with the corrected premise; then Tasks 3–13 with the amendments in the W0 card (Task 9 manifest
   from `repo_map.py`; Task 12 → codegen report; five canonical baselines in the exit gate).
4. One implementation plan per wave, written just before the wave (W1 next).

## Open questions

- Is `unordered_map` move-assignment under `noexcept` a real `std::terminate` risk on MSVC's STL (S-13)?
- S-01: true negative Soft-DTW diagonal or the Soft-DTW divergence — settle inside derivation D7.
- Some explicit test floors are said to sit below the measured table deliberately (build-flavour branches);
  read each site before W0-R3.
- Latency- vs memory-bound: still no PMU artefact (O-22).

## Status honesty

Built and tested once on macOS, Release, serial; nothing on Linux, Windows, CUDA, MPI, MATLAB or Python.
No benchmark was run. No library code changed, so there is no regression claim to make. The cleanup is
uncommitted and fully reversible (`git restore --staged --worktree .`), and every removed file exists at `9c08074`.
