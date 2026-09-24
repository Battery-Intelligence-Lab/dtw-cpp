# Handoff — 2026-09-23 — YAGNI pass, then GT-1…4a, FX-6/10/11/13, PF-6

## Base

Branch `design-2.0`, HEAD `e784e5c`; **nothing committed** (Volkan commits). Working tree: 70 modified, 12 deleted,
10 untracked paths (+2,232 / −8,468). Two agent worktrees remain under `.claude/worktrees/` (gitignored, merged,
removable with `git worktree remove`). Scratch evidence: the session scratchpad (`eigen/`, patches) — not durable.

## Done (all uncommitted)

- Plan: `PLAN.md` rewritten (75.9 KB → ~33 KB) around the K1–K4 test; `design.md` §12 A19–A25; `CHARTER.md` holds
  Volkan's two 2026-09-23 messages verbatim; `DECISIONS.md` log entries; A-10 guard reverted.
- Volkan decided: no Highway (PF-5 is plain C++ lanes with a 1.5× kill criterion); 1.x output filenames stay;
  the Eigen gap is investigated (PF-6), not accepted.
- GT-2: docs gate 2,391 → 1,854 lines, record pins gone; `check_record_hygiene.py`, the 223 KB plan archive and
  CHANGELOG's 127 KB history deleted. GT-1: pass = ≥ 1 assertion in ≥ 1 case, no failure, no skip unless
  `MAY_SKIP`; floors deleted. GT-3: F22 apparatus (3,804 lines) → `test_deprecated_shims_warn`; T-01 deleted.
  GT-4a: manifest count and the `check_ipo_inlining` test removed.
- FX-13: LB_Keogh admissible on CPU and Metal (CUDA blind → V-10); F12's Metal half verified on device.
- FX-6/10/11: fast_float 8.3.0 vendored; reader fixes; empty series rejected; macOS target 13.3; `--dist-matrix`
  strict; Python BOM and HPC precision fixed. PF-6 closed: no Eigen gap (quiet re-run within ±1 %). PF-7: 64-byte
  loop alignment FALSIFIED both ways; the tree loses up to 30 % on banded DTW to a split `fcmp`/`fcsel` pair. GT-7 done.
- CLAUDE.md runbook (gate line, rule 5, Key files) and two skills updated; LESSONS: two entries.

## Verified by me

- `std::from_chars(double)` compiles only at `-mmacosx-version-min=26.0`; `to_chars` at 13.3; `dtwc_cl` `minos 26.0`.
- Cited lines opened for every agent K3 claim that entered the plan (GPU LB, Metal FP64→FP32, view-mode asserts,
  MPI int loop, CLARANS, writers, Arrow cast, hierarchical NaN, `_hpc.py`, `Dataset.m`, `DataLoader.hpp:452-466`).
- Merged tree: `cmake --build --preset clang-macos` exit 0; `ctest --test-dir build -C Release -j1` **131/131 pass,
  2 CUDA `MAY_SKIP` skips**; `check_repo_hygiene`, `check_docs_contract`, `check_supply_chain_pins` PASS.
- The 124 build warnings are one pre-existing driver warning (`-fno-signaling-nans` on Clang) → GT-7.
- 2026-09-24, after merging loud + pkg + device: I read every library hunk of the three patches; the loud reviewer's
  "High" (CLARANS `max_neighbor = 0`) was against a stale diff — the merged code keeps 0 legal and F13 passes. Build 0
  warnings; `ctest -j1` 134 / 134 (2 CUDA skips); gates PASS; `repo_map.py layers` 17 upward edges, unchanged.

## Reported by agents, unverified by me

- PF-6 A/B numbers and the placement cliff (8/8 split placements slow, 28–46 % on relink), the ~2.8× SoA figure
  (from LESSONS), fast_float bit-identity (9.7M values) and 491 → 775 MB/s, FX-13 before/after counts, the harness
  and probe mutation proofs, Homebrew libomp `minos 26.0` (FX-14).

## Decisions awaiting Volkan

`PLAN.md` §6: D-E (Metal narrow-envelope error on every route), D-11, D-13, D-14, D-4, D-6; the rewrite itself and
`design.md` §12. Kept against Fable's advice: `mip/index_guard.hpp`.

## Next steps

1. **Merged and verified 2026-09-24 16:00 (uncommitted):** FX-1 (◐), FX-3, FX-4, FX-5, FX-14, FX-15, FX-16, FX-17, FX-18 ☑; GT-4 (◐, GT-4b
   left); IF-1 ☑; IF-2 S1 + S2 ☑ (◐ overall); IF-3 and RL-2 half. Main tree: build 0 warnings; **ctest -j1 140 / 140** (2 CUDA skips);
   **pytest 1,231 pass, 0 fail**, 11 skip (fresh venv, no `DYLD_*`); gates PASS; upward edges 17.
2. **MATLAB R2026a is installed** (`/Applications/MATLAB_R2026a.app`, not on `PATH`) — PLAN §2.1 corrected; V-12 / V-14 / V-15 can run here.
3. **Running (snapshots `scratchpad/sync5/`, `sync6/`):** matlab (build the MEX, run `matlab_suite`, fix the MATLAB layer →
   `sync5/matlab.patch`); S3 (`run(Config)`, the CLI rewired onto `bind` + `run`, `cluster()` as a wrapper, `--print-config`, the method ×
   device table, DOC-1's contract text → `sync6/s3.patch`); GT-4b (untyped filesystem / llfio leaks, one type for "made for other data",
   `skip_cols`, PDLP's GPU fallback → `sync6/gt4b.patch`); FX-19 (`distance::dtw` drops `metric` for five variants; two broken examples →
   `sync6/fx19.patch`). Then S4 (bindings + hpc by config file), FX-2, IF-3's `run` half, IF-4…IF-8.
4. Timing (GT-6, PF-7 with plain `fmin` now that FX-15 landed) waits for Low Power Mode off — still `powermode 1` on AC.
5. Volkan: review and commit (suggested groups in PLAN order: docs/plan; GT-1…4; FX-13; FX-6/10/11 + GT-7; FX-3/4; FX-5/14/16; IF-1 / FX-1 /
   FX-15; IF-3 grammar + RL-2; FX-17; IF-2 S1 + S2 + FX-18). Commit `.claude/plans/2026-09-23-fx6-reader-prototype/` without its two
   zero-byte fixtures or `check_repo_hygiene` fails. Agent worktrees under `.claude/worktrees/` are removable. GT-5 needs a push. The IF-2
   design's decisions (§10) and today's DECISIONS entries are the main session's — overturnable.

## Open questions

- Do CI's macOS runners have floating-point `from_chars` at all? CI has never run on this branch (GT-5).
- Answered 2026-09-24: a built wheel does **not** repair or import without `DYLD_LIBRARY_PATH` (libhighs) → FX-16.

## Status honesty

macOS / Apple clang only. CUDA, GCC, MSVC, MATLAB, MPI, Arrow and Linux were not built or run; the CUDA half of
FX-13 and the GCC/MSVC probe branches are unverified. Timing numbers are laptop wall-clock (advisory).
