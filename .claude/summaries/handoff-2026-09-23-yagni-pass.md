# Handoff — 2026-09-23 — YAGNI pass, then GT-1…4a, FX-6/10/11/13, PF-6

## Base

Branch `design-2.0`. **Volkan committed `899bb65` "test improvement" on 2026-09-24 16:58** (302 files: everything merged before IF-2
S3). Uncommitted on top of it: IF-2 S3 (`run(Config)`, the CLI rewired, Tier-1 as a wrapper, DOC-1), the hygiene allowlist for the
FX-6 prototype's two zero-byte fixtures (they went into `899bb65`), and record updates. Agent worktrees must now sync from `899bb65`
(`git diff --binary HEAD`), not `e784e5c`. Agent worktrees under `.claude/worktrees/` are merged and removable.

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

1. **Verified 2026-09-24 ~18:00 on `899bb65` + S3:** build 0 warnings; **ctest -j1 142 / 142** (2 CUDA skips); **pytest 1,245 pass, 0 fail**,
   11 skip; **`matlab_suite` passes** (run with the default `TMPDIR` — see LESSONS); gates PASS; upward edges 17. IF-2 S1–S3 done;
   FX-1…FX-19 mostly ☑ (see PLAN); GT-4 / GT-4b ◐ (small residue); V-12 / V-14 / V-15 ☑ (MATLAB R2026a runs here).
2. **Running (snapshot `scratchpad/sync7/`, base `899bb65`):** S4a (Python: one `run(Config)` call — IF-3's deletion of `_api.py`'s dispatch — hpc submitting a config file,
   `Problem.metric`, checkpoint defaults from the Problem's metric, DDTW's `metric` argument); S4b (MATLAB: the same through the MEX, its
   seven parsers → the `Name<E>` tables); FX-2 (Soft-DTW self-distance with derivation D7).
3. Timing (GT-6, PF-7 with plain `fmin` now that FX-15 landed) waits for Low Power Mode off — still `powermode 1` on AC.
4. Volkan: commit S3 and the record updates when ready; 35 `~/matlab_crash_dump.*` files (all from 2026-09-24 agent runs) can be
   deleted; D-15 (MATLAB's thread setting governs the MEX) and the IF-2 §10 decisions are the main session's, overturnable. GT-5 needs a push.

## Open questions

- Do CI's macOS runners have floating-point `from_chars` at all? CI has never run on this branch (GT-5).
- Answered 2026-09-24: a built wheel does **not** repair or import without `DYLD_LIBRARY_PATH` (libhighs) → FX-16.

## Status honesty

macOS / Apple clang only. CUDA, GCC, MSVC, MATLAB, MPI, Arrow and Linux were not built or run; the CUDA half of
FX-13 and the GCC/MSVC probe branches are unverified. Timing numbers are laptop wall-clock (advisory).
