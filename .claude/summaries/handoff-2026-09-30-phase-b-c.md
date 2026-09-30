# Handoff — 2026-09-30 — phase B nearly closed, phase C under way (Windows)

## Base

Branch `design-2.0`, HEAD after this file's commit (on `218127d`). Session base `a073113`; first commit `cf0cf4a`.
Interim version (updated 09-30 evening): W11a (index_t, Opus) and C1 (long-series CUDA kernel, Opus) were running.
Agent briefs and inventories: session scratchpad `…/79bd8991-b0e9-441b-bc55-95584dcfaf55/scratchpad/`
(`brief_*.md`, `gates_*.md`, `y4_inventory.md`, `w11_inventory.md`, `brief_w11a.md`, `brief_w12a.md`, `brief_b1.md`).

## Done (merged into design-2.0, one gated merge each)

- X2 `4de2ce9` (W3a–g): Benders, PDLP, CLARANS, the OneBatch weightings go; `decode_assignment` + `set_result`.
- P3 `5e15459`: unbanded per-pair Standard DTW runs `dtwFull_L`; EAPruned deleted (−573/+15); no CHANGELOG line.
- Y4 `93eefe1`: the MEX compiles again; a clang MEX on Windows links MATLAB's `libiomp5md`; F43 did not reproduce.
- G1 `1bb9413`: `test_run_resolution` asserts `SolverError` for `mip` without HiGHS (LR-core needs no solver).
- R1 `1ea8327`: no critical, atomic or mutex in `dtwc/algorithms`; `run_openmp` rethrows the lowest-index failure.
- W6e `fe1bf9b` (Python) and W6m `e053594` (MATLAB): never-released aliases go; exact MEX integers; labels are
  values; `dist_by_ind` bounds-checked at both language boundaries.
- F1 `b5c7048`: every C++ test writes to its own scratch directory; `ctest -j 8` passes.
- W4d `febd25f`: `KernelOverride` gone; FP32 L 4095–4096 fixed; shared memory opened once per device (a
  two-thread race, now tested) and checked before any allocation; typed stubs; empty series → `InvalidInput`.
- B1 `4968d44`, B2 `e0085fd`, B3 `65edcf7`: the bindings check indices; outputs stay empty until a clustering; one
  `require_clustered`; `set_data` clears it; `fast_pam` `max_iter = 0` is BUILD only everywhere (crashes before).
- X3b `6e346a7`: validator tails gone; `-Werror=switch` and MSVC C4062 both bite. H1 `1a079cd`: msvc preset, docs tool.
- W13a `62d5822`: GPU writes the packed matrix; int64 chunks (N > 65,536); FP32 L 100 fill 0.756×; host memory −74 %.
- Records: DECISIONS §3 (09-30 rulings), LESSONS (+3), MAP, runbook (overlapping test runs), PLAN marks.

## Verified by me

- X2's sync merge kept its `cli/run.cpp` edits; every X2-deleted name has the same count as its pre-sync tip.
- Integrator logs for X2: ctest 123 / 0 failed; `ctest -N` diff = the 4 registered removals + 1 addition;
  pytest 1116 / 19 / 0. Read P3's routing diff, R1's `run_openmp` (the per-slot minimum is the global lowest
  failing index under any schedule), Y4's OpenMP CMake branch.
- `stress_test_cli.sh`'s failing `p1_pam_standard_band5` is a stale expectation: the binary refuses with the typed
  "smallest feasible band is 4257".
- Workflow agents ran on `claude-sonnet-5-5` (the Y4 inventory's progress record).

## Reported by agents, unverified

- Last green (B3 merge `1cc4fcc`): clang ctest 122 = 119 + 3 MAY_SKIP; pytest 1172 / 19 / 0; `matlab_suite` 147 run /
  146 passed / 1 allowed incomplete; CUDA tree 121 / 0 failed (W13a merge; llfio-ON tree 7145 assertions);
  conformance digit-identical at every merge.
- R1: TSan with LLVM libomp 18 + Archer (WSL, user space, `~/tsan`): 0 reports at base and head; controls bite.
- W4d: band within 1 % on a quiet machine (cv ≤ 0.6 %); SASS byte-identical.

## Decisions

- Volkan 09-30: "Please continue but don't call fable, for delegating simpler tasks use Sonnet 5.5 xhigh".
  Later: "It is fine, we will never cluster as many as Nb^2 more than int64"; "You are overthiking about simple
  things. Just keep the code simple, we think about it when it overflows" (CHARTER).
  Sonnet units run as one-agent workflows (`model: sonnet`, `effort: xhigh`); harder ones on the default model.
- Awaiting Volkan: (1) clang-cl Windows wheels (libomp shipped); (2) drop `/fp:contract` from MSVC `fast`;
  (3) VERSIONINFO for `dtwc_cl.exe`; (4) NEW — the permission system refused W12a's `git rm` of tests ("Security
  Test Removal"); W12a's deletions (~3,700 lines, each with its kept oracle named) and W6f's trims need his OK.

## Next steps (PLAN)

1. B: W12a's additive half (`pb/W12a`, 3 commits) with or without its deletions (Volkan's OK needed).
2. C: C1 (global-memory wavefront for long series; carveout; preload wavefront — each behind its band), then W4e
   (macOS CI).
3. D: W11a (in flight) → W11b (Python `np.int64`, MATLAB double labels, drop `_CPP_INT_MAX`).
4. B/F: W6f after X3b (permission); W12b–d; the tracker-id comment sweep when no code unit is in flight.

## Open questions

- Metal edits (W4d, W13a) are compiled blind here; macOS CI after a push is their gate.
- Do ld.lld / ld64 LTO keep the lanes' SLP packing, and do llfio-ON wheels build on manylinux / macOS?

## Status honesty

Windows only (clang, MSVC for the MEX/CUDA). Not run: macOS/Metal, Linux, the full MATLAB suite on the MSVC MEX
with HiGHS. Timing under load is [inferred]; W4d's final band ran on a quiet machine.
