# Handoff — 2026-10-05 (evening) — the Mac pass ran; macOS fixes landed; W4e merged

## Base

Branch `design-2.0` on the Mac, base `4dd4dcaf` (the morning handoff); nine commits since, HEAD = this file's
commit (2026-10-06, past midnight). Volkan pushed `f5c58764` at 00:16 and pushes this one; the Windows machine then
`git pull --no-rebase`. W9b was still unmerged (origin unchanged all day). Tree clean after this commit.

## Done (each a local commit on design-2.0, proven on the main tree before committing)

- `a332d671` lanes kernel: both 64-byte fills copy one constant row; Apple clang had made each a `memset_pattern16`
  call, which failed `test_codegen_no_calls` on macOS (`calls=2`). The only test failure of the pass.
- `6cb0c3c1` W4e (PLAN C): one pipeline helper, one wavefront MSL body, the dead pair-index plumbing gone, every Metal
  buffer and the autorelease pool owned by one RAII holder (six leaking exits closed; leak probe 191.8 → 0.0 MB).
- `bc9469fd` Metal: every chunk's command buffer is waited for and checked; a failed earlier chunk left its pairs at
  0 (W4e review finding, confirmed by reading the loop; CHANGELOG line covers both Metal fixes).
- `55911b2a` banded kernel: the per-column fills go (two `memset_pattern16` calls per column on macOS, invisible to
  the gate); the first in-band row is seeded from the constant. Faster at every band ≥ 2, slower at band 1 (table in
  the baseline).
- `51a9944a` the codegen probe's cost is the library's `std::abs(a - b)` (its listing showed four instructions where
  the shipped code has one `fabd`); this commit also carries the old handoff's deletion (index was staged).
- `98e986fc` the gate reads any loop of a probe kernel (the banded calls sat one loop out) and the probe gains the
  f32 per-pair kernels; bites on both old headers (12 and 8 calls), passes at HEAD (114 innermost loops, 0 calls).
- `822225bc` merge of `pb/banded-bounds-arith` (`a2d5e4e5`): the banded kernel's bounds vectors and `diag` guard
  removed (bit-identical, −25 lines). FALSIFIED on the Mac as a no-regression change (1.08–1.20× slower at bands
  5–12 without early abandon, faster with it; table in the baseline); Volkan merged it anyway (DECISIONS §3, 10-06):
  x86 timing on Windows decides. Band 1 explained (store→load handoff through `col`); its fix is a dead store, not shipped.
- Records: `baselines/2026-10-05-macos-design-2-0.md` (every number of the pass), PLAN (W4e ☑, the Mac entry),
  DECISIONS §3 (10-05 Mac line), LESSONS (the memset_pattern16 idiom), CHANGELOG (Metal fixes); MAP §7, DECISIONS
  (K1 line) and LESSONS now say the gate reads any loop.

## Verified by me (commands in the baseline)

- `-DDTWC_ENABLE_GUROBI=OFF` once (pre-W14b cache); 206 targets, 31 s, zero warnings; the 22 Metal commits compiled
  first time. `ctest -j1` at base 93/95 + 1 CUDA skip + the codegen failure; at HEAD 95/95 + 1 skip (four runs).
  Metal 2168/16 and 169/5 assertions, no SKIP, unchanged; GPU CLI outputs byte-identical to the base CLI, 4 routes.
- `cpp_conformance`: labels and medoids digit-identical; silhouette one ulp off the Windows reference
  (0.96894972764334841 → 0.9689497276433483); accepted under D-19, reference untouched.
- Gates green (check_docs 392 flags, check_pins, generate_docs); pytest 1102/11/0 from a fresh venv at base (skips: 9
  CUDA, 1 GPU-present, 1 scipy-present); MEX built, matlab_suite 140 run, 139 passed, 0 failed, 1 registered filter.
- arm64 codegen: the lanes DP loop is NEON-packed (`fabd.2d`, compare-and-select mins, `fadd.2d`, `stp q`); the
  loop-vectorise remark hides it (SLP after full unroll). The banded A/B timing table (quiet machine).
- Leak probe run by me on base, the agent's build and the merged library: 191.8 / 0.0 / 0.0 MB.
- ASan + UBSan (`build-asan/`, halt on error): 94/94 passed, 1 CUDA skip, no sanitizer report; the codegen gate
  excluded there by design. The first sanitizer run of design-2.0 on macOS.
- After the merge `822225bc`: gate PASS (108 innermost loops, 0 calls), ctest 95/95 + 1 skip, Metal counts unchanged,
  conformance the same ulp, GPU routes byte-identical, pytest 1102 passed / 11 skipped (wheel rebuilt there).

## Reported by agents, unverified

- Lanes agent: 38,928-value output hash identical before/after; x86-64 Darwin cross-compile had the same idiom.
- Banded agent: 30.3 M-call sweep identical and equal to an independent reference; deliberate breaks detected; the
  `diag` guard and the bounds vectors provably removable (unit A); the probe lacked an f32 banded kernel (unit B).
- Gate agent: 274 calls in the probe's dtwc functions, all outside loops; the `BLOCK` regex misses x86-64 Darwin's
  `##  %bb.N:`. Bounds agent: cycles per column via `proc_pid_rusage` (band 1: 21.6 with the old call, 33.5 without).

## Decisions

- Volkan (chat): "you can [use] 1-2 more agents"; "I enabled ultracode" — no session confirmation arrived, so no
  Workflow ran; two ordinary agents instead. Awaiting him: nothing blocking; the 1-ulp silhouette stays (D-19).

## Next steps

1. Windows: merge W9b; `git pull --no-rebase` design-2.0 (these commits); re-run the Windows clang tree, the CUDA
   tree and pytest (the kernels changed: lanes, banded; `test_codegen_no_calls` under the hardened rule).
2. Mac, when W9b reaches origin: pull, rebuild, pytest again (PLAN "Blocked on another machine").
3. E: W9e, M1, W9c, L2b; the second short Mac pass after L2b; W9f. G: W14a, WM, W14c.
4. Windows: time the merged bounds change (`822225bc` vs its parent) on x86 with a registered band; revert
   `a2d5e4e5` if it loses there too. Candidates: `-v` naming the Metal kernel; squared-L2, multivariate and Soft
   wrappers in the probe; Metal `context()` retains the device twice (harmless).

## Open questions

- None for Volkan beyond the silhouette re-record choice. 20 stale September worktrees sit under
  `.claude/worktrees/agent-*` (at e784e5c3/899bb653 with stray `.claude/*.md` edits): his call to delete.

## Status honesty

Mac only: `build/` (clang-macos Release), `build-matlab/`, `build-asan/` (at `98e986fc`), a fresh-venv wheel; pytest
at base, `98e986fc` and `822225bc`. Not run here: CUDA, Linux, CI, the release archives, any x86 timing of `a2d5e4e5`.
