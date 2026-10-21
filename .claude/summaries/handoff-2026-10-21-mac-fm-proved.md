# Handoff — 2026-10-21 (Mac) — the merged FM tree proved on macOS

## Base

- `design-2.0` at 705ada87 (origin, Volkan's push from Windows, fetched 15:16) plus this session's one docs commit,
  local only, not pushed. Tree clean after the commit.
- At 11:52 origin still had 694341fa: the Windows push landed between 11:52 and 15:16.

## Done

- Proved the merged tree (FM 0548796b, pb/FM-metal bc36375a, clang fix 2ef3e075) on macOS; no re-timing.
- Record: `.claude/baselines/2026-10-21-mac-gates-fm-merged.md` and its folder (logs, assembly, `asm/count.py`).
- PLAN's FM row: the Mac's gates added. The 10-12 handoff pruned.

## Verified by me

- `build/` (clang-macos): exit 0, 0 warnings. `ctest -j1`: `100% tests passed out of 95`, only 18
  `test_cuda_correctness` Skipped (MAY_SKIP). `test_metal_correctness` `All tests passed (21972 assertions in 17 test
  cases)`, `test_metal_mmap` `(10072 assertions in 6 test cases)`: ran, no skip. `test_codegen_no_calls`
  `inner_loops=63 calls=0 verdict=PASS`.
- `check_docs.py --cli`: `flags checked=447 … undocumented=0`, PASS; `check_pins.py` `failures=0`; `generate_docs.py
  --check` current.
- pytest, fresh venv in the scratchpad: `907 passed, 11 skipped, 1 warning`.
- `build-matlab/` rebuilt (MEX relinked 15:17:59, 0 warnings); R2026a Update 5: `matlab_suite: 142 run, 142 passed`.
- Apple clang 21's `calculate_medoids` float32 sum: scalar (`ldr s` → `fcvt` → `fadd d`, one accumulator), no lane
  ops, no sp access in the loop; float32 = float64 + one `fcvt` per pair, in `-S` (280 / 271) and in the ThinLTO
  `dtwc_cl` (279 / 270). Without the pragma the outlined bodies are identical (`diff`): on arm64 the cost model
  already declines; the pragma bites only where clang would gather (x86).

## Reported by agents, unverified

- None; no agents this session.

## Decisions

- Volkan: "Prove the merged tree on macOS, no re-timing"; record, PLAN's FM row, handoff, commit locally, no push.

## Next steps

1. Volkan: push design-2.0 (this commit).
2. PLAN MM: S1 (the reader) by a new implementer on `pb/MM` from design-2.0.
3. PLAN "Before the 2.0 tag": the docs audit's fixes (`baselines/2026-10-21-docs-audit.md`, on the Windows box).
4. FM's E1 when Volkan runs `C:/D/git/wt/tmp/FM/kit/profile_fm_etw.cmd` elevated; then FM's scratch and `wt/FM`.

## Open questions

- Unchanged from the Windows handoff: MM and HG in 2.0 or after; `ext4.vhdx`; the history purge; which `.claude`
  files to untrack.

## Status honesty

- Run: the two builds, serial ctest, the three docs gates, pytest, matlab_suite, the assembly and its counterfactual.
- Not run: timing, Arrow (no Arrow tree here), ASan, the conformance regen diff. Runs overlapped across separate
  trees (ctest with build-matlab's build and the wheel's; matlab_suite with pytest); counts, not times, are the evidence.
